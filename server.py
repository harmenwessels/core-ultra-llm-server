r"""Phase 3/5: lean OpenAI-compatible API server for OpenVINO GenAI on Intel Arc iGPU.

Serves one or more models behind the OpenAI surface Continue.dev needs:
  GET  /v1/models           -> all loaded model ids
  POST /v1/chat/completions -> chat (routed by the request's "model" field),
                               incl. OpenAI tool calling (hermes-style prompt
                               injection + <tool_call> output parsing) so
                               native-tool agent frontends (Kilo CLI/OpenCode)
                               work against local models
  POST /v1/completions      -> legacy/raw completions (Continue autocomplete FIM)

Design choices (see README):
  - Pipelines loaded ONCE at startup, compile cache enabled (CACHE_DIR).
  - Single-flight across ALL models: one generation at a time, serialized with
    a lock (single-user box; the iGPU is bandwidth-bound — concurrency only
    thrashes).
  - VLM-shaped IRs (Gemma 4) are auto-detected and served text-only through
    VLMPipeline; plain LLM IRs (Qwen) go through LLMPipeline.

Config via environment variables:
  MODEL_DIRS (";"-separated list; default: models/gemma-4-E2B-it-int4-ov;models/Qwen2.5-Coder-1.5B-Instruct-int4-ov)
  MODEL_DIR  (single-model override, kept for backward compat)
  DEVICE     (default: GPU)
  HOST       (default: 127.0.0.1)
  PORT       (default: 8000)
  CACHE_DIR  (default: ./.ovcache)
  SCHEDULER_MODELS ("model_id=GB" pairs, ";"-separated; default: granite-8b=4)
             Loads listed models with prefix caching + chunked prefill.
             Warm-prefix TTFT collapses ~60x (agent/chat-history turns);
             the GB value is a permanently reserved KV pool — budget it.
  PROMPT_LOOKUP_MODELS (";"-separated model ids; default: Qwen2.5-Coder-1.5B-Instruct-int4-ov)
             Enables prompt-lookup speculative decoding for the listed models.
             Measured +25% decode on FIM/code-edit workloads for the coder
             model, but it *hurts* general chat models (-20..-33%) — enable
             only for echo-faithful (FIM-trained) models. LLMPipeline only.

Run:
    .\.venv\Scripts\python.exe server.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import queue
import re
import threading
import time
import uuid

import openvino_genai as ov_genai
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = pathlib.Path(__file__).resolve().parent

_default_dirs = (
    f"{ROOT / 'models' / 'HarmenWessels' / 'gemma-4-E2B-it-qat-int4-ov'};"
    f"{ROOT / 'models' / 'OpenVINO' / 'Qwen2.5-Coder-1.5B-Instruct-int4-ov'}"
)
if os.environ.get("MODEL_DIR"):  # single-model override
    _default_dirs = os.environ["MODEL_DIR"]
MODEL_DIRS = [
    pathlib.Path(p) for p in os.environ.get("MODEL_DIRS", _default_dirs).split(";") if p
]
DEVICE = os.environ.get("DEVICE", "GPU")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
CACHE_DIR = pathlib.Path(os.environ.get("CACHE_DIR", ROOT / ".ovcache"))
MAX_NEW_TOKENS_CAP = int(os.environ.get("MAX_NEW_TOKENS_CAP", "8192"))
# Per-model device targeting ("model_id=NPU;..."). NPU models get their own
# generation lock, so autocomplete on NPU never queues behind GPU chat/agent
# turns (measured: ~2.1s FIM during a granite generation vs 30s+ queued).
# NPU requires channel-wise-sym int4 IRs and a per-device probe pass
# (RESEARCH.md) — serve only certified artifacts there.
MODEL_DEVICES: dict[str, str] = {}
for _entry in os.environ.get("MODEL_DEVICES", "").split(";"):
    if _entry:
        _name, _, _dev = _entry.partition("=")
        MODEL_DEVICES[_name] = (_dev or "GPU").upper()
NPU_MAX_PROMPT_LEN = int(os.environ.get("NPU_MAX_PROMPT_LEN", "4096"))
# Prefix caching + chunked prefill (measured: warm-prefix TTFT 63s -> 0.9s on
# granite-8b; clears the 16k single-allocation wall). Format: "model_id=GB;..."
# where GB is the reserved KV block pool size. The pool is held permanently —
# budget it against the iGPU ceiling alongside model weights.
SCHEDULER_MODELS: dict[str, int] = {}
for _entry in os.environ.get(
        "SCHEDULER_MODELS",
        "HarmenWessels/granite-4.1-8b-int4-cw-ov=4").split(";"):
    if _entry:
        _name, _, _gb = _entry.partition("=")
        SCHEDULER_MODELS[_name] = int(_gb or "4")
PROMPT_LOOKUP_MODELS = set(
    m for m in os.environ.get(
        "PROMPT_LOOKUP_MODELS", "OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov"
    ).split(";") if m
)


_ALIASES: dict[str, str] = {}          # resolved dir -> served alias (env path)
_REGISTRY_ALIASES: list[str] = []      # positional ids for models.yaml entries
_THINKING_POLICY: dict[str, str] = {}  # model id -> none | switchable
_TOOL_FORMAT_OVERRIDE: dict[str, str] = {}
_PROMPT_LEN_OVERRIDE: dict[str, int] = {}


def _model_id(model_dir: pathlib.Path) -> str:
    """Served id: the registry alias if set, else the HF-style 'owner/name'."""
    alias = _ALIASES.get(str(model_dir.resolve()))
    if alias:
        return alias
    try:
        return model_dir.resolve().relative_to((ROOT / "models").resolve()).as_posix()
    except ValueError:
        return model_dir.name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("server")

app = FastAPI(title="openvino-windows-openai-api")

_pipes: dict[str, object] = {}  # model id -> pipeline
_prompt_lookup_enabled: dict[str, bool] = {}  # model id -> PL active
_think_variants: dict[str, dict | None] = {}  # model id -> {think, nothink} templates
_think_mode: dict[str, str] = {}  # model id -> currently applied mode
_model_device: dict[str, str] = {}  # model id -> device it is served on
# single-flight generation PER DEVICE: GPU stays serialized (bandwidth-bound),
# but an NPU model generates concurrently with it
_device_locks: dict[str, threading.Lock] = {}


def _lock_for(model_id: str) -> threading.Lock:
    dev = _model_device.get(model_id, DEVICE)
    return _device_locks.setdefault(dev, threading.Lock())

_NOTHINK_PREFIX = "<think>\n\n</think>\n\n"
_THINK_PREFIX = "<think>\n"


def _derive_think_variants(template: str) -> dict | None:
    """For hybrid-thinking models, build think/nothink template variants.

    Handles the two patterns seen in the wild:
      A) our hardcoded no-think prefix (rt_info-patched artifacts)
      B) the vendor `enable_thinking` conditional (unusable through GenAI,
         which cannot pass template kwargs)
    Returns None for models without thinking support (e.g. Gemma).
    """
    if _NOTHINK_PREFIX in template:  # pattern A
        return {
            "nothink": template,
            "think": template.replace(_NOTHINK_PREFIX, _THINK_PREFIX),
        }
    if "enable_thinking" in template and "<think>" in template:  # pattern B
        cond = re.compile(
            r"\{%- if enable_thinking is defined.*?\{%- endif %\}", re.DOTALL)
        if cond.search(template):
            nothink = cond.sub("{{- '" + _NOTHINK_PREFIX.replace("\n", "\\n") + "' }}",
                               template)
            think = cond.sub("{{- '" + _THINK_PREFIX.replace("\n", "\\n") + "' }}",
                             template)
            return {"nothink": nothink, "think": think}
    if "enable_thinking" in template and "<|think|>" in template:  # pattern C:
        # Gemma 4 — thinking is gated on an enable_thinking kwarg GenAI cannot
        # pass; force the gate per variant instead
        gate = "enable_thinking is defined and enable_thinking"
        if gate in template:
            return {"nothink": template.replace(gate, "false"),
                    "think": template.replace(gate, "true")}
    return None


def _apply_think_mode(model_id: str, pipe, mode: str) -> None:
    """Swap the chat template if the requested mode differs. Call under _gen_lock."""
    variants = _think_variants.get(model_id)
    if not variants or _think_mode.get(model_id) == mode:
        return
    template = variants[mode]
    if hasattr(pipe, "set_chat_template"):  # VLMPipeline
        pipe.set_chat_template(template)
    else:  # LLMPipeline: via its tokenizer (verified to propagate)
        pipe.get_tokenizer().set_chat_template(template)
    _think_mode[model_id] = mode
    log.info("[%s] thinking mode -> %s", model_id, mode)


def _requested_think_mode(body: dict) -> str:
    """OpenAI-style: reasoning_effort 'none' (or absent) -> nothink; any other
    value, or enable_thinking=true, -> think."""
    if body.get("enable_thinking") is True:
        return "think"
    effort = body.get("reasoning_effort")
    if effort and str(effort).lower() != "none":
        return "think"
    return "nothink"


def _split_reasoning(text: str, think_mode: str = "nothink") -> tuple[str | None, str]:
    """Split '<think>...</think>' (or an unopened '...</think>') prefix into
    (reasoning_content, content). In think mode, an output that never closed
    its think block is all reasoning (budget ran out mid-thought)."""
    if "</think>" not in text:
        if think_mode == "think":
            return text.replace("<think>", "").strip("\n") or None, ""
        return None, text
    head, _, tail = text.partition("</think>")
    reasoning = head.replace("<think>", "").strip("\n")
    return (reasoning or None), tail.lstrip("\n")


# --- OpenAI tool calling on local models ------------------------------------
# Per-model-family tool language (RESEARCH.md findings 9 + Gemma/LFM addenda):
# models trained on a native function-calling protocol ignore instructed
# formats, so the server speaks each family's own language.
#   "native" families (gemma, lfm): the model's chat_template.jinja is
#     rendered SERVER-SIDE with jinja2 — passing tools / enable_thinking /
#     tool-role messages that GenAI's template application cannot — and
#     generation runs raw; emissions are parsed with a per-family parser.
#   "hermes" fallback (qwen & friends): definitions injected into the system
#     message, calls expected as <tool_call>{json}</tool_call> blocks.

_TOOLS_PROMPT = """

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments \
within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

_TOOL_FORMATS: dict[str, str] = {}  # model id -> gemma | lfm | hermes
_NATIVE_TEMPLATES: dict[str, object] = {}  # model id -> compiled jinja template


def _detect_tool_format(model_dir: pathlib.Path, model_id: str) -> str:
    """Pick the tool language by inspecting the model's own chat template
    (or the models.yaml override)."""
    tf = model_dir / "chat_template.jinja"
    if not tf.exists():
        return "hermes"
    template = tf.read_text(encoding="utf-8")
    fmt = _TOOL_FORMAT_OVERRIDE.get(model_id)
    if fmt is None:
        fmt = "hermes"
        if "declaration:" in template and "<|tool" in template:
            fmt = "gemma"
        elif "<|tool_call_start|>" in template or "List of tools:" in template:
            fmt = "lfm"
        elif re.search(r"\{%-?\s*(?:if|for)[^%]*tools", template):
            # template natively renders hermes-style tools (granite, qwen,
            # minicpm): render natively for faithful placement, parse hermes
            fmt = "native-hermes"
    if fmt != "hermes":
        try:
            import jinja2
            env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
            env.filters["tojson"] = lambda v, **kw: json.dumps(v)
            _NATIVE_TEMPLATES[model_id] = env.from_string(template)
        except Exception as e:  # noqa: BLE001 — fall back rather than break
            log.warning("%s: native template compile failed (%s) — hermes "
                        "fallback", model_id, str(e)[:80])
            return "hermes"
    return fmt


def _render_native(model_id: str, messages: list, tools: list | None,
                   think: bool) -> str:
    """Render the model's own template with everything GenAI cannot pass."""
    return _NATIVE_TEMPLATES[model_id].render(
        messages=messages, tools=tools or None, add_generation_prompt=True,
        enable_thinking=think, bos_token="")


def _coerce_arg(value: str):
    v = value.strip().strip('"').strip("'")
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


_GEMMA_CALL_RE = re.compile(r"call:(\w+)\{(.*?)\}(?:\s|$)", re.DOTALL)


def _parse_gemma_calls(text: str, tools: list, id_prefix: str
                       ) -> tuple[str, list]:
    """Parse 'call:name{key:value,...}' — quote tokens are stripped by the
    detokenizer, so argument values arrive bare; keys are schema-known."""
    known = {t["function"]["name"]: list(
        t["function"].get("parameters", {}).get("properties", {}))
        for t in tools}
    calls: list[dict] = []

    def _consume(m: re.Match) -> str:
        name, argstr = m.group(1), m.group(2)
        if name not in known:
            return m.group(0)
        keys = known[name]
        args: dict = {}
        # split on commas that precede a known key
        parts = re.split(
            r",(?=(?:" + "|".join(map(re.escape, keys)) + r"):)", argstr) \
            if keys else [argstr]
        for part in parts:
            k, sep, v = part.partition(":")
            if sep and k.strip() in keys:
                args[k.strip()] = _coerce_arg(v)
        calls.append({
            "id": f"{id_prefix}{uuid.uuid4().hex[:20]}",
            "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(args, ensure_ascii=False)},
        })
        return ""

    content = _GEMMA_CALL_RE.sub(_consume, text).strip()
    return content, calls


_LFM_CALL_RE = re.compile(
    r"(?:<\|tool_call_start\|>)?\[(\w+)\((.*?)\)\](?:<\|tool_call_end\|>)?",
    re.DOTALL)


def _parse_lfm_calls(text: str, tools: list, id_prefix: str
                     ) -> tuple[str, list]:
    """Parse LFM's Pythonic '[name(key="value", ...)]' calls."""
    known = {t["function"]["name"] for t in tools}
    calls: list[dict] = []

    def _consume(m: re.Match) -> str:
        name, argstr = m.group(1), m.group(2)
        if name not in known:
            return m.group(0)
        args = {k: _coerce_arg(v) for k, v in
                re.findall(r"(\w+)\s*=\s*(\"[^\"]*\"|'[^']*'|[^,]+)", argstr)}
        calls.append({
            "id": f"{id_prefix}{uuid.uuid4().hex[:20]}",
            "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(args, ensure_ascii=False)},
        })
        return ""

    content = _LFM_CALL_RE.sub(_consume, text).strip()
    content = content.replace("<|tool_call_start|>", "").replace(
        "<|tool_call_end|>", "").strip()
    return content, calls


_NATIVE_PARSERS = {
    "gemma": _parse_gemma_calls,
    "lfm": _parse_lfm_calls,
    "native-hermes": lambda text, tools, id_prefix:
        _extract_tool_calls(text, id_prefix),
}


def _lenient_tool_json(raw: str) -> dict | None:
    """Parse a tool-call JSON object, repairing the slips small models make.

    Observed in the wild (E2B, 5B-class): missing commas between pairs
    ('"a": 1 "b": 2'), trailing commas, and argument keys at the top level
    instead of nested under "arguments".
    """
    for attempt in (
        raw,
        re.sub(r"([\"\]}0-9e])\s+\"", r'\1, "', raw),  # missing commas
        re.sub(r",\s*([}\]])", r"\1", raw),            # trailing commas
        re.sub(r",\s*([}\]])", r"\1",
               re.sub(r"([\"\]}0-9e])\s+\"", r'\1, "', raw)),  # both
    ):
        try:
            obj = json.loads(attempt)
            break
        except json.JSONDecodeError:
            obj = None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    if "arguments" not in obj:  # args emitted at top level
        obj = {"name": obj["name"],
               "arguments": {k: v for k, v in obj.items() if k != "name"}}
    return obj


def _inject_tools(messages: list, tools: list) -> list:
    """Return a template-safe message list with tool definitions in the system
    message and tool-call/-result turns folded into plain assistant/user text."""
    defs = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if isinstance(content, list):  # OpenAI content-parts form
            content = "".join(
                p.get("text", "") for p in content if p.get("type") == "text")
        if role == "assistant" and m.get("tool_calls"):
            blocks = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                blocks.append("<tool_call>\n"
                              + json.dumps({"name": fn.get("name"),
                                            "arguments": json.loads(args)
                                            if isinstance(args, str) else args},
                                           ensure_ascii=False)
                              + "\n</tool_call>")
            content = (content + "\n" if content else "") + "\n".join(blocks)
        elif role == "tool":
            role = "user"
            content = f"<tool_response>\n{content}\n</tool_response>"
        if out and out[-1]["role"] == role:  # merge consecutive same-role turns
            out[-1]["content"] += "\n" + content  # (Gemma requires alternation)
        else:
            out.append({"role": role, "content": content})
    prompt = _TOOLS_PROMPT.format(tools=defs)
    if out and out[0]["role"] == "system":
        out[0]["content"] += prompt
    else:
        out.insert(0, {"role": "system", "content": prompt.lstrip("\n")})
    return out


def _extract_tool_calls(text: str, id_prefix: str = "call_") -> tuple[str, list]:
    """Split generated text into (content, OpenAI tool_calls list).

    id_prefix lets the virtual-model layer encode the active role into call
    ids ("call_arch_…"), which is how a tool-result continuation is routed
    back to the same brain statelessly.
    """
    calls: list[dict] = []

    def _consume(m: re.Match) -> str:
        obj = _lenient_tool_json(m.group(1))
        if obj is None:
            return m.group(0)  # unrepairable: leave in content, agent will retry
        args = obj.get("arguments", {})
        calls.append({
            "id": f"{id_prefix}{uuid.uuid4().hex[:20]}",
            "type": "function",
            "function": {
                "name": obj.get("name", ""),
                "arguments": args if isinstance(args, str)
                else json.dumps(args, ensure_ascii=False),
            },
        })
        return ""

    content = _TOOL_CALL_RE.sub(_consume, text).strip()
    if not calls:
        # fallback: smaller models (Coder-1.5B) emit the call JSON bare or in a
        # ```json fence, without the <tool_call> wrapper
        candidate = content
        fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
        if fence:
            candidate = fence.group(1)
        obj = _lenient_tool_json(candidate)
        # bare JSON is ambiguous — only treat as a call when the model clearly
        # meant one (explicit "arguments" key), unlike wrapped <tool_call> blocks
        if obj is not None and '"arguments"' in candidate:
            args = obj["arguments"]
            calls.append({
                "id": f"{id_prefix}{uuid.uuid4().hex[:20]}",
                "type": "function",
                "function": {
                    "name": obj["name"],
                    "arguments": args if isinstance(args, str)
                    else json.dumps(args, ensure_ascii=False),
                },
            })
            content = ""
    return content, calls


class QueueStreamer:
    """Decodes streamed tokens via ov TextStreamer into a thread-safe queue.

    StreamerBase.write receives raw token ids from VLMPipeline (and str pieces
    from LLMPipeline) — TextStreamer normalizes both to decoded text.
    """

    _DONE = object()

    def __init__(self, tokenizer: ov_genai.Tokenizer) -> None:
        self.q: queue.Queue = queue.Queue()
        self.token_count = 0
        self.first_token_time: float | None = None
        self.cancel = False  # set by the SSE generator when the client is gone
        self.text_streamer = ov_genai.TextStreamer(tokenizer, self._on_text)

    def _on_text(self, text: str) -> ov_genai.StreamingStatus:
        if self.cancel:
            return ov_genai.StreamingStatus.CANCEL
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self.token_count += 1
        if text:
            self.q.put(text)
        return ov_genai.StreamingStatus.RUNNING

    def finish(self) -> None:
        self.q.put(self._DONE)


def _load_pipelines() -> None:
    for idx, model_dir in enumerate(MODEL_DIRS):
        if not model_dir.exists():
            # registry entries may reference not-yet-downloaded/-converted
            # artifacts — serve what exists rather than refuse to start
            log.warning("model dir not found, skipping: %s", model_dir)
            continue
        is_vlm = (model_dir / "openvino_vision_embeddings_model.xml").exists()
        pipe_cls = ov_genai.VLMPipeline if is_vlm else ov_genai.LLMPipeline
        kwargs: dict = {"CACHE_DIR": str(CACHE_DIR)}
        model_id = _REGISTRY_ALIASES[idx] if idx < len(_REGISTRY_ALIASES) \
            else _model_id(model_dir)
        device = MODEL_DEVICES.get(model_id, DEVICE)
        if device == "NPU":
            kwargs["MAX_PROMPT_LEN"] = _PROMPT_LEN_OVERRIDE.get(
                model_id, NPU_MAX_PROMPT_LEN)
        use_pl = model_id in PROMPT_LOOKUP_MODELS and not is_vlm
        if device == "NPU" and use_pl:
            log.warning("%s: prompt lookup unsupported on NPU — ignoring",
                        model_id)
            use_pl = False
        if use_pl:
            kwargs["prompt_lookup"] = True
        if model_id in PROMPT_LOOKUP_MODELS and is_vlm:
            log.warning("%s: prompt lookup requested but unsupported on "
                        "VLM-shaped IRs — ignoring", model_id)
        pool_gb = SCHEDULER_MODELS.get(model_id)
        if pool_gb and use_pl:
            log.warning("%s: both prompt_lookup and scheduler requested — "
                        "keeping prompt_lookup, ignoring scheduler", model_id)
            pool_gb = None
        if pool_gb and device == "NPU":
            log.warning("%s: scheduler/prefix-caching config is GPU-path only "
                        "— NPU uses NPUW properties instead; ignoring", model_id)
            pool_gb = None
        if pool_gb:
            sch = ov_genai.SchedulerConfig()
            sch.enable_prefix_caching = True
            sch.max_num_batched_tokens = 2048  # chunked prefill: clears the
            sch.cache_size = pool_gb           # 16k single-allocation wall
            kwargs["scheduler_config"] = sch
        log.info("Loading %s from %s on %s (cache: %s, prompt_lookup: %s, "
                 "prefix_caching: %s)", pipe_cls.__name__, model_dir, device,
                 CACHE_DIR, use_pl, f"{pool_gb}GB pool" if pool_gb else False)
        t0 = time.perf_counter()
        pipe = pipe_cls(str(model_dir), device, **kwargs)
        _pipes[model_id] = pipe
        _model_device[model_id] = device
        _prompt_lookup_enabled[model_id] = use_pl
        _TOOL_FORMATS[model_id] = _detect_tool_format(model_dir, model_id)
        if _TOOL_FORMATS[model_id] != "hermes":
            log.info("%s: native tool language '%s' (server-side template "
                     "rendering)", model_id, _TOOL_FORMATS[model_id])
        try:
            if _THINKING_POLICY.get(model_id) == "none":
                variants = None  # registry says no thinking machinery
            else:
                variants = _derive_think_variants(
                    pipe.get_tokenizer().chat_template)
        except Exception:  # noqa: BLE001 — template introspection is best-effort
            variants = None
        _think_variants[model_id] = variants
        if variants:
            try:
                _apply_think_mode(model_id, pipe, "nothink")  # server default
                log.info("%s: hybrid-thinking model — per-request mode switch "
                         "enabled (default nothink)", model_id)
            except RuntimeError as e:  # e.g. VLM CB adapter: "Chat mode is not
                _think_variants[model_id] = None  # supported" -> serve as-shipped
                log.warning("%s: thinking-mode switching unsupported on this "
                            "pipeline (%s) — serving template as-shipped",
                            model_id, str(e).strip().splitlines()[-1])
        log.info("%s ready in %.1fs", model_id, time.perf_counter() - t0)


def _resolve_pipe(body: dict):
    """Pick the pipeline for the request's model field (default: first loaded)."""
    model_id = body.get("model")
    if not model_id:
        model_id = next(iter(_pipes))
    pipe = _pipes.get(model_id)
    if pipe is None:
        raise HTTPException(
            status_code=404,
            detail=f"model '{model_id}' not loaded; available: {list(_pipes)}",
        )
    return model_id, pipe


def _build_generation_config(pipe, body: dict, default_max: int = 1024,
                             model_id: str | None = None
                             ) -> ov_genai.GenerationConfig:
    cfg = pipe.get_generation_config()
    # cap runaway client values (agent frontends ask for model-context-sized
    # budgets; at ~20 tok/s that is a half-hour generation)
    cfg.max_new_tokens = min(int(
        body.get("max_completion_tokens") or body.get("max_tokens") or default_max
    ), MAX_NEW_TOKENS_CAP)
    if model_id and _prompt_lookup_enabled.get(model_id):
        # speculative prompt-lookup decoding (measured +25% on FIM workloads)
        cfg.num_assistant_tokens = 5
        cfg.max_ngram_size = 3
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    if temperature is not None and temperature > 0:
        cfg.do_sample = True
        cfg.temperature = float(temperature)
        if top_p is not None:
            cfg.top_p = float(top_p)
    else:
        cfg.do_sample = False
    stop = body.get("stop")
    if stop:
        cfg.stop_strings = set([stop] if isinstance(stop, str) else stop)
    return cfg


def _to_chat_history(messages: list) -> ov_genai.ChatHistory:
    """Map OpenAI messages to ov ChatHistory; flatten content-part lists to text."""
    history = []
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, list):  # OpenAI content-parts form
            content = "".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
        history.append({"role": m["role"], "content": content})
    return ov_genai.ChatHistory(history)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def _run_streaming(pipe, model_id: str, inputs, gen_cfg,
                   think_mode: str | None = None) -> QueueStreamer:
    """Start a generation thread that feeds a QueueStreamer; returns the streamer."""
    streamer = QueueStreamer(pipe.get_tokenizer())

    def _generate() -> None:
        with _lock_for(model_id):
            if think_mode is not None:
                _apply_think_mode(model_id, pipe, think_mode)
            t0 = time.perf_counter()
            try:
                pipe.generate(inputs, generation_config=gen_cfg,
                              streamer=streamer.text_streamer)
            except Exception:  # noqa: BLE001 — surface in log; SSE just ends
                log.exception("generation failed (%s)", model_id)
                return
            finally:
                streamer.finish()
            dt = time.perf_counter() - t0
            ttft = (streamer.first_token_time - t0) if streamer.first_token_time else 0
            decode = streamer.token_count - 1
            tps = decode / (dt - ttft) if dt > ttft and decode > 0 else 0
            log.info("[%s] stream done: %d tokens, TTFT %.2fs, %.1f tok/s on %s",
                     model_id, streamer.token_count, ttft, tps, DEVICE)

    threading.Thread(target=_generate, daemon=True).start()
    return streamer


# --- virtual model: per-turn role router ------------------------------------
# One model id (default virtual/agent) that routes each turn to the best
# measured brain (BENCHMARKS.md role-fitness): a router classifies fresh
# requests, the architect analyzes/plans (read-only tools), the executor
# does edit->test->verify loops (full tools). Stateless across requests:
# tool-result continuations are routed by the role encoded in our call ids;
# once a plan marker exists in history, turns go to the executor.

VIRTUAL_MODEL_ID = os.environ.get("VIRTUAL_MODEL", "virtual/agent")
VIRTUAL_ROLES: dict[str, str] = {}
for _entry in os.environ.get(
        "VIRTUAL_ROLES",
        "router=OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov;"
        "architect=Echo9Zulu/Qwen3.5-2B-int4_sym-ov;"
        "executor=HarmenWessels/granite-4.1-8b-int4-cw-ov").split(";"):
    if _entry:
        _role, _, _mid = _entry.partition("=")
        VIRTUAL_ROLES[_role] = _mid

_PLAN_MARKER = "<!--virtual:plan-->"
_READONLY_TOOLS = re.compile(
    r"read|grep|glob|search|fetch|list|view|cat|find|web", re.IGNORECASE)

_ROLE_PROMPTS = {
    "reviewer": (
        "You are the reviewer of a role-split coding agent. You are given a "
        "plan and an implementation. Check: does the implementation follow "
        "the plan, and is the code correct (logic, edge cases)? If it is "
        "acceptable, reply with exactly OK. Otherwise reply with a short "
        "numbered list of concrete defects to fix — no praise, no rewrite."),
    "architect": (
        "You are the analyst of a role-split coding agent. Investigate and "
        "diagnose using the available read-only tools, then produce a short "
        "numbered plan (3-6 steps) naming the exact files, functions and "
        "changes for an executor to implement. Plan only — no code, no "
        "file modifications."),
    "executor": (
        "You are the executor of a role-split coding agent. Implement the "
        "requested change using the available tools, then verify by running "
        "tests. Rules: edits must copy old_string EXACTLY from file content "
        "you have read in this conversation. A failing test is ground truth "
        "— fix the code, never argue with the test. Never repeat a tool call "
        "you already made with identical arguments. When tests pass, stop "
        "calling tools and summarize what changed."),
}


def _virtual_ready() -> bool:
    return all(mid in _pipes for mid in VIRTUAL_ROLES.values())


def _route_request(messages: list) -> str:
    """Classify the latest user request via the router brain."""
    last_user = next((m.get("content") or "" for m in reversed(messages)
                      if m.get("role") == "user"), "")
    if isinstance(last_user, list):
        last_user = "".join(p.get("text", "") for p in last_user
                            if p.get("type") == "text")
    router_id = VIRTUAL_ROLES["router"]
    pipe = _pipes[router_id]
    cfg = pipe.get_generation_config()
    cfg.max_new_tokens = 24
    cfg.do_sample = False
    if _prompt_lookup_enabled.get(router_id):  # PL pipelines require these
        cfg.num_assistant_tokens = 5
        cfg.max_ngram_size = 3
    prompt = ('Classify the user request. Reply with ONLY a JSON object: '
              '{"route": "chat"} for questions/explanations, '
              '{"route": "edit"} for direct code changes, '
              '{"route": "design"} for architecture/planning/multi-step work.'
              f"\n\nUser request: {last_user[:2000]}\nJSON:")
    with _lock_for(router_id):
        out = pipe.generate(prompt, generation_config=cfg)
    text = out.texts[0] if hasattr(out, "texts") else str(out)
    try:
        route = json.loads(text[text.index("{"):text.rindex("}") + 1]).get("route")
    except Exception:  # noqa: BLE001
        route = None
    return route if route in ("chat", "edit", "design") else "design"


def _detect_role(messages: list) -> tuple[str, str]:
    """Return (role, reason) for this turn — stateless reconstruction."""
    # 1) continuation of a tool round-trip -> same brain that asked
    for m in reversed(messages):
        if m.get("role") == "tool":
            cid = str(m.get("tool_call_id", ""))
            if cid.startswith("call_arch"):
                return "architect", "tool-continuation"
            if cid.startswith("call_exec"):
                return "executor", "tool-continuation"
            break
        if m.get("role") != "tool":
            break
    # 2) a plan already exists -> execution phase
    if any(_PLAN_MARKER in str(m.get("content") or "") for m in messages
           if m.get("role") == "assistant"):
        return "executor", "plan-exists"
    # 3) fresh request -> ask the router
    route = _route_request(messages)
    if route == "edit":
        return "executor", "routed:edit"
    return "architect", f"routed:{route}"


def _seen_file_content(messages: list) -> str:
    """Concatenated tool results — the file content the executor has seen."""
    return "\n".join(str(m.get("content") or "") for m in messages
                     if m.get("role") == "tool")


def _prior_call_signatures(messages: list) -> set:
    sigs = set()
    for m in messages:
        for tc in (m.get("tool_calls") or []) if m.get("role") == "assistant" else []:
            fn = tc.get("function", {})
            sigs.add(f"{fn.get('name')}:{fn.get('arguments')}")
    return sigs


def _virtual_guard(role: str, message: dict, messages: list) -> str | None:
    """Return a corrective note if the executor output violates a rule."""
    if role != "executor":
        return None
    calls = message.get("tool_calls") or []
    prior = _prior_call_signatures(messages)
    for tc in calls:
        fn = tc["function"]
        if f"{fn['name']}:{fn['arguments']}" in prior:
            return (f"You already called {fn['name']} with those exact "
                    "arguments — its result is above. Take the next step "
                    "instead of repeating it.")
        if fn["name"].lower() in ("edit_file", "edit", "str_replace"):
            try:
                old = json.loads(fn["arguments"]).get("old_string", "")
            except Exception:  # noqa: BLE001
                old = ""
            if old and old not in _seen_file_content(messages):
                return ("Your edit's old_string does not match any file "
                        "content you have read in this conversation. Re-read "
                        "the file and copy the exact current text.")
    return None


def _role_call(role: str, messages: list, tools: list | None, body: dict
               ) -> tuple[dict, str]:
    """One internal completion on the role's brain; returns (message, finish)."""
    model_id = VIRTUAL_ROLES[role]
    pipe = _pipes[model_id]
    msgs = list(messages)
    sys_prompt = _ROLE_PROMPTS[role]
    if msgs and msgs[0].get("role") == "system":
        msgs[0] = {"role": "system",
                   "content": f"{sys_prompt}\n\n{msgs[0].get('content') or ''}"}
    else:
        msgs.insert(0, {"role": "system", "content": sys_prompt})
    scoped = tools
    if tools and role == "architect":
        scoped = [t for t in tools
                  if _READONLY_TOOLS.search(t.get("function", {}).get("name", ""))]
    use_tools = bool(scoped)
    if use_tools:
        msgs = _inject_tools(msgs, scoped)
    history = _to_chat_history(msgs)
    gen_cfg = _build_generation_config(
        pipe, body, default_max=4096 if use_tools else 2048, model_id=model_id)
    with _lock_for(model_id):
        _apply_think_mode(model_id, pipe, "nothink")
        result = pipe.generate(history, generation_config=gen_cfg)
    text = result.texts[0] if hasattr(result, "texts") else str(result)
    _, content = _split_reasoning(text, "nothink")
    message: dict = {"role": "assistant", "content": content}
    finish = "stop"
    if use_tools:
        prefix = "call_arch_" if role == "architect" else "call_exec_"
        content, tool_calls = _extract_tool_calls(content, id_prefix=prefix)
        message["content"] = content or None
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish = "tool_calls"
    return message, finish


@app.get("/v1/models")
def list_models() -> dict:
    ids = list(_pipes)
    if _virtual_ready():
        ids.append(VIRTUAL_MODEL_ID)
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": 0, "owned_by": "local"}
            for mid in ids
        ],
    }


def _virtual_compute(body: dict) -> tuple[dict, str, str | None]:
    """All blocking work of a virtual turn (runs in a worker thread)."""
    messages = body["messages"]
    tools = body.get("tools")
    role, reason = _detect_role(messages)
    log.info("[%s] turn -> %s (%s)", VIRTUAL_MODEL_ID, role, reason)

    reasoning: str | None = None
    if not tools and role == "architect" and reason == "routed:design":
        # no-tools design request: plan (architect) then implement (executor)
        plan_msg, _ = _role_call("architect", messages, None, body)
        reasoning = plan_msg.get("content") or ""
        exec_msgs = messages + [{
            "role": "assistant", "content": f"{_PLAN_MARKER}\n{reasoning}"},
            {"role": "user", "content": "Implement the plan above."}]
        message, finish = _role_call("executor", exec_msgs, None, body)
        # review phase — OPT-IN per request ("review": true). Tournament
        # verdict (castings.jsonl): six reviewed cells, zero catches; LLM
        # review at this scale doesn't earn its latency — execution is the
        # only reviewer that works. ONE corrective pass max when enabled.
        want_review = bool(body.get("review", False))
        if want_review and VIRTUAL_ROLES.get("reviewer") in _pipes:
            review_msgs = [{"role": "user", "content":
                            f"PLAN:\n{reasoning}\n\nIMPLEMENTATION:\n"
                            f"{message.get('content') or ''}"}]
            verdict_msg, _ = _role_call("reviewer", review_msgs, None, body)
            verdict = (verdict_msg.get("content") or "").strip()
            if verdict and not verdict.upper().startswith("OK"):
                log.info("[%s] reviewer found issues -> one corrective pass",
                         VIRTUAL_MODEL_ID)
                fix_msgs = exec_msgs + [
                    {"role": "assistant",
                     "content": message.get("content") or ""},
                    {"role": "user", "content":
                     "A reviewer found these defects — fix them and output "
                     f"the corrected implementation:\n{verdict[:2000]}"}]
                message, finish = _role_call("executor", fix_msgs, None, body)
                reasoning = (reasoning or "") + \
                    f"\n\n[review]\n{verdict[:1200]}"
    else:
        message, finish = _role_call(role, messages, tools, body)
        note = _virtual_guard(role, message, messages)
        if note:
            log.info("[%s] guard tripped: %s", VIRTUAL_MODEL_ID, note[:60])
            retry_msgs = messages + [{"role": "user", "content": note}]
            message, finish = _role_call(role, retry_msgs, tools, body)
        if role == "architect" and tools and finish == "stop" \
                and message.get("content"):
            # final architect answer in an agentic flow = the plan; mark it so
            # the next turn routes to the executor
            message["content"] = f"{_PLAN_MARKER}\n{message['content']}"
    return message, finish, reasoning


async def _virtual_chat(body: dict):
    """The virtual model: route this turn to the best brain, guard, respond."""
    message, finish, reasoning = await asyncio.to_thread(_virtual_compute, body)

    comp_id = _new_id("chatcmpl")
    created = int(time.time())
    if reasoning:
        message["reasoning_content"] = reasoning

    if not body.get("stream", False):
        return JSONResponse({
            "id": comp_id, "object": "chat.completion", "created": created,
            "model": VIRTUAL_MODEL_ID,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": finish}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0},
        })

    def _sse():
        def chunk(delta: dict, fin: str | None = None) -> str:
            return "data: " + json.dumps({
                "id": comp_id, "object": "chat.completion.chunk",
                "created": created, "model": VIRTUAL_MODEL_ID,
                "choices": [{"index": 0, "delta": delta,
                             "finish_reason": fin}],
            }) + "\n\n"

        yield chunk({"role": "assistant", "content": ""})
        if message.get("reasoning_content"):
            yield chunk({"reasoning_content": message["reasoning_content"]})
        if message.get("content"):
            yield chunk({"content": message["content"]})
        for i, tc in enumerate(message.get("tool_calls") or []):
            yield chunk({"tool_calls": [{"index": i, "id": tc["id"],
                                         "type": "function",
                                         "function": tc["function"]}]})
        yield chunk({}, fin=finish)
        yield "data: [DONE]\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages")
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    if body.get("model") == VIRTUAL_MODEL_ID:
        if not _virtual_ready():
            raise HTTPException(status_code=503, detail=(
                f"virtual model needs all role models loaded: {VIRTUAL_ROLES}"))
        return await _virtual_chat(body)

    model_id, pipe = _resolve_pipe(body)
    tools = body.get("tools")
    use_tools = bool(tools) and body.get("tool_choice") != "none"
    think_mode = _requested_think_mode(body)
    native_fmt = _TOOL_FORMATS.get(model_id, "hermes") \
        if (use_tools or model_id in _NATIVE_TEMPLATES) else "hermes"
    if native_fmt != "hermes":
        # native tool language: render the model's OWN template server-side
        # (tools + enable_thinking + tool-role turns), generate raw
        history = _render_native(model_id, messages, tools if use_tools
                                 else None, think_mode == "think")
    else:
        if use_tools:
            messages = _inject_tools(messages, tools)
        history = _to_chat_history(messages)
    # tool-call JSON (file edits!) must not be truncated mid-arguments
    gen_cfg = _build_generation_config(
        pipe, body, default_max=4096 if use_tools else 1024, model_id=model_id)
    if native_fmt != "hermes":
        gen_cfg.apply_chat_template = False
        think_mode = "nothink"  # native render handled thinking via kwarg;
        # (gemma reasoning has no decodable end-delimiter to split on)
    elif not _think_variants.get(model_id):
        # model has no switchable thinking — never treat output as reasoning
        # (otherwise a no-</think> answer would be swallowed whole)
        think_mode = "nothink"
    comp_id = _new_id("chatcmpl")
    created = int(time.time())
    prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
    log.info("[%s] request: %d msgs, ~%d prompt chars, max_new=%d, stream=%s, "
             "tools=%d", model_id, len(messages), prompt_chars,
             gen_cfg.max_new_tokens, bool(body.get("stream")),
             len(tools or []))

    def _completion_message(text: str) -> tuple[dict, str]:
        """Build the response message dict + finish_reason from raw output."""
        reasoning, content = _split_reasoning(text, think_mode)
        message: dict = {"role": "assistant", "content": content}
        finish = "stop"
        if use_tools:
            if native_fmt != "hermes":
                content, tool_calls = _NATIVE_PARSERS[native_fmt](
                    content, tools, "call_")
            else:
                content, tool_calls = _extract_tool_calls(content)
            message["content"] = content or None
            if tool_calls:
                message["tool_calls"] = tool_calls
                finish = "tool_calls"
                log.info("[%s] tool calls (%s): %s", model_id, native_fmt,
                         "; ".join(f"{c['function']['name']}("
                                   f"{c['function']['arguments'][:80]})"
                                   for c in tool_calls))
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        return message, finish

    if not body.get("stream", False):
        def _blocking_generate():
            with _lock_for(model_id):
                _apply_think_mode(model_id, pipe, think_mode)
                t0 = time.perf_counter()
                res = pipe.generate(history, generation_config=gen_cfg)
                return res, time.perf_counter() - t0
        # off the event loop: a long generate must not freeze the server
        result, dt = await asyncio.to_thread(_blocking_generate)
        text = result.texts[0] if hasattr(result, "texts") else str(result)
        message, finish = _completion_message(text)
        log.info("[%s] non-stream done in %.2fs (mode=%s, finish=%s)",
                 model_id, dt, think_mode, finish)
        return JSONResponse({
            "id": comp_id,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish,
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    streamer = _run_streaming(pipe, model_id, history, gen_cfg, think_mode=think_mode)

    def _sse():
        def chunk(delta: dict, finish: str | None = None) -> str:
            return "data: " + json.dumps({
                "id": comp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }) + "\n\n"

        try:
            yield from _sse_body(chunk)
        finally:
            streamer.cancel = True  # client gone (or done): stop wasting GPU

    def _sse_body(chunk):
        yield chunk({"role": "assistant", "content": ""})

        if use_tools:
            # Tool-call blocks must be parsed whole — aggregate, then emit.
            # (Agent frontends act on the complete call anyway; correctness
            # over token-level streaming UX.) SSE comment pings during
            # aggregation keep the connection alive AND give uvicorn a yield
            # point to detect client disconnects (-> cancel the generation).
            parts: list[str] = []
            last_ping = time.perf_counter()
            while True:
                piece = streamer.q.get()
                if piece is QueueStreamer._DONE:
                    break
                parts.append(piece)
                if time.perf_counter() - last_ping > 2.0:
                    yield ": ping\n\n"
                    last_ping = time.perf_counter()
            message, finish = _completion_message("".join(parts))
            if message.get("reasoning_content"):
                yield chunk({"reasoning_content": message["reasoning_content"]})
            if message.get("content"):
                yield chunk({"content": message["content"]})
            for i, tc in enumerate(message.get("tool_calls") or []):
                yield chunk({"tool_calls": [{
                    "index": i,
                    "id": tc["id"],
                    "type": "function",
                    "function": tc["function"],
                }]})
            yield chunk({}, finish=finish)
            yield "data: [DONE]\n\n"
            return

        # In think mode, route the reasoning phase to delta.reasoning_content
        # (DeepSeek streaming convention) so clients never see raw think tags.
        in_reasoning = think_mode == "think"
        CLOSE = "</think>"
        buf = ""  # holds back text that might be a partial closing tag
        while True:
            piece = streamer.q.get()
            if piece is QueueStreamer._DONE:
                break
            if not in_reasoning:
                yield chunk({"content": piece})
                continue
            buf += piece.replace("<think>", "")
            if CLOSE in buf:
                reasoning, _, rest = buf.partition(CLOSE)
                if reasoning.strip("\n"):
                    yield chunk({"reasoning_content": reasoning.strip("\n")})
                in_reasoning = False
                buf = ""
                rest = rest.lstrip("\n")
                if rest:
                    yield chunk({"content": rest})
            else:
                # emit everything that cannot be the start of a partial CLOSE tag
                holdback = 0
                for k in range(min(len(CLOSE) - 1, len(buf)), 0, -1):
                    if buf.endswith(CLOSE[:k]):
                        holdback = k
                        break
                emit, buf = buf[:len(buf) - holdback], buf[len(buf) - holdback:]
                if emit:
                    yield chunk({"reasoning_content": emit})
        if buf and in_reasoning:  # budget exhausted mid-thought
            yield chunk({"reasoning_content": buf})
        yield chunk({}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")


@app.post("/v1/completions")
async def completions(request: Request):
    """Legacy completions: raw prompt continuation (Continue.dev autocomplete/FIM)."""
    body = await request.json()
    prompt = body.get("prompt")
    if prompt is None:
        raise HTTPException(status_code=400, detail="prompt is required")
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""

    model_id, pipe = _resolve_pipe(body)
    gen_cfg = _build_generation_config(pipe, body, default_max=256, model_id=model_id)
    # Raw continuation: do NOT wrap the prompt in the chat template (defaults to
    # True in openvino_genai and would break FIM autocomplete prompts).
    gen_cfg.apply_chat_template = False
    comp_id = _new_id("cmpl")
    created = int(time.time())

    if not body.get("stream", False):
        def _blocking_generate():
            with _lock_for(model_id):
                return pipe.generate(prompt, generation_config=gen_cfg)
        result = await asyncio.to_thread(_blocking_generate)
        text = result.texts[0] if hasattr(result, "texts") else str(result)
        return JSONResponse({
            "id": comp_id,
            "object": "text_completion",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    streamer = _run_streaming(pipe, model_id, prompt, gen_cfg)

    def _sse():
        def chunk(text: str, finish: str | None = None) -> str:
            return "data: " + json.dumps({
                "id": comp_id,
                "object": "text_completion",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "text": text, "finish_reason": finish}],
            }) + "\n\n"

        try:
            while True:
                piece = streamer.q.get()
                if piece is QueueStreamer._DONE:
                    break
                yield chunk(piece)
            yield chunk("", finish="stop")
            yield "data: [DONE]\n\n"
        finally:
            streamer.cancel = True  # client gone (or done): stop wasting GPU

    return StreamingResponse(_sse(), media_type="text/event-stream")


def _load_models_config() -> None:
    """models.yaml: the per-model registry (alias, device, scheduler pool,
    prompt-lookup, tool language, thinking policy, prompt-length, virtual
    roles). Explicit MODEL_DIRS/MODEL_DIR env keeps override power for
    quick experiments."""
    global MODEL_DIRS, VIRTUAL_MODEL_ID
    path = pathlib.Path(os.environ.get("MODELS_CONFIG", ROOT / "models.yaml"))
    if not path.exists():
        return
    if os.environ.get("MODEL_DIRS") or os.environ.get("MODEL_DIR"):
        log.info("models.yaml present but MODEL_DIRS env set — env wins")
        return
    import yaml
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    dirs: list[pathlib.Path] = []
    for m in cfg.get("models") or []:
        d = pathlib.Path(m["dir"])
        if not d.is_absolute():
            d = ROOT / d
        d = d.resolve()
        dirs.append(d)
        # one dir may be served multiple times under different aliases
        # (e.g. the same IR on GPU and CPU) — entries are positional
        mid = m.get("alias") or _model_id(d)
        _REGISTRY_ALIASES.append(mid)
        if m.get("device"):
            MODEL_DEVICES[mid] = str(m["device"]).upper()
        if (m.get("scheduler") or {}).get("kv_pool_gb"):
            SCHEDULER_MODELS[mid] = int(m["scheduler"]["kv_pool_gb"])
        if m.get("prompt_lookup"):
            PROMPT_LOOKUP_MODELS.add(mid)
        if m.get("tool_format"):
            _TOOL_FORMAT_OVERRIDE[mid] = str(m["tool_format"])
        if m.get("thinking"):
            _THINKING_POLICY[mid] = str(m["thinking"])
        if m.get("max_prompt_len"):
            _PROMPT_LEN_OVERRIDE[mid] = int(m["max_prompt_len"])
    if dirs:
        MODEL_DIRS = dirs
        # registry replaces (not merges with) the env defaults
        for k in list(SCHEDULER_MODELS):
            if k not in {_model_id(d) for d in dirs}:
                del SCHEDULER_MODELS[k]
    v = cfg.get("virtual") or {}
    if v.get("id"):
        VIRTUAL_MODEL_ID = v["id"]
    if v.get("roles"):
        VIRTUAL_ROLES.clear()
        VIRTUAL_ROLES.update({str(r): str(mid) for r, mid in v["roles"].items()})
    log.info("models.yaml: %d models, virtual=%s", len(dirs),
             VIRTUAL_MODEL_ID if VIRTUAL_ROLES else "off")


if __name__ == "__main__":
    _load_models_config()
    _load_pipelines()
    if not _pipes:
        raise SystemExit("no models could be loaded — check models.yaml / "
                         "MODEL_DIRS and download models first")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
