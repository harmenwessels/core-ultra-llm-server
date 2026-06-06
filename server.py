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


def _model_id(model_dir: pathlib.Path) -> str:
    """Model id mirrors the HF repo id: 'owner/name' for dirs under models/,
    plain folder name otherwise."""
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
_gen_lock = threading.Lock()  # single-flight generation across all models

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
# GenAI applies the model's chat template, which (for our models) has no tools
# support — so tools are injected hermes-style: definitions in the system
# message, calls expected as <tool_call>{json}</tool_call> blocks (the exact
# format Qwen2.5 was trained on; Gemma follows it from the instruction).

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


def _extract_tool_calls(text: str) -> tuple[str, list]:
    """Split generated text into (content, OpenAI tool_calls list)."""
    calls: list[dict] = []

    def _consume(m: re.Match) -> str:
        obj = _lenient_tool_json(m.group(1))
        if obj is None:
            return m.group(0)  # unrepairable: leave in content, agent will retry
        args = obj.get("arguments", {})
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
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
                "id": f"call_{uuid.uuid4().hex[:24]}",
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
    for model_dir in MODEL_DIRS:
        if not model_dir.exists():
            raise SystemExit(f"model dir not found: {model_dir}")
        is_vlm = (model_dir / "openvino_vision_embeddings_model.xml").exists()
        pipe_cls = ov_genai.VLMPipeline if is_vlm else ov_genai.LLMPipeline
        kwargs: dict = {"CACHE_DIR": str(CACHE_DIR)}
        model_id = _model_id(model_dir)
        use_pl = model_id in PROMPT_LOOKUP_MODELS and not is_vlm
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
        if pool_gb:
            sch = ov_genai.SchedulerConfig()
            sch.enable_prefix_caching = True
            sch.max_num_batched_tokens = 2048  # chunked prefill: clears the
            sch.cache_size = pool_gb           # 16k single-allocation wall
            kwargs["scheduler_config"] = sch
        log.info("Loading %s from %s on %s (cache: %s, prompt_lookup: %s, "
                 "prefix_caching: %s)", pipe_cls.__name__, model_dir, DEVICE,
                 CACHE_DIR, use_pl, f"{pool_gb}GB pool" if pool_gb else False)
        t0 = time.perf_counter()
        pipe = pipe_cls(str(model_dir), DEVICE, **kwargs)
        _pipes[model_id] = pipe
        _prompt_lookup_enabled[model_id] = use_pl
        try:
            variants = _derive_think_variants(pipe.get_tokenizer().chat_template)
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
        with _gen_lock:
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


@app.get("/v1/models")
def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": 0, "owned_by": "local"}
            for mid in _pipes
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages")
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    model_id, pipe = _resolve_pipe(body)
    tools = body.get("tools")
    use_tools = bool(tools) and body.get("tool_choice") != "none"
    if use_tools:
        messages = _inject_tools(messages, tools)
    history = _to_chat_history(messages)
    # tool-call JSON (file edits!) must not be truncated mid-arguments
    gen_cfg = _build_generation_config(
        pipe, body, default_max=4096 if use_tools else 1024, model_id=model_id)
    think_mode = _requested_think_mode(body)
    if not _think_variants.get(model_id):
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
            content, tool_calls = _extract_tool_calls(content)
            message["content"] = content or None
            if tool_calls:
                message["tool_calls"] = tool_calls
                finish = "tool_calls"
                log.info("[%s] tool calls: %s", model_id,
                         "; ".join(f"{c['function']['name']}("
                                   f"{c['function']['arguments'][:80]})"
                                   for c in tool_calls))
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        return message, finish

    if not body.get("stream", False):
        with _gen_lock:
            _apply_think_mode(model_id, pipe, think_mode)
            t0 = time.perf_counter()
            result = pipe.generate(history, generation_config=gen_cfg)
            dt = time.perf_counter() - t0
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
        with _gen_lock:
            result = pipe.generate(prompt, generation_config=gen_cfg)
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


if __name__ == "__main__":
    _load_pipelines()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
