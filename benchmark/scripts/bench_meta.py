"""Shared benchmark metadata: per-model cards, IR quant provenance, engine
version, and the run-record writer. Single source of truth for "how to run a
model" (decoding + think + serving), consumed by the benchmark AND (via the
card schema) the production server.

Path rule: this file lives at <repo>/benchmark/scripts/bench_meta.py.
  BENCH_ROOT = <repo>/benchmark      (results, combos)
  REPO_ROOT  = <repo>                (models/, cards/)
Only engine_info() imports openvino; everything else is stdlib + PyYAML so the
assembler runs in any venv.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import time
import xml.etree.ElementTree as ET

import yaml

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
BENCH_ROOT = SCRIPTS_DIR.parent
REPO_ROOT = BENCH_ROOT.parent
MODELS_DIR = REPO_ROOT / "models"
CARDS_DIR = REPO_ROOT / "cards"
RESULTS_DIR = BENCH_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
COMBOS_FILE = BENCH_ROOT / "combos.yaml"

SCHEMA_VERSION = 1

# task type -> decoding class (which card.decoding.<class> block applies)
TASK_CLASS = {
    "codegen": "generative",     # open-ended -> sampling helps (rule 0f)
    "edit": "structured",        # exact edits -> greedy
    "autocomplete-fim": "fim",   # FIM -> greedy
    "agent-loop": "structured",  # tool discipline -> greedy
    "analysis": "structured",    # routing/diagnose/plan -> greedy (+think policy)
}
TASK_TYPES = list(TASK_CLASS)


# --------------------------------------------------------------------------- #
# Quant provenance — read straight from the IR rt_info (free, no model run).
# Recipe is derived from the rt_info FLAGS, never the dir name (a *-cw- dir can
# be AWQ+SE — confirmed on granite-4.1-3b-int4-cw-ov).
# --------------------------------------------------------------------------- #
def _ir_path(model_dir: pathlib.Path) -> pathlib.Path | None:
    for name in ("openvino_language_model.xml", "openvino_model.xml"):
        p = model_dir / name
        if p.exists():
            return p
    return None


def _coerce(v):
    if v is None:
        return None
    s = str(v).strip()
    if s in ("True", "true"):
        return True
    if s in ("False", "false"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _leaf(parent, tag):
    el = parent.find(tag) if parent is not None else None
    return _coerce(el.get("value")) if el is not None else None


def read_quant(model_dir) -> dict:
    """Parse the model-level <rt_info> for the quant recipe. Never raises."""
    model_dir = pathlib.Path(model_dir)
    out = {"recipe": "unknown", "ir_file": None}
    try:
        ir = _ir_path(model_dir)
        if ir is None:
            out["error"] = "no IR xml"
            return out
        out["ir_file"] = ir.name
        # the model-level rt_info is a direct child of the root <net>
        root = ET.parse(ir).getroot()
        rt = root.find("rt_info")
        if rt is None:
            out["error"] = "no rt_info"
            return out
        wc = rt.find("nncf/weight_compression")
        for k in ("mode", "group_size", "ratio", "awq", "gptq",
                  "scale_estimation", "lora_correction", "all_layers",
                  "backup_mode"):
            out[k] = _leaf(wc, k)
        ro = rt.find("runtime_options")
        out["activations_scale_factor"] = _leaf(ro, "ACTIVATIONS_SCALE_FACTOR")
        opt = rt.find("optimum")
        for k in ("nncf_version", "optimum_intel_version", "transformers_version"):
            out[k] = _leaf(opt, k)
        out["ir_runtime_version"] = _leaf(rt, "Runtime_version")
        out.update(_config_meta(model_dir))
        out["recipe"] = _recipe(out, model_dir)
    except Exception as e:  # noqa: BLE001 — provenance is best-effort
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def _config_meta(model_dir: pathlib.Path) -> dict:
    cfg = model_dir / "config.json"
    meta = {"model_type": None, "base_model": None, "quant_method": None}
    if cfg.exists():
        try:
            d = json.loads(cfg.read_text(encoding="utf-8"))
            meta["model_type"] = d.get("model_type")
            meta["base_model"] = (d.get("base_model")
                                  or (d.get("base_model_name_or_path")))
            qc = d.get("quantization_config") or {}
            meta["quant_method"] = qc.get("quant_method")
        except Exception:  # noqa: BLE001
            pass
    return meta


def _recipe(q: dict, model_dir: pathlib.Path) -> str:
    if q.get("quant_method") and q["quant_method"] not in ("nncf", None):
        return str(q["quant_method"])          # e.g. torchao
    if q.get("awq") and q.get("scale_estimation"):
        return "awq+se"
    if q.get("awq"):
        return "awq"
    if q.get("gptq"):
        return "gptq"
    if q.get("scale_estimation"):
        return "scale_estimation"
    if "-qat-" in model_dir.name.lower():
        return "qat"
    if q.get("mode"):
        return "data-free"
    return "unknown"


# --------------------------------------------------------------------------- #
# Engine version — guarded so the assembler runs without openvino installed.
# --------------------------------------------------------------------------- #
def engine_info() -> dict:
    info = {"name": "openvino-genai", "version": "unknown",
            "openvino_version": "unknown", "source": "unknown"}
    try:
        import openvino_genai as ov_genai  # noqa: PLC0415
        v = getattr(ov_genai, "__version__", None)
        if v:
            info["version"], info["source"] = v, "import"
    except Exception:  # noqa: BLE001
        pass
    if info["version"] == "unknown":
        try:
            import importlib.metadata as md  # noqa: PLC0415
            info["version"] = md.version("openvino-genai")
            info["source"] = "metadata"
        except Exception:  # noqa: BLE001
            pass
    try:
        import openvino  # noqa: PLC0415
        info["openvino_version"] = getattr(openvino, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        pass
    return info


# --------------------------------------------------------------------------- #
# Model id <-> dir <-> card.
# --------------------------------------------------------------------------- #
def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def resolve_model_dir(model_id: str) -> pathlib.Path:
    """owner/name -> models/owner/name. Also accepts an explicit path."""
    p = pathlib.Path(model_id)
    if p.exists() and (p / "config.json").exists():
        return p
    return MODELS_DIR / model_id


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


_DEFAULTS_CACHE = None


def _defaults() -> dict:
    global _DEFAULTS_CACHE
    if _DEFAULTS_CACHE is None:
        f = CARDS_DIR / "_defaults.yaml"
        _DEFAULTS_CACHE = yaml.safe_load(f.read_text(encoding="utf-8")) if f.exists() else {}
    return _DEFAULTS_CACHE


def card_path(model_id: str) -> pathlib.Path:
    return CARDS_DIR / f"{model_slug(model_id)}.yaml"


def load_card(model_id: str) -> dict:
    """The model's card merged onto its family + global defaults."""
    f = card_path(model_id)
    card = yaml.safe_load(f.read_text(encoding="utf-8")) if f.exists() else {}
    card = card or {}
    if "hf_id" not in card:
        card["hf_id"] = model_id
    d = _defaults()
    fam = card.get("family")
    merged_dec = dict(d.get("default", {}))
    if fam and fam in (d.get("families") or {}):
        merged_dec = _deep_merge(merged_dec, d["families"][fam])
    merged_dec = _deep_merge(merged_dec, card.get("decoding", {}))
    card["decoding"] = merged_dec
    card["think_by_task"] = _deep_merge(d.get("think_by_task", {}),
                                        card.get("think_by_task", {}))
    return card


def card_for(model_id: str, task_type: str) -> dict:
    """Resolve {decoding{...}, think} for a model on a given task type."""
    card = load_card(model_id)
    cls = TASK_CLASS.get(task_type, "generative")
    dec = dict(card["decoding"].get(cls, {}))
    if dec.get("greedy"):            # greedy replaces sampling params, not merges
        dec = {"greedy": True}
    blocks = card["decoding"].get("blocks", 1)
    think = card["think_by_task"].get(task_type, "nothink")
    # Per-model reasoning budget: reasoners (e.g. Ministral-3) need a large think
    # cap; default keeps every other model at the prior hardcoded value.
    think_max = int(card.get("think_max_tokens", 3072))
    return {"task_class": cls, "decoding": dec,
            "blocks": int(blocks), "think": think,
            "think_max_tokens": think_max}


def alias_to_hf(alias: str) -> str:
    """Resolve a card alias (or hf_id) to its hf_id via the cards dir."""
    for f in CARDS_DIR.glob("*.yaml"):
        if f.name.startswith("_"):
            continue
        try:
            c = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            continue
        if c.get("alias") == alias or c.get("hf_id") == alias:
            return c.get("hf_id", alias)
    return alias


def combo_info(name: str) -> dict:
    combos = (yaml.safe_load(COMBOS_FILE.read_text(encoding="utf-8"))
              if COMBOS_FILE.exists() else {}) or {}
    return (combos.get("combos") or {}).get(name, {})


# --------------------------------------------------------------------------- #
# Run-record: <task_type>__<slug>__<stamp>.jsonl  (header + cell rows).
# --------------------------------------------------------------------------- #
def build_run_header(*, target, task_type, suite, suite_budget, decoding,
                     think, model_dir=None, engine=None, driver=None,
                     subject=None, composition=None, notes=None,
                     stamp=None, confidence="high", warmup_seconds=None) -> dict:
    h = {
        "_type": "run_header", "schema_version": SCHEMA_VERSION,
        "run_id": f"{model_slug(target)}__{task_type}__{stamp or _stamp()}",
        "date": _date(stamp),
        "subject": subject or target,          # model id, or "combo:<name>"
        "target": target, "task_type": task_type, "suite": suite,
        "suite_budget": suite_budget,
        "engine": engine or engine_info(),
        "decoding": decoding, "think": think,
        "driver": driver, "notes": notes or [], "confidence": confidence,
    }
    if warmup_seconds is not None:
        # First-inference warmup (OV kernel JIT + prefix-cache prime, measured
        # ~60s on cold .ovcache). Untimed for scoring — recorded here so the
        # model's start cost is visible, separate from steady-state task times.
        h["warmup_seconds"] = round(float(warmup_seconds), 1)
    if composition is not None:
        h["composition"] = composition        # combo: {role: {model,quant,card}}
    else:
        md = pathlib.Path(model_dir) if model_dir else resolve_model_dir(target)
        # NB: no absolute path in the record — quant is read here; size is
        # recovered from the model id at assemble time. Keeps records portable.
        h["quant"] = read_quant(md)
    return h


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _date(stamp=None) -> str:
    return time.strftime("%Y-%m-%d")


_HOME = os.path.expanduser("~")
_USER_PATH_RE = re.compile(r'([A-Za-z]:\\Users\\|/Users/|/home/)[^\\/\s)\'"]+')


def redact_paths(text):
    """Strip the local username from absolute paths so it never reaches a
    committed artifact (standing rule: no local/username paths in git). Probe
    verdicts can embed exec tracebacks that include the local Python install
    path; this collapses the home dir to ~ and any Users/home path's username
    to <user>. Idempotent; non-strings pass through unchanged."""
    if not isinstance(text, str):
        return text
    for h in {_HOME, _HOME.replace("\\", "/")}:
        if h and h not in ("", "~"):
            text = text.replace(h, "~")
    return _USER_PATH_RE.sub(r"\1<user>", text)


def _redact_response(turns):
    """Normalize + path-redact captured model turns for storage. `turns` is a
    list of assistant message dicts; keep only the fields useful for offline
    grading/research (content, reasoning_content, tool_calls, finish_reason).
    Storing the raw output lets us re-grade or analyse without re-hitting the
    LLM — and see exactly what the model wrote (e.g. unclosed <think>)."""
    out = []
    for m in turns or []:
        rm = {}
        for k in ("content", "reasoning_content"):
            if m.get(k):
                rm[k] = redact_paths(m[k])
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or c
            rm.setdefault("tool_calls", []).append(
                {"name": fn.get("name"),
                 "arguments": redact_paths(str(fn.get("arguments", "")))})
        if m.get("finish_reason"):
            rm["finish_reason"] = m["finish_reason"]
        out.append(rm)
    return out


class RunWriter:
    """Writes one <task_type>__<slug>__<stamp>.jsonl atomically (.tmp->rename)."""

    def __init__(self, header: dict, out_dir: pathlib.Path = RUNS_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.header = header
        slug = model_slug(header["target"])
        stamp = header["run_id"].rsplit("__", 1)[-1] or _stamp()
        self.path = out_dir / f"{header['task_type']}__{slug}__{stamp}.jsonl"
        self._tmp = self.path.with_suffix(".jsonl.tmp")
        self._lines = [json.dumps(header)]

    def cell(self, cell_id, *, passed, verdict, seconds, blocks_used=1,
             extra=None, response=None):
        row = {"_type": "cell", "run_id": self.header["run_id"],
               "suite": self.header["suite"], "task_type": self.header["task_type"],
               "cell_id": cell_id,
               "quality": {"pass": bool(passed), "verdict": redact_paths(verdict)},
               "runtime": {"seconds": round(float(seconds), 1),
                           "blocks_used": blocks_used}}
        if response:
            row["response"] = _redact_response(response)
        if extra:
            row["extra"] = extra
        self._lines.append(json.dumps(row))

    def close(self) -> pathlib.Path:
        self._tmp.write_text("\n".join(self._lines) + "\n", encoding="utf-8")
        self._tmp.replace(self.path)
        return self.path
