"""NPU inference optimization sweep on a probe-certified artifact.

Variants over the certified cw model (default: Coder-1.5B-cw):
  baseline       MAX_PROMPT_LEN=2048, library defaults
  best-perf      + GENERATE_HINT=BEST_PERF
  pyramid        + NPUW_LLM_PREFILL_ATTENTION_HINT=PYRAMID
  prefix-cache   + NPUW_LLM_ENABLE_PREFIX_CACHING=YES (warm-prefix TTFT test)
  all            every knob together
Each variant: load time, FIM x3 (warm), probe verdict, and for the
prefix-cache variants a repeated-2k-prefix TTFT pair.

Run: .venv/Scripts/python.exe scripts/bench_npu_opt.py [model_dir]
"""

import importlib.util
import json
import pathlib
import sys
import time

import openvino_genai as ov_genai

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "bench_results"
MODEL = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                     ROOT / "models" / "HarmenWessels" /
                     "Qwen2.5-Coder-1.5B-int4-cw-ov")

spec = importlib.util.spec_from_file_location(
    "bw", ROOT / "scripts" / "bench_workloads.py")
bw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bw)
FIM_PROMPT, USED_FIM = bw.fim_prompt(MODEL)

FILLER = ("# project context: utilities for parsing, caching and retrying "
          "HTTP fetches against the internal API gateway.\n") * 40  # ~2k tok

VARIANTS = {
    "baseline": {},
    "best-perf": {"GENERATE_HINT": "BEST_PERF"},
    "pyramid": {"NPUW_LLM_PREFILL_ATTENTION_HINT": "PYRAMID"},
    "prefix-cache": {"NPUW_LLM_ENABLE_PREFIX_CACHING": "YES"},
    "all": {"GENERATE_HINT": "BEST_PERF",
            "NPUW_LLM_PREFILL_ATTENTION_HINT": "PYRAMID",
            "NPUW_LLM_ENABLE_PREFIX_CACHING": "YES"},
}


def run_variant(name: str, extra: dict) -> dict:
    out: dict = {}
    try:
        t0 = time.perf_counter()
        pipe = ov_genai.LLMPipeline(str(MODEL), "NPU",
                                    MAX_PROMPT_LEN=4096, **extra)
        out["load_s"] = round(time.perf_counter() - t0, 0)
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e).strip().splitlines()[-1][:100]
        print(f"{name}: LOAD FAILED ({out['error']})", flush=True)
        return out
    cfg = pipe.get_generation_config()
    cfg.max_new_tokens = 96
    cfg.do_sample = False
    cfg.apply_chat_template = False
    pipe.generate(FIM_PROMPT, generation_config=cfg)  # warmup
    lat = []
    for _ in range(3):
        t0 = time.perf_counter()
        comp = pipe.generate(FIM_PROMPT, generation_config=cfg)
        lat.append(round(time.perf_counter() - t0, 2))
    out["fim_x3"] = lat
    out["probe"] = bw.probe_autocomplete(comp, USED_FIM)
    # warm-prefix TTFT pair: long shared prefix, different tails
    cfg.max_new_tokens = 8
    t0 = time.perf_counter()
    pipe.generate(FILLER + "def fetch_with_retry(url):\n",
                  generation_config=cfg)
    cold = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    pipe.generate(FILLER + "def parse_response(payload):\n",
                  generation_config=cfg)
    warm = round(time.perf_counter() - t0, 2)
    out["prefix_2k"] = {"cold": cold, "warm": warm}
    print(f"{name}: load {out['load_s']:.0f}s, FIM {lat}, "
          f"probe {out['probe']}, 2k-prefix cold {cold}s -> warm {warm}s",
          flush=True)
    del pipe
    return out


if __name__ == "__main__":
    results = {"model": str(MODEL)}
    for name, extra in VARIANTS.items():
        results[name] = run_variant(name, extra)
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"npu_opt__{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {out}")
