"""KV-cache precision experiment: f16 (default) vs u8 on long-context prefill.

Loads the model standalone (no server, no co-residents) in each precision and
measures: prefill time at increasing context, whether the GPU memory ceiling
moves, and output equivalence on a short determinism check.

Run: .venv/Scripts/python.exe scripts/bench_kv_precision.py [model_dir]
"""

import json
import pathlib
import sys
import time

import openvino as ov
import openvino_genai as ov_genai

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "bench_results"
MODEL = sys.argv[1] if len(sys.argv) > 1 else str(
    ROOT / "models" / "HarmenWessels" / "granite-4.1-8b-int4-cw-ov")

FILLER = (
    "The quick brown fox jumps over the lazy dog while the seasoned engineer "
    "reviews pull requests, refactors legacy modules, and documents the build "
    "pipeline for the next release cycle of the platform. "
)
SIZES = (8000, 16000, 24000)


def run_variant(label: str, extra_props: dict) -> dict:
    print(f"\n--- {label} ---", flush=True)
    t0 = time.perf_counter()
    pipe = ov_genai.LLMPipeline(MODEL, "GPU",
                                CACHE_DIR=str(ROOT / ".ovcache"),
                                **extra_props)
    print(f"  loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    out: dict = {"prefill": {}, "check": None}
    cfg = pipe.get_generation_config()
    cfg.max_new_tokens = 8
    cfg.do_sample = False
    for k in SIZES:
        n = (k * 4) // len(FILLER) + 1
        prompt = FILLER * n + "\nReply with exactly: OK"
        t0 = time.perf_counter()
        try:
            pipe.generate(prompt, generation_config=cfg)
            dt = round(time.perf_counter() - t0, 1)
            out["prefill"][k] = dt
            print(f"  prefill ~{k:>5} tok: {dt:6.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001 — memory ceiling shows up here
            msg = str(e).strip().splitlines()[-1][:90]
            out["prefill"][k] = f"FAIL: {msg}"
            print(f"  prefill ~{k:>5} tok: FAIL ({msg})", flush=True)
            break
    # determinism / quality canary: short greedy completion
    cfg.max_new_tokens = 48
    out["check"] = pipe.generate(
        "Write a Python function that merges overlapping intervals.",
        generation_config=cfg)
    del pipe
    return out


if __name__ == "__main__":
    results = {"model": MODEL}
    results["f16 (default)"] = run_variant("f16 (default)", {})
    results["u8"] = run_variant(
        "u8", {"KV_CACHE_PRECISION": ov.Type.u8})
    same = results["f16 (default)"]["check"] == results["u8"]["check"]
    print(f"\ncanary outputs identical: {same}")
    results["canary_identical"] = same
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"kv_precision__{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {out}")
