"""KV-cache precision experiment on long-context prefill.

On this GPU the plugin default (`KV_CACHE_PRECISION = dynamic`) RESOLVES to u8 —
verified by reading the property off the compiled granite-8b model. So u8 is the
production baseline, NOT f16; each variant below forces its precision explicitly
so the labels are honest (an empty-props run would silently re-test the u8 default).
int4 (u4) only earns its accuracy cost at very long context (>~32k tokens); it is
included here to measure that tradeoff, not because 8k workloads need it.

Loads the model standalone (no server, no co-residents) in each precision and
measures: prefill time at increasing context, whether the GPU memory ceiling
moves, and output equivalence on a short determinism check.

Run: .venv/Scripts/python.exe benchmark/scripts/hw/bench_kv_precision.py [model_dir]
"""

import json
import pathlib
import sys
import time

import openvino as ov
import openvino_genai as ov_genai

ROOT = pathlib.Path(__file__).resolve().parents[3]  # repo root (../../../ from hw/)
OUT_DIR = ROOT / "bench_results"  # gitignored; exploratory, not leaderboard data
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
    out: dict = {"prefill": {}, "check": None}
    t0 = time.perf_counter()
    try:  # a precision the GPU rejects (e.g. u4 block-size) must not kill the run
        pipe = ov_genai.LLMPipeline(MODEL, "GPU",
                                    CACHE_DIR=str(ROOT / ".ovcache"),
                                    **extra_props)
    except Exception as e:  # noqa: BLE001
        msg = str(e).strip().splitlines()[-1][:90]
        out["check"] = f"LOAD FAIL: {msg}"
        print(f"  load: FAIL ({msg})", flush=True)
        return out
    print(f"  loaded in {time.perf_counter() - t0:.0f}s", flush=True)
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
    try:
        out["check"] = pipe.generate(
            "Write a Python function that merges overlapping intervals.",
            generation_config=cfg)
    except Exception as e:  # noqa: BLE001 — same precision rejection as prefill
        msg = str(e).strip().splitlines()[-1][:90]
        out["check"] = f"FAIL: {msg}"
        print(f"  canary: FAIL ({msg})", flush=True)
    del pipe
    return out


if __name__ == "__main__":
    results = {"model": MODEL}
    # Force each precision explicitly — the default ({}) resolves to u8 on GPU,
    # so an empty-props run would mislabel u8 as the "f16 baseline".
    results["f16 (forced)"] = run_variant(
        "f16 (forced)", {"KV_CACHE_PRECISION": ov.Type.f16})
    results["u8 (default)"] = run_variant(
        "u8 (default)", {"KV_CACHE_PRECISION": ov.Type.u8})
    # u4 is rejected on this GPU build: BY_CHANNEL int4 key cache wants a paged-
    # attention block size of 12, but the plugin default is 16 and LLMPipeline
    # exposes no knob for it. Arm kept so it self-reports if a future driver fixes
    # the block size; today it records a FAIL and the run continues.
    results["u4 (int4)"] = run_variant(
        "u4 (int4)", {"KV_CACHE_PRECISION": ov.Type.u4})
    base = results["f16 (forced)"]["check"]
    results["canary_identical"] = {
        "u8": results["u8 (default)"]["check"] == base,
        "u4": results["u4 (int4)"]["check"] == base,
    }
    print(f"\ncanary vs f16: {results['canary_identical']}")
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"kv_precision__{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {out}")
