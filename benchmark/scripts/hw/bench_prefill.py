"""Per-model prefill curve + decode-at-depth.

Measures, for each loaded model:
  - TTFT-proxy (time to a tiny 8-token completion) at ~1k/4k/8k/16k prompt
  - decode tok/s at shallow (1k) vs deep (12k) context, derived from the
    delta between an 8-token and a 136-token completion at the same prompt

Gives the per-role context budgets an orchestration layer should enforce.

Run: .venv/Scripts/python.exe scripts/bench_prefill.py [model_id ...]
"""

import json
import pathlib
import sys
import time
import urllib.request

BASE = "http://localhost:8000/v1"
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "bench_results"

FILLER = (
    "The quick brown fox jumps over the lazy dog while the seasoned engineer "
    "reviews pull requests, refactors legacy modules, and documents the build "
    "pipeline for the next release cycle of the platform. "
)


def _ask(model, approx_tokens, max_tokens, timeout=900):
    n = (approx_tokens * 4) // len(FILLER) + 1
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. "
             + FILLER * n},
            {"role": "user", "content": "Count from 1 upwards, one number "
             "per line, until told to stop."},
        ],
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        json.load(r)
    return time.perf_counter() - t0


def main():
    with urllib.request.urlopen(f"{BASE}/models") as r:
        loaded = [m["id"] for m in json.load(r)["data"]]
    models = sys.argv[1:] or loaded
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results = {}
    for model in models:
        print(f"\n=== {model} ===")
        results[model] = {"prefill": {}, "decode": {}}
        for k in (1000, 4000, 8000, 16000):
            try:
                dt = _ask(model, k, 8)
            except Exception as e:  # noqa: BLE001 — e.g. GPU mem ceiling
                results[model]["prefill"][k] = f"FAIL: {e}"
                print(f"  prefill ~{k:>5} tok: FAIL ({e})")
                break  # larger sizes will fail too
            results[model]["prefill"][k] = round(dt, 1)
            print(f"  prefill ~{k:>5} tok: {dt:6.1f}s")
        for label, k in (("shallow-1k", 1000), ("deep-12k", 12000)):
            try:
                t_short = _ask(model, k, 8)
                t_long = _ask(model, k, 136)
            except Exception as e:  # noqa: BLE001
                results[model]["decode"][label] = f"FAIL: {e}"
                print(f"  decode {label}: FAIL ({e})")
                continue
            tps = 128 / (t_long - t_short) if t_long > t_short else 0
            results[model]["decode"][label] = round(tps, 1)
            print(f"  decode {label}: {tps:5.1f} tok/s")
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"prefill__{stamp}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
