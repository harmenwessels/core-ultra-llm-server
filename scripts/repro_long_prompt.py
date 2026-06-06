"""Repro: do long prompts hang the VLM pipeline?

Sends non-stream chat requests of increasing prompt size to each model with a
hard client timeout, reporting where things stop coming back.

Run: .venv/Scripts/python.exe scripts/repro_long_prompt.py [model_id ...]
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://localhost:8000/v1"

# ~4 chars/token filler paragraph, repeated to hit target token counts
FILLER = (
    "The quick brown fox jumps over the lazy dog while the seasoned engineer "
    "reviews pull requests, refactors legacy modules, and documents the build "
    "pipeline for the next release cycle of the platform. "
)


def probe(model: str, approx_tokens: int, timeout: int = 180) -> str:
    n = (approx_tokens * 4) // len(FILLER) + 1
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. " + FILLER * n},
            {"role": "user", "content": "Reply with exactly: OK"},
        ],
        "max_tokens": 8,
    }
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            json.load(r)
        return f"ok in {time.perf_counter() - t0:.1f}s"
    except TimeoutError:
        return f"TIMEOUT after {timeout}s"
    except urllib.error.URLError as e:
        if isinstance(getattr(e, "reason", None), TimeoutError):
            return f"TIMEOUT after {timeout}s"
        return f"error: {e}"


if __name__ == "__main__":
    models = sys.argv[1:] or [
        "HarmenWessels/gemma-4-E2B-it-qat-int4-ov",
        "OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov",
    ]
    for model in models:
        print(f"=== {model} ===")
        for k in (1000, 4000, 8000, 12000, 16000):
            print(f"  ~{k:>5} tok prompt: ", end="", flush=True)
            result = probe(model, k)
            print(result)
            if "TIMEOUT" in result:
                print("  (stopping this model — server gen thread likely wedged)")
                break

# extended probe: python scripts/repro_long_prompt.py --extend
