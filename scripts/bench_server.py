"""Fair casting bench via the server — hits a model DIRECTLY by id (single-shot
chat, not virtual/agent), so every model gets identical nothink/template/engine
handling. Run against the server started under .venv-genai (the gemma4_unified
GenAI build), so Gemma-12B and the Qwen models share ONE engine = truly equal.

Same tasks + robust probe as the leaderboard. Greedy (server default), 3072.

Run:  python scripts/bench_server.py <owner/name>   (server must be up on :8000)
"""
import json
import os
import pathlib
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench_castings import TASKS, probe  # noqa: E402

BASE = "http://127.0.0.1:8000/v1"
MODEL = sys.argv[1]
MAX = int(os.environ.get("MAX_TOKENS", "3072"))
THINK = os.environ.get("THINK") not in (None, "", "0")  # reasoning_effort on
LABEL = MODEL.split("/")[-1] + ("-genai-think-3072" if THINK else "-genai-3072")
OUT = pathlib.Path(__file__).resolve().parent.parent / "bench_results" / "genai_server_castings.jsonl"


def ask(prompt: str) -> tuple:
    body = {"model": MODEL, "max_tokens": MAX,
            "messages": [{"role": "user", "content": prompt}]}
    if THINK:
        body["reasoning_effort"] = "medium"
    req = urllib.request.Request(f"{BASE}/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1200) as r:
        d = json.load(r)
    return (d["choices"][0]["message"].get("content") or "",
            round(time.perf_counter() - t0, 1))


if __name__ == "__main__":
    passes = cells = 0
    for tname, task in TASKS.items():
        for pi, prompt in enumerate(task["asks"]):
            try:
                content, dt = ask(prompt)
                verdict = probe(task, content)
            except Exception as e:  # noqa: BLE001
                content, dt, verdict = "", 0.0, f"FAIL (EXC: {type(e).__name__})"
            cells += 1
            passes += verdict == "PASS"
            row = {"model": LABEL, "task": tname, "phrasing": pi,
                   "probe": verdict, "seconds": dt}
            print(json.dumps(row), flush=True)
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    print(f"TOTAL {LABEL}: {passes}/{cells}", flush=True)
