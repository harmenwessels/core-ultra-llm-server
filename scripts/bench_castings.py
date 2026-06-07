"""Casting tournament cell runner: design tasks through virtual/agent.

Runs the no-tools design flow for each task x review={on,off} against the
currently-served casting, executes the returned code against probe tests,
and appends results to bench_results/castings.jsonl.

Usage: .venv/Scripts/python.exe scripts/bench_castings.py <casting-label>
"""

import json
import pathlib
import re
import sys
import time
import urllib.request

BASE = "http://localhost:8000/v1"
OUT = pathlib.Path(__file__).resolve().parent.parent / "bench_results" / "castings.jsonl"
CASTING = sys.argv[1]

TASKS = {
    "merge-intervals": {
        "ask": ("Design and implement a Python function "
                "merge_intervals(intervals) that merges overlapping closed "
                "intervals given as [start, end] lists and returns them "
                "sorted. Think about edge cases. Provide the complete "
                "function."),
        "tests": [
            ("merge_intervals([[1,3],[2,6],[8,10]])", [[1, 6], [8, 10]]),
            ("merge_intervals([[1,4],[4,5]])", [[1, 5]]),
            ("merge_intervals([])", []),
            ("merge_intervals([[5,7]])", [[5, 7]]),
        ],
        "fn": "merge_intervals",
    },
    "rate-limiter": {
        "ask": ("Design and implement a Python class SlidingWindowLimiter "
                "with __init__(self, max_calls, window_seconds) and a method "
                "allow(self, timestamp) -> bool that returns True and counts "
                "the call if fewer than max_calls happened in the trailing "
                "window_seconds before (and including) timestamp, else False "
                "without counting. Provide the complete class."),
        "tests": [
            ("_l(2,10).allow(1)", True),
            ("_seq()", [True, True, False, True]),
        ],
        "fn": "SlidingWindowLimiter",
        "harness": ("def _l(m, w):\n    return SlidingWindowLimiter(m, w)\n"
                    "def _seq():\n    l = SlidingWindowLimiter(2, 10)\n"
                    "    return [l.allow(1), l.allow(2), l.allow(3), "
                    "l.allow(20)]\n"),
    },
}


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return "\n\n".join(blocks) if blocks else text


def probe(task: dict, content: str) -> str:
    code = extract_code(content)
    ns: dict = {}
    try:
        exec(code, ns)  # noqa: S102 — our own benchmark task
        if task["fn"] not in ns:
            return "FAIL (missing definition)"
        if task.get("harness"):
            exec(task["harness"], ns)  # noqa: S102
        for expr, want in task["tests"]:
            got = eval(expr, ns)  # noqa: S307
            if got != want:
                return f"FAIL ({expr} -> {got!r})"
        return "PASS"
    except Exception as e:  # noqa: BLE001
        return f"FAIL ({type(e).__name__}: {e})"


def ask_virtual(prompt: str, review: bool) -> tuple[str, float]:
    body = {"model": "virtual/agent", "max_tokens": 2048, "review": review,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"{BASE}/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    return msg.get("content") or "", round(time.perf_counter() - t0, 1)


if __name__ == "__main__":
    for tname, task in TASKS.items():
        for review in (False, True):
            content, dt = ask_virtual(task["ask"], review)
            verdict = probe(task, content)
            row = {"casting": CASTING, "task": tname, "review": review,
                   "probe": verdict, "seconds": dt}
            print(json.dumps(row), flush=True)
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
