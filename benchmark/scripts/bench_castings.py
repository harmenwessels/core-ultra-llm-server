"""Casting tournament runner v2: breadth over repetition.

Greedy decoding makes identical-cell reruns byte-identical, so statistical
weight comes from task and phrasing breadth: 6 exec-probed design tasks x 2
phrasings per casting. Appends to bench_results/castings.jsonl.

Usage: .venv/Scripts/python.exe scripts/bench_castings.py <label> [--review]
           [--temp=0.6] [--top-p=0.95]

Default decoding is greedy (the tournament condition). --temp enables
sampling — use for card-advised parameter blocks; scores then carry
sampling variance on every serving path, not just the VLM one.
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

BASE = "http://localhost:8000/v1"
OUT = pathlib.Path(__file__).resolve().parent.parent / "bench_results" / "castings.jsonl"
CASTING = sys.argv[1] if len(sys.argv) > 1 else ""  # lazy: importable for reuse
REVIEW = "--review" in sys.argv
TEMP = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--temp=")), None)
TOP_P = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--top-p=")), None)
TOP_K = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--top-k=")), None)

TASKS = {
    "merge-intervals": {
        "asks": [
            ("Design and implement a Python function merge_intervals(intervals) "
             "that merges overlapping closed intervals given as [start, end] "
             "lists and returns them sorted. Think about edge cases. Provide "
             "the complete function."),
            ("I need a Python function: merge_intervals(intervals). Input is a "
             "list of [start, end] pairs (closed intervals, unsorted, may "
             "overlap). Return the merged, sorted list of intervals as lists. "
             "Plan it first, then give the full implementation."),
        ],
        "fn": "merge_intervals",
        "tests": [
            ("merge_intervals([[1,3],[2,6],[8,10]])", [[1, 6], [8, 10]]),
            ("merge_intervals([[1,4],[4,5]])", [[1, 5]]),
            ("merge_intervals([])", []),
        ],
    },
    "rate-limiter": {
        "asks": [
            ("Design and implement a Python class SlidingWindowLimiter with "
             "__init__(self, max_calls, window_seconds) and a method "
             "allow(self, timestamp) -> bool that returns True and counts the "
             "call if fewer than max_calls happened in the trailing "
             "window_seconds before (and including) timestamp, else False "
             "without counting. Provide the complete class."),
            ("Build a sliding-window rate limiter in Python: class "
             "SlidingWindowLimiter(max_calls, window_seconds), method "
             "allow(timestamp)->bool. A call at time T is allowed (and "
             "recorded) when strictly fewer than max_calls recorded calls lie "
             "in (T - window_seconds, T]. Disallowed calls are not recorded. "
             "Design first, then implement fully."),
        ],
        "fn": "SlidingWindowLimiter",
        "harness": ("def _seq():\n    l = SlidingWindowLimiter(2, 10)\n"
                    "    return [l.allow(1), l.allow(2), l.allow(3), "
                    "l.allow(20)]\n"),
        "tests": [("_seq()", [True, True, False, True])],
    },
    "lru-cache": {
        "asks": [
            ("Design and implement a Python class LRUCache with "
             "__init__(self, capacity), get(self, key) returning the value or "
             "-1, and put(self, key, value); evict the least-recently-used "
             "entry when capacity is exceeded. get counts as use. Provide the "
             "complete class."),
            ("I want an LRU cache in Python: LRUCache(capacity) with "
             "get(key)->value or -1 and put(key, value). Reads and writes both "
             "refresh recency; inserting beyond capacity evicts the least "
             "recently used key. Plan, then implement completely."),
        ],
        "fn": "LRUCache",
        "harness": ("def _lru():\n    c = LRUCache(2)\n    c.put(1, 1)\n"
                    "    c.put(2, 2)\n    a = c.get(1)\n    c.put(3, 3)\n"
                    "    return [a, c.get(2), c.get(3), c.get(1)]\n"),
        "tests": [("_lru()", [1, -1, 3, 1])],
    },
    "parse-duration": {
        "asks": [
            ("Design and implement a Python function parse_duration(s) that "
             "converts strings like '2h45m', '90s', '1h1s' into total seconds "
             "(int). Units: h, m, s; any subset, in that order; empty string "
             "gives 0. Provide the complete function."),
            ("Write parse_duration(s) in Python: turn duration strings such as "
             "'1h30m15s', '45m', '10s' into the total number of seconds as an "
             "integer. h/m/s units, each optional, ordered h then m then s; "
             "'' -> 0. Think it through, then give the full function."),
        ],
        "fn": "parse_duration",
        "tests": [
            ("parse_duration('2h45m')", 9900),
            ("parse_duration('90s')", 90),
            ("parse_duration('1h1s')", 3601),
            ("parse_duration('')", 0),
        ],
    },
    "rle-codec": {
        "asks": [
            ("Design and implement two Python functions: rle_encode(s) "
             "compressing runs of characters as char+count (e.g. 'aaabccc' -> "
             "'a3b1c3'), and rle_decode(s) reversing it. Assume input letters "
             "only. Provide both complete functions."),
            ("Implement run-length encoding in Python: rle_encode('aaabccc') "
             "should give 'a3b1c3' and rle_decode('a3b1c3') should return "
             "'aaabccc'. Letters-only input; empty string maps to empty "
             "string. Plan briefly, then write both functions in full."),
        ],
        "fn": "rle_encode",
        "tests": [
            ("rle_encode('aaabccc')", "a3b1c3"),
            ("rle_encode('')", ""),
            ("rle_decode('a3b1c3')", "aaabccc"),
            ("rle_decode(rle_encode('zzzzzzzzzzzz'))", "zzzzzzzzzzzz"),
        ],
    },
    "group-anagrams": {
        "asks": [
            ("Design and implement a Python function group_anagrams(words) "
             "that groups a list of lowercase words into lists of mutual "
             "anagrams and returns the groups. Provide the complete "
             "function."),
            ("Write group_anagrams(words) in Python: given lowercase strings, "
             "return a list of groups where each group contains words that "
             "are anagrams of each other. Order does not matter. Plan, then "
             "implement in full."),
        ],
        "fn": "group_anagrams",
        "harness": ("def _ga():\n    r = group_anagrams(['eat','tea','tan',"
                    "'ate','nat','bat'])\n    return sorted(sorted(g) for g "
                    "in r)\n"),
        "tests": [("_ga()", [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]])],
    },
}


def extract_code(text: str) -> str:  # kept for back-compat
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return "\n\n".join(blocks) if blocks else text


def _code_candidates(text: str) -> list:
    """Extraction strategies, in order — verbose models emit several ``` blocks
    (snippets + the final answer); joining them all yields invalid Python, which
    unfairly fails them. Try joined first (single-block / helper+main), then each
    block largest-first. PASS if ANY candidate runs the tests."""
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if not blocks:
        return [text]
    return ["\n\n".join(blocks)] + sorted(blocks, key=len, reverse=True)


# Exec-grading runs model-written code. Weak models occasionally emit a runaway
# (infinite loop / pathological recursion), which would hang an in-process exec
# forever. Grade each candidate in a throwaway subprocess with a hard timeout so
# a runaway is killed (TerminateProcess) and scored FAIL instead of wedging the
# whole sweep. Override the per-candidate budget with BENCH_EXEC_TIMEOUT seconds.
_EXEC_TIMEOUT = float(os.environ.get("BENCH_EXEC_TIMEOUT", "10"))


def _grade_candidate(code: str, task: dict, timeout: float) -> str:
    payload = json.dumps({"code": code, "fn": task["fn"],
                          "harness": task.get("harness", ""),
                          "tests": task["tests"]})
    try:
        r = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--grade-worker"], input=payload,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "FAIL (exec timeout)"
    out = (r.stdout or "").strip().splitlines()
    return out[-1] if out else f"FAIL (no verdict; rc={r.returncode})"


def probe(task: dict, content: str) -> str:
    """Grade in subprocess isolation; PASS if ANY extracted candidate passes."""
    last = "FAIL (no code)"
    for code in _code_candidates(content):
        verdict = _grade_candidate(code, task, _EXEC_TIMEOUT)
        if verdict == "PASS":
            return "PASS"
        last = verdict
    return last


def _grade_worker() -> int:
    """Subprocess entrypoint: exec one candidate + harness, run tests, print the
    verdict. Isolated so a runaway is killed by the parent's timeout."""
    p = json.load(sys.stdin)
    ns: dict = {}
    try:
        exec(p["code"], ns)  # noqa: S102 — our own benchmark task
        if p["fn"] not in ns:
            print("FAIL (missing definition)")
            return 0
        if p.get("harness"):
            exec(p["harness"], ns)  # noqa: S102
        for expr, want in p["tests"]:
            got = eval(expr, ns)  # noqa: S307
            if got != want:
                print(f"FAIL ({expr} -> {got!r})")
                return 0
        print("PASS")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL ({type(e).__name__}: {e})")
    return 0


_MAX_TOKENS = int(__import__("os").environ.get("MAX_TOKENS", "2048"))


def ask_virtual(prompt: str) -> tuple[str, float]:
    body = {"model": "virtual/agent", "max_tokens": _MAX_TOKENS, "review": REVIEW,
            "messages": [{"role": "user", "content": prompt}]}
    if TEMP is not None:
        body["temperature"] = TEMP
        if TOP_P is not None:
            body["top_p"] = TOP_P
        if TOP_K is not None:
            body["top_k"] = TOP_K
    req = urllib.request.Request(f"{BASE}/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return d["choices"][0]["message"].get("content") or "", \
        round(time.perf_counter() - t0, 1)


if __name__ == "__main__":
    if "--grade-worker" in sys.argv:
        sys.exit(_grade_worker())
    passes = 0
    cells = 0
    for tname, task in TASKS.items():
        for pi, ask in enumerate(task["asks"]):
            content, dt = ask_virtual(ask)
            verdict = probe(task, content)
            cells += 1
            passes += verdict == "PASS"
            row = {"casting": CASTING, "task": tname, "phrasing": pi,
                   "review": REVIEW, "probe": verdict, "seconds": dt}
            if TEMP is not None:
                row["temperature"] = TEMP
                row["top_p"] = TOP_P
                row["top_k"] = TOP_K
            print(json.dumps(row), flush=True)
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    print(f"TOTAL {CASTING}: {passes}/{cells}", flush=True)
