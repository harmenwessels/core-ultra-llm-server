"""Casting tournament runner v2: breadth over repetition.

Greedy decoding makes identical-cell reruns byte-identical, so statistical
weight comes from task and phrasing breadth: 6 exec-probed design tasks x 2
phrasings per casting. Appends to bench_results/castings.jsonl.

Usage: .venv/Scripts/python.exe scripts/bench_castings.py <label> [--review]
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
REVIEW = "--review" in sys.argv

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


def ask_virtual(prompt: str) -> tuple[str, float]:
    body = {"model": "virtual/agent", "max_tokens": 2048, "review": REVIEW,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"{BASE}/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return d["choices"][0]["message"].get("content") or "", \
        round(time.perf_counter() - t0, 1)


if __name__ == "__main__":
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
            print(json.dumps(row), flush=True)
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    print(f"TOTAL {CASTING}: {passes}/{cells}", flush=True)
