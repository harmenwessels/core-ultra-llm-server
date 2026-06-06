"""Role-fitness benchmark: which local model fits which agent role?

Probes are distilled from observed agent-loop failures (see RESEARCH.md):
tool-call JSON discipline, tool selection/restraint, result usage, repeat
loops, edit precision (byte-exact old_string), full-file rewrites, stop
discipline, and routing classification. Each probe is objective pass/fail
with latency, run directly against the server API.

Run: .venv/Scripts/python.exe scripts/bench_roles.py [model_id ...]
     (default: all loaded models)
"""

import json
import pathlib
import sys
import time
import urllib.request

BASE = "http://localhost:8000/v1"
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "bench_results"

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file's content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace an exact existing substring of a file. "
                       "old_string must match the file content exactly.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old_string": {"type": "string"},
            "new_string": {"type": "string"}},
            "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the project's test suite.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "web_search", "description": "Search the web.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
]

STATS_PY = '''"""Small statistics helpers."""


def mean(values):
    """Arithmetic mean of a non-empty list."""
    if not values:
        raise ValueError("values must be non-empty")
    return sum(values) / len(values)


def moving_average(values, window):
    """Moving average; window must satisfy 1 <= window <= len(values)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return [sum(values[i:i + window]) / window
            for i in range(len(values) - window + 1)]
'''
# bug for edit-exact: missing the window > len(values) guard


EXTRA_BODY: dict = {}  # axis overrides (e.g. {"reasoning_effort": "high"})


def _chat(model, messages, tools=None, max_tokens=1024, timeout=600):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            **EXTRA_BODY}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    msg = data["choices"][0]["message"]
    return msg, time.perf_counter() - t0


def _calls(msg):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"]))
            for c in msg.get("tool_calls") or []]


# --- probes -------------------------------------------------------------


def probe_call_simple(model):
    msg, dt = _chat(model, [
        {"role": "user", "content": "Read the file config.yaml"}], TOOLS,
        max_tokens=128)
    calls = _calls(msg)
    ok = len(calls) == 1 and calls[0][0] == "read_file" \
        and calls[0][1].get("path", "").endswith("config.yaml")
    return ok, dt, f"calls={calls}"


def probe_call_choose(model):
    msg, dt = _chat(model, [
        {"role": "user", "content":
         "Find out what the latest stable Python version is."}], TOOLS,
        max_tokens=128)
    calls = _calls(msg)
    ok = len(calls) == 1 and calls[0][0] == "web_search"
    return ok, dt, f"calls={calls}"


def probe_call_restraint(model):
    msg, dt = _chat(model, [
        {"role": "user", "content": "What does the acronym API stand for?"}],
        TOOLS, max_tokens=128)
    calls = _calls(msg)
    content = msg.get("content") or ""
    ok = not calls and "application programming interface" in content.lower()
    return ok, dt, f"calls={calls} content={content[:60]!r}"


def probe_result_use(model):
    msg, dt = _chat(model, [
        {"role": "user", "content": "What value does MAX_RETRIES have in "
                                    "config.yaml?"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": "read_file",
                "arguments": "{\"path\": \"config.yaml\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": "timeout: 30\nMAX_RETRIES: 7\nlog_level: info"},
    ], TOOLS, max_tokens=128)
    calls = _calls(msg)
    ok = not calls and "7" in (msg.get("content") or "")
    return ok, dt, f"calls={calls} content={(msg.get('content') or '')[:60]!r}"


def probe_no_repeat(model):
    msg, dt = _chat(model, [
        {"role": "user", "content":
         "Fix the bug in stats.py: moving_average crashes on empty input. "
         "Read the file first, then fix it."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": "read_file",
                "arguments": "{\"path\": \"stats.py\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": STATS_PY},
    ], TOOLS)
    calls = _calls(msg)
    repeated = any(n == "read_file" and a.get("path") == "stats.py"
                   for n, a in calls)
    ok = bool(calls) and not repeated  # must act, must not re-read
    return ok, dt, f"calls={[(n, str(a)[:40]) for n, a in calls]}"


def probe_edit_exact(model):
    msg, dt = _chat(model, [
        {"role": "user", "content":
         "This is stats.py:\n```python\n" + STATS_PY + "```\n"
         "Bug: moving_average accepts window larger than len(values) and "
         "returns an empty list instead of raising ValueError. Fix it with "
         "a single edit_file call. old_string must match the file exactly."},
    ], TOOLS)
    calls = _calls(msg)
    edits = [(n, a) for n, a in calls if n == "edit_file"]
    if len(edits) != 1:
        return False, dt, f"edits={len(edits)} calls={[n for n, _ in calls]}"
    args = edits[0][1]
    old = args.get("old_string", "")
    ok = bool(old) and old in STATS_PY and bool(args.get("new_string"))
    if ok:  # the patched file must actually contain the new guard
        patched = STATS_PY.replace(old, args["new_string"])
        ns = {}
        try:
            exec(patched, ns)
            ns["moving_average"]([1, 2], 5)
            ok = False  # should have raised
        except ValueError:
            ok = True
        except Exception:
            ok = False
    return ok, dt, f"old-match={old in STATS_PY} old={old[:50]!r}"


def probe_write_full(model):
    msg, dt = _chat(model, [
        {"role": "user", "content":
         "This is stats.py:\n```python\n" + STATS_PY + "```\n"
         "Bug: moving_average accepts window larger than len(values). Fix it "
         "by writing the complete corrected file with one write_file call. "
         "Keep all existing functions."},
    ], TOOLS, max_tokens=2048)
    calls = _calls(msg)
    writes = [(n, a) for n, a in calls if n == "write_file"]
    if len(writes) != 1:
        return False, dt, f"writes={len(writes)} calls={[n for n, _ in calls]}"
    content = writes[0][1].get("content", "")
    ns = {}
    try:
        exec(content, ns)
        assert ns["mean"]([2, 4]) == 3.0           # untouched function intact
        assert ns["moving_average"]([1, 2, 3, 4, 5], 3) == [2.0, 3.0, 4.0]
        try:
            ns["moving_average"]([1, 2], 5)
            return False, dt, "no ValueError on window > len"
        except ValueError:
            pass
        try:
            ns["moving_average"]([1, 2], 0)
            return False, dt, "no ValueError on window < 1"
        except ValueError:
            pass
    except Exception as e:  # noqa: BLE001
        return False, dt, f"exec/assert failed: {e}"
    return True, dt, f"file={len(content)} chars"


def probe_stop_done(model):
    msg, dt = _chat(model, [
        {"role": "user", "content":
         "Fix stats.py so all tests pass, then confirm."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": "edit_file", "arguments": json.dumps({
                    "path": "stats.py",
                    "old_string": "if window < 1:",
                    "new_string": "if window < 1 or window > len(values):"})}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "edit applied"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_2", "type": "function", "function": {
                "name": "run_tests", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_2",
         "content": "7 passed in 0.05s"},
    ], TOOLS, max_tokens=256)
    calls = _calls(msg)
    ok = not calls and bool(msg.get("content"))
    return ok, dt, f"calls={[n for n, _ in calls]}"


ROUTE_CASES = [
    ("How does a Python decorator work?", "chat"),
    ("Rename the variable `cnt` to `count` in utils.py", "edit"),
    ("Design a caching layer for our API server", "design"),
    ("Why am I getting KeyError in this dict lookup?", "chat"),
    ("Add type hints to the parse_config function", "edit"),
    ("How should we structure the new plugin system?", "design"),
]


def probe_route(model):
    correct, total_dt = 0, 0.0
    details = []
    for text, want in ROUTE_CASES:
        msg, dt = _chat(model, [
            {"role": "system", "content":
             'Classify the user request. Reply with ONLY a JSON object: '
             '{"route": "chat"} for questions/explanations, '
             '{"route": "edit"} for direct code changes, '
             '{"route": "design"} for architecture/planning work.'},
            {"role": "user", "content": text},
        ], max_tokens=32)
        total_dt += dt
        content = (msg.get("content") or "").strip()
        try:
            got = json.loads(content[content.index("{"):
                                     content.rindex("}") + 1]).get("route")
        except Exception:  # noqa: BLE001
            got = None
        correct += got == want
        details.append(f"{want}:{got}")
    ok = correct == len(ROUTE_CASES)
    return ok, total_dt, f"{correct}/{len(ROUTE_CASES)} [{', '.join(details)}]"


# --- v2 probes: analysis quality + loop endurance -------------------------

BUGGY_MEDIAN = '''def median(values):
    """Median of a non-empty sorted-or-not list."""
    if not values:
        raise ValueError("values must be non-empty")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid] + s[mid + 1]) / 2   # even case
'''
PYTEST_FAIL = """FAILED test_stats.py::test_median_even - assert 3.5 == 2.5
 where median([4, 1, 3, 2]) returned 3.5 (expected 2.5)
========================= 1 failed, 2 passed in 0.04s"""


def probe_diagnose(model):
    msg, dt = _chat(model, [
        {"role": "user", "content":
         "This test fails:\n```\n" + PYTEST_FAIL + "\n```\n"
         "The code:\n```python\n" + BUGGY_MEDIAN + "```\n"
         "In one or two sentences: what exactly is wrong, and on which "
         "expression?"},
    ], max_tokens=1024)
    content = (msg.get("content") or "").lower()
    # ground truth: even branch averages s[mid] and s[mid+1] instead of
    # s[mid-1] and s[mid]
    names_indices = ("mid + 1" in content or "mid+1" in content
                     or "mid - 1" in content or "mid-1" in content)
    ok = names_indices and ("even" in content or "index" in content
                            or "indices" in content or "off" in content)
    return ok, dt, f"content={content[:100]!r}"


PLAN_CONTEXT = """Module overview (config.py):
- parse_config(path) -> dict : reads and validates the YAML config
- load_cache(key) -> bytes | None : local disk cache lookup
- fetch_remote(url, timeout) -> bytes : HTTP fetch, raises FetchError
"""


def probe_plan(model):
    msg, dt = _chat(model, [
        {"role": "user", "content":
         PLAN_CONTEXT + "\nWrite a short numbered implementation plan "
         "(3-6 steps) for adding an offline mode: when fetch_remote fails, "
         "fall back to the cache. Reference the actual functions involved. "
         "Plan only, no code."},
    ], max_tokens=1024)
    content = msg.get("content") or ""
    real = sum(s in content for s in ("fetch_remote", "load_cache",
                                      "parse_config", "FetchError"))
    numbered = sum(content.count(f"{i}.") > 0 for i in range(1, 7)) >= 3
    has_code = "```" in content and "def " in content
    ok = real >= 2 and numbered and not has_code
    return ok, dt, f"symbols={real} numbered={numbered} code={has_code}"


CHAIN_FILE = STATS_PY  # buggy: missing window > len(values) guard


def probe_chain_depth(model):
    """Scripted multi-turn agent loop with synthetic tool results.

    Stages the model must traverse: read -> fix (edit/write) -> test ->
    (tests fail once) -> fix again -> test -> stop after green.
    Fails on: identical repeated call, no tool call before done, >8 turns.
    """
    messages = [{"role": "user", "content":
                 "stats.py has a bug: moving_average accepts window larger "
                 "than len(values) and silently returns []. It must raise "
                 "ValueError. Fix it and verify with run_tests. Files can "
                 "only be inspected via tools."}]
    seen_calls: set = set()
    edits = 0
    tests_after_fix = 0
    total_dt = 0.0
    for turn in range(8):
        msg, dt = _chat(model, messages, TOOLS, max_tokens=2048)
        total_dt += dt
        calls = _calls(msg)
        if not calls:
            done_ok = edits >= 1 and tests_after_fix >= 1
            return done_ok, total_dt, (
                f"stopped at turn {turn}: edits={edits} "
                f"green-tests-seen={tests_after_fix}")
        tool_msgs = []
        for i, (name, args) in enumerate(calls):
            sig = f"{name}:{json.dumps(args, sort_keys=True)}"
            if sig in seen_calls:
                return False, total_dt, f"turn {turn}: repeated call {name}"
            seen_calls.add(sig)
            if name == "read_file":
                result = CHAIN_FILE
            elif name == "edit_file":
                if args.get("old_string", "") not in CHAIN_FILE:
                    result = "ERROR: old_string not found in file"
                else:
                    edits += 1
                    result = "edit applied"
            elif name == "write_file":
                edits += 1
                result = "file written"
            elif name == "run_tests":
                if edits == 0:
                    result = ("FAILED test_stats.py::test_invalid_window - "
                              "expected ValueError\n1 failed, 6 passed")
                else:
                    tests_after_fix += 1
                    result = "7 passed in 0.05s"
            else:
                result = "unsupported tool"
            tool_msgs.append((f"call_{turn}_{i}", name, args, result))
        messages.append({"role": "assistant", "content":
                         msg.get("content"), "tool_calls": [
            {"id": cid, "type": "function", "function": {
                "name": n, "arguments": json.dumps(a)}}
            for cid, n, a, _ in tool_msgs]})
        for cid, _, _, result in tool_msgs:
            messages.append({"role": "tool", "tool_call_id": cid,
                             "content": result})
    return False, total_dt, f">8 turns: edits={edits}"


FILLER_TURNS = [
    "Summarize what a REST API is.",
    "What is the difference between a list and a tuple in Python?",
    "Explain git rebase briefly.",
    "What does CI/CD stand for?",
]


def probe_recall_deep(model):
    messages = [
        {"role": "user", "content":
         "Remember this for later: our deployment API key lives in the "
         "environment variable FROBNICATE_KEY_77. Now, some questions."}]
    messages.append({"role": "assistant",
                     "content": "Noted — FROBNICATE_KEY_77. Ask away."})
    for q in FILLER_TURNS:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content":
                         "(answer omitted from transcript for brevity) " * 40})
    messages.append({"role": "user", "content":
                     "Which environment variable holds our deployment "
                     "API key?"})
    msg, dt = _chat(model, messages, max_tokens=64)
    content = msg.get("content") or ""
    ok = "FROBNICATE_KEY_77" in content
    return ok, dt, f"content={content[:80]!r}"


PROBES = [
    ("call-simple", probe_call_simple),
    ("call-choose", probe_call_choose),
    ("call-restraint", probe_call_restraint),
    ("result-use", probe_result_use),
    ("no-repeat", probe_no_repeat),
    ("edit-exact", probe_edit_exact),
    ("write-full", probe_write_full),
    ("stop-done", probe_stop_done),
    ("route", probe_route),
    # v2: analysis quality + loop endurance
    ("diagnose", probe_diagnose),
    ("plan", probe_plan),
    ("chain-depth", probe_chain_depth),
    ("recall-deep", probe_recall_deep),
]

THINK_AXIS = {"diagnose", "plan"}  # probes rerun with reasoning_effort=high


def main():
    global EXTRA_BODY
    args = sys.argv[1:]
    only = None
    think_axis = False
    if "--only" in args:
        i = args.index("--only")
        only = set(args[i + 1].split(","))
        del args[i:i + 2]
    if "--think" in args:
        think_axis = True
        args.remove("--think")
    with urllib.request.urlopen(f"{BASE}/models") as r:
        loaded = [m["id"] for m in json.load(r)["data"]]
    models = args or loaded
    probes = [(n, f) for n, f in PROBES if only is None or n in only]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results = {}
    for model in models:
        print(f"\n=== {model} ===")
        results[model] = {}
        for name, fn in probes:
            variants = [("", {})]
            if think_axis and name in THINK_AXIS:
                # thinking needs headroom: budget-exhausted-mid-thought
                # yields reasoning_content but empty content
                variants.append(("+think", {"reasoning_effort": "high",
                                            "max_tokens": 3072}))
            for suffix, extra in variants:
                EXTRA_BODY = extra
                try:
                    ok, dt, detail = fn(model)
                except Exception as e:  # noqa: BLE001
                    ok, dt, detail = False, 0.0, f"EXC: {e}"
                EXTRA_BODY = {}
                key = name + suffix
                results[model][key] = {"pass": ok, "seconds": round(dt, 1),
                                       "detail": detail}
                print(f"  {key:<17} {'PASS' if ok else 'FAIL':<5} "
                      f"{dt:6.1f}s  {detail[:90]}")
        passed = sum(r["pass"] for r in results[model].values())
        print(f"  -> {passed}/{len(results[model])}")
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"roles__{stamp}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
