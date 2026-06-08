"""Assemble the fair GenAI leaderboard from genai_server_castings.jsonl.

Every row was produced by bench_server.py hitting a model SOLO by id through the
one source-built gemma4_unified GenAI engine (nothink, greedy, 3072, robust
probe), so the comparison is truly equal. Ranks quality first (passes/cells),
then total wall-clock (quality matters most, then time-to-solve — tok/s is not
the metric since some models need more tokens for the same task).

Run: .venv-genai/Scripts/python.exe scripts/assemble_genai_leaderboard.py
"""
import collections
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "bench_results" / "genai_server_castings.jsonl"

agg = collections.defaultdict(lambda: {"pass": 0, "cells": 0, "sec": 0.0,
                                       "fails": []})
for line in SRC.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    a = agg[r["model"]]
    a["cells"] += 1
    a["sec"] += r.get("seconds", 0.0)
    if r["probe"] == "PASS":
        a["pass"] += 1
    else:
        a["fails"].append(f'{r["task"]}#{r["phrasing"]}: {r["probe"]}')

rows = []
for model, a in agg.items():
    rows.append((a["pass"], a["cells"], a["sec"], model, a["fails"]))
# quality desc, then total seconds asc
rows.sort(key=lambda x: (-x[0], x[2]))

print(f"{'model':<42} {'quality':>9} {'total_s':>9} {'avg_s':>7}")
print("-" * 72)
for passes, cells, sec, model, fails in rows:
    avg = sec / cells if cells else 0
    print(f"{model:<42} {passes:>4}/{cells:<4} {sec:>9.0f} {avg:>7.0f}")

print("\nfailures:")
for passes, cells, sec, model, fails in rows:
    if fails:
        print(f"  {model}:")
        for f in fails:
            print(f"    - {f}")
