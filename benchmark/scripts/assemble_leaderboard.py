"""Assemble the benchmark leaderboard from provenance run-records.

Reads benchmark/results/runs/*.jsonl (the ONLY source — no legacy adapters),
enriches single models with their IR quant recipe + size, and regenerates:
  - per-task-type leaderboards (single models + combos as peer rows) + retest queue,
    written between <!--LEADERBOARD START/END--> markers in benchmark/README.md
  - a best-setup summary between <!--BEST-SETUP START/END--> markers in the root README
  - benchmark/results/leaderboard.json

  assemble_leaderboard.py            # write into the docs
  assemble_leaderboard.py --check    # print to stdout, touch nothing
"""
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bench_meta as bm  # noqa: E402

CHECK = "--check" in sys.argv
TASK_ORDER = ["codegen", "edit", "agent-loop", "analysis", "autocomplete-fim"]


def _size_gb(model_dir: pathlib.Path) -> float:
    try:
        return round(sum(p.stat().st_size for p in model_dir.glob("*.bin")) / 1e9, 1)
    except Exception:  # noqa: BLE001
        return 0.0


def _is_vlm(subject: str) -> bool:
    """VLM-shaped IRs legitimately skip autocomplete-fim — not a coverage gap."""
    try:
        return (bm.resolve_model_dir(subject)
                / "openvino_vision_embeddings_model.xml").exists()
    except Exception:  # noqa: BLE001
        return False


def load_fleet() -> list:
    """Canonical fleet (benchmark/fleet.txt) — lets us surface members that
    produced NO records (attempted-but-wedged) instead of silently dropping them."""
    f = bm.BENCH_ROOT / "fleet.txt"
    if not f.exists():
        return []
    return [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def load_runs():
    runs = []
    for f in sorted(bm.RUNS_DIR.glob("*.jsonl")):
        lines = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not lines or lines[0].get("_type") != "run_header":
            continue
        runs.append((lines[0], lines[1:]))
    return runs


def aggregate(runs):
    """Latest run per (task_type, subject). -> entry dicts."""
    by_key = {}
    for header, cells in runs:
        key = (header["task_type"], header["subject"])
        # run_id ends in the stamp; later stamp wins
        if key in by_key and by_key[key][0]["run_id"] >= header["run_id"]:
            continue
        by_key[key] = (header, cells)
    entries = collections.defaultdict(list)
    for (task_type, subject), (header, cells) in by_key.items():
        npass = sum(c["quality"]["pass"] for c in cells)
        total = len(cells)
        secs = round(sum(c["runtime"]["seconds"] for c in cells), 0)
        is_combo = "composition" in header
        e = {"subject": subject, "kind": "combo" if is_combo else "single",
             "quality": npass, "cells": total, "seconds": secs,
             "decoding": header["decoding"].get("strategy", "?"),
             "think": header.get("think", "nothink"),
             "engine": header["engine"]["version"],
             "fails": [c["cell_id"] + ": " + c["quality"]["verdict"]
                       for c in cells if not c["quality"]["pass"]]}
        if is_combo:
            e["detail"] = " / ".join(f"{r}={m}" for r, m in header["composition"].items())
        else:
            q = header.get("quant", {})
            e["detail"] = q.get("recipe", "?")
            e["recipe"] = q.get("recipe", "?")
            e["size"] = _size_gb(bm.resolve_model_dir(subject))  # from id, no stored path
        entries[task_type].append(e)
    return entries


def rank(entries_for_task):
    return sorted(entries_for_task, key=lambda e: (-e["quality"], e["seconds"]))


def render_overall(entries) -> str:
    """One row per model: passes + wall-clock summed across every task type,
    ranked by total passed (desc) then total time (asc). Single models link to
    their Hugging Face repo."""
    agg = {}
    for rows in entries.values():
        for e in rows:
            a = agg.setdefault(e["subject"], {
                "subject": e["subject"], "kind": e["kind"], "pass": 0,
                "cells": 0, "seconds": 0.0, "size": e.get("size"),
                "recipe": e.get("recipe"), "detail": e.get("detail")})
            a["pass"] += e["quality"]
            a["cells"] += e["cells"]
            a["seconds"] += e["seconds"]
    rows = sorted(agg.values(), key=lambda a: (-a["pass"], a["seconds"]))
    out = ["| # | Model | Passed | Total s | Size/Roles | Recipe |",
           "|---|---|---|---|---|---|"]
    for i, a in enumerate(rows, 1):
        if a["kind"] == "single":
            name = f"[{a['subject']}](https://huggingface.co/{a['subject']})"
            sr = f'{a.get("size", "?")} GB'
            rec = a.get("recipe") or "—"
        else:
            name, sr, rec = a["subject"], (a.get("detail") or "—"), "combo"
        out.append(f"| {i} | {name} | {a['pass']}/{a['cells']} | "
                   f"{a['seconds']:.0f} | {sr} | {rec} |")
    return "\n".join(out) + "\n"


def render_tables(entries) -> str:
    out = []
    for tt in TASK_ORDER:
        rows = rank(entries.get(tt, []))
        if not rows:
            continue
        out.append(f"### {tt}\n")
        out.append("| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for i, e in enumerate(rows, 1):
            avg = round(e["seconds"] / e["cells"], 0) if e["cells"] else 0
            sr = (f'{e.get("size","?")} GB' if e["kind"] == "single" else e["detail"])
            rec = e.get("recipe", "—") if e["kind"] == "single" else "combo"
            eng = (e["engine"] or "?").split("-")[0]
            out.append(f"| {i} | {e['subject']} | {e['kind']} | {sr} | "
                       f"{e['quality']}/{e['cells']} | {e['seconds']:.0f} | {avg:.0f} | "
                       f"{rec} | {e['decoding']} | {e['think']} | {eng} |")
        out.append("")
    return "\n".join(out)


def render_retest(entries, runs) -> str:
    versions = [h["engine"]["version"] for h, _ in runs if h["engine"]["version"] != "unknown"]
    newest = max(versions) if versions else None
    queued = []
    singles = {e["subject"] for rows in entries.values() for e in rows if e["kind"] == "single"}
    # fleet members with NO records — attempted-but-wedged or not yet run (surface,
    # don't silently drop)
    for m in load_fleet():
        if m not in singles:
            queued.append(f"{m}: NO records — attempted-but-incomplete or not yet run")
    # entries benched on an older engine than the newest seen
    for tt, rows in entries.items():
        for e in rows:
            if newest and e["engine"] != newest:
                queued.append(f"{e['subject']} / {tt}: engine {e['engine']} != newest {newest}")
    # coverage gaps: single models missing task types (autocomplete-fim is N/A for VLM IRs)
    for s in sorted(singles):
        have = {tt for tt, rows in entries.items() if any(e["subject"] == s for e in rows)}
        vlm = _is_vlm(s)
        miss = [t for t in TASK_ORDER if t not in have
                and not (t == "autocomplete-fim" and vlm)]
        if miss:
            queued.append(f"{s}: not yet run on {', '.join(miss)}")
    if not queued:
        return "_None — all entries current._\n"
    return "\n".join(f"- {q}" for q in queued) + "\n"


def _q(e):
    return f"{e['quality']}/{e['cells']}" if e else "—"


def _name(e):
    return e["subject"] if e else "—"


def render_summary(entries) -> str:
    out = ["| Task type | Best single | Q | Best combo | Q |",
           "|---|---|---|---|---|"]
    for tt in TASK_ORDER:
        rows = rank(entries.get(tt, []))
        if not rows:
            continue
        singles = [e for e in rows if e["kind"] == "single"]
        combos = [e for e in rows if e["kind"] == "combo"]
        bs = singles[0] if singles else None
        bc_ = combos[0] if combos else None
        out.append(f"| {tt} | {_name(bs)} | {_q(bs)} | {_name(bc_)} | {_q(bc_)} |")
    return "\n".join(out) + "\n"


def _splice(path: pathlib.Path, start: str, end: str, content: str):
    txt = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"{start}\n{content}\n{end}"
    if start in txt and end in txt:
        pre = txt[:txt.index(start)]
        post = txt[txt.index(end) + len(end):]
        path.write_text(pre + block + post, encoding="utf-8")
    else:
        sep = "\n\n" if txt and not txt.endswith("\n\n") else ""
        path.write_text(txt + sep + block + "\n", encoding="utf-8")


def main():
    runs = load_runs()
    entries = aggregate(runs)
    overall = render_overall(entries)
    tables = render_tables(entries)
    retest = render_retest(entries, runs)
    summary = render_summary(entries)
    failures = []
    for tt in TASK_ORDER:
        for e in rank(entries.get(tt, [])):
            if e["fails"]:
                failures.append(f"**{e['subject']} / {tt}**:\n" +
                                "\n".join(f"  - {x}" for x in e["fails"]))
    lb = (f"## Overall\n\nEvery tested model, passes and wall-clock summed across all "
          f"task types — ranked by total passed, then total time.\n\n{overall}\n"
          f"## Per-task-type leaderboard\n\n_{len(runs)} runs._\n\n{tables}\n"
          f"## Retest queue\n\n{retest}\n"
          + ("## Failures\n\n" + "\n\n".join(failures) + "\n" if failures else ""))

    if CHECK:
        print("===== benchmark/README.md leaderboard block =====\n")
        print(lb)
        print("===== root README best-setup block =====\n")
        print(summary)
        return
    _splice(bm.BENCH_ROOT / "README.md", "<!--LEADERBOARD START-->",
            "<!--LEADERBOARD END-->", lb)
    _splice(bm.REPO_ROOT / "README.md", "<!--BEST-SETUP START-->",
            "<!--BEST-SETUP END-->", summary)
    (bm.RESULTS_DIR / "leaderboard.json").write_text(
        json.dumps({tt: rank(rows) for tt, rows in entries.items()}, indent=2),
        encoding="utf-8")
    print(f"assembled {len(runs)} runs -> benchmark/README.md + root README + leaderboard.json")


if __name__ == "__main__":
    main()
