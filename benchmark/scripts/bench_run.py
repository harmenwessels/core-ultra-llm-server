"""Unified GenAI benchmark runner. Scores one target (a single model id OR
`virtual/agent` for a combo) across task types, using each model's CARD for
decoding + think policy, and writes provenance-tagged run-records.

  bench_run.py <target> --tasks codegen,edit,agent-loop,analysis,autocomplete-fim
  bench_run.py virtual/agent --combo small-trio --tasks codegen,analysis

Server must be up on :8000 (started under .venv-genai). Probes are reused as-is:
  codegen           -> bench_castings.TASKS + probe        (/v1/chat/completions)
  edit/agent/analysis -> bench_roles probe fns             (/v1/chat/completions, tools)
  autocomplete-fim  -> bench_workloads fim_prompt + probe  (/v1/completions, raw)
"""
import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bench_meta as bm          # noqa: E402
import bench_castings as bc      # noqa: E402
import bench_roles as br         # noqa: E402
from bench_workloads import fim_prompt, probe_autocomplete  # noqa: E402

BASE = "http://127.0.0.1:8000/v1"
CODEGEN_MAX = int(os.environ.get("CODEGEN_MAX", "3072"))

ROLE_TASKS = {
    "edit": ["edit-exact", "write-full"],
    "agent-loop": ["call-simple", "call-choose", "call-restraint", "result-use",
                   "no-repeat", "stop-done", "chain-depth"],
    "analysis": ["route", "diagnose", "plan", "recall-deep"],
}
_PROBE_BY_NAME = dict(br.PROBES)
SUITE = {"codegen": "castings-v3", "edit": "roles-edit-v2",
         "agent-loop": "roles-agent-v2", "analysis": "roles-analysis-v2",
         "autocomplete-fim": "fim-v3"}
ALL_TASKS = list(SUITE)


def _is_vlm(target: str) -> bool:
    if target == "virtual/agent":
        return True                       # composed -> treat as non-deterministic
    d = bm.resolve_model_dir(target)
    return (d / "openvino_vision_embeddings_model.xml").exists()


def _sampling(dec: dict) -> dict:
    if dec.get("greedy") or not dec:
        return {}
    out = {}
    for src, dst in (("temp", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if dec.get(src) is not None:
            out[dst] = dec[src]
    return out


def _blocks(cf: dict, is_vlm: bool) -> int:
    if cf["decoding"].get("greedy") and not is_vlm:
        return 1                          # greedy non-VLM is deterministic
    return cf["blocks"]


def _decoding_record(cf: dict) -> dict:
    dec = cf["decoding"]
    if dec.get("greedy") or not _sampling(dec):
        rec = {"strategy": "greedy"}
    else:
        rec = {"strategy": "sampling",
               **{k: dec.get(k) for k in ("temp", "top_p", "top_k")}}
    rec["blocks"] = cf["blocks"]
    rec["task_class"] = cf["task_class"]
    return rec


def _post(path: str, body: dict, timeout: int = 1800):
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d, round(time.perf_counter() - t0, 1)


def _chat(target, messages, max_tokens, sampling, think):
    body = {"model": target, "messages": messages, "max_tokens": max_tokens,
            **sampling}
    if think == "think":
        body["reasoning_effort"] = "high"
    d, dt = _post("/chat/completions", body)
    return (d["choices"][0]["message"].get("content") or ""), dt


def _header(target, task_type, combo, stamp):
    cf = bm.card_for(target, task_type)
    budget = {"max_tokens": CODEGEN_MAX if task_type == "codegen" else None,
              "blocks": cf["blocks"]}
    kw = dict(target=target, task_type=task_type, suite=SUITE[task_type],
              suite_budget=budget, decoding=_decoding_record(cf),
              think=cf["think"], driver="benchmark/scripts/bench_run.py",
              stamp=stamp)
    if target == "virtual/agent":
        info = bm.combo_info(combo) if combo else {}
        kw["composition"] = info.get("roles", {})
        kw["subject"] = f"combo:{combo}" if combo else "virtual/agent"
        kw["notes"] = ["composition decoding applies per-role server-side"]
    return bm.build_run_header(**kw), cf


# --------------------------------------------------------------------------- #
def run_codegen(target, combo, stamp, is_vlm):
    header, cf = _header(target, "codegen", combo, stamp)
    w = bm.RunWriter(header)
    sampling, think = _sampling(cf["decoding"]), cf["think"]
    nblocks = _blocks(cf, is_vlm)
    for tname, task in bc.TASKS.items():
        for pi, prompt in enumerate(task["asks"]):
            verdict, secs, used = "FAIL (no run)", 0.0, 0
            for _ in range(nblocks):
                used += 1
                try:
                    content, dt = _chat(target, [{"role": "user", "content": prompt}],
                                        CODEGEN_MAX, sampling, think)
                    v = bc.probe(task, content)
                except Exception as e:  # noqa: BLE001
                    dt, v = 0.0, f"FAIL (EXC: {type(e).__name__})"
                secs += dt
                verdict = v
                if v == "PASS":
                    break
            w.cell(f"{tname}#{pi}", passed=verdict == "PASS", verdict=verdict,
                   seconds=secs, blocks_used=used)
    return w.close()


def run_roles(target, task_type, combo, stamp, is_vlm):
    header, cf = _header(target, task_type, combo, stamp)
    w = bm.RunWriter(header)
    extra = _sampling(cf["decoding"])
    if cf["think"] == "think":
        extra = {**extra, "reasoning_effort": "high", "max_tokens": 3072}
    nblocks = _blocks(cf, is_vlm)
    for name in ROLE_TASKS[task_type]:
        fn = _PROBE_BY_NAME[name]
        passed, verdict, secs, used = False, "FAIL (no run)", 0.0, 0
        for _ in range(nblocks):
            used += 1
            br.EXTRA_BODY = dict(extra)
            try:
                ok, dt, detail = fn(target)
            except Exception as e:  # noqa: BLE001
                ok, dt, detail = False, 0.0, f"EXC: {e}"
            br.EXTRA_BODY = {}
            secs += dt
            passed, verdict = ok, ("PASS" if ok else f"FAIL ({detail[:70]})")
            if ok:
                break
        w.cell(name, passed=passed, verdict=verdict, seconds=secs, blocks_used=used)
    return w.close()


def run_fim(target, combo, stamp):
    if target == "virtual/agent":
        print("  autocomplete-fim: skipped (single-model role, not composed)")
        return None
    header, cf = _header(target, "autocomplete-fim", combo, stamp)
    w = bm.RunWriter(header)
    prompt, used_fim = fim_prompt(bm.resolve_model_dir(target))
    body = {"model": target, "prompt": prompt, "max_tokens": 96,
            **_sampling(cf["decoding"])}
    try:
        d, dt = _post("/completions", body, timeout=300)
        text = d["choices"][0].get("text") or ""
        verdict = probe_autocomplete(text, used_fim)
    except Exception as e:  # noqa: BLE001
        dt, verdict = 0.0, f"FAIL (EXC: {type(e).__name__})"
    w.cell("merge-fim", passed=verdict == "PASS", verdict=verdict, seconds=dt)
    return w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--combo", default=None)
    args = ap.parse_args()
    tasks = ALL_TASKS if args.tasks == "all" else args.tasks.split(",")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    is_vlm = _is_vlm(args.target)
    print(f"=== {args.target}{' ['+args.combo+']' if args.combo else ''} "
          f"(vlm={is_vlm}) tasks={tasks} ===", flush=True)
    for tt in tasks:
        t0 = time.perf_counter()
        if tt == "codegen":
            p = run_codegen(args.target, args.combo, stamp, is_vlm)
        elif tt in ROLE_TASKS:
            p = run_roles(args.target, tt, args.combo, stamp, is_vlm)
        elif tt == "autocomplete-fim":
            if is_vlm:
                print("  autocomplete-fim: skipped (VLM-shaped IR)")
                continue
            p = run_fim(args.target, args.combo, stamp)
        else:
            print(f"  unknown task type: {tt}")
            continue
        if p:
            rows = p.read_text(encoding="utf-8").splitlines()[1:]
            npass = sum(json.loads(r)["quality"]["pass"] for r in rows)
            print(f"  {tt:18} {npass}/{len(rows)}  ({time.perf_counter()-t0:.0f}s)  "
                  f"-> {p.name}", flush=True)


if __name__ == "__main__":
    main()
