"""Interleaved engine A/B: decode speed + output identity for the same IR under two venvs.

    python engine_ab.py <venv_a> <venv_b> <model_dir> [rounds=3] [device=GPU] [greedy|sampled] [max_new_tokens=200]

Runs A, B, A, B, … in fresh processes (one pipeline per process, its own CACHE_DIR
per venv so compile blobs never mix), greedy by default, a fixed ~200-token coding
prompt. `sampled` uses the fleet's structured decoding (temp 0.7 / top_p 0.8 /
top_k 20, seeded) — the sampler is a separate cost from decode and can regress on
its own (2026.4: halved tiny-model codegen throughput while greedy got faster).
Prints per-round tok/s and TTFT and whether the two engines produced identical
text. Exists because this box drifts ~30 % across a session (RESEARCH 13b): a
single-invocation comparison once "found" a regression that interleaving showed
was drift. Medians over adjacent pairs are the only trustworthy number.
"""
import json
import statistics
import subprocess
import sys
import time

CHILD = r'''
import json, sys, time, openvino_genai as g
venv, model, device, mode, ntok = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[5], int(sys.argv[6])
p = g.LLMPipeline(model, device, CACHE_DIR=sys.argv[4])
t = p.get_tokenizer()
c = g.GenerationConfig(); c.max_new_tokens = ntok; c.do_sample = False; c.ignore_eos = True
if mode == "sampled":
    c.do_sample = True; c.temperature = 0.7; c.top_p = 0.8; c.top_k = 20; c.rng_seed = 1
prompt = t.apply_chat_template([{"role": "user", "content":
    "Write a Python class `LRUCache` with get and put in O(1). Explain the approach briefly, then give the full code."}],
    add_generation_prompt=True)
p.generate(prompt, c)  # warm (kernels + cache), excluded
r = p.generate([prompt], c)
pm = r.perf_metrics
print(json.dumps({"engine": g.__version__, "tps": pm.get_throughput().mean, "ttft_ms": pm.get_ttft().mean,
                  "ntok": pm.get_num_generated_tokens(), "text": r.texts[0]}))
'''


def run(venv: str, model: str, device: str, mode: str, ntok: int) -> dict:
    py = f"{venv}\\Scripts\\python.exe"
    cache = f".ovcache-ab-{venv.strip('.').replace('/', '_')}"
    out = subprocess.run([py, "-c", CHILD, venv, model, device, cache, mode, str(ntok)], capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    line = [l for l in out.stdout.splitlines() if l.startswith("{")]
    if not line:
        raise SystemExit(f"{venv}: no result\n{out.stderr[-800:]}")
    return json.loads(line[-1])


def main() -> None:
    a, b, model = sys.argv[1], sys.argv[2], sys.argv[3]
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    device = sys.argv[5] if len(sys.argv) > 5 else "GPU"
    mode = sys.argv[6] if len(sys.argv) > 6 else "greedy"
    ntok = int(sys.argv[7]) if len(sys.argv) > 7 else 200   # long runs expose per-token cost growth with context
    print(f"model={model} device={device} rounds={rounds} mode={mode} max_new_tokens={ntok}  A={a}  B={b}", flush=True)
    res = {a: [], b: []}
    wins_b = 0
    for i in range(rounds):
        for v in (a, b):
            r = run(v, model, device, mode, ntok); res[v].append(r)
            print(f"  round {i+1} {'A' if v == a else 'B'} {r['engine'][:14]:14} {r['tps']:6.2f} tok/s  ttft {r['ttft_ms']:7.0f} ms  {r['ntok']} tok", flush=True)
        wins_b += res[b][-1]["tps"] > res[a][-1]["tps"]
    ma = statistics.median(x["tps"] for x in res[a]); mb = statistics.median(x["tps"] for x in res[b])
    same = all(x["text"] == y["text"] for x, y in zip(res[a], res[b]))
    print(f"\n  median decode: A {ma:.2f}  B {mb:.2f}  ({(mb/ma-1)*100:+.1f}%)  B won {wins_b}/{rounds} adjacent pairs")
    print(f"  median TTFT:   A {statistics.median(x['ttft_ms'] for x in res[a]):.0f} ms  B {statistics.median(x['ttft_ms'] for x in res[b]):.0f} ms")
    print(f"  {mode} output identical across engines: {same}")


if __name__ == "__main__":
    main()
