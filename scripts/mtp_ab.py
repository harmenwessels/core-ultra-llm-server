"""MTP (multi-token-prediction) speculative-decoding A/B on one Qwen3.5 IR.

    python scripts/mtp_ab.py <model_dir> [rounds=3] [device=GPU] [greedy|sampled] [max_new_tokens=300]

OpenVINO GenAI 2026.4 runs Qwen3.5's own MTP head as the draft: the IR must carry
`openvino_mtp_model.xml` (optimum-intel >= 2.2.0 exports it when the checkpoint has
`mtp_num_hidden_layers > 0`; the MTP submodel needs `--group-size-fallback ignore` because its
rotary matmul has channel size 1). The entry point is **VLMPipeline** (Qwen3.5 IRs are
decomposed: language model + text embeddings; LLMPipeline has no `input_ids` port there) with
`draft_model=ov_genai.draft_model(SAME_DIR, device)`, a SchedulerConfig with
`enable_prefix_caching=False` (the linear-attention verifier refuses prefix caching) and
`num_assistant_tokens` on the GenerationConfig. Same discipline as engine_ab.py: fresh
process per arm, interleaved A (plain) / B (MTP), medians over adjacent pairs, because this
box drifts ~30 % across a session (RESEARCH 13b).
"""
import json
import statistics
import subprocess
import sys

CHILD = r'''
import json, sys, time, openvino_genai as g
model, device, mode, ntok, mtp, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5] == "mtp", int(sys.argv[6])
kw = {"CACHE_DIR": sys.argv[7]}
if mtp:
    sch = g.SchedulerConfig(); sch.enable_prefix_caching = False
    kw["draft_model"] = g.draft_model(model, device); kw["scheduler_config"] = sch
p = g.VLMPipeline(model, device, **kw)
t = p.get_tokenizer()
c = g.GenerationConfig(); c.max_new_tokens = ntok; c.do_sample = False
if mode == "sampled":
    c.do_sample = True; c.temperature = 0.6; c.top_p = 0.95; c.top_k = 20; c.rng_seed = 1
if mtp:
    c.num_assistant_tokens = k
prompt = t.apply_chat_template([{"role": "user", "content":
    "Write a Python class `LRUCache` with get and put in O(1). Explain the approach briefly, then give the full code."}],
    add_generation_prompt=True)
p.generate(prompt, generation_config=c)  # warm (kernels + cache), excluded
t0 = time.perf_counter(); r = p.generate(prompt, generation_config=c); wall = time.perf_counter() - t0
pm = r.perf_metrics
n = pm.get_num_generated_tokens()
print(json.dumps({"tps": pm.get_throughput().mean, "wall_tps": n / wall, "ttft_ms": pm.get_ttft().mean, "ntok": n, "text": r.texts[0]}))
'''


def run(model: str, device: str, mode: str, ntok: int, arm: str, k: int) -> dict:
    cache = ".ovcache-mtp-" + arm
    out = subprocess.run([sys.executable, "-c", CHILD, model, device, mode, str(ntok), arm, str(k), cache],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    line = [l for l in out.stdout.splitlines() if l.startswith("{")]
    if not line:
        raise SystemExit(f"{arm}: no result\n{out.stderr[-1500:]}")
    return json.loads(line[-1])


def main() -> None:
    model = sys.argv[1]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    device = sys.argv[3] if len(sys.argv) > 3 else "GPU"
    mode = sys.argv[4] if len(sys.argv) > 4 else "greedy"
    ntok = int(sys.argv[5]) if len(sys.argv) > 5 else 300
    k = int(sys.argv[6]) if len(sys.argv) > 6 else 1     # MTP head = 1 draft layer → 1 assistant token
    print(f"model={model} device={device} rounds={rounds} mode={mode} max_new_tokens={ntok} num_assistant_tokens={k}", flush=True)
    res = {"plain": [], "mtp": []}
    wins = 0
    for i in range(rounds):
        for arm in ("plain", "mtp"):
            r = run(model, device, mode, ntok, arm, k); res[arm].append(r)
            print(f"  round {i+1} {arm:5} {r['wall_tps']:6.2f} tok/s wall ({r['tps']:6.2f} reported)  ttft {r['ttft_ms']:6.0f} ms  {r['ntok']} tok", flush=True)
        wins += res["mtp"][-1]["wall_tps"] > res["plain"][-1]["wall_tps"]
    ma = statistics.median(x["wall_tps"] for x in res["plain"]); mb = statistics.median(x["wall_tps"] for x in res["mtp"])
    same = all(x["text"] == y["text"] for x, y in zip(res["plain"], res["mtp"]))
    print(f"\n  median wall tok/s: plain {ma:.2f}  mtp {mb:.2f}  ({(mb/ma-1)*100:+.1f}%)  mtp won {wins}/{rounds} adjacent pairs")
    print(f"  {mode} output identical plain vs mtp: {same}")


if __name__ == "__main__":
    main()
