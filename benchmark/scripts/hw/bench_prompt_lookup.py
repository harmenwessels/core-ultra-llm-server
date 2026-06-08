r"""A/B benchmark: prompt-lookup decoding vs plain decoding on a code-edit prompt.

Prompt-lookup speculates upcoming tokens from n-grams already in the prompt and
verifies them in batched passes — output is identical to plain decoding, but
tokens that overlap the prompt cost a fraction of the bandwidth. Code-editing
prompts (where the model regenerates most of the input) are the best case.

Usage:
    .\.venv\Scripts\python.exe scripts\bench_prompt_lookup.py --model-dir models\Qwen2.5-Coder-1.5B-Instruct-int4-ov
"""

from __future__ import annotations

import argparse
import pathlib
import time

import openvino_genai as ov_genai

DEFAULT_CACHE = pathlib.Path(__file__).resolve().parent.parent / ".ovcache"

CODE = '''
def process_orders(orders, inventory, prices):
    total = 0
    shipped = []
    backordered = []
    for order in orders:
        item = order["item"]
        qty = order["qty"]
        if item not in inventory:
            backordered.append(order)
            continue
        if inventory[item] >= qty:
            inventory[item] -= qty
            cost = prices.get(item, 0) * qty
            total += cost
            shipped.append({"item": item, "qty": qty, "cost": cost})
        else:
            available = inventory[item]
            if available > 0:
                inventory[item] = 0
                cost = prices.get(item, 0) * available
                total += cost
                shipped.append({"item": item, "qty": available, "cost": cost})
                backordered.append({"item": item, "qty": qty - available})
            else:
                backordered.append(order)
    return total, shipped, backordered
'''

PROMPT = (
    "Here is a Python function:\n```python\n" + CODE + "\n```\n"
    "Rewrite this exact function with type hints added to the signature and a "
    "one-line docstring. Keep the logic and variable names identical. Output "
    "only the rewritten function in a python code block."
)


def run(pipe, cfg, label: str, runs: int = 3) -> str:
    """Benchmark using GenAI's own perf_metrics (correct under speculative
    decoding, where streamer callbacks deliver token batches). Returns the
    last output text so callers can verify both modes produce identical output."""
    results = []
    text = ""
    for i in range(runs + 1):  # first run is warm-up
        # list-form generate returns DecodedResults (with perf_metrics);
        # single-string form returns a bare str
        res = pipe.generate([PROMPT], generation_config=cfg)
        text = res.texts[0]
        pm = res.perf_metrics
        tps = pm.get_throughput().mean
        ttft = pm.get_ttft().mean / 1000.0
        ntok = pm.get_num_generated_tokens()
        if i == 0:
            print(f"{label} warm: {ntok:4d} tok  {tps:6.1f} tok/s (ignored)")
        else:
            results.append(tps)
            print(f"{label} run{i}: {ntok:4d} tok  TTFT={ttft:5.2f}s  {tps:6.1f} tok/s")
    med = sorted(results)[len(results) // 2]
    print(f"{label} median: {med:.1f} tok/s\n")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    base_kwargs = {"CACHE_DIR": str(DEFAULT_CACHE)}

    print("--- plain decoding ---")
    pipe = ov_genai.LLMPipeline(str(args.model_dir), "GPU", **base_kwargs)
    cfg = pipe.get_generation_config()
    cfg.max_new_tokens = args.max_new_tokens
    cfg.do_sample = False
    text_plain = run(pipe, cfg, "plain")
    del pipe

    print("--- prompt-lookup decoding ---")
    pipe = ov_genai.LLMPipeline(str(args.model_dir), "GPU",
                                prompt_lookup=True, **base_kwargs)
    cfg = pipe.get_generation_config()
    cfg.max_new_tokens = args.max_new_tokens
    cfg.do_sample = False
    cfg.num_assistant_tokens = 5
    cfg.max_ngram_size = 3
    text_pl = run(pipe, cfg, "PL   ")

    print("outputs identical:", text_plain == text_pl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
