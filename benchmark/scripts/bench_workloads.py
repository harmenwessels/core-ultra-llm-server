r"""Workload-representative benchmark: autocomplete / assistant / architect.

Models serve different phases of software work, with different prompt shapes:

  autocomplete       FIM or raw continuation, no chat template, short output —
                     what matters is total latency per completion
  assistant-edit     chat-templated "rewrite this exact function" (echo-heavy)
  assistant-explain  chat-templated "explain this code" (medium echo)
  architect          chat-templated design/trade-off question, no code in the
                     prompt (pure generation)

For LLM-shaped models the suite optionally repeats the three chat profiles with
prompt-lookup decoding (--pl) to show where speculation helps and where it
hurts. VLM-shaped IRs (Gemma 4, Qwen3.5, ...) run plain only.

Usage:
    .\.venv\Scripts\python.exe scripts\bench_workloads.py --model-dir models\OpenVINO\Qwen2.5-Coder-1.5B-Instruct-int4-ov --pl
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import re
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

AUTOCOMPLETE_PREFIX = (
    "def merge_sorted_lists(a: list, b: list) -> list:\n"
    '    """Merge two sorted lists into one sorted list."""\n'
    "    result = []\n"
    "    i = j = 0\n"
    "    while i < len(a) and j < len(b):\n"
)
AUTOCOMPLETE_SUFFIX = "\n    return result\n"

PROFILES = {
    "assistant-edit": {
        "prompt": ("Here is a Python function:\n```python\n" + CODE + "\n```\n"
                   "Rewrite this exact function with type hints added to the "
                   "signature and a one-line docstring. Keep the logic and "
                   "variable names identical. Output only the rewritten "
                   "function in a python code block."),
        "max_new": 512,
        "chat": True,
    },
    "assistant-explain": {
        "prompt": ("Here is a Python function:\n```python\n" + CODE + "\n```\n"
                   "Explain step by step what this function does, including "
                   "the edge cases it handles and one potential bug or "
                   "improvement."),
        "max_new": 256,
        "chat": True,
    },
    "architect": {
        "prompt": ("We are designing an order-fulfilment service for a "
                   "mid-size webshop: ~50k orders/day, items in multiple "
                   "warehouses, partial shipments allowed. Propose a service "
                   "architecture: components, data model, APIs between them, "
                   "and the two most important trade-offs you considered."),
        "max_new": 512,
        "chat": True,
    },
}


def fim_prompt(model_dir: pathlib.Path) -> tuple[str, bool]:
    """Autocomplete prompt: Qwen-Coder gets true FIM tokens, others raw prefix."""
    if "Coder" in model_dir.name:
        return (f"<|fim_prefix|>{AUTOCOMPLETE_PREFIX}<|fim_suffix|>"
                f"{AUTOCOMPLETE_SUFFIX}<|fim_middle|>", True)
    return AUTOCOMPLETE_PREFIX, False


def measure(pipe, prompt: str, max_new: int, chat: bool, runs: int = 3,
            pl: bool = False) -> dict:
    cfg = pipe.get_generation_config()
    cfg.max_new_tokens = max_new
    cfg.do_sample = False
    if hasattr(cfg, "apply_chat_template"):
        cfg.apply_chat_template = chat
    if pl:
        cfg.num_assistant_tokens = 5
        cfg.max_ngram_size = 3
    is_vlm = isinstance(pipe, ov_genai.VLMPipeline)
    tps_list, ttft_list, wall_list = [], [], []
    text = ""
    for i in range(runs + 1):  # first is warm-up
        t0 = time.perf_counter()
        if is_vlm:  # VLMPipeline takes a single str (no list overload)
            res = pipe.generate(prompt, generation_config=cfg)
        else:
            res = pipe.generate([prompt], generation_config=cfg)
        wall = time.perf_counter() - t0
        text = res.texts[0]
        if i == 0:
            continue
        pm = res.perf_metrics
        tps_list.append(pm.get_throughput().mean)
        ttft_list.append(pm.get_ttft().mean / 1000.0)
        wall_list.append(wall)
    mid = len(tps_list) // 2
    return {
        "tps": sorted(tps_list)[mid],
        "ttft": sorted(ttft_list)[mid],
        "wall": sorted(wall_list)[mid],
        "text": text,
    }


# --- pass/fail correctness probes -----------------------------------------
# These validate the ARTIFACT (conversion + quantization + serving path), not
# the model's intelligence — quality ranking should come from the base model's
# official benchmarks. Greedy decoding makes them deterministic.

_PROBE_ORDERS = [
    {"item": "a", "qty": 2}, {"item": "b", "qty": 5},
    {"item": "c", "qty": 1}, {"item": "a", "qty": 9},
]
_PROBE_INV = {"a": 3, "b": 2}
_PROBE_PRICES = {"a": 10.0, "b": 2.5}


def _extract_code(text: str) -> str | None:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    # tolerate a missing closing fence (output truncated at max_new_tokens)
    m = re.search(r"```(?:python)?\s*\n(.*)", text, re.DOTALL)
    return m.group(1) if m else None


def probe_autocomplete(completion: str, used_fim: bool) -> str:
    """PASS if the completed merge function actually merges correctly."""
    candidates = [
        AUTOCOMPLETE_PREFIX + completion + AUTOCOMPLETE_SUFFIX,  # FIM middle
        AUTOCOMPLETE_PREFIX + completion,                         # raw continuation
    ]
    for src in candidates:
        try:
            ns: dict = {}
            exec(src, ns)  # noqa: S102 — local, deterministic, our own prompt
            fn = ns.get("merge_sorted_lists")
            if fn and fn([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6] \
                    and fn([], [1]) == [1]:
                return "PASS"
        except Exception:  # noqa: BLE001
            continue
    return "FAIL"


def probe_edit(output: str) -> str:
    """PASS if the rewritten function preserves the original's behavior and
    actually gained type hints."""
    code = _extract_code(output)
    if not code:
        return "FAIL (no code block)"
    try:
        ref_ns: dict = {}
        exec(CODE, ref_ns)  # noqa: S102
        new_ns: dict = {}
        exec(code, new_ns)  # noqa: S102
        ref_fn = ref_ns["process_orders"]
        new_fn = new_ns.get("process_orders")
        if new_fn is None:
            return "FAIL (function renamed/missing)"
        expected = ref_fn(copy.deepcopy(_PROBE_ORDERS), copy.deepcopy(_PROBE_INV),
                          copy.deepcopy(_PROBE_PRICES))
        actual = new_fn(copy.deepcopy(_PROBE_ORDERS), copy.deepcopy(_PROBE_INV),
                        copy.deepcopy(_PROBE_PRICES))
        if actual != expected:
            return "FAIL (behavior changed)"
        hints = "->" in code.split("\n")[next(
            i for i, l in enumerate(code.split("\n")) if "def process_orders" in l)]
        return "PASS" if hints else "PASS (no return hint)"
    except StopIteration:
        return "FAIL (function missing)"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL ({type(exc).__name__})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--pl", action="store_true",
                        help="also run chat profiles with prompt-lookup")
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent.parent
                        / "bench_results",
                        help="directory for raw result JSON (outputs included, "
                             "so probes can be re-scored without re-running)")
    args = parser.parse_args()

    import datetime
    import json
    import openvino as ov

    is_vlm = (args.model_dir / "openvino_vision_embeddings_model.xml").exists()
    pipe_cls = ov_genai.VLMPipeline if is_vlm else ov_genai.LLMPipeline
    name = args.model_dir.name
    owner = args.model_dir.parent.name
    print(f"model: {owner}/{name}  ({pipe_cls.__name__})")

    results: dict = {
        "model": f"{owner}/{name}",
        "pipeline": pipe_cls.__name__,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "openvino": ov.__version__,
        "profiles": {},
        "probes": {},
    }

    pipe = pipe_cls(str(args.model_dir), "GPU", CACHE_DIR=str(DEFAULT_CACHE))

    # --- autocomplete (plain pipeline; PL adds little on 48-token outputs) ---
    if not is_vlm:
        prompt, used_fim = fim_prompt(args.model_dir)
        r = measure(pipe, prompt, max_new=96, chat=False)
        verdict = probe_autocomplete(r["text"], used_fim)
        results["profiles"]["autocomplete"] = {"plain": r, "fim": used_fim}
        results["probes"]["autocomplete"] = verdict
        print(f"autocomplete{' (FIM)' if used_fim else ' (raw)':8s}: "
              f"TTFT {r['ttft']:.2f}s  {r['tps']:5.1f} tok/s  "
              f"completion in {r['wall']:.2f}s  probe: {verdict}")
    else:
        print("autocomplete: skipped (VLM-shaped IR; not an autocomplete candidate)")

    # --- chat profiles, plain ---
    plain = {}
    for pname, p in PROFILES.items():
        plain[pname] = measure(pipe, p["prompt"], p["max_new"], p["chat"])
        r = plain[pname]
        results["profiles"][pname] = {"plain": r}
        line = f"{pname:18s}: TTFT {r['ttft']:.2f}s  {r['tps']:5.1f} tok/s"
        if pname == "assistant-edit":
            verdict = probe_edit(r["text"])
            results["probes"]["assistant-edit"] = verdict
            line += f"  probe: {verdict}"
        print(line)

    # --- chat profiles, prompt-lookup ---
    if args.pl and not is_vlm:
        del pipe
        pipe = ov_genai.LLMPipeline(str(args.model_dir), "GPU",
                                    prompt_lookup=True,
                                    CACHE_DIR=str(DEFAULT_CACHE))
        for pname, p in PROFILES.items():
            r = measure(pipe, p["prompt"], p["max_new"], p["chat"], pl=True)
            delta = (r["tps"] / plain[pname]["tps"] - 1) * 100
            results["profiles"][pname]["pl"] = r
            results["profiles"][pname]["pl_delta_pct"] = round(delta, 1)
            line = (f"{pname:18s} +PL: TTFT {r['ttft']:.2f}s  {r['tps']:5.1f} tok/s "
                    f"({delta:+.0f}%)")
            if pname == "assistant-edit":
                verdict = probe_edit(r["text"])
                results["probes"]["assistant-edit+PL"] = verdict
                line += f"  probe: {verdict}"
            print(line)
    elif args.pl:
        print("(PL skipped: unsupported on VLM-shaped IRs)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out_dir / f"{owner}__{name}__{stamp}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"results saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
