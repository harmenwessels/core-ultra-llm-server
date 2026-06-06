"""SchedulerConfig experiment: chunked prefill + prefix caching on granite.

Tests, against the plain-LLMPipeline baseline:
  1. does chunked prefill (max_num_batched_tokens) clear the 16k single-
     allocation wall ("Exceeded max size of memory object allocation")?
  2. does enable_prefix_caching collapse TTFT on a repeated prefix
     (the agent-loop turn shape)?

Run: .venv/Scripts/python.exe scripts/bench_prefix_caching.py [model_dir]
"""

import json
import pathlib
import sys
import time

import openvino_genai as ov_genai

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "bench_results"
MODEL = sys.argv[1] if len(sys.argv) > 1 else str(
    ROOT / "models" / "HarmenWessels" / "granite-4.1-8b-int4-cw-ov")

FILLER = (
    "The quick brown fox jumps over the lazy dog while the seasoned engineer "
    "reviews pull requests, refactors legacy modules, and documents the build "
    "pipeline for the next release cycle of the platform. "
)


def make_prompt(approx_tokens: int, question: str) -> str:
    n = (approx_tokens * 4) // len(FILLER) + 1
    return FILLER * n + "\n" + question


def timed_generate(pipe, prompt: str, max_new: int = 8) -> float:
    cfg = pipe.get_generation_config()
    cfg.max_new_tokens = max_new
    cfg.do_sample = False
    t0 = time.perf_counter()
    pipe.generate(prompt, generation_config=cfg)
    return round(time.perf_counter() - t0, 1)


if __name__ == "__main__":
    sch = ov_genai.SchedulerConfig()
    sch.enable_prefix_caching = True
    sch.max_num_batched_tokens = 2048   # chunked prefill
    sch.cache_size = 4                  # GB of KV block pool
    is_vlm = (pathlib.Path(MODEL) / "openvino_vision_embeddings_model.xml").exists()
    pipe_cls = ov_genai.VLMPipeline if is_vlm else ov_genai.LLMPipeline
    print(f"loading {pipe_cls.__name__} with SchedulerConfig "
          "(prefix caching, 2048-tok chunks)...", flush=True)
    t0 = time.perf_counter()
    pipe = pipe_cls(MODEL, "GPU", scheduler_config=sch,
                    CACHE_DIR=str(ROOT / ".ovcache"))
    print(f"loaded in {time.perf_counter() - t0:.0f}s", flush=True)

    results: dict = {"model": MODEL, "prefill": {}, "prefix": {}}

    # 1) chunked prefill vs the allocation wall
    for k in (8000, 16000, 24000):
        try:
            dt = timed_generate(pipe, make_prompt(k, "Reply with exactly: OK"))
            results["prefill"][k] = dt
            print(f"  prefill ~{k:>5} tok: {dt:6.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            msg = str(e).strip().splitlines()[-1][:90]
            results["prefill"][k] = f"FAIL: {msg}"
            print(f"  prefill ~{k:>5} tok: FAIL ({msg})", flush=True)
            break

    # 2) prefix caching: identical 8k prefix, different questions
    prefix_prompt_a = make_prompt(8000, "Question 1: what animal jumps? "
                                        "One word answer.")
    prefix_prompt_b = make_prompt(8000, "Question 2: who reviews pull "
                                        "requests? One word answer.")
    cold = timed_generate(pipe, prefix_prompt_a, max_new=8)
    warm = timed_generate(pipe, prefix_prompt_b, max_new=8)
    warm2 = timed_generate(pipe, make_prompt(
        8000, "Question 3: what gets documented? One word answer."), 8)
    results["prefix"] = {"cold": cold, "warm": warm, "warm2": warm2}
    print(f"  prefix test: cold {cold}s -> warm {warm}s / {warm2}s "
          f"({cold / warm:.1f}x)" if warm else "", flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"prefix_caching__{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {out}")
