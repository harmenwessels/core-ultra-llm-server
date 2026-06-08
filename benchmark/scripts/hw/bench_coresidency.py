"""Co-residency validation: scheduler pipelines + plain pipeline together.

Loads the realistic next-gen lineup in ONE process:
  granite-8b   LLMPipeline + SchedulerConfig (prefix caching, 4 GB pool)
  Qwen3.5-2B   VLMPipeline + SchedulerConfig (prefix caching, 2 GB pool)
  Coder-1.5B   LLMPipeline + prompt_lookup (as in production)

Verifies: (1) everything fits, (2) granite's warm-prefix TTFT survives
co-residency, (3) coder FIM latency is unaffected.

Run: .venv/Scripts/python.exe scripts/bench_coresidency.py
"""

import json
import pathlib
import time

import openvino_genai as ov_genai

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "bench_results"
M = ROOT / "models"

FILLER = (
    "The quick brown fox jumps over the lazy dog while the seasoned engineer "
    "reviews pull requests, refactors legacy modules, and documents the build "
    "pipeline for the next release cycle of the platform. "
)


def prompt8k(question: str) -> str:
    n = (8000 * 4) // len(FILLER) + 1
    return FILLER * n + "\n" + question


def gen(pipe, prompt: str, max_new: int = 8, raw: bool = False,
        pl: bool = False) -> float:
    cfg = pipe.get_generation_config()
    cfg.max_new_tokens = max_new
    cfg.do_sample = False
    if raw:
        cfg.apply_chat_template = False
    if pl:  # prompt-lookup pipelines require these (server always sets them)
        cfg.num_assistant_tokens = 5
        cfg.max_ngram_size = 3
    t0 = time.perf_counter()
    pipe.generate(prompt, generation_config=cfg)
    return round(time.perf_counter() - t0, 2)


def sched(gb: int) -> ov_genai.SchedulerConfig:
    s = ov_genai.SchedulerConfig()
    s.enable_prefix_caching = True
    s.max_num_batched_tokens = 2048
    s.cache_size = gb
    return s


results: dict = {}
print("loading lineup...", flush=True)
t0 = time.perf_counter()
granite = ov_genai.LLMPipeline(
    str(M / "HarmenWessels" / "granite-4.1-8b-int4-cw-ov"), "GPU",
    scheduler_config=sched(4), CACHE_DIR=str(ROOT / ".ovcache"))
print(f"  granite + 4GB pool: {time.perf_counter() - t0:.0f}s", flush=True)
t0 = time.perf_counter()
twob = ov_genai.VLMPipeline(
    str(M / "Echo9Zulu" / "Qwen3.5-2B-int4_sym-ov"), "GPU",
    scheduler_config=sched(2), CACHE_DIR=str(ROOT / ".ovcache"))
print(f"  2B + 2GB pool: {time.perf_counter() - t0:.0f}s", flush=True)
t0 = time.perf_counter()
coder = ov_genai.LLMPipeline(
    str(M / "OpenVINO" / "Qwen2.5-Coder-1.5B-Instruct-int4-ov"), "GPU",
    prompt_lookup=True, CACHE_DIR=str(ROOT / ".ovcache"))
print(f"  coder + PL: {time.perf_counter() - t0:.0f}s", flush=True)
results["fit"] = "all three loaded"

# granite: cold 8k prefix, then two warm hits (the agent-turn shape)
cold = gen(granite, prompt8k("Q1: what animal jumps? One word."))
warm1 = gen(granite, prompt8k("Q2: who reviews pull requests? One word."))
warm2 = gen(granite, prompt8k("Q3: what gets documented? One word."))
results["granite"] = {"cold_8k": cold, "warm1": warm1, "warm2": warm2}
print(f"granite 8k: cold {cold}s -> warm {warm1}s / {warm2}s", flush=True)

# 2B: same shape on the architect seat
cold2 = gen(twob, prompt8k("Q1: what animal jumps? One word."))
warm2b = gen(twob, prompt8k("Q2: who reviews pull requests? One word."))
results["twob"] = {"cold_8k": cold2, "warm": warm2b}
print(f"2B 8k: cold {cold2}s -> warm {warm2b}s", flush=True)

# coder: FIM-style raw completion latency (autocomplete shape)
fim = "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists.\"\"\"\n"
lat = [gen(coder, fim, max_new=48, raw=True, pl=True) for _ in range(3)]
results["coder_fim_3x"] = lat
print(f"coder FIM x3: {lat}", flush=True)

OUT_DIR.mkdir(exist_ok=True)
out = OUT_DIR / f"coresidency__{time.strftime('%Y%m%d-%H%M%S')}.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"saved: {out}")
