# Benchmark

A clean, provenance-logged benchmark that scores each locally-runnable model — and
combinations of models — per task type, to decide which single models and which
`virtual/agent` combos are worth serving. The repo's product is the server; this is the
tool that tells us what to put in it.

## Method

- **Backend:** OpenVINO **GenAI only** (`.venv-genai`). One engine, no cross-engine confounds.
  Since 2026-09-20 every record is on **OpenVINO 2026.4.0 / GenAI 2026.4.0.0** (`-3407`, full fleet
  re-sweep 2026-09-18 → 09-20). Engine history: 2026.3.1 (`-3290`, 2026-09-14) → 2026.4 was
  quality-neutral (net −5 cells over 40 models, all seeded-trajectory flips) and +0–6% decode in an
  interleaved A-B (finding 20); 2026.3.0 (`-3277`) → 2026.3.1 was +6–10% decode (finding 13d);
  the source-built gemma-4 fork (`2026.3.0.0-1`) → 2026.3.0 was decode-neutral (finding 13b).
  Greedy output is *not* byte-identical across engines, so ±1–3 cells per model is engine noise —
  a cross-sweep **time** delta on a small model is drift until an interleaved re-run says otherwise
  (finding 20: a "+67%" on Qwen3-0.6B re-ran at parity). The Intel GPU driver (`32.0.101.8974` →
  `.8991` on 2026-08-24) changed nothing measurable; records carry `engine.gpu_driver` since 2026-09-13.
- **Task types (5 suites):** `codegen` · `edit` · `autocomplete-fim` · `agent-loop`
  · `analysis` (diagnose / plan / route / recall).
- **Scoring:** per (entry, task type) → **quality** = probe pass-rate, **runtime** = total
  wall-clock to solve the suite. Ranked **quality first, then total runtime**. tok/s is logged
  but is *never* the rank key — more tokens for the same task is not per se better.
- **Decoding:** every model runs at the operating point in **its card** (`cards/<owner>__<name>.yaml`):
  sampling for open-ended `codegen` (generative class), greedy for `edit`/`agent-loop`/`analysis`/`fim`
  (structured/fim classes — rule 0f); hybrid thinkers and reasoners sample on the structured
  class too (greedy think-mode loops — MiniCPM5-2B, Ornith). Best-of-N blocks: since 2026-09-14
  each block carries OpenAI `seed` = block index (GenAI's `rng_seed` otherwise stays 0, which made
  every sampled run one deterministic trajectory and every extra block a replay — RESEARCH,
  MiniCPM5/K2 entry); block 0 reproduces older records, later blocks are real draws.
- **Think:** per-task policy from the card — `nothink` for codegen/edit/autocomplete, `think`
  for analysis (where reasoning may help).
- **Provenance:** every run records the engine version, the model's quant recipe (read from the
  IR rt_info — qat/awq+se/data-free, mode, group_size, ratio, ASF), decoding, think, suite, date.
  → reproducible, and a **retest queue** of what to re-run on a newer engine/recipe.

### Two steps
1. **Single-model role-fitness** — `run_fleet.ps1` benchmarks each model solo across the task types.
2. **Combinations** — `run_combos.ps1` loads a `combos.yaml` combo into the `virtual/agent` roles
   (router/architect/executor/reviewer) and benchmarks `virtual/agent` on the same suites. A combo
   is just another peer row in the tables; the best-setup summary (single or combo) is in the root README.

## Run

```powershell
# Step 1 — single models (per-model solo server, all task types)
benchmark\scripts\run_fleet.ps1
benchmark\scripts\run_fleet.ps1 -Models "OpenVINO/Qwen3-14B-int4-ov" -Tasks codegen   # subset

# Step 2 — combinations
benchmark\scripts\run_combos.ps1

# regenerate the tables below + the root README best-setup summary
.venv-genai\Scripts\python.exe benchmark\scripts\assemble_leaderboard.py
.venv-genai\Scripts\python.exe benchmark\scripts\assemble_leaderboard.py --check   # preview only
```

Cards are generated from the on-disk fleet by `benchmark\scripts\scaffold_cards.py` (run it after
downloading a new model); hand-edit a card to pin a model's best-use decoding/think.

## Layout
- `cards/` (repo root) — per-model best-use config, shared with the server.
- `benchmark/combos.yaml` — named virtual/agent combinations.
- `benchmark/scripts/` — `bench_meta.py` (provenance + cards), `bench_run.py` (runner),
  `run_fleet.ps1` / `run_combos.ps1` (orchestrators), `assemble_leaderboard.py`, `bench_castings.py`
  / `bench_roles.py` / `bench_workloads.py` (probes), `hw/` (hardware microbenches).
- `benchmark/results/runs/*.jsonl` — provenance run-records (source of truth).

## What each task tests

Every cell is an **objective** pass/fail: code-producing probes are graded by *executing* the
output against hidden assertions (not text matching); the rest by checking required content or
tool-call shape. Probe source: `bench_castings.py` (codegen), `bench_roles.py` (edit/agent-loop/
analysis), `bench_workloads.py` (fim). Greedy probes are deterministic; sampling/VLM cells use
best-of-N (pass if any block passes — `blocks_used` is logged).

### codegen — 12 cells (6 tasks × 2 phrasings) · `bench_castings.py`
Write correct, self-contained code from a natural-language spec. Tasks: **merge-intervals,
rate-limiter** (sliding window), **lru-cache, parse-duration, rle-codec, group-anagrams** — each
asked two ways (one "design and implement…", one "plan first, then implement…"). The ```python
block(s) are extracted and **executed** against assertions; PASS if a candidate produces every
expected output (e.g. `merge_intervals([[1,3],[2,6],[8,10]]) == [[1,6],[8,10]]`).

### edit — 2 cells · `bench_roles.py`
Precise code modification via tool calls on a buggy `stats.py` (`moving_average` lacks a
`window > len(values)` guard).
- **edit-exact** — fix with **one `edit_file` call** whose `old_string` matches the file byte-exact;
  PASS only if the patched code actually raises `ValueError` on `window > len`.
- **write-full** — fix by **rewriting the whole file** with one `write_file` call, keeping all
  functions; PASS if the rewrite passes behavior asserts (other functions intact, correct results,
  raises on `window > len` *and* `window < 1`).

### agent-loop — 7 cells · `bench_roles.py` (tool-call discipline)
- **call-simple** — "Read config.yaml" → exactly one `read_file(path=…config.yaml)`.
- **call-choose** — "Find the latest stable Python version" → picks `web_search` (right tool).
- **call-restraint** — "What does API stand for?" → **no** tool call; answers inline (don't over-call).
- **result-use** — given a `read_file` result in context, answer the value without re-calling.
- **no-repeat** — file already read → must act (edit) and **not** re-read the same file.
- **stop-done** — edit applied + tests green → must **stop** (no further calls) and confirm.
- **chain-depth** — full scripted loop read→fix→test(fails once)→fix→test→stop; fails on a repeated
  identical call, acting with no tool call, or exceeding 8 turns.

### analysis — 4 cells · `bench_roles.py` (reasoning; runs with **think on**)
- **route** — classify 6 requests into chat/edit/design as JSON; PASS only if **all 6** correct.
- **diagnose** — given a failing pytest + buggy `median`, name the exact wrong expression
  (`s[mid]+s[mid+1]` should be `s[mid-1]+s[mid]` in the even branch).
- **plan** — numbered 3–6 step offline-mode plan referencing the real module functions; no code.
- **recall-deep** — a fact planted early (`FROBNICATE_KEY_77`), then filler turns, then recall it
  (long-context retention).

### autocomplete-fim — 1 cell · `bench_workloads.py` (non-VLM only)
Fill-in-the-middle completion. Coder models get true FIM tokens
(`<|fim_prefix|>…<|fim_suffix|>…<|fim_middle|>`); others a raw prefix. Task: complete the body of
`merge_sorted_lists`. PASS if the assembled function **executes** and merges correctly
(`[1,3,5]+[2,4,6] → [1,2,3,4,5,6]`, `[]+[1] → [1]`). VLM-shaped IRs skip this (not autocomplete
candidates).

<!--LEADERBOARD START-->
## Overall

Every tested model, passes and wall-clock summed across all task types — ranked by total passed, then total time.

| # | Model | Passed | Total s | Size/Roles | Recipe |
|---|---|---|---|---|---|
| 1 | [OpenVINO/Qwen3-14B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-14B-int4-ov) | 25/26 | 1775 | 9.7 GB | data-free |
| 2 | [HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov](https://huggingface.co/HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov) | 24/26 | 697 | 4.8 GB | awq+se |
| 3 | [HarmenWessels/K2-Horizon-7B-int4-symg128-ov](https://huggingface.co/HarmenWessels/K2-Horizon-7B-int4-symg128-ov) | 24/26 | 1995 | 5.7 GB | awq+se |
| 4 | [HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov](https://huggingface.co/HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov) | 23/26 | 1452 | 7.6 GB | awq+se |
| 5 | [HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov](https://huggingface.co/HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov) | 23/26 | 2910 | 4.9 GB | awq+se |
| 6 | [OpenVINO/Qwen3.5-9B-int4-ov](https://huggingface.co/OpenVINO/Qwen3.5-9B-int4-ov) | 23/25 | 2973 | 6.1 GB | data-free |
| 7 | [HarmenWessels/granite-4.1-8b-int4-cw-code-ov](https://huggingface.co/HarmenWessels/granite-4.1-8b-int4-cw-code-ov) | 22/26 | 824 | 4.4 GB | awq+se |
| 8 | [HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov](https://huggingface.co/HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov) | 22/26 | 1238 | 4.9 GB | awq+se |
| 9 | [HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov](https://huggingface.co/HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov) | 22/26 | 1467 | 2.0 GB | awq+se |
| 10 | [OpenVINO/Qwen3.5-4B-int4-ov](https://huggingface.co/OpenVINO/Qwen3.5-4B-int4-ov) | 22/25 | 2089 | 3.5 GB | data-free |
| 11 | [HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov](https://huggingface.co/HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov) | 22/26 | 4298 | 7.6 GB | awq+se |
| 12 | [HarmenWessels/granite-4.1-8b-int4-cw-ov](https://huggingface.co/HarmenWessels/granite-4.1-8b-int4-cw-ov) | 21/26 | 809 | 4.4 GB | awq+se |
| 13 | [HarmenWessels/gemma-4-E2B-it-qat-int4-ov](https://huggingface.co/HarmenWessels/gemma-4-E2B-it-qat-int4-ov) | 21/25 | 1279 | 4.4 GB | qat |
| 14 | [HarmenWessels/gemma-4-E4B-it-qat-int4-ov](https://huggingface.co/HarmenWessels/gemma-4-E4B-it-qat-int4-ov) | 21/25 | 1597 | 6.6 GB | qat |
| 15 | [HarmenWessels/Ornith-1.0-9B-int4-symg128-ov](https://huggingface.co/HarmenWessels/Ornith-1.0-9B-int4-symg128-ov) | 21/25 | 2567 | 6.1 GB | data-free |
| 16 | [OpenVINO/Qwen3-4B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-4B-int4-ov) | 20/26 | 868 | 2.3 GB | awq |
| 17 | [HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov](https://huggingface.co/HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov) | 20/26 | 950 | 3.2 GB | awq+se |
| 18 | [OpenVINO/Qwen3-8B-int4-cw-ov](https://huggingface.co/OpenVINO/Qwen3-8B-int4-cw-ov) | 20/26 | 1416 | 4.7 GB | data-free |
| 19 | [HarmenWessels/Spark-X2.5-4B-int4-symg128-ov](https://huggingface.co/HarmenWessels/Spark-X2.5-4B-int4-symg128-ov) | 20/26 | 2434 | 2.3 GB | awq+se |
| 20 | [HarmenWessels/gemma-4-12B-it-qat-int4-ov](https://huggingface.co/HarmenWessels/gemma-4-12B-it-qat-int4-ov) | 20/25 | 2733 | 8.2 GB | qat |
| 21 | [HarmenWessels/MiniCPM5-2B-int4-symg128-ov](https://huggingface.co/HarmenWessels/MiniCPM5-2B-int4-symg128-ov) | 19/26 | 680 | 1.6 GB | awq+se |
| 22 | [Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov](https://huggingface.co/Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov) | 19/25 | 1963 | 6.2 GB | awq+se |
| 23 | [HarmenWessels/granite-4.1-3b-int4-cw-code-ov](https://huggingface.co/HarmenWessels/granite-4.1-3b-int4-cw-code-ov) | 18/26 | 505 | 1.8 GB | awq+se |
| 24 | [OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov](https://huggingface.co/OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov) | 17/26 | 420 | 1.8 GB | scale_estimation |
| 25 | [HarmenWessels/SmolLM3-3B-int4-symg128-ov](https://huggingface.co/HarmenWessels/SmolLM3-3B-int4-symg128-ov) | 17/26 | 447 | 1.7 GB | awq+se |
| 26 | [HarmenWessels/Spark-X2.5-1.7B-int4-symg128-ov](https://huggingface.co/HarmenWessels/Spark-X2.5-1.7B-int4-symg128-ov) | 17/26 | 1776 | 1.0 GB | awq+se |
| 27 | [HarmenWessels/Agents-A1-4B-int4-asymg128-ov](https://huggingface.co/HarmenWessels/Agents-A1-4B-int4-asymg128-ov) | 17/25 | 2863 | 3.5 GB | data-free |
| 28 | [Echo9Zulu/OmniCoder-9B-int4_sym-ov](https://huggingface.co/Echo9Zulu/OmniCoder-9B-int4_sym-ov) | 17/25 | 4101 | 6.1 GB | data-free |
| 29 | [HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov](https://huggingface.co/HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov) | 15/26 | 235 | 0.7 GB | data-free |
| 30 | [HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov](https://huggingface.co/HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov) | 15/26 | 675 | 0.7 GB | awq+se |
| 31 | [HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov](https://huggingface.co/HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov) | 15/26 | 1137 | 2.0 GB | awq+se |
| 32 | [HarmenWessels/granite-4.1-3b-int4-cw-ov](https://huggingface.co/HarmenWessels/granite-4.1-3b-int4-cw-ov) | 12/26 | 285 | 1.8 GB | awq+se |
| 33 | [HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov](https://huggingface.co/HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov) | 11/26 | 269 | 0.9 GB | data-free |
| 34 | [OpenVINO/Qwen3-1.7B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-1.7B-int4-ov) | 11/26 | 665 | 1.2 GB | data-free |
| 35 | [Echo9Zulu/Qwen3.5-2B-int4_sym-ov](https://huggingface.co/Echo9Zulu/Qwen3.5-2B-int4_sym-ov) | 11/25 | 1578 | 2.1 GB | data-free |
| 36 | [OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov](https://huggingface.co/OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov) | 10/26 | 310 | 0.9 GB | data-free |
| 37 | [HarmenWessels/MiniCPM5-1B-int4-g128-ov](https://huggingface.co/HarmenWessels/MiniCPM5-1B-int4-g128-ov) | 8/26 | 401 | 0.8 GB | data-free |
| 38 | [OpenVINO/Qwen3-0.6B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-0.6B-int4-ov) | 7/26 | 305 | 0.4 GB | data-free |
| 39 | [OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov](https://huggingface.co/OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov) | 5/26 | 199 | 0.3 GB | data-free |
| 40 | [Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov](https://huggingface.co/Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov) | 5/26 | 1051 | 0.7 GB | data-free |

## Per-task-type leaderboard

_190 runs._

### codegen

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov | single | 4.8 GB | 12/12 | 537 | 45 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 2 | HarmenWessels/gemma-4-E2B-it-qat-int4-ov | single | 4.4 GB | 12/12 | 848 | 71 | qat | sampling | nothink | 2026.4.0.0-3407 |
| 3 | HarmenWessels/K2-Horizon-7B-int4-symg128-ov | single | 5.7 GB | 12/12 | 891 | 74 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 4 | HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov | single | 4.9 GB | 12/12 | 960 | 80 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 5 | OpenVINO/Qwen3-14B-int4-ov | single | 9.7 GB | 12/12 | 1004 | 84 | data-free | sampling | nothink | 2026.4.0.0-3407 |
| 6 | HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov | single | 7.6 GB | 12/12 | 1099 | 92 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 7 | HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov | single | 7.6 GB | 12/12 | 1884 | 157 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 8 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 11/12 | 384 | 32 | awq | sampling | nothink | 2026.4.0.0-3407 |
| 9 | HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov | single | 2.0 GB | 11/12 | 567 | 47 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 10 | HarmenWessels/granite-4.1-8b-int4-cw-code-ov | single | 4.4 GB | 11/12 | 635 | 53 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 11 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 11/12 | 722 | 60 | data-free | sampling | nothink | 2026.4.0.0-3407 |
| 12 | HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov | single | 4.9 GB | 11/12 | 827 | 69 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 13 | OpenVINO/Qwen3.5-4B-int4-ov | single | 3.5 GB | 11/12 | 1179 | 98 | data-free | sampling | nothink | 2026.4.0.0-3407 |
| 14 | OpenVINO/Qwen3.5-9B-int4-ov | single | 6.1 GB | 11/12 | 1379 | 115 | data-free | sampling | nothink | 2026.4.0.0-3407 |
| 15 | HarmenWessels/Ornith-1.0-9B-int4-symg128-ov | single | 6.1 GB | 11/12 | 1487 | 124 | data-free | sampling | nothink | 2026.4.0.0-3407 |
| 16 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 10/12 | 637 | 53 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 17 | HarmenWessels/Agents-A1-4B-int4-asymg128-ov | single | 3.5 GB | 10/12 | 1246 | 104 | data-free | sampling | nothink | 2026.4.0.0-3407 |
| 18 | HarmenWessels/gemma-4-E4B-it-qat-int4-ov | single | 6.6 GB | 10/12 | 1264 | 105 | qat | sampling | nothink | 2026.4.0.0-3407 |
| 19 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 1.8 GB | 9/12 | 224 | 19 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 20 | HarmenWessels/SmolLM3-3B-int4-symg128-ov | single | 1.7 GB | 9/12 | 308 | 26 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 21 | Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov | single | 6.2 GB | 9/12 | 1367 | 114 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 22 | HarmenWessels/gemma-4-12B-it-qat-int4-ov | single | 8.2 GB | 9/12 | 2056 | 171 | qat | sampling | nothink | 2026.4.0.0-3407 |
| 23 | HarmenWessels/MiniCPM5-2B-int4-symg128-ov | single | 1.6 GB | 8/12 | 291 | 24 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 24 | HarmenWessels/Spark-X2.5-4B-int4-symg128-ov | single | 2.3 GB | 8/12 | 1438 | 120 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 25 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 7/12 | 184 | 15 | data-free | sampling | nothink | 2026.4.0.0-3407 |
| 26 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 7/12 | 331 | 28 | scale_estimation | sampling | nothink | 2026.4.0.0-3407 |
| 27 | HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov | single | 3.2 GB | 7/12 | 608 | 51 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 28 | HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov | single | 2.0 GB | 6/12 | 855 | 71 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 29 | Echo9Zulu/OmniCoder-9B-int4_sym-ov | single | 6.1 GB | 6/12 | 2368 | 197 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 30 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 1.8 GB | 5/12 | 200 | 17 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 31 | HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov | single | 0.9 GB | 5/12 | 200 | 17 | data-free | sampling | nothink | 2026.4.0.0-3407 |
| 32 | HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov | single | 0.7 GB | 5/12 | 466 | 39 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 33 | HarmenWessels/Spark-X2.5-1.7B-int4-symg128-ov | single | 1.0 GB | 5/12 | 1101 | 92 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 34 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 4/12 | 242 | 20 | data-free | sampling | nothink | 2026.4.0.0-3407 |
| 35 | OpenVINO/Qwen3-1.7B-int4-ov | single | 1.2 GB | 4/12 | 443 | 37 | data-free | sampling | nothink | 2026.4.0.0-3407 |
| 36 | HarmenWessels/MiniCPM5-1B-int4-g128-ov | single | 0.8 GB | 3/12 | 292 | 24 | data-free | sampling | nothink | 2026.4.0.0-3407 |
| 37 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.7 GB | 3/12 | 822 | 68 | data-free | sampling | nothink | 2026.4.0.0-3407 |
| 38 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.4 GB | 2/12 | 130 | 11 | data-free | sampling | nothink | 2026.4.0.0-3407 |
| 39 | Echo9Zulu/Qwen3.5-2B-int4_sym-ov | single | 2.1 GB | 2/12 | 713 | 59 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 40 | OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov | single | 0.3 GB | 1/12 | 139 | 12 | data-free | sampling | nothink | 2026.4.0.0-3407 |

### edit

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov | single | 0.7 GB | 2/2 | 17 | 8 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 2 | HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov | single | 3.2 GB | 2/2 | 19 | 10 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 3 | OpenVINO/Qwen3.5-4B-int4-ov | single | 3.5 GB | 2/2 | 30 | 15 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 4 | HarmenWessels/K2-Horizon-7B-int4-symg128-ov | single | 5.7 GB | 2/2 | 33 | 16 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 5 | HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov | single | 4.9 GB | 2/2 | 35 | 18 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 6 | HarmenWessels/Spark-X2.5-1.7B-int4-symg128-ov | single | 1.0 GB | 2/2 | 39 | 20 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 7 | OpenVINO/Qwen3.5-9B-int4-ov | single | 6.1 GB | 2/2 | 43 | 22 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 8 | Echo9Zulu/OmniCoder-9B-int4_sym-ov | single | 6.1 GB | 2/2 | 57 | 28 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 9 | OpenVINO/Qwen3-14B-int4-ov | single | 9.7 GB | 2/2 | 63 | 32 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 10 | HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov | single | 4.9 GB | 2/2 | 68 | 34 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 11 | HarmenWessels/MiniCPM5-2B-int4-symg128-ov | single | 1.6 GB | 1/2 | 13 | 6 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 12 | Echo9Zulu/Qwen3.5-2B-int4_sym-ov | single | 2.1 GB | 1/2 | 15 | 8 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 13 | HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov | single | 4.8 GB | 1/2 | 19 | 10 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 14 | HarmenWessels/granite-4.1-8b-int4-cw-code-ov | single | 4.4 GB | 1/2 | 26 | 13 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 15 | HarmenWessels/Spark-X2.5-4B-int4-symg128-ov | single | 2.3 GB | 1/2 | 26 | 13 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 16 | HarmenWessels/gemma-4-E2B-it-qat-int4-ov | single | 4.4 GB | 1/2 | 31 | 16 | qat | greedy | nothink | 2026.4.0.0-3407 |
| 17 | HarmenWessels/gemma-4-E4B-it-qat-int4-ov | single | 6.6 GB | 1/2 | 35 | 18 | qat | greedy | nothink | 2026.4.0.0-3407 |
| 18 | HarmenWessels/Ornith-1.0-9B-int4-symg128-ov | single | 6.1 GB | 1/2 | 39 | 20 | data-free | sampling | nothink | 2026.4.0.0-3407 |
| 19 | HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov | single | 7.6 GB | 1/2 | 40 | 20 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 20 | HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov | single | 7.6 GB | 1/2 | 65 | 32 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 21 | HarmenWessels/gemma-4-12B-it-qat-int4-ov | single | 8.2 GB | 1/2 | 73 | 36 | qat | greedy | nothink | 2026.4.0.0-3407 |
| 22 | HarmenWessels/Agents-A1-4B-int4-asymg128-ov | single | 3.5 GB | 1/2 | 258 | 129 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 23 | OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov | single | 0.3 GB | 0/2 | 5 | 2 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 24 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 0/2 | 6 | 3 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 25 | HarmenWessels/MiniCPM5-1B-int4-g128-ov | single | 0.8 GB | 0/2 | 7 | 4 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 26 | HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov | single | 0.9 GB | 0/2 | 7 | 4 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 27 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.4 GB | 0/2 | 7 | 4 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 28 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 0/2 | 10 | 5 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 29 | OpenVINO/Qwen3-1.7B-int4-ov | single | 1.2 GB | 0/2 | 11 | 6 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 30 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 0/2 | 14 | 7 | scale_estimation | greedy | nothink | 2026.4.0.0-3407 |
| 31 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 1.8 GB | 0/2 | 16 | 8 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 32 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 0/2 | 19 | 10 | awq | greedy | nothink | 2026.4.0.0-3407 |
| 33 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 0/2 | 19 | 10 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 34 | HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov | single | 2.0 GB | 0/2 | 23 | 12 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 35 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 0/2 | 26 | 13 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 36 | HarmenWessels/SmolLM3-3B-int4-symg128-ov | single | 1.7 GB | 0/2 | 26 | 13 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 37 | Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov | single | 6.2 GB | 0/2 | 30 | 15 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 38 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.7 GB | 0/2 | 40 | 20 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 39 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 1.8 GB | 0/2 | 85 | 42 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 40 | HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov | single | 2.0 GB | 0/2 | 87 | 44 | awq+se | sampling | nothink | 2026.4.0.0-3407 |

### agent-loop

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HarmenWessels/gemma-4-E4B-it-qat-int4-ov | single | 6.6 GB | 7/7 | 44 | 6 | qat | greedy | nothink | 2026.4.0.0-3407 |
| 2 | HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov | single | 3.2 GB | 7/7 | 58 | 8 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 3 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 7/7 | 60 | 9 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 4 | HarmenWessels/granite-4.1-8b-int4-cw-code-ov | single | 4.4 GB | 7/7 | 72 | 10 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 5 | HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov | single | 4.8 GB | 7/7 | 73 | 10 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 6 | HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov | single | 7.6 GB | 7/7 | 89 | 13 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 7 | HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov | single | 2.0 GB | 7/7 | 89 | 13 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 8 | HarmenWessels/Spark-X2.5-4B-int4-symg128-ov | single | 2.3 GB | 7/7 | 110 | 16 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 9 | HarmenWessels/gemma-4-12B-it-qat-int4-ov | single | 8.2 GB | 7/7 | 137 | 20 | qat | greedy | nothink | 2026.4.0.0-3407 |
| 10 | HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov | single | 7.6 GB | 7/7 | 145 | 21 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 11 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 6/7 | 32 | 5 | scale_estimation | greedy | nothink | 2026.4.0.0-3407 |
| 12 | HarmenWessels/MiniCPM5-2B-int4-symg128-ov | single | 1.6 GB | 6/7 | 43 | 6 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 13 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 6/7 | 59 | 8 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 14 | HarmenWessels/K2-Horizon-7B-int4-symg128-ov | single | 5.7 GB | 6/7 | 84 | 12 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 15 | HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov | single | 4.9 GB | 6/7 | 87 | 12 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 16 | OpenVINO/Qwen3-14B-int4-ov | single | 9.7 GB | 6/7 | 110 | 16 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 17 | HarmenWessels/Ornith-1.0-9B-int4-symg128-ov | single | 6.1 GB | 6/7 | 116 | 17 | data-free | sampling | nothink | 2026.4.0.0-3407 |
| 18 | HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov | single | 4.9 GB | 6/7 | 121 | 17 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 19 | Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov | single | 6.2 GB | 6/7 | 210 | 30 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 20 | OpenVINO/Qwen3.5-9B-int4-ov | single | 6.1 GB | 6/7 | 505 | 72 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 21 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 5/7 | 20 | 3 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 22 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 1.8 GB | 5/7 | 24 | 3 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 23 | OpenVINO/Qwen3-1.7B-int4-ov | single | 1.2 GB | 5/7 | 29 | 4 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 24 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 5/7 | 31 | 4 | awq | greedy | nothink | 2026.4.0.0-3407 |
| 25 | HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov | single | 2.0 GB | 5/7 | 44 | 6 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 26 | HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov | single | 0.7 GB | 5/7 | 56 | 8 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 27 | HarmenWessels/Spark-X2.5-1.7B-int4-symg128-ov | single | 1.0 GB | 5/7 | 110 | 16 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 28 | OpenVINO/Qwen3.5-4B-int4-ov | single | 3.5 GB | 5/7 | 128 | 18 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 29 | HarmenWessels/gemma-4-E2B-it-qat-int4-ov | single | 4.4 GB | 5/7 | 155 | 22 | qat | greedy | nothink | 2026.4.0.0-3407 |
| 30 | Echo9Zulu/Qwen3.5-2B-int4_sym-ov | single | 2.1 GB | 5/7 | 306 | 44 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 31 | Echo9Zulu/OmniCoder-9B-int4_sym-ov | single | 6.1 GB | 5/7 | 530 | 76 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 32 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.4 GB | 4/7 | 19 | 3 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 33 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 1.8 GB | 4/7 | 27 | 4 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 34 | HarmenWessels/SmolLM3-3B-int4-symg128-ov | single | 1.7 GB | 4/7 | 57 | 8 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 35 | HarmenWessels/MiniCPM5-1B-int4-g128-ov | single | 0.8 GB | 3/7 | 26 | 4 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 36 | HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov | single | 0.9 GB | 3/7 | 27 | 4 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 37 | HarmenWessels/Agents-A1-4B-int4-asymg128-ov | single | 3.5 GB | 3/7 | 428 | 61 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 38 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 2/7 | 22 | 3 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 39 | OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov | single | 0.3 GB | 2/7 | 26 | 4 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 40 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.7 GB | 0/7 | 62 | 9 | data-free | greedy | nothink | 2026.4.0.0-3407 |

### analysis

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov | single | 4.8 GB | 4/4 | 63 | 16 | awq+se | greedy | think | 2026.4.0.0-3407 |
| 2 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 4/4 | 78 | 20 | awq+se | greedy | think | 2026.4.0.0-3407 |
| 3 | HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov | single | 3.2 GB | 4/4 | 260 | 65 | awq+se | sampling | think | 2026.4.0.0-3407 |
| 4 | Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov | single | 6.2 GB | 4/4 | 356 | 89 | awq+se | sampling | think | 2026.4.0.0-3407 |
| 5 | HarmenWessels/Spark-X2.5-1.7B-int4-symg128-ov | single | 1.0 GB | 4/4 | 520 | 130 | awq+se | sampling | think | 2026.4.0.0-3407 |
| 6 | OpenVINO/Qwen3-14B-int4-ov | single | 9.7 GB | 4/4 | 584 | 146 | data-free | greedy | think | 2026.4.0.0-3407 |
| 7 | OpenVINO/Qwen3.5-4B-int4-ov | single | 3.5 GB | 4/4 | 752 | 188 | data-free | greedy | think | 2026.4.0.0-3407 |
| 8 | OpenVINO/Qwen3.5-9B-int4-ov | single | 6.1 GB | 4/4 | 1046 | 262 | data-free | greedy | think | 2026.4.0.0-3407 |
| 9 | Echo9Zulu/OmniCoder-9B-int4_sym-ov | single | 6.1 GB | 4/4 | 1146 | 286 | data-free | greedy | think | 2026.4.0.0-3407 |
| 10 | HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov | single | 0.9 GB | 3/4 | 34 | 8 | data-free | greedy | think | 2026.4.0.0-3407 |
| 11 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 3/4 | 34 | 8 | data-free | greedy | think | 2026.4.0.0-3407 |
| 12 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 1.8 GB | 3/4 | 40 | 10 | awq+se | greedy | think | 2026.4.0.0-3407 |
| 13 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 3/4 | 41 | 10 | scale_estimation | greedy | think | 2026.4.0.0-3407 |
| 14 | HarmenWessels/SmolLM3-3B-int4-symg128-ov | single | 1.7 GB | 3/4 | 52 | 13 | awq+se | sampling | nothink | 2026.4.0.0-3407 |
| 15 | HarmenWessels/granite-4.1-8b-int4-cw-code-ov | single | 4.4 GB | 3/4 | 83 | 21 | awq+se | greedy | think | 2026.4.0.0-3407 |
| 16 | HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov | single | 0.7 GB | 3/4 | 134 | 34 | awq+se | sampling | think | 2026.4.0.0-3407 |
| 17 | HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov | single | 7.6 GB | 3/4 | 157 | 39 | awq+se | greedy | think | 2026.4.0.0-3407 |
| 18 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 1.8 GB | 3/4 | 168 | 42 | awq+se | greedy | think | 2026.4.0.0-3407 |
| 19 | HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov | single | 2.0 GB | 3/4 | 211 | 53 | awq+se | greedy | think | 2026.4.0.0-3407 |
| 20 | HarmenWessels/gemma-4-E2B-it-qat-int4-ov | single | 4.4 GB | 3/4 | 245 | 61 | qat | greedy | think | 2026.4.0.0-3407 |
| 21 | HarmenWessels/gemma-4-E4B-it-qat-int4-ov | single | 6.6 GB | 3/4 | 254 | 64 | qat | greedy | think | 2026.4.0.0-3407 |
| 22 | HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov | single | 4.9 GB | 3/4 | 281 | 70 | awq+se | greedy | think | 2026.4.0.0-3407 |
| 23 | HarmenWessels/MiniCPM5-2B-int4-symg128-ov | single | 1.6 GB | 3/4 | 330 | 82 | awq+se | sampling | think | 2026.4.0.0-3407 |
| 24 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 3/4 | 429 | 107 | awq | greedy | think | 2026.4.0.0-3407 |
| 25 | HarmenWessels/gemma-4-12B-it-qat-int4-ov | single | 8.2 GB | 3/4 | 467 | 117 | qat | greedy | think | 2026.4.0.0-3407 |
| 26 | Echo9Zulu/Qwen3.5-2B-int4_sym-ov | single | 2.1 GB | 3/4 | 544 | 136 | data-free | greedy | think | 2026.4.0.0-3407 |
| 27 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 3/4 | 609 | 152 | data-free | greedy | think | 2026.4.0.0-3407 |
| 28 | HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov | single | 2.0 GB | 3/4 | 720 | 180 | awq+se | sampling | think | 2026.4.0.0-3407 |
| 29 | HarmenWessels/Spark-X2.5-4B-int4-symg128-ov | single | 2.3 GB | 3/4 | 785 | 196 | awq+se | sampling | think | 2026.4.0.0-3407 |
| 30 | HarmenWessels/Ornith-1.0-9B-int4-symg128-ov | single | 6.1 GB | 3/4 | 925 | 231 | data-free | sampling | think | 2026.4.0.0-3407 |
| 31 | HarmenWessels/Agents-A1-4B-int4-asymg128-ov | single | 3.5 GB | 3/4 | 931 | 233 | data-free | greedy | think | 2026.4.0.0-3407 |
| 32 | HarmenWessels/K2-Horizon-7B-int4-symg128-ov | single | 5.7 GB | 3/4 | 979 | 245 | awq+se | sampling | think | 2026.4.0.0-3407 |
| 33 | HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov | single | 4.9 GB | 3/4 | 1754 | 438 | awq+se | sampling | think | 2026.4.0.0-3407 |
| 34 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 2/4 | 24 | 6 | data-free | greedy | think | 2026.4.0.0-3407 |
| 35 | HarmenWessels/MiniCPM5-1B-int4-g128-ov | single | 0.8 GB | 2/4 | 75 | 19 | data-free | greedy | think | 2026.4.0.0-3407 |
| 36 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.7 GB | 2/4 | 125 | 31 | data-free | greedy | think | 2026.4.0.0-3407 |
| 37 | OpenVINO/Qwen3-1.7B-int4-ov | single | 1.2 GB | 2/4 | 179 | 45 | data-free | greedy | think | 2026.4.0.0-3407 |
| 38 | HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov | single | 7.6 GB | 2/4 | 2249 | 562 | awq+se | sampling | think | 2026.4.0.0-3407 |
| 39 | OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov | single | 0.3 GB | 1/4 | 28 | 7 | data-free | greedy | think | 2026.4.0.0-3407 |
| 40 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.4 GB | 1/4 | 147 | 37 | data-free | greedy | think | 2026.4.0.0-3407 |

### autocomplete-fim

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 1/1 | 1 | 1 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 2 | OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov | single | 0.3 GB | 1/1 | 1 | 1 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 3 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 1/1 | 2 | 2 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 4 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 1/1 | 2 | 2 | scale_estimation | greedy | nothink | 2026.4.0.0-3407 |
| 5 | HarmenWessels/MiniCPM5-2B-int4-symg128-ov | single | 1.6 GB | 1/1 | 3 | 3 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 6 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 1.8 GB | 1/1 | 4 | 4 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 7 | HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov | single | 2.0 GB | 1/1 | 4 | 4 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 8 | HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov | single | 2.0 GB | 1/1 | 4 | 4 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 9 | HarmenWessels/SmolLM3-3B-int4-symg128-ov | single | 1.7 GB | 1/1 | 4 | 4 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 10 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 1/1 | 5 | 5 | awq | greedy | nothink | 2026.4.0.0-3407 |
| 11 | HarmenWessels/Spark-X2.5-1.7B-int4-symg128-ov | single | 1.0 GB | 1/1 | 6 | 6 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 12 | HarmenWessels/K2-Horizon-7B-int4-symg128-ov | single | 5.7 GB | 1/1 | 8 | 8 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 13 | OpenVINO/Qwen3-14B-int4-ov | single | 9.7 GB | 1/1 | 14 | 14 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 14 | HarmenWessels/Spark-X2.5-4B-int4-symg128-ov | single | 2.3 GB | 1/1 | 75 | 75 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 15 | HarmenWessels/MiniCPM5-1B-int4-g128-ov | single | 0.8 GB | 0/1 | 1 | 1 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 16 | HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov | single | 0.9 GB | 0/1 | 1 | 1 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 17 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.7 GB | 0/1 | 2 | 2 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 18 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 1.8 GB | 0/1 | 2 | 2 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 19 | HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov | single | 0.7 GB | 0/1 | 2 | 2 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 20 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.4 GB | 0/1 | 2 | 2 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 21 | OpenVINO/Qwen3-1.7B-int4-ov | single | 1.2 GB | 0/1 | 3 | 3 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 22 | HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov | single | 3.2 GB | 0/1 | 5 | 5 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 23 | HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov | single | 4.8 GB | 0/1 | 5 | 5 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 24 | HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov | single | 4.9 GB | 0/1 | 7 | 7 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 25 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 0/1 | 7 | 7 | data-free | greedy | nothink | 2026.4.0.0-3407 |
| 26 | HarmenWessels/granite-4.1-8b-int4-cw-code-ov | single | 4.4 GB | 0/1 | 8 | 8 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 27 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 0/1 | 8 | 8 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 28 | HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov | single | 4.9 GB | 0/1 | 8 | 8 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 29 | HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov | single | 7.6 GB | 0/1 | 11 | 11 | awq+se | greedy | nothink | 2026.4.0.0-3407 |
| 30 | HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov | single | 7.6 GB | 0/1 | 11 | 11 | awq+se | greedy | nothink | 2026.4.0.0-3407 |

## Retest queue

_None — all entries current._

## Failures

**OpenVINO/Qwen3-4B-int4-ov / codegen**:
  - group-anagrams#0: FAIL (missing definition)

**HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov / codegen**:
  - rle-codec#1: FAIL (rle_decode('a3b1c3') -> '3b1c3')

**HarmenWessels/granite-4.1-8b-int4-cw-code-ov / codegen**:
  - rle-codec#0: FAIL (NameError: name 'aaabccc' is not defined)

**OpenVINO/Qwen3-8B-int4-cw-ov / codegen**:
  - parse-duration#1: FAIL (NameError: name 'parse_duration' is not defined)

**HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov / codegen**:
  - parse-duration#0: FAIL (NameError: name 'parse_duration' is not defined)

**OpenVINO/Qwen3.5-4B-int4-ov / codegen**:
  - parse-duration#1: FAIL (SyntaxError: unterminated string literal (detected at line 5) (<string>, line 5))

**OpenVINO/Qwen3.5-9B-int4-ov / codegen**:
  - rle-codec#0: FAIL (SyntaxError: invalid syntax (<string>, line 5))

**HarmenWessels/Ornith-1.0-9B-int4-symg128-ov / codegen**:
  - rle-codec#1: FAIL (SyntaxError: invalid syntax. Perhaps you forgot a comma? (<string>, line 3))

**HarmenWessels/granite-4.1-8b-int4-cw-ov / codegen**:
  - parse-duration#0: FAIL (parse_duration('2h45m') -> 3660)
  - parse-duration#1: FAIL (parse_duration('2h45m') -> 162000)

**HarmenWessels/Agents-A1-4B-int4-asymg128-ov / codegen**:
  - lru-cache#1: FAIL (NameError: name 'Node' is not defined)
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)

**HarmenWessels/gemma-4-E4B-it-qat-int4-ov / codegen**:
  - group-anagrams#0: FAIL (missing definition)
  - group-anagrams#1: FAIL (missing definition)

**HarmenWessels/granite-4.1-3b-int4-cw-code-ov / codegen**:
  - parse-duration#0: FAIL (ValueError: invalid literal for int() with base 10: '2h45')
  - rle-codec#0: FAIL (IndexError: string index out of range)
  - rle-codec#1: FAIL (rle_decode('a3b1c3') -> 'bbbc')

**HarmenWessels/SmolLM3-3B-int4-symg128-ov / codegen**:
  - rate-limiter#1: FAIL (TypeError: '<' not supported between instances of 'int' and 'NoneType')
  - parse-duration#0: FAIL (parse_duration('2h45m') -> 0)
  - rle-codec#0: FAIL (IndexError: string index out of range)

**Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov / codegen**:
  - merge-intervals#0: FAIL (NameError: name 'merge_intervals' is not defined)
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)
  - rle-codec#1: FAIL (NameError: name 'rle_encode' is not defined)

**HarmenWessels/gemma-4-12B-it-qat-int4-ov / codegen**:
  - merge-intervals#0: FAIL (NameError: name '__name__Ä™_main__' is not defined)
  - parse-duration#0: FAIL (error: cannot refer to an open group at position 1)
  - rle-codec#0: FAIL (SyntaxError: invalid syntax (<string>, line 28))

**HarmenWessels/MiniCPM5-2B-int4-symg128-ov / codegen**:
  - rate-limiter#0: FAIL (TypeError: '>=' not supported between instances of 'tuple' and 'int')
  - rate-limiter#1: FAIL (NameError: name 'deque' is not defined)
  - parse-duration#0: FAIL (parse_duration('2h45m') -> 0)
  - parse-duration#1: FAIL (SyntaxError: invalid syntax (<string>, line 28))

**HarmenWessels/Spark-X2.5-4B-int4-symg128-ov / codegen**:
  - merge-intervals#0: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [(1, 6), [8, 10]])
  - lru-cache#0: FAIL (missing definition)
  - parse-duration#1: FAIL (ValueError: invalid literal for int() with base 10: '2h45')
  - rle-codec#1: FAIL (IndexError: string index out of range)

**HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / codegen**:
  - rate-limiter#0: FAIL (_seq() -> [False, False, False, False])
  - rate-limiter#1: FAIL (NameError: name 'SlidingWindowLimiter' is not defined)
  - parse-duration#0: FAIL (SyntaxError: '(' was never closed (<string>, line 27))
  - parse-duration#1: FAIL (SyntaxError: '(' was never closed (<string>, line 4))
  - rle-codec#0: FAIL (exec timeout)

**OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / codegen**:
  - merge-intervals#0: FAIL (IndexError: list index out of range)
  - rate-limiter#1: FAIL (_seq() -> [True, True, True, True])
  - parse-duration#0: FAIL (parse_duration('2h45m') -> 60)
  - parse-duration#1: FAIL (parse_duration('2h45m') -> 0)
  - rle-codec#1: FAIL (ValueError: invalid literal for int() with base 10: '3b')

**HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov / codegen**:
  - rate-limiter#1: FAIL (TypeError: _thread.allocate_lock() takes no arguments (1 given))
  - parse-duration#0: FAIL (parse_duration('2h45m') -> 0)
  - parse-duration#1: FAIL (SyntaxError: invalid syntax (<string>, line 1))
  - rle-codec#1: FAIL (SyntaxError: invalid syntax. Perhaps you forgot a comma? (<string>, line 1))
  - group-anagrams#0: FAIL (_ga() -> [['ate'], ['bat'], ['eat'], ['nat'], ['tan'], ['tea']])

**HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov / codegen**:
  - lru-cache#0: FAIL (NameError: name 'LRUCache' is not defined)
  - lru-cache#1: FAIL (NameError: name 'LRUCache' is not defined)
  - parse-duration#0: FAIL (NameError: name 'parse_duration' is not defined)
  - parse-duration#1: FAIL (SyntaxError: invalid syntax (<string>, line 1))
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)
  - rle-codec#1: FAIL (NameError: name 'rle_encode' is not defined)

**Echo9Zulu/OmniCoder-9B-int4_sym-ov / codegen**:
  - merge-intervals#0: FAIL (NameError: name 'merge_intervals' is not defined)
  - rate-limiter#0: FAIL (SyntaxError: invalid syntax (<string>, line 1))
  - lru-cache#1: FAIL (NameError: name 'LRUCache' is not defined)
  - parse-duration#0: FAIL (UnboundLocalError: cannot access local variable 'unit' where it is not associated with a value)
  - parse-duration#1: FAIL (SyntaxError: invalid syntax (<string>, line 3))
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)

**HarmenWessels/granite-4.1-3b-int4-cw-ov / codegen**:
  - merge-intervals#0: FAIL (NameError: name 'merge_intervals' is not defined)
  - rate-limiter#1: FAIL (ImportError: cannot import name 'deque' from 'datetime' (~\AppData\Local\Programs\Python\Python312\Lib\datetime.py))
  - lru-cache#0: FAIL (_lru() -> [1, 2, 3, -1])
  - lru-cache#1: FAIL (SyntaxError: unmatched ')' (<string>, line 52))
  - parse-duration#0: FAIL (KeyError: '2')
  - rle-codec#0: FAIL (rle_decode('a3b1c3') -> 'a3b1c3')
  - rle-codec#1: FAIL (ValueError: invalid literal for int() with base 10: '')

**HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov / codegen**:
  - rate-limiter#0: FAIL (_seq() -> [True, False, False, False])
  - rate-limiter#1: FAIL (_seq() -> [False, False, False, False])
  - parse-duration#0: FAIL (ValueError: invalid literal for int() with base 10: '2h45')
  - parse-duration#1: FAIL (ValueError: invalid literal for int() with base 10: '1h30m15')
  - rle-codec#0: FAIL (ValueError: invalid literal for int() with base 10: '3b1c3')
  - rle-codec#1: FAIL (IndexError: string index out of range)
  - group-anagrams#0: FAIL (_ga() -> [['a', 'b', 't'], ['a', 'e', 't'], ['a', 'n', 't']])

**HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov / codegen**:
  - merge-intervals#1: FAIL (SyntaxError: invalid syntax (<string>, line 1))
  - rate-limiter#0: FAIL (_seq() -> [True, False, False, False])
  - rate-limiter#1: FAIL (SyntaxError: invalid syntax (<string>, line 2))
  - lru-cache#1: FAIL (IndentationError: unexpected indent (<string>, line 2))
  - parse-duration#1: FAIL (SyntaxError: invalid syntax (<string>, line 2))
  - rle-codec#0: FAIL (SyntaxError: invalid syntax (<string>, line 1))
  - rle-codec#1: FAIL (missing definition)

**HarmenWessels/Spark-X2.5-1.7B-int4-symg128-ov / codegen**:
  - rate-limiter#0: FAIL (NameError: name 'SlidingWindowLimiter' is not defined)
  - lru-cache#0: FAIL (_lru() -> [1, 2, 3, 1])
  - parse-duration#0: FAIL (NameError: name 'parse_duration' is not defined)
  - parse-duration#1: FAIL (ValueError: empty separator)
  - rle-codec#0: FAIL (exec timeout)
  - rle-codec#1: FAIL (rle_encode('aaabccc') -> 'a3a1a3')
  - group-anagrams#0: FAIL (_ga() -> [['a', 'b', 't'], ['a', 'e', 't'], ['a', 'n', 't']])

**OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov / codegen**:
  - merge-intervals#0: FAIL (IndexError: list index out of range)
  - rate-limiter#0: FAIL (_seq() -> [True, True, False, False])
  - rate-limiter#1: FAIL (_seq() -> [False, False, False, False])
  - parse-duration#0: FAIL (ValueError: invalid literal for int() with base 10: '2h45m')
  - parse-duration#1: FAIL (ValueError: invalid literal for int() with base 10: '1h30m15')
  - rle-codec#0: FAIL (TypeError: sequence item 1: expected str instance, int found)
  - rle-codec#1: FAIL (NameError: name 'rle_encode' is not defined)
  - group-anagrams#0: FAIL (_ga() -> [['a', 'b', 't'], ['a', 'e', 't'], ['a', 'n', 't']])

**OpenVINO/Qwen3-1.7B-int4-ov / codegen**:
  - merge-intervals#0: FAIL (NameError: name 'merge_intervals' is not defined)
  - rate-limiter#0: FAIL (NameError: name 'SlidingWindowLimiter' is not defined)
  - rate-limiter#1: FAIL (NameError: name 'SlidingWindowLimiter' is not defined)
  - lru-cache#0: FAIL (AttributeError: 'NoneType' object has no attribute 'next')
  - lru-cache#1: FAIL (missing definition)
  - parse-duration#1: FAIL (NameError: name 'parse_duration' is not defined)
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)
  - rle-codec#1: FAIL (NameError: name 'rle_encode' is not defined)

**HarmenWessels/MiniCPM5-1B-int4-g128-ov / codegen**:
  - rate-limiter#0: FAIL (TypeError: '>=' not supported between instances of 'datetime.datetime' and 'int')
  - rate-limiter#1: FAIL (SyntaxError: invalid syntax (<string>, line 1))
  - lru-cache#0: FAIL (SyntaxError: invalid syntax (<string>, line 1))
  - lru-cache#1: FAIL (missing definition)
  - parse-duration#0: FAIL (ValueError: Invalid format)
  - parse-duration#1: FAIL (ValueError: not enough values to unpack (expected 2, got 1))
  - rle-codec#0: FAIL (rle_decode('a3b1c3') -> 'a0b0c0')
  - rle-codec#1: FAIL (SyntaxError: invalid syntax (<string>, line 1))
  - group-anagrams#0: FAIL (_ga() -> [['a', 'e', 't'], ['a', 'n', 't']])

**Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / codegen**:
  - merge-intervals#0: FAIL (missing definition)
  - merge-intervals#1: FAIL (missing definition)
  - rate-limiter#0: FAIL (_seq() -> [True, True, False, False])
  - rate-limiter#1: FAIL (missing definition)
  - lru-cache#1: FAIL (missing definition)
  - parse-duration#0: FAIL (missing definition)
  - parse-duration#1: FAIL (missing definition)
  - rle-codec#0: FAIL (missing definition)
  - rle-codec#1: FAIL (rle_decode('a3b1c3') -> 'a4b2c4')

**OpenVINO/Qwen3-0.6B-int4-ov / codegen**:
  - merge-intervals#1: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [[1, 3], [8, 10]])
  - rate-limiter#0: FAIL (_seq() -> [True, True, True, True])
  - rate-limiter#1: FAIL (_seq() -> [True, False, False, False])
  - lru-cache#0: FAIL (NameError: name 'LRUCache' is not defined)
  - lru-cache#1: FAIL (NameError: name 'lru_cache' is not defined)
  - parse-duration#0: FAIL (NameError: name 'parse_duration' is not defined)
  - parse-duration#1: FAIL (NameError: name 'parse_duration' is not defined)
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)
  - rle-codec#1: FAIL (rle_decode('a3b1c3') -> 'a1b1c1')
  - group-anagrams#0: FAIL (NameError: name 'group_anagrams' is not defined)

**Echo9Zulu/Qwen3.5-2B-int4_sym-ov / codegen**:
  - merge-intervals#0: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [[1, 3], [8, 10], [1, 6]])
  - rate-limiter#0: FAIL (_seq() -> [True, True, False, False])
  - rate-limiter#1: FAIL (_seq() -> [True, True, True, True])
  - lru-cache#1: FAIL (AttributeError: 'dict' object has no attribute 'move_to_end')
  - parse-duration#0: FAIL (ValueError: Invalid unit: 2h45m. Allowed units: h, m, s.)
  - parse-duration#1: FAIL (SyntaxError: invalid syntax (<string>, line 1))
  - rle-codec#0: FAIL (rle_decode(rle_encode('zzzzzzzzzzzz')) -> 'z')
  - rle-codec#1: FAIL (SyntaxError: invalid syntax (<string>, line 2))
  - group-anagrams#0: FAIL (_ga() -> [['a', 'b', 't'], ['a', 'e', 't'], ['a', 'n', 't']])
  - group-anagrams#1: FAIL (TypeError: unhashable type: 'list')

**OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov / codegen**:
  - merge-intervals#0: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [(1, 6), [2, 6], [8, 10]])
  - merge-intervals#1: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [(1, 3), [2, 6], [2, 6]])
  - rate-limiter#0: FAIL (_seq() -> [True, True, True, True])
  - rate-limiter#1: FAIL (_seq() -> [True, True, False, False])
  - lru-cache#0: FAIL (KeyError: 'b')
  - lru-cache#1: FAIL (NameError: name 'capacity' is not defined)
  - parse-duration#0: FAIL (ValueError: Invalid duration format)
  - parse-duration#1: FAIL (parse_duration('2h45m') -> 0)
  - rle-codec#0: FAIL (NameError: name 'count' is not defined)
  - rle-codec#1: FAIL (IndexError: list index out of range)
  - group-anagrams#0: FAIL (TypeError: unhashable type: 'list')

**HarmenWessels/MiniCPM5-2B-int4-symg128-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['read_file'])

**Echo9Zulu/Qwen3.5-2B-int4_sym-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['read_file'])

**HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov / edit**:
  - write-full: FAIL (writes=0 calls=['read_file'])

**HarmenWessels/granite-4.1-8b-int4-cw-code-ov / edit**:
  - edit-exact: FAIL (old-match=False old='if window < 1:\n        raise ValueError("window )

**HarmenWessels/Spark-X2.5-4B-int4-symg128-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['read_file'])

**HarmenWessels/gemma-4-E2B-it-qat-int4-ov / edit**:
  - write-full: FAIL (exec/assert failed: unterminated triple-quoted string literal (detecte)

**HarmenWessels/gemma-4-E4B-it-qat-int4-ov / edit**:
  - write-full: FAIL (exec/assert failed: unterminated triple-quoted string literal (detecte)

**HarmenWessels/Ornith-1.0-9B-int4-symg128-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['read_file'])

**HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov / edit**:
  - write-full: FAIL (exec/assert failed: '[' was never closed (<string>, line 15))

**HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov / edit**:
  - write-full: FAIL (exec/assert failed: '[' was never closed (<string>, line 15))

**HarmenWessels/gemma-4-12B-it-qat-int4-ov / edit**:
  - write-full: FAIL (exec/assert failed: unterminated triple-quoted string literal (detecte)

**HarmenWessels/Agents-A1-4B-int4-asymg128-ov / edit**:
  - write-full: FAIL (writes=3 calls=['write_file', 'write_file', 'write_file'])

**OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['mean'])
  - write-full: FAIL (writes=0 calls=['mean'])

**HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / edit**:
  - edit-exact: FAIL (old-match=False old='def moving_average(values, window):\\n    ')
  - write-full: FAIL (exec/assert failed: 'mean')

**HarmenWessels/MiniCPM5-1B-int4-g128-ov / edit**:
  - edit-exact: FAIL (old-match=True old='moving_average')
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov / edit**:
  - edit-exact: FAIL (old-match=False old='moving_average = lambda values, window:')
  - write-full: FAIL (writes=0 calls=[])

**OpenVINO/Qwen3-0.6B-int4-ov / edit**:
  - edit-exact: FAIL (old-match=True old='')
  - write-full: FAIL (writes=0 calls=[])

**OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov / edit**:
  - edit-exact: FAIL (old-match=True old='window must satisfy 1 <= window <= len(values)')
  - write-full: FAIL (writes=0 calls=[])

**OpenVINO/Qwen3-1.7B-int4-ov / edit**:
  - edit-exact: FAIL (old-match=True old='return [sum(values[i:i + window]) / window\n      )
  - write-full: FAIL (writes=0 calls=[])

**OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=[])
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/granite-4.1-3b-int4-cw-ov / edit**:
  - edit-exact: FAIL (old-match=True old='')
  - write-full: FAIL (writes=0 calls=[])

**OpenVINO/Qwen3-4B-int4-ov / edit**:
  - edit-exact: FAIL (old-match=True old='return [sum(values[i:i + window]) / window\n      )
  - write-full: FAIL (writes=0 calls=[])

**OpenVINO/Qwen3-8B-int4-cw-ov / edit**:
  - edit-exact: FAIL (old-match=True old='return [sum(values[i:i + window]) / window\n      )
  - write-full: FAIL (writes=0 calls=['edit_file'])

**HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=[])
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/granite-4.1-8b-int4-cw-ov / edit**:
  - edit-exact: FAIL (old-match=True old='')
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/SmolLM3-3B-int4-symg128-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=[])
  - write-full: FAIL (writes=0 calls=[])

**Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['read_file'])
  - write-full: FAIL (writes=0 calls=['read_file'])

**Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=[])
  - write-full: FAIL (exec/assert failed: '(' was never closed (<string>, line 1))

**HarmenWessels/granite-4.1-3b-int4-cw-code-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=[])
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=[])
  - write-full: FAIL (exec/assert failed: unterminated triple-quoted string literal (detecte)

**OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / agent-loop**:
  - result-use: FAIL (calls=[('read_file', {'path': 'config.yaml'})] content='')

**HarmenWessels/MiniCPM5-2B-int4-symg128-ov / agent-loop**:
  - call-restraint: FAIL (calls=[('web_search', {'query': 'API acronym stands for'})] content='')

**OpenVINO/Qwen3-8B-int4-cw-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])

**HarmenWessels/K2-Horizon-7B-int4-symg128-ov / agent-loop**:
  - stop-done: FAIL (calls=[])

**HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov / agent-loop**:
  - chain-depth: FAIL (turn 3: repeated call run_tests)

**OpenVINO/Qwen3-14B-int4-ov / agent-loop**:
  - chain-depth: FAIL (turn 1: repeated call edit_file)

**HarmenWessels/Ornith-1.0-9B-int4-symg128-ov / agent-loop**:
  - chain-depth: FAIL (turn 4: repeated call read_file)

**HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])

**Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov / agent-loop**:
  - chain-depth: FAIL (turn 4: repeated call run_tests)

**OpenVINO/Qwen3.5-9B-int4-ov / agent-loop**:
  - chain-depth: FAIL (turn 1: repeated call read_file)

**HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / agent-loop**:
  - no-repeat: FAIL (calls=[('read_file', "{'path': 'stats.py'}")])
  - chain-depth: FAIL (turn 1: repeated call run_tests)

**HarmenWessels/granite-4.1-3b-int4-cw-code-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**OpenVINO/Qwen3-1.7B-int4-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 1: edits=0 green-tests-seen=0)

**OpenVINO/Qwen3-4B-int4-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 1: edits=0 green-tests-seen=0)

**HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov / agent-loop**:
  - call-choose: FAIL (calls=[])
  - chain-depth: FAIL (turn 4: repeated call read_file)

**HarmenWessels/Spark-X2.5-1.7B-int4-symg128-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (turn 3: repeated call read_file)

**OpenVINO/Qwen3.5-4B-int4-ov / agent-loop**:
  - stop-done: FAIL (calls=['run_tests'])
  - chain-depth: FAIL (turn 4: repeated call read_file)

**HarmenWessels/gemma-4-E2B-it-qat-int4-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (turn 3: repeated call run_tests)

**Echo9Zulu/Qwen3.5-2B-int4_sym-ov / agent-loop**:
  - call-simple: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 2: edits=0 green-tests-seen=0)

**Echo9Zulu/OmniCoder-9B-int4_sym-ov / agent-loop**:
  - no-repeat: FAIL (calls=[('run_tests', '{}'), ('read_file', "{'path': 'stats.py'}"), ('r)
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**OpenVINO/Qwen3-0.6B-int4-ov / agent-loop**:
  - call-choose: FAIL (calls=[])
  - call-restraint: FAIL (calls=[] content='The acronym API stands for **API**.')
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**HarmenWessels/granite-4.1-3b-int4-cw-ov / agent-loop**:
  - call-simple: FAIL (calls=[])
  - call-restraint: FAIL (calls=[('web_search', {'query': 'API acronym'})] content='')
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**HarmenWessels/SmolLM3-3B-int4-symg128-ov / agent-loop**:
  - call-restraint: FAIL (calls=[('web_search', {'query': 'What does the acronym API stand for?')
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 1: edits=0 green-tests-seen=0)

**HarmenWessels/MiniCPM5-1B-int4-g128-ov / agent-loop**:
  - call-choose: FAIL (calls=[])
  - call-restraint: FAIL (calls=[] content="I don't have access to a dictionary or a list of acr)
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 1: edits=0 green-tests-seen=0)

**HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov / agent-loop**:
  - call-choose: FAIL (calls=[('get_python_version', {'version_type': 'stable'})])
  - result-use: FAIL (calls=[] content='```json\n{\n  "function": {\n    "name": "read_file")
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (turn 2: repeated call unknown)

**HarmenWessels/Agents-A1-4B-int4-asymg128-ov / agent-loop**:
  - call-choose: FAIL (calls=[])
  - no-repeat: FAIL (calls=[])
  - stop-done: FAIL (calls=['read_file', 'run_tests', 'read_file', 'run_tests', 'read_file')
  - chain-depth: FAIL (turn 0: repeated call read_file)

**OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov / agent-loop**:
  - call-choose: FAIL (calls=[('run_tests', {})])
  - result-use: FAIL (calls=[('read_file', {'path': 'config.yaml'})] content='')
  - no-repeat: FAIL (calls=[])
  - stop-done: FAIL (calls=['run_tests'])
  - chain-depth: FAIL (turn 1: repeated call run_tests)

**OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov / agent-loop**:
  - call-choose: FAIL (calls=[('latest_python_version', {'python_version': '3.10'})])
  - result-use: FAIL (calls=[('read_file', {'path': 'config.yaml'})] content='')
  - no-repeat: FAIL (calls=[])
  - stop-done: FAIL (calls=['run_tests'])
  - chain-depth: FAIL (turn 4: repeated call unsupported tool)

**Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / agent-loop**:
  - call-simple: FAIL (calls=[])
  - call-choose: FAIL (calls=[])
  - call-restraint: FAIL (calls=[] content='')
  - result-use: FAIL (calls=[] content='')
  - no-repeat: FAIL (calls=[])
  - stop-done: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov / analysis**:
  - route: FAIL (4/6 [chat:design, edit:edit, design:design, chat:debug, edit:edit, des)

**OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov / analysis**:
  - diagnose: FAIL (content='the error is that the function `median` is returning 3.5 when)

**HarmenWessels/granite-4.1-3b-int4-cw-ov / analysis**:
  - recall-deep: FAIL (content='(remembered from earlier) FROBNICATE_77.')

**OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / analysis**:
  - diagnose: FAIL (content='the issue lies in the line `return (s[mid] + s[mid + 1]) / 2`)

**HarmenWessels/SmolLM3-3B-int4-symg128-ov / analysis**:
  - recall-deep: FAIL (content='(answer omitted from transcript for brevity) (answer omitted )

**HarmenWessels/granite-4.1-8b-int4-cw-code-ov / analysis**:
  - diagnose: FAIL (content='the test fails because the expected median value for the list)

**HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov / analysis**:
  - route: FAIL (4/6 [chat:chat, edit:edit, design:design, chat:None, edit:None, design)

**HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov / analysis**:
  - diagnose: FAIL (content="the issue is that the function doesn't handle the case where )

**HarmenWessels/granite-4.1-3b-int4-cw-code-ov / analysis**:
  - recall-deep: FAIL (content='(answer omitted from transcript for brevity) (answer omitted )

**HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov / analysis**:
  - route: FAIL (4/6 [chat:chat, edit:edit, design:None, chat:chat, edit:chat, design:d)

**HarmenWessels/gemma-4-E2B-it-qat-int4-ov / analysis**:
  - route: FAIL (3/6 [chat:None, edit:edit, design:design, chat:chat, edit:None, design)

**HarmenWessels/gemma-4-E4B-it-qat-int4-ov / analysis**:
  - route: FAIL (3/6 [chat:chat, edit:edit, design:None, chat:None, edit:edit, design:N)

**HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov / analysis**:
  - route: FAIL (4/6 [chat:chat, edit:edit, design:None, chat:None, edit:edit, design:d)

**HarmenWessels/MiniCPM5-2B-int4-symg128-ov / analysis**:
  - route: FAIL (5/6 [chat:chat, edit:edit, design:design, chat:chat, edit:edit, design)

**OpenVINO/Qwen3-4B-int4-ov / analysis**:
  - plan: FAIL (symbols=0 numbered=False code=False)

**HarmenWessels/gemma-4-12B-it-qat-int4-ov / analysis**:
  - route: FAIL (2/6 [chat:None, edit:None, design:None, chat:None, edit:edit, design:d)

**Echo9Zulu/Qwen3.5-2B-int4_sym-ov / analysis**:
  - route: FAIL (4/6 [chat:None, edit:edit, design:design, chat:chat, edit:edit, design)

**OpenVINO/Qwen3-8B-int4-cw-ov / analysis**:
  - plan: FAIL (symbols=0 numbered=False code=False)

**HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov / analysis**:
  - route: FAIL (4/6 [chat:chat, edit:edit, design:edit, chat:chat, edit:chat, design:d)

**HarmenWessels/Spark-X2.5-4B-int4-symg128-ov / analysis**:
  - recall-deep: FAIL (EXC: timed out)

**HarmenWessels/Ornith-1.0-9B-int4-symg128-ov / analysis**:
  - route: FAIL (5/6 [chat:chat, edit:edit, design:design, chat:None, edit:edit, design)

**HarmenWessels/Agents-A1-4B-int4-asymg128-ov / analysis**:
  - route: FAIL (5/6 [chat:chat, edit:edit, design:design, chat:chat, edit:edit, design)

**HarmenWessels/K2-Horizon-7B-int4-symg128-ov / analysis**:
  - route: FAIL (3/6 [chat:None, edit:edit, design:None, chat:None, edit:edit, design:d)

**HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov / analysis**:
  - route: FAIL (5/6 [chat:chat, edit:edit, design:design, chat:chat, edit:design, desi)

**HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / analysis**:
  - route: FAIL (3/6 [chat:chat, edit:edit, design:chat, chat:edit, edit:edit, design:c)
  - diagnose: FAIL (content='the test fails because the expected median of `[4, 1, 3, 2]` )

**HarmenWessels/MiniCPM5-1B-int4-g128-ov / analysis**:
  - route: FAIL (3/6 [chat:chat, edit:edit, design:design, chat:None, edit:None, design)
  - recall-deep: FAIL (content='The environment variable that holds our deployment API key is)

**Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / analysis**:
  - route: FAIL (4/6 [chat:chat, edit:edit, design:design, chat:edit, edit:chat, design)
  - diagnose: FAIL (content='')

**OpenVINO/Qwen3-1.7B-int4-ov / analysis**:
  - route: FAIL (5/6 [chat:chat, edit:edit, design:design, chat:questions/explanations,)
  - plan: FAIL (symbols=0 numbered=False code=False)

**HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov / analysis**:
  - route: FAIL (4/6 [chat:None, edit:None, design:design, chat:chat, edit:edit, design)
  - plan: FAIL (EXC: timed out)

**OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov / analysis**:
  - route: FAIL (3/6 [chat:edit, edit:edit, design:design, chat:edit, edit:edit, design)
  - diagnose: FAIL (content='the error message indicates that the median function is retur)
  - plan: FAIL (symbols=3 numbered=False code=True)

**OpenVINO/Qwen3-0.6B-int4-ov / analysis**:
  - route: FAIL (4/6 [chat:design, edit:edit, design:design, chat:design, edit:edit, de)
  - diagnose: FAIL (content='')
  - plan: FAIL (symbols=0 numbered=False code=False)

**HarmenWessels/MiniCPM5-1B-int4-g128-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

**Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/granite-4.1-3b-int4-cw-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

**OpenVINO/Qwen3-0.6B-int4-ov / autocomplete-fim**:
  - merge-fim: FAIL

**OpenVINO/Qwen3-1.7B-int4-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

**OpenVINO/Qwen3-8B-int4-cw-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/granite-4.1-8b-int4-cw-code-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/granite-4.1-8b-int4-cw-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

<!--LEADERBOARD END-->
