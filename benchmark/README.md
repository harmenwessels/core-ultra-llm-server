# Benchmark

A clean, provenance-logged benchmark that scores each locally-runnable model — and
combinations of models — per task type, to decide which single models and which
`virtual/agent` combos are worth serving. The repo's product is the server; this is the
tool that tells us what to put in it.

## Method

- **Backend:** OpenVINO **GenAI only** (`.venv-genai`). One engine, no cross-engine confounds.
  Since 2026-09-14 that is the **2026.3.1.0 patch release** (`-3290`); records tagged `2026.3.0.0-3277`
  came from the 2026.3.0 stable release (2026-08-18 → 2026-09-13) and `2026.3.0.0-1` from the earlier
  source-built gemma-4 fork — hence the build id in the Engine column. All three engines score
  **identically** cell-for-cell on the models re-run across them; 2026.3.0 vs the fork was decode-neutral
  (finding 13b), and 2026.3.1 is **+6-10% decode** over 2026.3.0 in an interleaved A-B (finding 13d),
  so older records remain comparable in practice. The retest queue below still flags them as
  engine-stale — the conservative default, worth honouring for any close ranking. The Intel GPU driver
  (`32.0.101.8974` → `.8991` on 2026-08-24) changed nothing measurable; records carry `engine.gpu_driver`
  since 2026-09-13.
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
| 1 | [HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov](https://huggingface.co/HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov) | 25/26 | 679 | 4.8 GB | awq+se |
| 2 | [OpenVINO/Qwen3-14B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-14B-int4-ov) | 25/26 | 1606 | 9.7 GB | data-free |
| 3 | [HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov](https://huggingface.co/HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov) | 24/26 | 1215 | 3.2 GB | awq+se |
| 4 | [HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov](https://huggingface.co/HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov) | 24/26 | 1767 | 7.6 GB | awq+se |
| 5 | [HarmenWessels/K2-Horizon-7B-int4-symg128-ov](https://huggingface.co/HarmenWessels/K2-Horizon-7B-int4-symg128-ov) | 24/26 | 1938 | 5.7 GB | awq+se |
| 6 | [HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov](https://huggingface.co/HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov) | 23/26 | 1505 | 2.0 GB | awq+se |
| 7 | [HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov](https://huggingface.co/HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov) | 23/26 | 1508 | 4.9 GB | awq+se |
| 8 | [OpenVINO/Qwen3.5-4B-int4-ov](https://huggingface.co/OpenVINO/Qwen3.5-4B-int4-ov) | 23/25 | 1928 | 3.5 GB | data-free |
| 9 | [HarmenWessels/gemma-4-12B-it-qat-int4-ov](https://huggingface.co/HarmenWessels/gemma-4-12B-it-qat-int4-ov) | 23/25 | 2517 | 8.2 GB | qat |
| 10 | [HarmenWessels/granite-4.1-8b-int4-cw-code-ov](https://huggingface.co/HarmenWessels/granite-4.1-8b-int4-cw-code-ov) | 22/26 | 787 | 4.4 GB | awq+se |
| 11 | [OpenVINO/Qwen3.5-9B-int4-ov](https://huggingface.co/OpenVINO/Qwen3.5-9B-int4-ov) | 22/25 | 3038 | 6.1 GB | data-free |
| 12 | [HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov](https://huggingface.co/HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov) | 22/26 | 3426 | 4.9 GB | awq+se |
| 13 | [HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov](https://huggingface.co/HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov) | 22/26 | 4442 | 7.6 GB | awq+se |
| 14 | [OpenVINO/Qwen3-4B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-4B-int4-ov) | 21/26 | 797 | 2.3 GB | awq |
| 15 | [HarmenWessels/gemma-4-E2B-it-qat-int4-ov](https://huggingface.co/HarmenWessels/gemma-4-E2B-it-qat-int4-ov) | 21/25 | 1433 | 4.4 GB | qat |
| 16 | [OpenVINO/Qwen3-8B-int4-cw-ov](https://huggingface.co/OpenVINO/Qwen3-8B-int4-cw-ov) | 21/26 | 1612 | 4.7 GB | data-free |
| 17 | [HarmenWessels/gemma-4-E4B-it-qat-int4-ov](https://huggingface.co/HarmenWessels/gemma-4-E4B-it-qat-int4-ov) | 21/25 | 1621 | 6.6 GB | qat |
| 18 | HarmenWessels/Spark-X2.5-4B-int4-symg128-ov | 20/26 | 2287 | 2.3 GB | awq+se |
| 19 | [HarmenWessels/MiniCPM5-2B-int4-symg128-ov](https://huggingface.co/HarmenWessels/MiniCPM5-2B-int4-symg128-ov) | 19/26 | 551 | 1.6 GB | awq+se |
| 20 | [HarmenWessels/granite-4.1-8b-int4-cw-ov](https://huggingface.co/HarmenWessels/granite-4.1-8b-int4-cw-ov) | 19/26 | 820 | 4.4 GB | awq+se |
| 21 | [HarmenWessels/Ornith-1.0-9B-int4-symg128-ov](https://huggingface.co/HarmenWessels/Ornith-1.0-9B-int4-symg128-ov) | 19/25 | 1967 | 6.1 GB | data-free |
| 22 | [HarmenWessels/granite-4.1-3b-int4-cw-code-ov](https://huggingface.co/HarmenWessels/granite-4.1-3b-int4-cw-code-ov) | 18/26 | 486 | 1.8 GB | awq+se |
| 23 | [Echo9Zulu/OmniCoder-9B-int4_sym-ov](https://huggingface.co/Echo9Zulu/OmniCoder-9B-int4_sym-ov) | 18/25 | 3978 | 6.1 GB | data-free |
| 24 | HarmenWessels/Spark-X2.5-1.7B-int4-symg128-ov | 17/26 | 1920 | 1.0 GB | awq+se |
| 25 | [OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov](https://huggingface.co/OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov) | 16/26 | 502 | 1.8 GB | scale_estimation |
| 26 | [HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov](https://huggingface.co/HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov) | 16/26 | 947 | 2.0 GB | awq+se |
| 27 | [Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov](https://huggingface.co/Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov) | 16/25 | 1877 | 0.0 GB | awq+se |
| 28 | HarmenWessels/Agents-A1-4B-int4-asymg128-ov | 16/25 | 2558 | 3.5 GB | data-free |
| 29 | HarmenWessels/Ornith-1.5-9B-int4-symg128-ov | 16/25 | 2719 | 0.0 GB | data-free |
| 30 | [HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov](https://huggingface.co/HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov) | 15/26 | 233 | 0.7 GB | data-free |
| 31 | [OpenVINO/Qwen3-1.7B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-1.7B-int4-ov) | 15/26 | 615 | 1.2 GB | data-free |
| 32 | HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov | 14/25 | 3294 | 0.0 GB | data-free |
| 33 | HarmenWessels/SmolLM3-3B-int4-symg128-ov | 12/26 | 914 | 0.0 GB | awq+se |
| 34 | [Echo9Zulu/Qwen3.5-2B-int4_sym-ov](https://huggingface.co/Echo9Zulu/Qwen3.5-2B-int4_sym-ov) | 12/25 | 1859 | 2.1 GB | data-free |
| 35 | [HarmenWessels/granite-4.1-3b-int4-cw-ov](https://huggingface.co/HarmenWessels/granite-4.1-3b-int4-cw-ov) | 11/26 | 327 | 1.8 GB | awq+se |
| 36 | [HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov](https://huggingface.co/HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov) | 11/26 | 688 | 0.7 GB | awq+se |
| 37 | HarmenWessels/Ornith-1.5-9B-int4-symg64-ov | 11/25 | 3350 | 0.0 GB | data-free |
| 38 | [HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov](https://huggingface.co/HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov) | 10/26 | 260 | 0.9 GB | data-free |
| 39 | [OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov](https://huggingface.co/OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov) | 10/26 | 337 | 0.9 GB | data-free |
| 40 | [HarmenWessels/MiniCPM5-1B-int4-g128-ov](https://huggingface.co/HarmenWessels/MiniCPM5-1B-int4-g128-ov) | 9/26 | 327 | 0.8 GB | data-free |
| 41 | [OpenVINO/Qwen3-0.6B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-0.6B-int4-ov) | 8/26 | 305 | 0.4 GB | data-free |
| 42 | [OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov](https://huggingface.co/OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov) | 8/26 | 307 | 0.3 GB | data-free |
| 43 | [Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov](https://huggingface.co/Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov) | 5/26 | 1209 | 0.7 GB | data-free |

## Per-task-type leaderboard

_202 runs._

### codegen

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov | single | 4.8 GB | 12/12 | 519 | 43 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 2 | OpenVINO/Qwen3-14B-int4-ov | single | 9.7 GB | 12/12 | 898 | 75 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 3 | OpenVINO/Qwen3.5-4B-int4-ov | single | 3.5 GB | 12/12 | 998 | 83 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 4 | HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov | single | 4.9 GB | 12/12 | 1040 | 87 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 5 | HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov | single | 4.9 GB | 12/12 | 1064 | 89 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 6 | HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov | single | 7.6 GB | 12/12 | 1422 | 118 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 7 | HarmenWessels/gemma-4-12B-it-qat-int4-ov | single | 8.2 GB | 12/12 | 1646 | 137 | qat | sampling | nothink | 2026.3.1.0-3290 |
| 8 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 11/12 | 386 | 32 | awq | sampling | nothink | 2026.3.1.0-3290 |
| 9 | HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov | single | 2.0 GB | 11/12 | 522 | 44 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 10 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 11/12 | 980 | 82 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 11 | HarmenWessels/K2-Horizon-7B-int4-symg128-ov | single | 5.7 GB | 11/12 | 1024 | 85 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 12 | HarmenWessels/gemma-4-E2B-it-qat-int4-ov | single | 4.4 GB | 11/12 | 1045 | 87 | qat | sampling | nothink | 2026.3.1.0-3290 |
| 13 | HarmenWessels/Ornith-1.0-9B-int4-symg128-ov | single | 6.1 GB | 11/12 | 1081 | 90 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 14 | HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov | single | 7.6 GB | 11/12 | 1988 | 166 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 15 | HarmenWessels/granite-4.1-8b-int4-cw-code-ov | single | 4.4 GB | 10/12 | 605 | 50 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 16 | HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov | single | 3.2 GB | 10/12 | 810 | 68 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 17 | HarmenWessels/gemma-4-E4B-it-qat-int4-ov | single | 6.6 GB | 10/12 | 1278 | 106 | qat | sampling | nothink | 2026.3.1.0-3290 |
| 18 | OpenVINO/Qwen3.5-9B-int4-ov | single | 6.1 GB | 10/12 | 1877 | 156 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 19 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 9/12 | 657 | 55 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 20 | HarmenWessels/Agents-A1-4B-int4-asymg128-ov | single | 3.5 GB | 9/12 | 1272 | 106 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 21 | HarmenWessels/Spark-X2.5-4B-int4-symg128-ov | single | 2.3 GB | 8/12 | 1362 | 114 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 22 | HarmenWessels/Ornith-1.5-9B-int4-symg128-ov | single | 0.0 GB | 8/12 | 1773 | 148 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 23 | Echo9Zulu/OmniCoder-9B-int4_sym-ov | single | 6.1 GB | 8/12 | 2007 | 167 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 24 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 7/12 | 177 | 15 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 25 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 1.8 GB | 7/12 | 264 | 22 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 26 | HarmenWessels/MiniCPM5-2B-int4-symg128-ov | single | 1.6 GB | 7/12 | 293 | 24 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 27 | Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov | single | 0.0 GB | 7/12 | 1334 | 111 | awq+se | sampling | nothink | 2026.3.0.0-3277 |
| 28 | HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov | single | 0.0 GB | 7/12 | 1545 | 129 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 29 | HarmenWessels/Ornith-1.5-9B-int4-symg64-ov | single | 0.0 GB | 7/12 | 2469 | 206 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 30 | OpenVINO/Qwen3-1.7B-int4-ov | single | 1.2 GB | 6/12 | 369 | 31 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 31 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 6/12 | 408 | 34 | scale_estimation | sampling | nothink | 2026.3.1.0-3290 |
| 32 | HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov | single | 2.0 GB | 6/12 | 651 | 54 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 33 | HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov | single | 0.9 GB | 5/12 | 191 | 16 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 34 | HarmenWessels/SmolLM3-3B-int4-symg128-ov | single | 0.0 GB | 5/12 | 457 | 38 | awq+se | sampling | nothink | 2026.3.0.0-1 |
| 35 | HarmenWessels/Spark-X2.5-1.7B-int4-symg128-ov | single | 1.0 GB | 5/12 | 1164 | 97 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 36 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 1.8 GB | 4/12 | 231 | 19 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 37 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 4/12 | 268 | 22 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 38 | Echo9Zulu/Qwen3.5-2B-int4_sym-ov | single | 2.1 GB | 4/12 | 811 | 68 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 39 | HarmenWessels/MiniCPM5-1B-int4-g128-ov | single | 0.8 GB | 3/12 | 210 | 18 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 40 | OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov | single | 0.3 GB | 3/12 | 244 | 20 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 41 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.7 GB | 3/12 | 937 | 78 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 42 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.4 GB | 1/12 | 177 | 15 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 43 | HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov | single | 0.7 GB | 1/12 | 447 | 37 | awq+se | sampling | nothink | 2026.3.1.0-3290 |

### edit

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov | single | 3.2 GB | 2/2 | 21 | 10 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 2 | HarmenWessels/granite-4.1-8b-int4-cw-code-ov | single | 4.4 GB | 2/2 | 26 | 13 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 3 | OpenVINO/Qwen3.5-4B-int4-ov | single | 3.5 GB | 2/2 | 26 | 13 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 4 | HarmenWessels/K2-Horizon-7B-int4-symg128-ov | single | 5.7 GB | 2/2 | 30 | 15 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 5 | HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov | single | 4.9 GB | 2/2 | 35 | 18 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 6 | OpenVINO/Qwen3.5-9B-int4-ov | single | 6.1 GB | 2/2 | 39 | 20 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 7 | Echo9Zulu/OmniCoder-9B-int4_sym-ov | single | 6.1 GB | 2/2 | 40 | 20 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 8 | HarmenWessels/Spark-X2.5-1.7B-int4-symg128-ov | single | 1.0 GB | 2/2 | 42 | 21 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 9 | HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov | single | 4.9 GB | 2/2 | 43 | 22 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 10 | OpenVINO/Qwen3-14B-int4-ov | single | 9.7 GB | 2/2 | 63 | 32 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 11 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 1.8 GB | 1/2 | 13 | 6 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 12 | HarmenWessels/MiniCPM5-2B-int4-symg128-ov | single | 1.6 GB | 1/2 | 14 | 7 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 13 | Echo9Zulu/Qwen3.5-2B-int4_sym-ov | single | 2.1 GB | 1/2 | 18 | 9 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 14 | HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov | single | 4.8 GB | 1/2 | 18 | 9 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 15 | HarmenWessels/Spark-X2.5-4B-int4-symg128-ov | single | 2.3 GB | 1/2 | 26 | 13 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 16 | HarmenWessels/gemma-4-E2B-it-qat-int4-ov | single | 4.4 GB | 1/2 | 29 | 14 | qat | greedy | nothink | 2026.3.1.0-3290 |
| 17 | HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov | single | 0.7 GB | 1/2 | 30 | 15 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 18 | HarmenWessels/Ornith-1.5-9B-int4-symg128-ov | single | 0.0 GB | 1/2 | 32 | 16 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 19 | HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov | single | 2.0 GB | 1/2 | 44 | 22 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 20 | HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov | single | 7.6 GB | 1/2 | 45 | 22 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 21 | HarmenWessels/gemma-4-E4B-it-qat-int4-ov | single | 6.6 GB | 1/2 | 50 | 25 | qat | greedy | nothink | 2026.3.1.0-3290 |
| 22 | HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov | single | 7.6 GB | 1/2 | 67 | 34 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 23 | HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov | single | 2.0 GB | 1/2 | 77 | 38 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 24 | HarmenWessels/gemma-4-12B-it-qat-int4-ov | single | 8.2 GB | 1/2 | 100 | 50 | qat | greedy | nothink | 2026.3.1.0-3290 |
| 25 | HarmenWessels/Agents-A1-4B-int4-asymg128-ov | single | 3.5 GB | 1/2 | 259 | 130 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 26 | OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov | single | 0.3 GB | 0/2 | 5 | 2 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 27 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 0/2 | 7 | 4 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 28 | HarmenWessels/MiniCPM5-1B-int4-g128-ov | single | 0.8 GB | 0/2 | 7 | 4 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 29 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.4 GB | 0/2 | 8 | 4 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 30 | HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov | single | 0.9 GB | 0/2 | 10 | 5 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 31 | OpenVINO/Qwen3-1.7B-int4-ov | single | 1.2 GB | 0/2 | 10 | 5 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 32 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 0/2 | 11 | 6 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 33 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 1.8 GB | 0/2 | 15 | 8 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 34 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 0/2 | 16 | 8 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 35 | HarmenWessels/SmolLM3-3B-int4-symg128-ov | single | 0.0 GB | 0/2 | 17 | 8 | awq+se | greedy | nothink | 2026.3.0.0-1 |
| 36 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 0/2 | 17 | 8 | scale_estimation | greedy | nothink | 2026.3.1.0-3290 |
| 37 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 0/2 | 18 | 9 | awq | greedy | nothink | 2026.3.1.0-3290 |
| 38 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 0/2 | 24 | 12 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 39 | HarmenWessels/Ornith-1.0-9B-int4-symg128-ov | single | 6.1 GB | 0/2 | 28 | 14 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 40 | Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov | single | 0.0 GB | 0/2 | 29 | 14 | awq+se | sampling | nothink | 2026.3.0.0-3277 |
| 41 | HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov | single | 0.0 GB | 0/2 | 33 | 16 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 42 | HarmenWessels/Ornith-1.5-9B-int4-symg64-ov | single | 0.0 GB | 0/2 | 37 | 18 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 43 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.7 GB | 0/2 | 45 | 22 | data-free | greedy | nothink | 2026.3.1.0-3290 |

### agent-loop

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HarmenWessels/gemma-4-E4B-it-qat-int4-ov | single | 6.6 GB | 7/7 | 45 | 6 | qat | greedy | nothink | 2026.3.1.0-3290 |
| 2 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 7/7 | 58 | 8 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 3 | HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov | single | 3.2 GB | 7/7 | 64 | 9 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 4 | HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov | single | 4.8 GB | 7/7 | 71 | 10 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 5 | HarmenWessels/granite-4.1-8b-int4-cw-code-ov | single | 4.4 GB | 7/7 | 72 | 10 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 6 | HarmenWessels/K2-Horizon-7B-int4-symg128-ov | single | 5.7 GB | 7/7 | 77 | 11 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 7 | HarmenWessels/Spark-X2.5-4B-int4-symg128-ov | single | 2.3 GB | 7/7 | 107 | 15 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 8 | HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov | single | 7.6 GB | 7/7 | 126 | 18 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 9 | HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov | single | 7.6 GB | 7/7 | 130 | 19 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 10 | HarmenWessels/gemma-4-12B-it-qat-int4-ov | single | 8.2 GB | 7/7 | 207 | 30 | qat | greedy | nothink | 2026.3.1.0-3290 |
| 11 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 1.8 GB | 6/7 | 23 | 3 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 12 | OpenVINO/Qwen3-1.7B-int4-ov | single | 1.2 GB | 6/7 | 28 | 4 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 13 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 6/7 | 32 | 5 | scale_estimation | greedy | nothink | 2026.3.1.0-3290 |
| 14 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 6/7 | 34 | 5 | awq | greedy | nothink | 2026.3.1.0-3290 |
| 15 | HarmenWessels/MiniCPM5-2B-int4-symg128-ov | single | 1.6 GB | 6/7 | 42 | 6 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 16 | HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov | single | 0.7 GB | 6/7 | 47 | 7 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 17 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 6/7 | 66 | 9 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 18 | HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov | single | 4.9 GB | 6/7 | 91 | 13 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 19 | OpenVINO/Qwen3-14B-int4-ov | single | 9.7 GB | 6/7 | 106 | 15 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 20 | HarmenWessels/Ornith-1.0-9B-int4-symg128-ov | single | 6.1 GB | 6/7 | 112 | 16 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 21 | HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov | single | 2.0 GB | 6/7 | 117 | 17 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 22 | HarmenWessels/gemma-4-E2B-it-qat-int4-ov | single | 4.4 GB | 6/7 | 151 | 22 | qat | greedy | nothink | 2026.3.1.0-3290 |
| 23 | Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov | single | 0.0 GB | 6/7 | 163 | 23 | awq+se | sampling | nothink | 2026.3.0.0-3277 |
| 24 | HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov | single | 4.9 GB | 6/7 | 192 | 27 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 25 | OpenVINO/Qwen3.5-9B-int4-ov | single | 6.1 GB | 6/7 | 207 | 30 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 26 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 5/7 | 22 | 3 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 27 | HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov | single | 2.0 GB | 5/7 | 45 | 6 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 28 | HarmenWessels/Spark-X2.5-1.7B-int4-symg128-ov | single | 1.0 GB | 5/7 | 121 | 17 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 29 | OpenVINO/Qwen3.5-4B-int4-ov | single | 3.5 GB | 5/7 | 239 | 34 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 30 | HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov | single | 0.0 GB | 5/7 | 316 | 45 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 31 | Echo9Zulu/OmniCoder-9B-int4_sym-ov | single | 6.1 GB | 5/7 | 443 | 63 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 32 | HarmenWessels/Ornith-1.5-9B-int4-symg128-ov | single | 0.0 GB | 5/7 | 540 | 77 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 33 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.4 GB | 4/7 | 20 | 3 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 34 | HarmenWessels/MiniCPM5-1B-int4-g128-ov | single | 0.8 GB | 4/7 | 25 | 4 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 35 | HarmenWessels/SmolLM3-3B-int4-symg128-ov | single | 0.0 GB | 4/7 | 36 | 5 | awq+se | greedy | nothink | 2026.3.0.0-1 |
| 36 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 1.8 GB | 4/7 | 39 | 6 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 37 | Echo9Zulu/Qwen3.5-2B-int4_sym-ov | single | 2.1 GB | 4/7 | 257 | 37 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 38 | HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov | single | 0.9 GB | 3/7 | 26 | 4 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 39 | HarmenWessels/Ornith-1.5-9B-int4-symg64-ov | single | 0.0 GB | 3/7 | 404 | 58 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 40 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 2/7 | 21 | 3 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 41 | OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov | single | 0.3 GB | 2/7 | 27 | 4 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 42 | HarmenWessels/Agents-A1-4B-int4-asymg128-ov | single | 3.5 GB | 2/7 | 438 | 63 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 43 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.7 GB | 0/7 | 66 | 9 | data-free | greedy | nothink | 2026.3.1.0-3290 |

### analysis

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov | single | 4.8 GB | 4/4 | 66 | 16 | awq+se | greedy | think | 2026.3.1.0-3290 |
| 2 | HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov | single | 7.6 GB | 4/4 | 161 | 40 | awq+se | greedy | think | 2026.3.1.0-3290 |
| 3 | HarmenWessels/MiniCPM5-2B-int4-symg128-ov | single | 1.6 GB | 4/4 | 199 | 50 | awq+se | sampling | think | 2026.3.1.0-3290 |
| 4 | HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov | single | 3.2 GB | 4/4 | 314 | 78 | awq+se | sampling | think | 2026.3.1.0-3290 |
| 5 | OpenVINO/Qwen3-14B-int4-ov | single | 9.7 GB | 4/4 | 526 | 132 | data-free | greedy | think | 2026.3.1.0-3290 |
| 6 | HarmenWessels/Spark-X2.5-1.7B-int4-symg128-ov | single | 1.0 GB | 4/4 | 585 | 146 | awq+se | sampling | think | 2026.3.1.0-3290 |
| 7 | HarmenWessels/Agents-A1-4B-int4-asymg128-ov | single | 3.5 GB | 4/4 | 589 | 147 | data-free | greedy | think | 2026.3.1.0-3290 |
| 8 | OpenVINO/Qwen3.5-4B-int4-ov | single | 3.5 GB | 4/4 | 665 | 166 | data-free | greedy | think | 2026.3.1.0-3290 |
| 9 | HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov | single | 2.0 GB | 4/4 | 785 | 196 | awq+se | sampling | think | 2026.3.1.0-3290 |
| 10 | OpenVINO/Qwen3.5-9B-int4-ov | single | 6.1 GB | 4/4 | 915 | 229 | data-free | greedy | think | 2026.3.1.0-3290 |
| 11 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 3/4 | 35 | 9 | data-free | greedy | think | 2026.3.1.0-3290 |
| 12 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 1.8 GB | 3/4 | 40 | 10 | awq+se | greedy | think | 2026.3.1.0-3290 |
| 13 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 3/4 | 43 | 11 | scale_estimation | greedy | think | 2026.3.1.0-3290 |
| 14 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 3/4 | 73 | 18 | awq+se | greedy | think | 2026.3.1.0-3290 |
| 15 | HarmenWessels/granite-4.1-8b-int4-cw-code-ov | single | 4.4 GB | 3/4 | 80 | 20 | awq+se | greedy | think | 2026.3.1.0-3290 |
| 16 | HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov | single | 0.7 GB | 3/4 | 162 | 40 | awq+se | sampling | think | 2026.3.1.0-3290 |
| 17 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 1.8 GB | 3/4 | 182 | 46 | awq+se | greedy | think | 2026.3.1.0-3290 |
| 18 | HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov | single | 2.0 GB | 3/4 | 203 | 51 | awq+se | greedy | think | 2026.3.1.0-3290 |
| 19 | OpenVINO/Qwen3-1.7B-int4-ov | single | 1.2 GB | 3/4 | 205 | 51 | data-free | greedy | think | 2026.3.1.0-3290 |
| 20 | HarmenWessels/gemma-4-E2B-it-qat-int4-ov | single | 4.4 GB | 3/4 | 208 | 52 | qat | greedy | think | 2026.3.1.0-3290 |
| 21 | HarmenWessels/gemma-4-E4B-it-qat-int4-ov | single | 6.6 GB | 3/4 | 248 | 62 | qat | greedy | think | 2026.3.1.0-3290 |
| 22 | HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov | single | 4.9 GB | 3/4 | 311 | 78 | awq+se | greedy | think | 2026.3.1.0-3290 |
| 23 | Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov | single | 0.0 GB | 3/4 | 351 | 88 | awq+se | sampling | think | 2026.3.0.0-3277 |
| 24 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 3/4 | 355 | 89 | awq | greedy | think | 2026.3.1.0-3290 |
| 25 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 3/4 | 542 | 136 | data-free | greedy | think | 2026.3.1.0-3290 |
| 26 | HarmenWessels/gemma-4-12B-it-qat-int4-ov | single | 8.2 GB | 3/4 | 564 | 141 | qat | greedy | think | 2026.3.1.0-3290 |
| 27 | Echo9Zulu/Qwen3.5-2B-int4_sym-ov | single | 2.1 GB | 3/4 | 773 | 193 | data-free | greedy | think | 2026.3.1.0-3290 |
| 28 | HarmenWessels/Spark-X2.5-4B-int4-symg128-ov | single | 2.3 GB | 3/4 | 786 | 196 | awq+se | sampling | think | 2026.3.1.0-3290 |
| 29 | HarmenWessels/K2-Horizon-7B-int4-symg128-ov | single | 5.7 GB | 3/4 | 799 | 200 | awq+se | sampling | think | 2026.3.1.0-3290 |
| 30 | Echo9Zulu/OmniCoder-9B-int4_sym-ov | single | 6.1 GB | 3/4 | 1488 | 372 | data-free | greedy | think | 2026.3.1.0-3290 |
| 31 | HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov | single | 7.6 GB | 3/4 | 2246 | 562 | awq+se | sampling | think | 2026.3.1.0-3290 |
| 32 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 2/4 | 25 | 6 | data-free | greedy | think | 2026.3.1.0-3290 |
| 33 | OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov | single | 0.3 GB | 2/4 | 30 | 8 | data-free | greedy | think | 2026.3.1.0-3290 |
| 34 | HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov | single | 0.9 GB | 2/4 | 32 | 8 | data-free | greedy | think | 2026.3.1.0-3290 |
| 35 | HarmenWessels/MiniCPM5-1B-int4-g128-ov | single | 0.8 GB | 2/4 | 84 | 21 | data-free | greedy | think | 2026.3.1.0-3290 |
| 36 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.4 GB | 2/4 | 98 | 24 | data-free | greedy | think | 2026.3.1.0-3290 |
| 37 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.7 GB | 2/4 | 159 | 40 | data-free | greedy | think | 2026.3.1.0-3290 |
| 38 | HarmenWessels/Ornith-1.5-9B-int4-symg128-ov | single | 0.0 GB | 2/4 | 374 | 94 | data-free | sampling | think | 2026.3.0.0-3277 |
| 39 | HarmenWessels/SmolLM3-3B-int4-symg128-ov | single | 0.0 GB | 2/4 | 400 | 100 | awq+se | greedy | think | 2026.3.0.0-1 |
| 40 | HarmenWessels/Ornith-1.0-9B-int4-symg128-ov | single | 6.1 GB | 2/4 | 746 | 186 | data-free | sampling | think | 2026.3.1.0-3290 |
| 41 | HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov | single | 0.0 GB | 2/4 | 1400 | 350 | data-free | sampling | think | 2026.3.0.0-3277 |
| 42 | HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov | single | 4.9 GB | 2/4 | 2143 | 536 | awq+se | sampling | think | 2026.3.1.0-3290 |
| 43 | HarmenWessels/Ornith-1.5-9B-int4-symg64-ov | single | 0.0 GB | 1/4 | 440 | 110 | data-free | sampling | think | 2026.3.0.0-3277 |

### autocomplete-fim

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov | single | 0.3 GB | 1/1 | 1 | 1 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 2 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 1/1 | 2 | 2 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 3 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 1/1 | 2 | 2 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 4 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 1/1 | 2 | 2 | scale_estimation | greedy | nothink | 2026.3.1.0-3290 |
| 5 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.4 GB | 1/1 | 2 | 2 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 6 | HarmenWessels/MiniCPM5-2B-int4-symg128-ov | single | 1.6 GB | 1/1 | 3 | 3 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 7 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 1.8 GB | 1/1 | 4 | 4 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 8 | HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov | single | 2.0 GB | 1/1 | 4 | 4 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 9 | HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov | single | 2.0 GB | 1/1 | 4 | 4 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 10 | HarmenWessels/SmolLM3-3B-int4-symg128-ov | single | 0.0 GB | 1/1 | 4 | 4 | awq+se | greedy | nothink | 2026.3.0.0-1 |
| 11 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 1/1 | 4 | 4 | awq | greedy | nothink | 2026.3.1.0-3290 |
| 12 | HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov | single | 4.8 GB | 1/1 | 5 | 5 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 13 | HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov | single | 3.2 GB | 1/1 | 6 | 6 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 14 | HarmenWessels/Spark-X2.5-4B-int4-symg128-ov | single | 2.3 GB | 1/1 | 6 | 6 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 15 | HarmenWessels/K2-Horizon-7B-int4-symg128-ov | single | 5.7 GB | 1/1 | 8 | 8 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 16 | HarmenWessels/Spark-X2.5-1.7B-int4-symg128-ov | single | 1.0 GB | 1/1 | 8 | 8 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 17 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 1/1 | 8 | 8 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 18 | OpenVINO/Qwen3-14B-int4-ov | single | 9.7 GB | 1/1 | 13 | 13 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 19 | HarmenWessels/MiniCPM5-1B-int4-g128-ov | single | 0.8 GB | 0/1 | 1 | 1 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 20 | HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov | single | 0.9 GB | 0/1 | 1 | 1 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 21 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.7 GB | 0/1 | 2 | 2 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 22 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 1.8 GB | 0/1 | 2 | 2 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 23 | HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov | single | 0.7 GB | 0/1 | 2 | 2 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 24 | OpenVINO/Qwen3-1.7B-int4-ov | single | 1.2 GB | 0/1 | 3 | 3 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 25 | HarmenWessels/granite-4.1-8b-int4-cw-code-ov | single | 4.4 GB | 0/1 | 4 | 4 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 26 | HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov | single | 4.9 GB | 0/1 | 7 | 7 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 27 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 0/1 | 8 | 8 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 28 | HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov | single | 4.9 GB | 0/1 | 8 | 8 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 29 | HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov | single | 7.6 GB | 0/1 | 11 | 11 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 30 | HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov | single | 7.6 GB | 0/1 | 13 | 13 | awq+se | greedy | nothink | 2026.3.1.0-3290 |

## Retest queue

- Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov / agent-loop: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov / agent-loop: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-symg128-ov / agent-loop: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-symg64-ov / agent-loop: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/SmolLM3-3B-int4-symg128-ov / agent-loop: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-symg128-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-symg64-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/SmolLM3-3B-int4-symg128-ov / analysis: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/SmolLM3-3B-int4-symg128-ov / autocomplete-fim: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-symg128-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-symg64-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/SmolLM3-3B-int4-symg128-ov / codegen: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-symg128-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-symg64-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/SmolLM3-3B-int4-symg128-ov / edit: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov: not yet run on autocomplete-fim
- HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov: not yet run on autocomplete-fim
- HarmenWessels/Ornith-1.5-9B-int4-symg128-ov: not yet run on autocomplete-fim
- HarmenWessels/Ornith-1.5-9B-int4-symg64-ov: not yet run on autocomplete-fim

## Failures

**OpenVINO/Qwen3-4B-int4-ov / codegen**:
  - lru-cache#0: FAIL (NameError: name 'LRUCache' is not defined)

**HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov / codegen**:
  - rle-codec#0: FAIL (rle_decode(rle_encode('zzzzzzzzzzzz')) -> 'zz')

**OpenVINO/Qwen3-8B-int4-cw-ov / codegen**:
  - parse-duration#0: FAIL (NameError: name 'parse_duration' is not defined)

**HarmenWessels/K2-Horizon-7B-int4-symg128-ov / codegen**:
  - lru-cache#1: FAIL (NameError: name 'LRU' is not defined)

**HarmenWessels/gemma-4-E2B-it-qat-int4-ov / codegen**:
  - group-anagrams#0: FAIL (missing definition)

**HarmenWessels/Ornith-1.0-9B-int4-symg128-ov / codegen**:
  - rle-codec#0: FAIL (SyntaxError: invalid syntax (<string>, line 3))

**HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov / codegen**:
  - parse-duration#0: FAIL (NameError: name 'parse_duration' is not defined)

**HarmenWessels/granite-4.1-8b-int4-cw-code-ov / codegen**:
  - lru-cache#0: FAIL (_lru() -> [1, 2, 3, 1])
  - rle-codec#0: FAIL (NameError: name 'aaabccc' is not defined)

**HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov / codegen**:
  - parse-duration#0: FAIL (parse_duration('2h45m') -> 0)
  - rle-codec#1: FAIL (rle_decode('a3b1c3') -> 'abc')

**HarmenWessels/gemma-4-E4B-it-qat-int4-ov / codegen**:
  - group-anagrams#0: FAIL (missing definition)
  - group-anagrams#1: FAIL (missing definition)

**OpenVINO/Qwen3.5-9B-int4-ov / codegen**:
  - parse-duration#1: FAIL (SyntaxError: unterminated string literal (detected at line 28) (<string>, line 28))
  - rle-codec#0: FAIL (NameError: name 'text' is not defined)

**HarmenWessels/granite-4.1-8b-int4-cw-ov / codegen**:
  - merge-intervals#1: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [[1, 3]])
  - lru-cache#1: FAIL (_lru() -> [1, 2, -1, 1])
  - parse-duration#1: FAIL (ValueError: invalid literal for int() with base 10: '')

**HarmenWessels/Agents-A1-4B-int4-asymg128-ov / codegen**:
  - lru-cache#0: FAIL (_lru() -> [None, -1, None, None])
  - lru-cache#1: FAIL (AttributeError: 'NoneType' object has no attribute 'next')
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)

**HarmenWessels/Spark-X2.5-4B-int4-symg128-ov / codegen**:
  - merge-intervals#0: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [(1, 6), [8, 10]])
  - lru-cache#0: FAIL (missing definition)
  - parse-duration#1: FAIL (ValueError: invalid literal for int() with base 10: '2h45')
  - rle-codec#1: FAIL (IndexError: string index out of range)

**HarmenWessels/Ornith-1.5-9B-int4-symg128-ov / codegen**:
  - merge-intervals#1: FAIL (missing definition)
  - rate-limiter#1: FAIL (SyntaxError: invalid syntax (<string>, line 1))
  - lru-cache#1: FAIL (NameError: name 'LRUCache' is not defined)
  - parse-duration#0: FAIL (missing definition)

**Echo9Zulu/OmniCoder-9B-int4_sym-ov / codegen**:
  - merge-intervals#0: FAIL (NameError: name 'merge_intervals' is not defined)
  - parse-duration#0: FAIL (ValueError: invalid literal for int() with base 10: '')
  - parse-duration#1: FAIL (SyntaxError: invalid decimal literal (<string>, line 5))
  - rle-codec#0: FAIL (NameError: name 'text' is not defined)

**HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / codegen**:
  - rate-limiter#0: FAIL (_seq() -> [False, False, False, False])
  - rate-limiter#1: FAIL (_seq() -> [True, True, True, True])
  - lru-cache#1: FAIL (NameError: name 'LRUCache' is not defined)
  - parse-duration#0: FAIL (parse_duration('2h45m') -> 247)
  - parse-duration#1: FAIL (parse_duration('2h45m') -> 0)

**HarmenWessels/granite-4.1-3b-int4-cw-code-ov / codegen**:
  - merge-intervals#0: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [(1, 6), [8, 10]])
  - parse-duration#0: FAIL (UnboundLocalError: cannot access local variable 'number' where it is not associated with a value)
  - parse-duration#1: FAIL (UnboundLocalError: cannot access local variable 'number_str' where it is not associated with a value)
  - rle-codec#0: FAIL (IndexError: string index out of range)
  - rle-codec#1: FAIL (ValueError: invalid literal for int() with base 10: '')

**HarmenWessels/MiniCPM5-2B-int4-symg128-ov / codegen**:
  - rate-limiter#0: FAIL (TypeError: unsupported operand type(s) for -: 'int' and 'datetime.timedelta')
  - rate-limiter#1: FAIL (TypeError: unsupported operand type(s) for -: 'int' and 'datetime.timedelta')
  - parse-duration#0: FAIL (parse_duration('2h45m') -> 0)
  - parse-duration#1: FAIL (ValueError: invalid literal for int() with base 10: '')
  - rle-codec#0: FAIL (NameError: name 'asyt' is not defined)

**Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov / codegen**:
  - merge-intervals#0: FAIL (NameError: name 'merge_intervals' is not defined)
  - merge-intervals#1: FAIL (SyntaxError: 'return' outside function (<string>, line 2))
  - rate-limiter#1: FAIL (SyntaxError: unterminated string literal (detected at line 5) (<string>, line 5))
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)
  - rle-codec#1: FAIL (SyntaxError: invalid syntax (<string>, line 1))

**HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov / codegen**:
  - merge-intervals#0: FAIL (missing definition)
  - parse-duration#0: FAIL (NameError: name 'parse_duration' is not defined)
  - parse-duration#1: FAIL (ValueError: not enough values to unpack (expected 2, got 1))
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)
  - rle-codec#1: FAIL (IndexError: string index out of range)

**HarmenWessels/Ornith-1.5-9B-int4-symg64-ov / codegen**:
  - lru-cache#0: FAIL (NameError: name 'OrderedDict' is not defined)
  - lru-cache#1: FAIL (missing definition)
  - parse-duration#1: FAIL (parse_duration('2h45m') -> None)
  - rle-codec#0: FAIL (rle_encode('aaabccc') -> '02002')
  - group-anagrams#0: FAIL (missing definition)

**OpenVINO/Qwen3-1.7B-int4-ov / codegen**:
  - merge-intervals#0: FAIL (NameError: name 'merge_intervals' is not defined)
  - rate-limiter#1: FAIL (NameError: name 'SlidingWindowLimiter' is not defined)
  - parse-duration#0: FAIL (NameError: name 'parse_duration' is not defined)
  - parse-duration#1: FAIL (NameError: name 'parse_duration' is not defined)
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)
  - rle-codec#1: FAIL (NameError: name 'rle_encode' is not defined)

**OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / codegen**:
  - merge-intervals#0: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [(1, 6), (8, 10)])
  - merge-intervals#1: FAIL (IndexError: list index out of range)
  - rate-limiter#0: FAIL (_seq() -> [True, True, False, False])
  - parse-duration#0: FAIL (parse_duration('2h45m') -> 0)
  - parse-duration#1: FAIL (IndexError: no such group)
  - rle-codec#1: FAIL (exec timeout)

**HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov / codegen**:
  - lru-cache#0: FAIL (NameError: name 'LRUCache' is not defined)
  - lru-cache#1: FAIL (SyntaxError: unterminated string literal (detected at line 1) (<string>, line 1))
  - parse-duration#0: FAIL (SyntaxError: unterminated string literal (detected at line 21) (<string>, line 21))
  - parse-duration#1: FAIL (NameError: name 'parse_duration' is not defined)
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)
  - rle-codec#1: FAIL (NameError: name 'rle_encode' is not defined)

**HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov / codegen**:
  - rate-limiter#0: FAIL (TypeError: '>=' not supported between instances of 'int' and 'datetime.datetime')
  - rate-limiter#1: FAIL (_seq() -> [True, True, True, True])
  - parse-duration#0: FAIL (ValueError: invalid literal for int() with base 10: '2h45')
  - parse-duration#1: FAIL (ValueError: invalid literal for int() with base 10: '1h30m15')
  - rle-codec#0: FAIL (ValueError: invalid literal for int() with base 10: 'a')
  - rle-codec#1: FAIL (NameError: name 'rle_encode' is not defined)
  - group-anagrams#0: FAIL (_ga() -> [['a', 'b', 't'], ['a', 'e', 't'], ['a', 'n', 't']])

**HarmenWessels/SmolLM3-3B-int4-symg128-ov / codegen**:
  - merge-intervals#0: FAIL (NameError: name 'merge_intervals' is not defined)
  - rate-limiter#0: FAIL (_seq() -> [True, True, True, True])
  - rate-limiter#1: FAIL (TypeError: '<' not supported between instances of 'int' and 'NoneType')
  - parse-duration#0: FAIL (ValueError: invalid literal for int() with base 10: '2h45')
  - parse-duration#1: FAIL (TypeError: 'int' object is not iterable)
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)
  - rle-codec#1: FAIL (ValueError: invalid literal for int() with base 10: 'a')

**HarmenWessels/Spark-X2.5-1.7B-int4-symg128-ov / codegen**:
  - rate-limiter#0: FAIL (NameError: name 'SlidingWindowLimiter' is not defined)
  - lru-cache#0: FAIL (_lru() -> [1, 2, 3, 1])
  - parse-duration#0: FAIL (NameError: name 'parse_duration' is not defined)
  - parse-duration#1: FAIL (ValueError: empty separator)
  - rle-codec#0: FAIL (exec timeout)
  - rle-codec#1: FAIL (rle_encode('aaabccc') -> 'a3a1a3')
  - group-anagrams#0: FAIL (_ga() -> [['a', 'b', 't'], ['a', 'e', 't'], ['a', 'n', 't']])

**HarmenWessels/granite-4.1-3b-int4-cw-ov / codegen**:
  - merge-intervals#0: FAIL (NameError: name 'merge_intervals' is not defined)
  - rate-limiter#0: FAIL (TypeError: unsupported operand type(s) for -: 'datetime.datetime' and 'int')
  - rate-limiter#1: FAIL (ImportError: cannot import name 'deque' from 'datetime' (~\AppData\Local\Programs\Python\Python312\Lib\datetime.py))
  - lru-cache#0: FAIL (_lru() -> [1, 2, 3, -1])
  - lru-cache#1: FAIL (SyntaxError: unmatched ')' (<string>, line 46))
  - parse-duration#0: FAIL (ValueError: invalid literal for int() with base 10: '45m')
  - parse-duration#1: FAIL (parse_duration('90s') -> 324000)
  - rle-codec#1: FAIL (ValueError: invalid literal for int() with base 10: '')

**OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov / codegen**:
  - rate-limiter#0: FAIL (_seq() -> [False, False, False, False])
  - rate-limiter#1: FAIL (_seq() -> [False, False, False, False])
  - lru-cache#1: FAIL (AttributeError: 'LRUCache' object has no attribute '_remove_node')
  - parse-duration#0: FAIL (parse_duration('2h45m') -> 0)
  - parse-duration#1: FAIL (parse_duration('2h45m') -> 0)
  - rle-codec#0: FAIL (IndexError: string index out of range)
  - rle-codec#1: FAIL (NameError: name 'rle_encode' is not defined)
  - group-anagrams#0: FAIL (_ga() -> [['a', 'b', 't'], ['a', 'e', 't'], ['a', 'n', 't']])

**Echo9Zulu/Qwen3.5-2B-int4_sym-ov / codegen**:
  - rate-limiter#0: FAIL (_seq() -> [True, False, False, True])
  - lru-cache#0: FAIL (_lru() -> [1, -1, -1, 1])
  - lru-cache#1: FAIL (AttributeError: 'dict' object has no attribute 'move_to_end')
  - parse-duration#0: FAIL (ValueError: Invalid unit: 2h45m)
  - parse-duration#1: FAIL (SyntaxError: invalid syntax (<string>, line 1))
  - rle-codec#0: FAIL (rle_decode('a3b1c3') -> 'abc')
  - rle-codec#1: FAIL (rle_encode('aaabccc') -> '3a1b3c')
  - group-anagrams#0: FAIL (_ga() -> [['a', 'b', 't'], ['a', 'e', 't'], ['a', 'n', 't']])

**HarmenWessels/MiniCPM5-1B-int4-g128-ov / codegen**:
  - merge-intervals#0: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [[1, 3], [8, 10]])
  - rate-limiter#0: FAIL (TypeError: '<' not supported between instances of 'int' and 'datetime.datetime')
  - rate-limiter#1: FAIL (SyntaxError: invalid syntax (<string>, line 1))
  - lru-cache#0: FAIL (SyntaxError: closing parenthesis ')' does not match opening parenthesis '[' (<string>, line 30))
  - lru-cache#1: FAIL (missing definition)
  - parse-duration#0: FAIL (parse_duration('2h45m') -> 46800)
  - parse-duration#1: FAIL (ValueError: not enough values to unpack (expected 2, got 1))
  - rle-codec#0: FAIL (rle_decode('a3b1c3') -> 'a3b1c3a131b111c131')
  - rle-codec#1: FAIL (NameError: name 'rle_encode' is not defined)

**OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov / codegen**:
  - merge-intervals#0: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [[8, 10]])
  - merge-intervals#1: FAIL (IndexError: list index out of range)
  - rate-limiter#0: FAIL (AttributeError: 'int' object has no attribute 'total_seconds')
  - rate-limiter#1: FAIL (_seq() -> [False, False, False, False])
  - lru-cache#1: FAIL (KeyError: ('key5', 'value5'))
  - parse-duration#0: FAIL (AssertionError: Test case 1 failed)
  - parse-duration#1: FAIL (parse_duration('2h45m') -> 0)
  - rle-codec#0: FAIL (AssertionError: Failed for input: aab)
  - rle-codec#1: FAIL (NameError: name 'rle_decode' is not defined)

**Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / codegen**:
  - merge-intervals#0: FAIL (missing definition)
  - rate-limiter#0: FAIL (missing definition)
  - rate-limiter#1: FAIL (missing definition)
  - lru-cache#0: FAIL (missing definition)
  - lru-cache#1: FAIL (missing definition)
  - parse-duration#0: FAIL (missing definition)
  - parse-duration#1: FAIL (missing definition)
  - rle-codec#0: FAIL (rle_decode('a3b1c3') -> 'a131b111c131')
  - rle-codec#1: FAIL (missing definition)

**OpenVINO/Qwen3-0.6B-int4-ov / codegen**:
  - merge-intervals#0: FAIL (NameError: name 'merge_intervals' is not defined)
  - merge-intervals#1: FAIL (NameError: name 'merge_intervals' is not defined)
  - rate-limiter#0: FAIL (NameError: name 'SlidingWindowLimiter' is not defined)
  - rate-limiter#1: FAIL (AttributeError: 'SlidingWindowLimiter' object has no attribute '_window_seconds')
  - lru-cache#0: FAIL (AttributeError: 'LRUCache' object has no attribute 'put')
  - lru-cache#1: FAIL (NameError: name 'key' is not defined)
  - parse-duration#0: FAIL (NameError: name 'parse_duration' is not defined)
  - parse-duration#1: FAIL (NameError: name 'parse_duration' is not defined)
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)
  - rle-codec#1: FAIL (IndexError: string index out of range)
  - group-anagrams#0: FAIL (NameError: name 'group_anagrams' is not defined)

**HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov / codegen**:
  - merge-intervals#0: FAIL (SyntaxError: invalid syntax (<string>, line 1))
  - merge-intervals#1: FAIL (missing definition)
  - rate-limiter#0: FAIL (_seq() -> [True, True, True, True])
  - rate-limiter#1: FAIL (NameError: name 'SlidingWindowLimiter' is not defined)
  - lru-cache#0: FAIL (_lru() -> [-1, 2, 3, -1])
  - lru-cache#1: FAIL (_lru() -> [2, 3, 3, 3])
  - parse-duration#0: FAIL (IndexError: list index out of range)
  - parse-duration#1: FAIL (missing definition)
  - rle-codec#0: FAIL (missing definition)
  - rle-codec#1: FAIL (SyntaxError: unterminated string literal (detected at line 1) (<string>, line 1))
  - group-anagrams#1: FAIL (SyntaxError: '[' was never closed (<string>, line 1))

**HarmenWessels/granite-4.1-3b-int4-cw-code-ov / edit**:
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/MiniCPM5-2B-int4-symg128-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['read_file'])

**Echo9Zulu/Qwen3.5-2B-int4_sym-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['read_file'])

**HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov / edit**:
  - write-full: FAIL (writes=0 calls=['read_file'])

**HarmenWessels/Spark-X2.5-4B-int4-symg128-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['read_file'])

**HarmenWessels/gemma-4-E2B-it-qat-int4-ov / edit**:
  - write-full: FAIL (exec/assert failed: unterminated triple-quoted string literal (detecte)

**HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov / edit**:
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/Ornith-1.5-9B-int4-symg128-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['read_file'])

**HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov / edit**:
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov / edit**:
  - write-full: FAIL (exec/assert failed: '[' was never closed (<string>, line 15))

**HarmenWessels/gemma-4-E4B-it-qat-int4-ov / edit**:
  - write-full: FAIL (exec/assert failed: unterminated triple-quoted string literal (detecte)

**HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov / edit**:
  - write-full: FAIL (exec/assert failed: '[' was never closed (<string>, line 15))

**HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=[])

**HarmenWessels/gemma-4-12B-it-qat-int4-ov / edit**:
  - write-full: FAIL (exec/assert failed: unterminated triple-quoted string literal (detecte)

**HarmenWessels/Agents-A1-4B-int4-asymg128-ov / edit**:
  - write-full: FAIL (writes=3 calls=['write_file', 'write_file', 'write_file'])

**OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['moving_average'])
  - write-full: FAIL (writes=0 calls=['mean'])

**HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / edit**:
  - edit-exact: FAIL (old-match=False old='def moving_average(values, window):\\n    ')
  - write-full: FAIL (exec/assert failed: 'mean')

**HarmenWessels/MiniCPM5-1B-int4-g128-ov / edit**:
  - edit-exact: FAIL (old-match=True old='moving_average')
  - write-full: FAIL (exec/assert failed: invalid syntax (<string>, line 1))

**OpenVINO/Qwen3-0.6B-int4-ov / edit**:
  - edit-exact: FAIL (old-match=True old='def moving_average(values, window):')
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov / edit**:
  - edit-exact: FAIL (old-match=False old='moving_average = lambda values, window:')
  - write-full: FAIL (writes=0 calls=[])

**OpenVINO/Qwen3-1.7B-int4-ov / edit**:
  - edit-exact: FAIL (old-match=True old='return [sum(values[i:i + window]) / window\n      )
  - write-full: FAIL (writes=0 calls=[])

**OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov / edit**:
  - edit-exact: FAIL (old-match=True old='window must satisfy 1 <= window <= len(values)')
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/granite-4.1-3b-int4-cw-ov / edit**:
  - edit-exact: FAIL (old-match=True old='')
  - write-full: FAIL (writes=0 calls=[])

**OpenVINO/Qwen3-8B-int4-cw-ov / edit**:
  - edit-exact: FAIL (old-match=True old='return [sum(values[i:i + window]) / window\n      )
  - write-full: FAIL (writes=0 calls=['edit_file'])

**HarmenWessels/SmolLM3-3B-int4-symg128-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=[])
  - write-full: FAIL (writes=0 calls=[])

**OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=[])
  - write-full: FAIL (writes=0 calls=[])

**OpenVINO/Qwen3-4B-int4-ov / edit**:
  - edit-exact: FAIL (old-match=True old='return [sum(values[i:i + window]) / window\n      )
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/granite-4.1-8b-int4-cw-ov / edit**:
  - edit-exact: FAIL (old-match=True old='')
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/Ornith-1.0-9B-int4-symg128-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['read_file'])
  - write-full: FAIL (writes=0 calls=['web_search'])

**Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['read_file'])
  - write-full: FAIL (writes=0 calls=['read_file'])

**HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['read_file'])
  - write-full: FAIL (writes=0 calls=['read_file'])

**HarmenWessels/Ornith-1.5-9B-int4-symg64-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['read_file'])
  - write-full: FAIL (writes=0 calls=['read_file'])

**Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=[])
  - write-full: FAIL (exec/assert failed: '(' was never closed (<string>, line 1))

**HarmenWessels/granite-4.1-3b-int4-cw-code-ov / agent-loop**:
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**OpenVINO/Qwen3-1.7B-int4-ov / agent-loop**:
  - chain-depth: FAIL (stopped at turn 1: edits=0 green-tests-seen=0)

**OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / agent-loop**:
  - result-use: FAIL (calls=[('read_file', {'path': 'config.yaml'})] content='')

**OpenVINO/Qwen3-4B-int4-ov / agent-loop**:
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**HarmenWessels/MiniCPM5-2B-int4-symg128-ov / agent-loop**:
  - call-restraint: FAIL (calls=[('web_search', {'query': 'API acronym stands for'})] content='')

**HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov / agent-loop**:
  - call-choose: FAIL (calls=[])

**OpenVINO/Qwen3-8B-int4-cw-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])

**HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov / agent-loop**:
  - chain-depth: FAIL (turn 3: repeated call run_tests)

**OpenVINO/Qwen3-14B-int4-ov / agent-loop**:
  - chain-depth: FAIL (turn 1: repeated call edit_file)

**HarmenWessels/Ornith-1.0-9B-int4-symg128-ov / agent-loop**:
  - chain-depth: FAIL (turn 4: repeated call read_file)

**HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov / agent-loop**:
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**HarmenWessels/gemma-4-E2B-it-qat-int4-ov / agent-loop**:
  - chain-depth: FAIL (turn 3: repeated call read_file)

**Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov / agent-loop**:
  - chain-depth: FAIL (turn 4: repeated call read_file)

**HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov / agent-loop**:
  - chain-depth: FAIL (turn 5: repeated call run_tests)

**OpenVINO/Qwen3.5-9B-int4-ov / agent-loop**:
  - chain-depth: FAIL (turn 1: repeated call read_file)

**HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / agent-loop**:
  - no-repeat: FAIL (calls=[('read_file', "{'path': 'stats.py'}")])
  - chain-depth: FAIL (turn 1: repeated call run_tests)

**HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 1: edits=0 green-tests-seen=0)

**HarmenWessels/Spark-X2.5-1.7B-int4-symg128-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (turn 3: repeated call read_file)

**OpenVINO/Qwen3.5-4B-int4-ov / agent-loop**:
  - stop-done: FAIL (calls=['run_tests'])
  - chain-depth: FAIL (turn 2: repeated call run_tests)

**HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 2: edits=1 green-tests-seen=0)

**Echo9Zulu/OmniCoder-9B-int4_sym-ov / agent-loop**:
  - no-repeat: FAIL (calls=[('run_tests', '{}'), ('read_file', "{'path': 'stats.py'}"), ('r)
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**HarmenWessels/Ornith-1.5-9B-int4-symg128-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (turn 7: repeated call read_file)

**OpenVINO/Qwen3-0.6B-int4-ov / agent-loop**:
  - call-choose: FAIL (calls=[])
  - call-restraint: FAIL (calls=[] content='The acronym API stands for **API**.')
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**HarmenWessels/MiniCPM5-1B-int4-g128-ov / agent-loop**:
  - call-choose: FAIL (calls=[])
  - call-restraint: FAIL (calls=[] content="I don't have access to a tool that can determine the)
  - chain-depth: FAIL (stopped at turn 1: edits=0 green-tests-seen=0)

**HarmenWessels/SmolLM3-3B-int4-symg128-ov / agent-loop**:
  - call-restraint: FAIL (calls=[('web_search', {'query': 'What does the acronym API stand for?')
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 1: edits=0 green-tests-seen=0)

**HarmenWessels/granite-4.1-3b-int4-cw-ov / agent-loop**:
  - call-simple: FAIL (calls=[])
  - call-restraint: FAIL (calls=[('web_search', {'query': 'API acronym'})] content='')
  - chain-depth: FAIL (stopped at turn 1: edits=0 green-tests-seen=0)

**Echo9Zulu/Qwen3.5-2B-int4_sym-ov / agent-loop**:
  - call-simple: FAIL (calls=[])
  - no-repeat: FAIL (calls=[('read_file', "{'path': 'stats.py'}"), ('edit_file', "{'path': )
  - chain-depth: FAIL (turn 0: repeated call run_tests)

**HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov / agent-loop**:
  - call-choose: FAIL (calls=[('get_python_version', {'version': '3.9'})])
  - result-use: FAIL (calls=[] content='```json\n{\n  "function": {\n    "name": "read_file")
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (turn 2: repeated call unknown)

**HarmenWessels/Ornith-1.5-9B-int4-symg64-ov / agent-loop**:
  - call-simple: FAIL (calls=[])
  - no-repeat: FAIL (calls=[])
  - stop-done: FAIL (calls=['read_file'])
  - chain-depth: FAIL (turn 2: repeated call run_tests)

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

**HarmenWessels/Agents-A1-4B-int4-asymg128-ov / agent-loop**:
  - call-simple: FAIL (calls=[('web_search', {'query': 'config.yaml file location'})])
  - call-choose: FAIL (calls=[])
  - no-repeat: FAIL (calls=[])
  - stop-done: FAIL (calls=['read_file', 'run_tests', 'read_file', 'run_tests', 'read_file')
  - chain-depth: FAIL (turn 0: repeated call read_file)

**Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / agent-loop**:
  - call-simple: FAIL (calls=[])
  - call-choose: FAIL (calls=[])
  - call-restraint: FAIL (calls=[] content='')
  - result-use: FAIL (calls=[] content='')
  - no-repeat: FAIL (calls=[])
  - stop-done: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov / analysis**:
  - diagnose: FAIL (content='the error is that the function `median` is returning 3.5 when)

**HarmenWessels/granite-4.1-3b-int4-cw-ov / analysis**:
  - recall-deep: FAIL (content='(remembered from earlier) FROBNICATE_77.')

**OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / analysis**:
  - diagnose: FAIL (content='the issue lies in the line `return (s[mid] + s[mid + 1]) / 2`)

**HarmenWessels/granite-4.1-8b-int4-cw-ov / analysis**:
  - diagnose: FAIL (content='the test fails because the `median` function returns `3.5` fo)

**HarmenWessels/granite-4.1-8b-int4-cw-code-ov / analysis**:
  - diagnose: FAIL (content='the test fails because the expected median value for the list)

**HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov / analysis**:
  - route: FAIL (4/6 [chat:chat, edit:edit, design:design, chat:None, edit:edit, design)

**HarmenWessels/granite-4.1-3b-int4-cw-code-ov / analysis**:
  - recall-deep: FAIL (content='(answer omitted from transcript for brevity) (answer omitted )

**HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov / analysis**:
  - route: FAIL (3/6 [chat:chat, edit:edit, design:None, chat:chat, edit:chat, design:N)

**OpenVINO/Qwen3-1.7B-int4-ov / analysis**:
  - plan: FAIL (symbols=0 numbered=False code=False)

**HarmenWessels/gemma-4-E2B-it-qat-int4-ov / analysis**:
  - route: FAIL (3/6 [chat:None, edit:None, design:design, chat:None, edit:edit, design)

**HarmenWessels/gemma-4-E4B-it-qat-int4-ov / analysis**:
  - route: FAIL (2/6 [chat:None, edit:edit, design:None, chat:None, edit:edit, design:N)

**HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov / analysis**:
  - route: FAIL (5/6 [chat:chat, edit:edit, design:design, chat:None, edit:edit, design)

**Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov / analysis**:
  - route: FAIL (5/6 [chat:chat, edit:edit, design:None, chat:chat, edit:edit, design:d)

**OpenVINO/Qwen3-4B-int4-ov / analysis**:
  - plan: FAIL (symbols=0 numbered=False code=False)

**OpenVINO/Qwen3-8B-int4-cw-ov / analysis**:
  - plan: FAIL (symbols=0 numbered=False code=False)

**HarmenWessels/gemma-4-12B-it-qat-int4-ov / analysis**:
  - route: FAIL (5/6 [chat:chat, edit:None, design:design, chat:chat, edit:edit, design)

**Echo9Zulu/Qwen3.5-2B-int4_sym-ov / analysis**:
  - route: FAIL (3/6 [chat:None, edit:edit, design:None, chat:chat, edit:edit, design:N)

**HarmenWessels/Spark-X2.5-4B-int4-symg128-ov / analysis**:
  - recall-deep: FAIL (EXC: timed out)

**HarmenWessels/K2-Horizon-7B-int4-symg128-ov / analysis**:
  - route: FAIL (2/6 [chat:None, edit:edit, design:None, chat:None, edit:None, design:d)

**Echo9Zulu/OmniCoder-9B-int4_sym-ov / analysis**:
  - route: FAIL (5/6 [chat:chat, edit:edit, design:design, chat:chat, edit:edit, design)

**HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov / analysis**:
  - plan: FAIL (EXC: timed out)

**HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / analysis**:
  - route: FAIL (3/6 [chat:chat, edit:edit, design:chat, chat:edit, edit:edit, design:c)
  - diagnose: FAIL (content='the test fails because the expected median of `[4, 1, 3, 2]` )

**OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov / analysis**:
  - route: FAIL (3/6 [chat:edit, edit:edit, design:design, chat:edit, edit:edit, design)
  - plan: FAIL (symbols=4 numbered=False code=True)

**HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov / analysis**:
  - route: FAIL (4/6 [chat:design, edit:edit, design:design, chat:debug, edit:edit, des)
  - plan: FAIL (symbols=3 numbered=True code=True)

**HarmenWessels/MiniCPM5-1B-int4-g128-ov / analysis**:
  - route: FAIL (2/6 [chat:chat, edit:chat, design:design, chat:None, edit:None, design)
  - recall-deep: FAIL (content='The environment variable that holds our deployment API key is)

**OpenVINO/Qwen3-0.6B-int4-ov / analysis**:
  - route: FAIL (4/6 [chat:design, edit:edit, design:design, chat:design, edit:edit, de)
  - diagnose: FAIL (content='')

**Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / analysis**:
  - route: FAIL (5/6 [chat:chat, edit:edit, design:design, chat:chat, edit:chat, design)
  - diagnose: FAIL (content='')

**HarmenWessels/Ornith-1.5-9B-int4-symg128-ov / analysis**:
  - route: FAIL (5/6 [chat:chat, edit:edit, design:design, chat:chat, edit:None, design)
  - plan: FAIL (symbols=4 numbered=False code=False)

**HarmenWessels/SmolLM3-3B-int4-symg128-ov / analysis**:
  - plan: FAIL (symbols=0 numbered=False code=False)
  - recall-deep: FAIL (content='')

**HarmenWessels/Ornith-1.0-9B-int4-symg128-ov / analysis**:
  - route: FAIL (5/6 [chat:chat, edit:edit, design:design, chat:None, edit:edit, design)
  - diagnose: FAIL (content='the expression `(s[mid] + s[mid + 1]) / 2` is wrong because `)

**HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov / analysis**:
  - route: FAIL (0/6 [chat:None, edit:None, design:None, chat:None, edit:None, design:N)
  - diagnose: FAIL (content='{"route": "chat"}')

**HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov / analysis**:
  - route: FAIL (4/6 [chat:chat, edit:edit, design:design, chat:None, edit:None, design)
  - recall-deep: FAIL (content='')

**HarmenWessels/Ornith-1.5-9B-int4-symg64-ov / analysis**:
  - route: FAIL (4/6 [chat:chat, edit:None, design:design, chat:chat, edit:None, design)
  - diagnose: FAIL (content='{"route": "chat"}')
  - recall-deep: FAIL (content='{"route": "chat"}')

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

**OpenVINO/Qwen3-1.7B-int4-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/granite-4.1-8b-int4-cw-code-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/granite-4.1-8b-int4-cw-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

<!--LEADERBOARD END-->
