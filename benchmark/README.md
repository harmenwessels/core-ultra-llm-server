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
| 1 | [OpenVINO/Qwen3-14B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-14B-int4-ov) | 25/26 | 1901 | 0.0 GB | data-free |
| 2 | HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov | 24/26 | 1486 | 7.6 GB | awq+se |
| 3 | [HarmenWessels/gemma-4-12B-it-qat-int4-ov](https://huggingface.co/HarmenWessels/gemma-4-12B-it-qat-int4-ov) | 23/25 | 2019 | 0.0 GB | qat |
| 4 | HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov | 23/26 | 3080 | 7.6 GB | awq+se |
| 5 | [HarmenWessels/granite-4.1-8b-int4-cw-code-ov](https://huggingface.co/HarmenWessels/granite-4.1-8b-int4-cw-code-ov) | 22/26 | 859 | 4.4 GB | awq+se |
| 6 | HarmenWessels/K2-Horizon-7B-int4-symg128-ov | 22/26 | 2023 | 5.7 GB | awq+se |
| 7 | [HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov](https://huggingface.co/HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov) | 21/26 | 676 | 4.8 GB | awq+se |
| 8 | [OpenVINO/Qwen3-4B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-4B-int4-ov) | 21/26 | 786 | 2.3 GB | awq |
| 9 | [HarmenWessels/gemma-4-E2B-it-qat-int4-ov](https://huggingface.co/HarmenWessels/gemma-4-E2B-it-qat-int4-ov) | 21/25 | 1438 | 4.4 GB | qat |
| 10 | [OpenVINO/Qwen3-8B-int4-cw-ov](https://huggingface.co/OpenVINO/Qwen3-8B-int4-cw-ov) | 21/26 | 1612 | 4.7 GB | data-free |
| 11 | [HarmenWessels/gemma-4-E4B-it-qat-int4-ov](https://huggingface.co/HarmenWessels/gemma-4-E4B-it-qat-int4-ov) | 21/25 | 1621 | 6.6 GB | qat |
| 12 | HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov | 21/26 | 2911 | 4.9 GB | awq+se |
| 13 | HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov | 20/26 | 1525 | 4.9 GB | awq+se |
| 14 | [OpenVINO/Qwen3.5-4B-int4-ov](https://huggingface.co/OpenVINO/Qwen3.5-4B-int4-ov) | 20/25 | 2196 | 3.5 GB | data-free |
| 15 | [OpenVINO/Qwen3.5-9B-int4-ov](https://huggingface.co/OpenVINO/Qwen3.5-9B-int4-ov) | 20/25 | 2791 | 6.1 GB | data-free |
| 16 | HarmenWessels/MiniCPM5-2B-int4-symg128-ov | 19/26 | 433 | 1.6 GB | awq+se |
| 17 | [HarmenWessels/granite-4.1-8b-int4-cw-ov](https://huggingface.co/HarmenWessels/granite-4.1-8b-int4-cw-ov) | 19/26 | 781 | 4.4 GB | awq+se |
| 18 | [HarmenWessels/Ornith-1.0-9B-int4-symg128-ov](https://huggingface.co/HarmenWessels/Ornith-1.0-9B-int4-symg128-ov) | 19/25 | 1967 | 6.1 GB | data-free |
| 19 | [HarmenWessels/granite-4.1-3b-int4-cw-code-ov](https://huggingface.co/HarmenWessels/granite-4.1-3b-int4-cw-code-ov) | 17/26 | 586 | 0.0 GB | awq+se |
| 20 | HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov | 17/26 | 1084 | 3.2 GB | awq+se |
| 21 | [OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov](https://huggingface.co/OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov) | 16/26 | 465 | 1.8 GB | scale_estimation |
| 22 | HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov | 16/26 | 828 | 2.0 GB | awq+se |
| 23 | [Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov](https://huggingface.co/Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov) | 16/25 | 1877 | 0.0 GB | awq+se |
| 24 | HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov | 16/26 | 1892 | 2.0 GB | awq+se |
| 25 | HarmenWessels/Ornith-1.5-9B-int4-symg128-ov | 16/25 | 2719 | 0.0 GB | data-free |
| 26 | [Echo9Zulu/OmniCoder-9B-int4_sym-ov](https://huggingface.co/Echo9Zulu/OmniCoder-9B-int4_sym-ov) | 16/25 | 4110 | 0.0 GB | data-free |
| 27 | [HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov](https://huggingface.co/HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov) | 15/26 | 159 | 0.7 GB | data-free |
| 28 | HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov | 14/25 | 3294 | 0.0 GB | data-free |
| 29 | [HarmenWessels/granite-4.1-3b-int4-cw-ov](https://huggingface.co/HarmenWessels/granite-4.1-3b-int4-cw-ov) | 13/26 | 266 | 0.0 GB | awq+se |
| 30 | [OpenVINO/Qwen3-1.7B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-1.7B-int4-ov) | 13/26 | 567 | 0.0 GB | data-free |
| 31 | HarmenWessels/SmolLM3-3B-int4-symg128-ov | 12/26 | 914 | 0.0 GB | awq+se |
| 32 | [Echo9Zulu/Qwen3.5-2B-int4_sym-ov](https://huggingface.co/Echo9Zulu/Qwen3.5-2B-int4_sym-ov) | 11/25 | 2725 | 2.1 GB | data-free |
| 33 | HarmenWessels/Ornith-1.5-9B-int4-symg64-ov | 11/25 | 3350 | 0.0 GB | data-free |
| 34 | [HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov](https://huggingface.co/HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov) | 10/26 | 255 | 0.9 GB | data-free |
| 35 | [OpenVINO/Qwen3-0.6B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-0.6B-int4-ov) | 10/26 | 255 | 0.0 GB | data-free |
| 36 | [OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov](https://huggingface.co/OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov) | 10/26 | 337 | 0.9 GB | data-free |
| 37 | [HarmenWessels/MiniCPM5-1B-int4-g128-ov](https://huggingface.co/HarmenWessels/MiniCPM5-1B-int4-g128-ov) | 9/26 | 327 | 0.8 GB | data-free |
| 38 | HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov | 8/26 | 703 | 0.7 GB | awq+se |
| 39 | [OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov](https://huggingface.co/OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov) | 6/26 | 233 | 0.3 GB | data-free |
| 40 | [Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov](https://huggingface.co/Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov) | 5/26 | 1113 | 0.0 GB | data-free |

## Per-task-type leaderboard

_188 runs._

### codegen

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov | single | 4.8 GB | 12/12 | 465 | 39 | awq+se | sampling | nothink | 2026.3.0.0-3277 |
| 2 | OpenVINO/Qwen3-14B-int4-ov | single | 0.0 GB | 12/12 | 920 | 77 | data-free | sampling | nothink | 2026.3.0.0-1 |
| 3 | HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov | single | 4.9 GB | 12/12 | 1072 | 89 | awq+se | sampling | nothink | 2026.3.0.0-3277 |
| 4 | HarmenWessels/K2-Horizon-7B-int4-symg128-ov | single | 5.7 GB | 12/12 | 1097 | 91 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 5 | HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov | single | 7.6 GB | 12/12 | 1183 | 99 | awq+se | sampling | nothink | 2026.3.0.0-3277 |
| 6 | HarmenWessels/gemma-4-12B-it-qat-int4-ov | single | 0.0 GB | 12/12 | 1502 | 125 | qat | sampling | nothink | 2026.3.0.0-1 |
| 7 | HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov | single | 7.6 GB | 12/12 | 1731 | 144 | awq+se | sampling | nothink | 2026.3.0.0-3277 |
| 8 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 11/12 | 370 | 31 | awq | sampling | nothink | 2026.3.0.0-3277 |
| 9 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 11/12 | 980 | 82 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 10 | HarmenWessels/gemma-4-E2B-it-qat-int4-ov | single | 4.4 GB | 11/12 | 1061 | 88 | qat | sampling | nothink | 2026.3.0.0-3277 |
| 11 | HarmenWessels/Ornith-1.0-9B-int4-symg128-ov | single | 6.1 GB | 11/12 | 1081 | 90 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 12 | HarmenWessels/granite-4.1-8b-int4-cw-code-ov | single | 4.4 GB | 10/12 | 664 | 55 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 13 | OpenVINO/Qwen3.5-4B-int4-ov | single | 3.5 GB | 10/12 | 963 | 80 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 14 | HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov | single | 4.9 GB | 10/12 | 1070 | 89 | awq+se | sampling | nothink | 2026.3.0.0-3277 |
| 15 | HarmenWessels/gemma-4-E4B-it-qat-int4-ov | single | 6.6 GB | 10/12 | 1278 | 106 | qat | sampling | nothink | 2026.3.1.0-3290 |
| 16 | HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov | single | 2.0 GB | 9/12 | 514 | 43 | awq+se | sampling | nothink | 2026.3.0.0-3277 |
| 17 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 9/12 | 622 | 52 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 18 | OpenVINO/Qwen3.5-9B-int4-ov | single | 6.1 GB | 9/12 | 1609 | 134 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 19 | Echo9Zulu/OmniCoder-9B-int4_sym-ov | single | 0.0 GB | 9/12 | 1992 | 166 | data-free | greedy | nothink | 2026.3.0.0-1 |
| 20 | HarmenWessels/MiniCPM5-2B-int4-symg128-ov | single | 1.6 GB | 8/12 | 206 | 17 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 21 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 0.0 GB | 8/12 | 264 | 22 | awq+se | greedy | nothink | 2026.3.0.0-1 |
| 22 | HarmenWessels/Ornith-1.5-9B-int4-symg128-ov | single | 0.0 GB | 8/12 | 1773 | 148 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 23 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 7/12 | 109 | 9 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 24 | Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov | single | 0.0 GB | 7/12 | 1334 | 111 | awq+se | sampling | nothink | 2026.3.0.0-3277 |
| 25 | HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov | single | 0.0 GB | 7/12 | 1545 | 129 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 26 | HarmenWessels/Ornith-1.5-9B-int4-symg64-ov | single | 0.0 GB | 7/12 | 2469 | 206 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 27 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 0.0 GB | 6/12 | 186 | 16 | awq+se | greedy | nothink | 2026.3.0.0-1 |
| 28 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 6/12 | 375 | 31 | scale_estimation | sampling | nothink | 2026.3.0.0-3277 |
| 29 | HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov | single | 2.0 GB | 6/12 | 532 | 44 | awq+se | sampling | nothink | 2026.3.0.0-3277 |
| 30 | HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov | single | 3.2 GB | 6/12 | 579 | 48 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 31 | HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov | single | 0.9 GB | 5/12 | 183 | 15 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 32 | HarmenWessels/SmolLM3-3B-int4-symg128-ov | single | 0.0 GB | 5/12 | 457 | 38 | awq+se | sampling | nothink | 2026.3.0.0-1 |
| 33 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.0 GB | 4/12 | 119 | 10 | data-free | sampling | nothink | 2026.3.0.0-1 |
| 34 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 4/12 | 268 | 22 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 35 | OpenVINO/Qwen3-1.7B-int4-ov | single | 0.0 GB | 4/12 | 343 | 29 | data-free | sampling | nothink | 2026.3.0.0-1 |
| 36 | Echo9Zulu/Qwen3.5-2B-int4_sym-ov | single | 2.1 GB | 4/12 | 827 | 69 | data-free | greedy | nothink | 2026.3.0.0-3277 |
| 37 | HarmenWessels/MiniCPM5-1B-int4-g128-ov | single | 0.8 GB | 3/12 | 210 | 18 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 38 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.0 GB | 3/12 | 898 | 75 | data-free | sampling | nothink | 2026.3.0.0-1 |
| 39 | OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov | single | 0.3 GB | 1/12 | 171 | 14 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 40 | HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov | single | 0.7 GB | 1/12 | 488 | 41 | awq+se | sampling | nothink | 2026.3.1.0-3290 |

### edit

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HarmenWessels/granite-4.1-8b-int4-cw-code-ov | single | 4.4 GB | 2/2 | 28 | 14 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 2 | OpenVINO/Qwen3.5-4B-int4-ov | single | 3.5 GB | 2/2 | 28 | 14 | data-free | greedy | nothink | 2026.3.0.0-3277 |
| 3 | HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov | single | 4.9 GB | 2/2 | 34 | 17 | awq+se | sampling | nothink | 2026.3.0.0-3277 |
| 4 | HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov | single | 4.9 GB | 2/2 | 35 | 18 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 5 | OpenVINO/Qwen3.5-9B-int4-ov | single | 6.1 GB | 2/2 | 40 | 20 | data-free | greedy | nothink | 2026.3.0.0-3277 |
| 6 | OpenVINO/Qwen3-14B-int4-ov | single | 0.0 GB | 2/2 | 64 | 32 | data-free | greedy | nothink | 2026.3.0.0-1 |
| 7 | HarmenWessels/MiniCPM5-2B-int4-symg128-ov | single | 1.6 GB | 1/2 | 13 | 6 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 8 | Echo9Zulu/Qwen3.5-2B-int4_sym-ov | single | 2.1 GB | 1/2 | 17 | 8 | data-free | greedy | nothink | 2026.3.0.0-3277 |
| 9 | HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov | single | 7.6 GB | 1/2 | 29 | 14 | awq+se | sampling | nothink | 2026.3.0.0-3277 |
| 10 | HarmenWessels/gemma-4-E2B-it-qat-int4-ov | single | 4.4 GB | 1/2 | 30 | 15 | qat | greedy | nothink | 2026.3.0.0-3277 |
| 11 | HarmenWessels/Ornith-1.5-9B-int4-symg128-ov | single | 0.0 GB | 1/2 | 32 | 16 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 12 | HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov | single | 3.2 GB | 1/2 | 38 | 19 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 13 | HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov | single | 7.6 GB | 1/2 | 39 | 20 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 14 | HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov | single | 2.0 GB | 1/2 | 44 | 22 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 15 | HarmenWessels/gemma-4-E4B-it-qat-int4-ov | single | 6.6 GB | 1/2 | 50 | 25 | qat | greedy | nothink | 2026.3.1.0-3290 |
| 16 | HarmenWessels/K2-Horizon-7B-int4-symg128-ov | single | 5.7 GB | 1/2 | 56 | 28 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 17 | HarmenWessels/gemma-4-12B-it-qat-int4-ov | single | 0.0 GB | 1/2 | 74 | 37 | qat | greedy | nothink | 2026.3.0.0-1 |
| 18 | OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov | single | 0.3 GB | 0/2 | 5 | 2 | data-free | greedy | nothink | 2026.3.0.0-3277 |
| 19 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 0/2 | 6 | 3 | data-free | greedy | nothink | 2026.3.0.0-3277 |
| 20 | HarmenWessels/MiniCPM5-1B-int4-g128-ov | single | 0.8 GB | 0/2 | 7 | 4 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 21 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.0 GB | 0/2 | 7 | 4 | data-free | greedy | nothink | 2026.3.0.0-1 |
| 22 | HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov | single | 0.9 GB | 0/2 | 10 | 5 | data-free | greedy | nothink | 2026.3.0.0-3277 |
| 23 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 0/2 | 11 | 6 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 24 | OpenVINO/Qwen3-1.7B-int4-ov | single | 0.0 GB | 0/2 | 11 | 6 | data-free | greedy | nothink | 2026.3.0.0-1 |
| 25 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 0.0 GB | 0/2 | 14 | 7 | awq+se | greedy | nothink | 2026.3.0.0-1 |
| 26 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 0/2 | 15 | 8 | scale_estimation | greedy | nothink | 2026.3.0.0-3277 |
| 27 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 0/2 | 16 | 8 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 28 | HarmenWessels/SmolLM3-3B-int4-symg128-ov | single | 0.0 GB | 0/2 | 17 | 8 | awq+se | greedy | nothink | 2026.3.0.0-1 |
| 29 | HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov | single | 4.8 GB | 0/2 | 18 | 9 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 30 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 0/2 | 18 | 9 | awq | greedy | nothink | 2026.3.0.0-3277 |
| 31 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 0/2 | 24 | 12 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 32 | HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov | single | 0.7 GB | 0/2 | 28 | 14 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 33 | HarmenWessels/Ornith-1.0-9B-int4-symg128-ov | single | 6.1 GB | 0/2 | 28 | 14 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 34 | Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov | single | 0.0 GB | 0/2 | 29 | 14 | awq+se | sampling | nothink | 2026.3.0.0-3277 |
| 35 | HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov | single | 0.0 GB | 0/2 | 33 | 16 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 36 | HarmenWessels/Ornith-1.5-9B-int4-symg64-ov | single | 0.0 GB | 0/2 | 37 | 18 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 37 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.0 GB | 0/2 | 40 | 20 | data-free | greedy | nothink | 2026.3.0.0-1 |
| 38 | Echo9Zulu/OmniCoder-9B-int4_sym-ov | single | 0.0 GB | 0/2 | 84 | 42 | data-free | greedy | nothink | 2026.3.0.0-1 |
| 39 | HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov | single | 2.0 GB | 0/2 | 85 | 42 | awq+se | sampling | nothink | 2026.3.0.0-3277 |
| 40 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 0.0 GB | 0/2 | 98 | 49 | awq+se | greedy | nothink | 2026.3.0.0-1 |

### agent-loop

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HarmenWessels/gemma-4-E4B-it-qat-int4-ov | single | 6.6 GB | 7/7 | 45 | 6 | qat | greedy | nothink | 2026.3.1.0-3290 |
| 2 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 7/7 | 56 | 8 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 3 | HarmenWessels/granite-4.1-8b-int4-cw-code-ov | single | 4.4 GB | 7/7 | 76 | 11 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 4 | HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov | single | 3.2 GB | 7/7 | 77 | 11 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 5 | HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov | single | 7.6 GB | 7/7 | 113 | 16 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 6 | HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov | single | 7.6 GB | 7/7 | 166 | 24 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 7 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 6/7 | 31 | 4 | scale_estimation | greedy | nothink | 2026.3.0.0-3277 |
| 8 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 6/7 | 33 | 5 | awq | greedy | nothink | 2026.3.0.0-3277 |
| 9 | OpenVINO/Qwen3-1.7B-int4-ov | single | 0.0 GB | 6/7 | 36 | 5 | data-free | greedy | nothink | 2026.3.0.0-1 |
| 10 | HarmenWessels/MiniCPM5-2B-int4-symg128-ov | single | 1.6 GB | 6/7 | 37 | 5 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 11 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 0.0 GB | 6/7 | 40 | 6 | awq+se | greedy | nothink | 2026.3.0.0-1 |
| 12 | HarmenWessels/K2-Horizon-7B-int4-symg128-ov | single | 5.7 GB | 6/7 | 65 | 9 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 13 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 6/7 | 66 | 9 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 14 | OpenVINO/Qwen3-14B-int4-ov | single | 0.0 GB | 6/7 | 106 | 15 | data-free | greedy | nothink | 2026.3.0.0-1 |
| 15 | HarmenWessels/Ornith-1.0-9B-int4-symg128-ov | single | 6.1 GB | 6/7 | 112 | 16 | data-free | sampling | nothink | 2026.3.1.0-3290 |
| 16 | HarmenWessels/gemma-4-12B-it-qat-int4-ov | single | 0.0 GB | 6/7 | 127 | 18 | qat | greedy | nothink | 2026.3.0.0-1 |
| 17 | HarmenWessels/gemma-4-E2B-it-qat-int4-ov | single | 4.4 GB | 6/7 | 134 | 19 | qat | greedy | nothink | 2026.3.0.0-3277 |
| 18 | Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov | single | 0.0 GB | 6/7 | 163 | 23 | awq+se | sampling | nothink | 2026.3.0.0-3277 |
| 19 | OpenVINO/Qwen3.5-9B-int4-ov | single | 6.1 GB | 6/7 | 214 | 31 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 20 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 5/7 | 20 | 3 | data-free | greedy | nothink | 2026.3.0.0-3277 |
| 21 | HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov | single | 2.0 GB | 5/7 | 45 | 6 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 22 | HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov | single | 4.9 GB | 5/7 | 100 | 14 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 23 | HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov | single | 4.8 GB | 5/7 | 107 | 15 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 24 | HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov | single | 4.9 GB | 5/7 | 156 | 22 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 25 | OpenVINO/Qwen3.5-4B-int4-ov | single | 3.5 GB | 5/7 | 250 | 36 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 26 | HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov | single | 0.0 GB | 5/7 | 316 | 45 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 27 | HarmenWessels/Ornith-1.5-9B-int4-symg128-ov | single | 0.0 GB | 5/7 | 540 | 77 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 28 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 0.0 GB | 4/7 | 22 | 3 | awq+se | greedy | nothink | 2026.3.0.0-1 |
| 29 | HarmenWessels/MiniCPM5-1B-int4-g128-ov | single | 0.8 GB | 4/7 | 25 | 4 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 30 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.0 GB | 4/7 | 26 | 4 | data-free | greedy | nothink | 2026.3.0.0-1 |
| 31 | HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov | single | 0.7 GB | 4/7 | 34 | 5 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 32 | HarmenWessels/SmolLM3-3B-int4-symg128-ov | single | 0.0 GB | 4/7 | 36 | 5 | awq+se | greedy | nothink | 2026.3.0.0-1 |
| 33 | HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov | single | 2.0 GB | 4/7 | 111 | 16 | awq+se | sampling | nothink | 2026.3.1.0-3290 |
| 34 | Echo9Zulu/Qwen3.5-2B-int4_sym-ov | single | 2.1 GB | 4/7 | 250 | 36 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 35 | Echo9Zulu/OmniCoder-9B-int4_sym-ov | single | 0.0 GB | 4/7 | 347 | 50 | data-free | greedy | nothink | 2026.3.0.0-1 |
| 36 | HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov | single | 0.9 GB | 3/7 | 28 | 4 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 37 | HarmenWessels/Ornith-1.5-9B-int4-symg64-ov | single | 0.0 GB | 3/7 | 404 | 58 | data-free | sampling | nothink | 2026.3.0.0-3277 |
| 38 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 2/7 | 21 | 3 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 39 | OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov | single | 0.3 GB | 2/7 | 27 | 4 | data-free | greedy | nothink | 2026.3.0.0-3277 |
| 40 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.0 GB | 0/7 | 53 | 8 | data-free | greedy | nothink | 2026.3.0.0-1 |

### analysis

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov | single | 7.6 GB | 4/4 | 140 | 35 | awq+se | greedy | think | 2026.3.0.0-3277 |
| 2 | HarmenWessels/gemma-4-12B-it-qat-int4-ov | single | 0.0 GB | 4/4 | 316 | 79 | qat | greedy | think | 2026.3.0.0-1 |
| 3 | OpenVINO/Qwen3-14B-int4-ov | single | 0.0 GB | 4/4 | 797 | 199 | data-free | greedy | think | 2026.3.0.0-1 |
| 4 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 3/4 | 35 | 9 | data-free | greedy | think | 2026.3.1.0-3290 |
| 5 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 0.0 GB | 3/4 | 42 | 10 | awq+se | greedy | think | 2026.3.0.0-1 |
| 6 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 3/4 | 42 | 10 | scale_estimation | greedy | think | 2026.3.0.0-3277 |
| 7 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 3/4 | 72 | 18 | awq+se | greedy | think | 2026.3.0.0-3277 |
| 8 | HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov | single | 4.8 GB | 3/4 | 81 | 20 | awq+se | greedy | think | 2026.3.0.0-3277 |
| 9 | HarmenWessels/granite-4.1-8b-int4-cw-code-ov | single | 4.4 GB | 3/4 | 86 | 22 | awq+se | greedy | think | 2026.3.0.0-3277 |
| 10 | HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov | single | 0.7 GB | 3/4 | 151 | 38 | awq+se | sampling | think | 2026.3.1.0-3290 |
| 11 | HarmenWessels/MiniCPM5-2B-int4-symg128-ov | single | 1.6 GB | 3/4 | 173 | 43 | awq+se | sampling | think | 2026.3.1.0-3290 |
| 12 | OpenVINO/Qwen3-1.7B-int4-ov | single | 0.0 GB | 3/4 | 175 | 44 | data-free | greedy | think | 2026.3.0.0-1 |
| 13 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 0.0 GB | 3/4 | 182 | 46 | awq+se | greedy | think | 2026.3.0.0-1 |
| 14 | HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov | single | 2.0 GB | 3/4 | 203 | 51 | awq+se | greedy | think | 2026.3.0.0-3277 |
| 15 | HarmenWessels/gemma-4-E2B-it-qat-int4-ov | single | 4.4 GB | 3/4 | 213 | 53 | qat | greedy | think | 2026.3.0.0-3277 |
| 16 | HarmenWessels/gemma-4-E4B-it-qat-int4-ov | single | 6.6 GB | 3/4 | 248 | 62 | qat | greedy | think | 2026.3.1.0-3290 |
| 17 | HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov | single | 4.9 GB | 3/4 | 313 | 78 | awq+se | greedy | think | 2026.3.0.0-3277 |
| 18 | Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov | single | 0.0 GB | 3/4 | 351 | 88 | awq+se | sampling | think | 2026.3.0.0-3277 |
| 19 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 3/4 | 361 | 90 | awq | greedy | think | 2026.3.0.0-3277 |
| 20 | HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov | single | 3.2 GB | 3/4 | 385 | 96 | awq+se | sampling | think | 2026.3.1.0-3290 |
| 21 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 3/4 | 542 | 136 | data-free | greedy | think | 2026.3.1.0-3290 |
| 22 | HarmenWessels/K2-Horizon-7B-int4-symg128-ov | single | 5.7 GB | 3/4 | 796 | 199 | awq+se | sampling | think | 2026.3.1.0-3290 |
| 23 | OpenVINO/Qwen3.5-9B-int4-ov | single | 6.1 GB | 3/4 | 928 | 232 | data-free | greedy | think | 2026.3.0.0-3277 |
| 24 | OpenVINO/Qwen3.5-4B-int4-ov | single | 3.5 GB | 3/4 | 955 | 239 | data-free | greedy | think | 2026.3.0.0-3277 |
| 25 | HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov | single | 7.6 GB | 3/4 | 1143 | 286 | awq+se | sampling | think | 2026.3.0.0-3277 |
| 26 | Echo9Zulu/OmniCoder-9B-int4_sym-ov | single | 0.0 GB | 3/4 | 1687 | 422 | data-free | greedy | think | 2026.3.0.0-1 |
| 27 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 2/4 | 23 | 6 | data-free | greedy | think | 2026.3.0.0-3277 |
| 28 | OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov | single | 0.3 GB | 2/4 | 29 | 7 | data-free | greedy | think | 2026.3.0.0-3277 |
| 29 | HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov | single | 0.9 GB | 2/4 | 33 | 8 | data-free | greedy | think | 2026.3.0.0-3277 |
| 30 | HarmenWessels/MiniCPM5-1B-int4-g128-ov | single | 0.8 GB | 2/4 | 84 | 21 | data-free | greedy | think | 2026.3.1.0-3290 |
| 31 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.0 GB | 2/4 | 102 | 26 | data-free | greedy | think | 2026.3.0.0-1 |
| 32 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.0 GB | 2/4 | 120 | 30 | data-free | greedy | think | 2026.3.0.0-1 |
| 33 | HarmenWessels/Ornith-1.5-9B-int4-symg128-ov | single | 0.0 GB | 2/4 | 374 | 94 | data-free | sampling | think | 2026.3.0.0-3277 |
| 34 | HarmenWessels/SmolLM3-3B-int4-symg128-ov | single | 0.0 GB | 2/4 | 400 | 100 | awq+se | greedy | think | 2026.3.0.0-1 |
| 35 | HarmenWessels/Ornith-1.0-9B-int4-symg128-ov | single | 6.1 GB | 2/4 | 746 | 186 | data-free | sampling | think | 2026.3.1.0-3290 |
| 36 | HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov | single | 2.0 GB | 2/4 | 1178 | 294 | awq+se | sampling | think | 2026.3.0.0-3277 |
| 37 | HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov | single | 0.0 GB | 2/4 | 1400 | 350 | data-free | sampling | think | 2026.3.0.0-3277 |
| 38 | Echo9Zulu/Qwen3.5-2B-int4_sym-ov | single | 2.1 GB | 2/4 | 1631 | 408 | data-free | greedy | think | 2026.3.0.0-3277 |
| 39 | HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov | single | 4.9 GB | 2/4 | 1642 | 410 | awq+se | sampling | think | 2026.3.0.0-3277 |
| 40 | HarmenWessels/Ornith-1.5-9B-int4-symg64-ov | single | 0.0 GB | 1/4 | 440 | 110 | data-free | sampling | think | 2026.3.0.0-3277 |

### autocomplete-fim

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 1/1 | 1 | 1 | data-free | greedy | nothink | 2026.3.0.0-3277 |
| 2 | OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov | single | 0.3 GB | 1/1 | 1 | 1 | data-free | greedy | nothink | 2026.3.0.0-3277 |
| 3 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 1/1 | 2 | 2 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 4 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 1/1 | 2 | 2 | scale_estimation | greedy | nothink | 2026.3.0.0-3277 |
| 5 | HarmenWessels/MiniCPM5-2B-int4-symg128-ov | single | 1.6 GB | 1/1 | 4 | 4 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 6 | HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov | single | 2.0 GB | 1/1 | 4 | 4 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 7 | HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov | single | 2.0 GB | 1/1 | 4 | 4 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 8 | HarmenWessels/SmolLM3-3B-int4-symg128-ov | single | 0.0 GB | 1/1 | 4 | 4 | awq+se | greedy | nothink | 2026.3.0.0-1 |
| 9 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 1/1 | 4 | 4 | awq | greedy | nothink | 2026.3.0.0-3277 |
| 10 | HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov | single | 4.8 GB | 1/1 | 5 | 5 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 11 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 1/1 | 8 | 8 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 12 | OpenVINO/Qwen3-14B-int4-ov | single | 0.0 GB | 1/1 | 14 | 14 | data-free | greedy | nothink | 2026.3.0.0-1 |
| 13 | HarmenWessels/MiniCPM5-1B-int4-g128-ov | single | 0.8 GB | 0/1 | 1 | 1 | data-free | greedy | nothink | 2026.3.1.0-3290 |
| 14 | HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov | single | 0.9 GB | 0/1 | 1 | 1 | data-free | greedy | nothink | 2026.3.0.0-3277 |
| 15 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.0 GB | 0/1 | 1 | 1 | data-free | greedy | nothink | 2026.3.0.0-1 |
| 16 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.0 GB | 0/1 | 2 | 2 | data-free | greedy | nothink | 2026.3.0.0-1 |
| 17 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 0.0 GB | 0/1 | 2 | 2 | awq+se | greedy | nothink | 2026.3.0.0-1 |
| 18 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 0.0 GB | 0/1 | 2 | 2 | awq+se | greedy | nothink | 2026.3.0.0-1 |
| 19 | HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov | single | 0.7 GB | 0/1 | 2 | 2 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 20 | OpenVINO/Qwen3-1.7B-int4-ov | single | 0.0 GB | 0/1 | 2 | 2 | data-free | greedy | nothink | 2026.3.0.0-1 |
| 21 | HarmenWessels/granite-4.1-8b-int4-cw-code-ov | single | 4.4 GB | 0/1 | 5 | 5 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 22 | HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov | single | 3.2 GB | 0/1 | 5 | 5 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 23 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 0/1 | 7 | 7 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 24 | HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov | single | 4.9 GB | 0/1 | 7 | 7 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 25 | HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov | single | 4.9 GB | 0/1 | 7 | 7 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 26 | HarmenWessels/K2-Horizon-7B-int4-symg128-ov | single | 5.7 GB | 0/1 | 9 | 9 | awq+se | greedy | nothink | 2026.3.1.0-3290 |
| 27 | HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov | single | 7.6 GB | 0/1 | 11 | 11 | awq+se | greedy | nothink | 2026.3.0.0-3277 |
| 28 | HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov | single | 7.6 GB | 0/1 | 11 | 11 | awq+se | greedy | nothink | 2026.3.0.0-3277 |

## Retest queue

- Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / agent-loop: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/OmniCoder-9B-int4_sym-ov / agent-loop: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov / agent-loop: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/gemma-4-12B-it-qat-int4-ov / agent-loop: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/gemma-4-E2B-it-qat-int4-ov / agent-loop: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-3b-int4-cw-code-ov / agent-loop: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-3b-int4-cw-ov / agent-loop: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-8b-int4-cw-code-ov / agent-loop: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-8b-int4-cw-ov / agent-loop: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / agent-loop: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov / agent-loop: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-symg128-ov / agent-loop: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-symg64-ov / agent-loop: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov / agent-loop: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/SmolLM3-3B-int4-symg128-ov / agent-loop: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov / agent-loop: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / agent-loop: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-0.6B-int4-ov / agent-loop: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-1.7B-int4-ov / agent-loop: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-14B-int4-ov / agent-loop: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-4B-int4-ov / agent-loop: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / analysis: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/OmniCoder-9B-int4_sym-ov / analysis: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/Qwen3.5-2B-int4_sym-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/gemma-4-12B-it-qat-int4-ov / analysis: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/gemma-4-E2B-it-qat-int4-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-3b-int4-cw-code-ov / analysis: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-3b-int4-cw-ov / analysis: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-8b-int4-cw-code-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-8b-int4-cw-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-symg128-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-symg64-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/SmolLM3-3B-int4-symg128-ov / analysis: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-0.6B-int4-ov / analysis: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-1.7B-int4-ov / analysis: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-14B-int4-ov / analysis: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-4B-int4-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3.5-4B-int4-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3.5-9B-int4-ov / analysis: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / autocomplete-fim: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-3b-int4-cw-code-ov / autocomplete-fim: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-3b-int4-cw-ov / autocomplete-fim: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-8b-int4-cw-code-ov / autocomplete-fim: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-8b-int4-cw-ov / autocomplete-fim: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / autocomplete-fim: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov / autocomplete-fim: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov / autocomplete-fim: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov / autocomplete-fim: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov / autocomplete-fim: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov / autocomplete-fim: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov / autocomplete-fim: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov / autocomplete-fim: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov / autocomplete-fim: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/SmolLM3-3B-int4-symg128-ov / autocomplete-fim: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov / autocomplete-fim: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / autocomplete-fim: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-0.6B-int4-ov / autocomplete-fim: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-1.7B-int4-ov / autocomplete-fim: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-14B-int4-ov / autocomplete-fim: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-4B-int4-ov / autocomplete-fim: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / codegen: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/OmniCoder-9B-int4_sym-ov / codegen: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/Qwen3.5-2B-int4_sym-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/gemma-4-12B-it-qat-int4-ov / codegen: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/gemma-4-E2B-it-qat-int4-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-3b-int4-cw-code-ov / codegen: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-3b-int4-cw-ov / codegen: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-8b-int4-cw-code-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-8b-int4-cw-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-symg128-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-symg64-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/SmolLM3-3B-int4-symg128-ov / codegen: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-0.6B-int4-ov / codegen: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-1.7B-int4-ov / codegen: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-14B-int4-ov / codegen: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-4B-int4-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3.5-4B-int4-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3.5-9B-int4-ov / codegen: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / edit: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/OmniCoder-9B-int4_sym-ov / edit: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/Qwen3.5-2B-int4_sym-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/gemma-4-12B-it-qat-int4-ov / edit: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/gemma-4-E2B-it-qat-int4-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-3b-int4-cw-code-ov / edit: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-3b-int4-cw-ov / edit: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-8b-int4-cw-code-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/granite-4.1-8b-int4-cw-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-symg128-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Ornith-1.5-9B-int4-symg64-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- HarmenWessels/SmolLM3-3B-int4-symg128-ov / edit: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-0.6B-int4-ov / edit: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-1.7B-int4-ov / edit: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-14B-int4-ov / edit: engine 2026.3.0.0-1-796cb43d0bf-enable/google-gemma-4-12B != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3-4B-int4-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3.5-4B-int4-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- OpenVINO/Qwen3.5-9B-int4-ov / edit: engine 2026.3.0.0-3277-bd8d6542e3c != newest 2026.3.1.0-3290-56d9685302d
- Echo9Zulu/OmniCoder-9B-int4_sym-ov: not yet run on autocomplete-fim
- Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov: not yet run on autocomplete-fim
- HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov: not yet run on autocomplete-fim
- HarmenWessels/Ornith-1.5-9B-int4-symg128-ov: not yet run on autocomplete-fim
- HarmenWessels/Ornith-1.5-9B-int4-symg64-ov: not yet run on autocomplete-fim
- HarmenWessels/gemma-4-12B-it-qat-int4-ov: not yet run on autocomplete-fim

## Failures

**OpenVINO/Qwen3-4B-int4-ov / codegen**:
  - rle-codec#1: FAIL (NameError: name 'rle_encode' is not defined)

**OpenVINO/Qwen3-8B-int4-cw-ov / codegen**:
  - parse-duration#0: FAIL (NameError: name 'parse_duration' is not defined)

**HarmenWessels/gemma-4-E2B-it-qat-int4-ov / codegen**:
  - lru-cache#0: FAIL (AttributeError: 'dict' object has no attribute 'move_to_end')

**HarmenWessels/Ornith-1.0-9B-int4-symg128-ov / codegen**:
  - rle-codec#0: FAIL (SyntaxError: invalid syntax (<string>, line 3))

**HarmenWessels/granite-4.1-8b-int4-cw-code-ov / codegen**:
  - lru-cache#0: FAIL (_lru() -> [1, 2, 3, 1])
  - rle-codec#0: FAIL (NameError: name 'aaabccc' is not defined)

**OpenVINO/Qwen3.5-4B-int4-ov / codegen**:
  - merge-intervals#0: FAIL (NameError: name 'merge_intervals' is not defined)
  - parse-duration#0: FAIL (SyntaxError: invalid syntax (<string>, line 2))

**HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov / codegen**:
  - rate-limiter#1: FAIL (SyntaxError: invalid decimal literal (<string>, line 1))
  - rle-codec#1: FAIL (rle_decode('a3b1c3') -> 'aaa')

**HarmenWessels/gemma-4-E4B-it-qat-int4-ov / codegen**:
  - group-anagrams#0: FAIL (missing definition)
  - group-anagrams#1: FAIL (missing definition)

**HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov / codegen**:
  - lru-cache#1: FAIL (missing definition)
  - parse-duration#0: FAIL (parse_duration('2h45m') -> 14400)
  - parse-duration#1: FAIL (ValueError: invalid literal for int() with base 10: '15s')

**HarmenWessels/granite-4.1-8b-int4-cw-ov / codegen**:
  - merge-intervals#1: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [[1, 3]])
  - lru-cache#1: FAIL (_lru() -> [1, 2, -1, 1])
  - parse-duration#1: FAIL (ValueError: invalid literal for int() with base 10: '')

**OpenVINO/Qwen3.5-9B-int4-ov / codegen**:
  - parse-duration#0: FAIL (parse_duration('2h45m') -> 2700)
  - parse-duration#1: FAIL (SyntaxError: unterminated string literal (detected at line 19) (<string>, line 19))
  - rle-codec#0: FAIL (NameError: name 'text' is not defined)

**Echo9Zulu/OmniCoder-9B-int4_sym-ov / codegen**:
  - merge-intervals#0: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [(1, 6), (8, 10)])
  - parse-duration#1: FAIL (ValueError: not enough values to unpack (expected 2, got 1))
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)

**HarmenWessels/MiniCPM5-2B-int4-symg128-ov / codegen**:
  - rate-limiter#0: FAIL (_seq() -> [True, True, False, False])
  - parse-duration#1: FAIL (parse_duration('90s') -> 0)
  - rle-codec#0: FAIL (IndexError: string index out of range)
  - rle-codec#1: FAIL (IndexError: string index out of range)

**HarmenWessels/granite-4.1-3b-int4-cw-code-ov / codegen**:
  - merge-intervals#0: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [(1, 6), [8, 10]])
  - parse-duration#0: FAIL (parse_duration('2h45m') -> 7500)
  - parse-duration#1: FAIL (parse_duration('2h45m') -> 0)
  - rle-codec#1: FAIL (ValueError: invalid literal for int() with base 10: '')

**HarmenWessels/Ornith-1.5-9B-int4-symg128-ov / codegen**:
  - merge-intervals#1: FAIL (missing definition)
  - rate-limiter#1: FAIL (SyntaxError: invalid syntax (<string>, line 1))
  - lru-cache#1: FAIL (NameError: name 'LRUCache' is not defined)
  - parse-duration#0: FAIL (missing definition)

**HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / codegen**:
  - rate-limiter#0: FAIL (_seq() -> [True, True, True, False])
  - rate-limiter#1: FAIL (NameError: name 'SlidingWindowLimiter' is not defined)
  - parse-duration#0: FAIL (ValueError: invalid literal for int() with base 10: '2h45')
  - parse-duration#1: FAIL (RecursionError: maximum recursion depth exceeded)
  - rle-codec#1: FAIL (NameError: name 'rle_encode' is not defined)

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

**HarmenWessels/granite-4.1-3b-int4-cw-ov / codegen**:
  - rate-limiter#0: FAIL (ImportError: cannot import name 'deque' from 'datetime' (~\AppData\Local\Programs\Python\Python312\Lib\datetime.py))
  - rate-limiter#1: FAIL (ImportError: cannot import name 'deque' from 'datetime' (~\AppData\Local\Programs\Python\Python312\Lib\datetime.py))
  - lru-cache#0: FAIL (_lru() -> [1, 2, 3, -1])
  - parse-duration#0: FAIL (ValueError: invalid literal for int() with base 10: 'h')
  - parse-duration#1: FAIL (TypeError: 'NoneType' object is not subscriptable)
  - rle-codec#1: FAIL (ValueError: invalid literal for int() with base 10: '')

**OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / codegen**:
  - merge-intervals#0: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [(1, 6), [8, 10]])
  - merge-intervals#1: FAIL (IndexError: list index out of range)
  - rate-limiter#0: FAIL (_seq() -> [True, True, True, True])
  - parse-duration#0: FAIL (ValueError: invalid literal for int() with base 10: '')
  - parse-duration#1: FAIL (parse_duration('2h45m') -> 0)
  - rle-codec#1: FAIL (ValueError: invalid literal for int() with base 10: '')

**HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov / codegen**:
  - lru-cache#0: FAIL (NameError: name 'LRUCache' is not defined)
  - lru-cache#1: FAIL (SyntaxError: closing parenthesis ')' does not match opening parenthesis '[' (<string>, line 28))
  - parse-duration#0: FAIL (AttributeError: module 're' has no attribute 'constants')
  - parse-duration#1: FAIL (NameError: name 'parse_duration' is not defined)
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)
  - rle-codec#1: FAIL (NameError: name 'rle_encode' is not defined)

**HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov / codegen**:
  - rate-limiter#1: FAIL (AttributeError: 'list' object has no attribute 'popleft')
  - lru-cache#0: FAIL (NameError: name 'LRUCache' is not defined)
  - parse-duration#0: FAIL (parse_duration('2h45m') -> 7200)
  - parse-duration#1: FAIL (ValueError: invalid literal for int() with base 10: '2h45')
  - rle-codec#0: FAIL (TypeError: unsupported operand type(s) for *: 'NoneType' and 'int')
  - rle-codec#1: FAIL (NameError: name 'count' is not defined)

**HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov / codegen**:
  - rate-limiter#0: FAIL (IndexError: deque index out of range)
  - rate-limiter#1: FAIL (NameError: name 'SlidingWindowLimiter' is not defined)
  - parse-duration#0: FAIL (KeyError: '2h45m')
  - parse-duration#1: FAIL (NameError: name 'parse_duration' is not defined)
  - rle-codec#0: FAIL (IndexError: string index out of range)
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

**OpenVINO/Qwen3-0.6B-int4-ov / codegen**:
  - rate-limiter#0: FAIL (_seq() -> [True, True, True, True])
  - rate-limiter#1: FAIL (NameError: name 'SlidingWindowLimiter' is not defined)
  - lru-cache#0: FAIL (_lru() -> [1, 2, 3, 1])
  - lru-cache#1: FAIL (missing definition)
  - parse-duration#0: FAIL (NameError: name 'parse_duration' is not defined)
  - parse-duration#1: FAIL (NameError: name 'parse_duration' is not defined)
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)
  - rle-codec#1: FAIL (NameError: name 'rle_encode' is not defined)

**OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov / codegen**:
  - rate-limiter#0: FAIL (_seq() -> [False, False, False, False])
  - rate-limiter#1: FAIL (_seq() -> [False, False, False, False])
  - lru-cache#1: FAIL (AttributeError: 'LRUCache' object has no attribute '_remove_node')
  - parse-duration#0: FAIL (parse_duration('2h45m') -> 0)
  - parse-duration#1: FAIL (parse_duration('2h45m') -> 0)
  - rle-codec#0: FAIL (IndexError: string index out of range)
  - rle-codec#1: FAIL (NameError: name 'rle_encode' is not defined)
  - group-anagrams#0: FAIL (_ga() -> [['a', 'b', 't'], ['a', 'e', 't'], ['a', 'n', 't']])

**OpenVINO/Qwen3-1.7B-int4-ov / codegen**:
  - rate-limiter#0: FAIL (NameError: name 'SlidingWindowLimiter' is not defined)
  - rate-limiter#1: FAIL (NameError: name 'SlidingWindowLimiter' is not defined)
  - lru-cache#0: FAIL (ValueError: 2 is not in deque)
  - lru-cache#1: FAIL (missing definition)
  - parse-duration#0: FAIL (NameError: name 'parse_duration' is not defined)
  - parse-duration#1: FAIL (NameError: name 'parse_duration' is not defined)
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)
  - rle-codec#1: FAIL (NameError: name 'rle_encode' is not defined)

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

**Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / codegen**:
  - merge-intervals#1: FAIL (missing definition)
  - rate-limiter#0: FAIL (missing definition)
  - rate-limiter#1: FAIL (missing definition)
  - lru-cache#0: FAIL (missing definition)
  - lru-cache#1: FAIL (missing definition)
  - parse-duration#0: FAIL (missing definition)
  - parse-duration#1: FAIL (missing definition)
  - rle-codec#0: FAIL (missing definition)
  - rle-codec#1: FAIL (missing definition)

**OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov / codegen**:
  - merge-intervals#0: FAIL (TypeError: 'tuple' object does not support item assignment)
  - merge-intervals#1: FAIL (TypeError: '<' not supported between instances of 'list' and 'int')
  - rate-limiter#0: FAIL (_seq() -> [True, True, True, True])
  - rate-limiter#1: FAIL (_seq() -> [True, True, False, False])
  - lru-cache#0: FAIL (_lru() -> [1, 2, 3, 1])
  - lru-cache#1: FAIL (_lru() -> [1, 2, 3, 1])
  - parse-duration#0: FAIL (ValueError: invalid literal for int() with base 10: '2h45m')
  - parse-duration#1: FAIL (parse_duration('2h45m') -> 0)
  - rle-codec#0: FAIL (NameError: name 'rle_decode' is not defined)
  - rle-codec#1: FAIL (NameError: name 'rle_decode' is not defined)
  - group-anagrams#0: FAIL (_ga() -> [['a', 'b', 't'], ['a', 'e', 't'], ['a', 'n', 't']])

**HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov / codegen**:
  - merge-intervals#0: FAIL (SyntaxError: invalid syntax (<string>, line 1))
  - merge-intervals#1: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [(1, 6), (8, 10)])
  - rate-limiter#0: FAIL (NameError: name 'threading' is not defined)
  - rate-limiter#1: FAIL (SyntaxError: invalid syntax (<string>, line 7))
  - lru-cache#0: FAIL (TypeError: 'dict_keys' object is not subscriptable)
  - lru-cache#1: FAIL (NameError: name 'LRUCache' is not defined)
  - parse-duration#0: FAIL (IndexError: no such group)
  - parse-duration#1: FAIL (SyntaxError: invalid syntax. Perhaps you forgot a comma? (<string>, line 1))
  - rle-codec#0: FAIL (rle_encode('aaabccc') -> 'a3,b1,c3')
  - rle-codec#1: FAIL (rle_encode('aaabccc') -> 'a1a1a1a1a1a1a1a1a1b1c1c1c1')
  - group-anagrams#1: FAIL (_ga() -> [['ate', 'eat', 'tea'], ['ate', 'eat', 'tea'], ['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan'], ['nat', 'tan']])

**HarmenWessels/MiniCPM5-2B-int4-symg128-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['read_file'])

**Echo9Zulu/Qwen3.5-2B-int4_sym-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['read_file'])

**HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov / edit**:
  - write-full: FAIL (writes=0 calls=['web_search'])

**HarmenWessels/gemma-4-E2B-it-qat-int4-ov / edit**:
  - write-full: FAIL (exec/assert failed: unterminated triple-quoted string literal (detecte)

**HarmenWessels/Ornith-1.5-9B-int4-symg128-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['read_file'])

**HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov / edit**:
  - edit-exact: FAIL (old-match=False old='def moving_average(values, window):\n     """Movi)

**HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov / edit**:
  - write-full: FAIL (exec/assert failed: '[' was never closed (<string>, line 15))

**HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov / edit**:
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/gemma-4-E4B-it-qat-int4-ov / edit**:
  - write-full: FAIL (exec/assert failed: unterminated triple-quoted string literal (detecte)

**HarmenWessels/K2-Horizon-7B-int4-symg128-ov / edit**:
  - edit-exact: FAIL (old-match=False old='def moving_average(values, window):\n     """Movi)

**HarmenWessels/gemma-4-12B-it-qat-int4-ov / edit**:
  - write-full: FAIL (exec/assert failed: unterminated triple-quoted string literal (detecte)

**OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=['moving_average'])
  - write-full: FAIL (writes=0 calls=['mean'])

**HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / edit**:
  - edit-exact: FAIL (old-match=False old='def moving_average(values, window):\\n    \\')
  - write-full: FAIL (exec/assert failed: 'mean')

**HarmenWessels/MiniCPM5-1B-int4-g128-ov / edit**:
  - edit-exact: FAIL (old-match=True old='moving_average')
  - write-full: FAIL (exec/assert failed: invalid syntax (<string>, line 1))

**OpenVINO/Qwen3-0.6B-int4-ov / edit**:
  - edit-exact: FAIL (old-match=True old='window must satisfy 1 <= window <= len(values)')
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov / edit**:
  - edit-exact: FAIL (old-match=False old='moving_average = lambda values, window:')
  - write-full: FAIL (writes=0 calls=[])

**OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov / edit**:
  - edit-exact: FAIL (old-match=True old='window must satisfy 1 <= window <= len(values)')
  - write-full: FAIL (writes=0 calls=[])

**OpenVINO/Qwen3-1.7B-int4-ov / edit**:
  - edit-exact: FAIL (old-match=True old='return [sum(values[i:i + window]) / window\n      )
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/granite-4.1-3b-int4-cw-ov / edit**:
  - edit-exact: FAIL (old-match=True old='')
  - write-full: FAIL (writes=0 calls=[])

**OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=[])
  - write-full: FAIL (writes=0 calls=[])

**OpenVINO/Qwen3-8B-int4-cw-ov / edit**:
  - edit-exact: FAIL (old-match=True old='return [sum(values[i:i + window]) / window\n      )
  - write-full: FAIL (writes=0 calls=['edit_file'])

**HarmenWessels/SmolLM3-3B-int4-symg128-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=[])
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=[])
  - write-full: FAIL (writes=0 calls=['read_file'])

**OpenVINO/Qwen3-4B-int4-ov / edit**:
  - edit-exact: FAIL (old-match=True old='return [sum(values[i:i + window]) / window\n      )
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/granite-4.1-8b-int4-cw-ov / edit**:
  - edit-exact: FAIL (old-match=True old='')
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov / edit**:
  - edit-exact: FAIL (old-match=False old='def moving_average(values, window):\n     """Movi)
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
  - write-full: FAIL (exec/assert failed: '(' was never closed (<string>, line 4))

**Echo9Zulu/OmniCoder-9B-int4_sym-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=[])
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=[])
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/granite-4.1-3b-int4-cw-code-ov / edit**:
  - edit-exact: FAIL (old-match=True old='def moving_average(values, window):\n    """Moving)
  - write-full: FAIL (writes=0 calls=[])

**OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / agent-loop**:
  - result-use: FAIL (calls=[('read_file', {'path': 'config.yaml'})] content='')

**OpenVINO/Qwen3-4B-int4-ov / agent-loop**:
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**OpenVINO/Qwen3-1.7B-int4-ov / agent-loop**:
  - chain-depth: FAIL (stopped at turn 2: edits=0 green-tests-seen=0)

**HarmenWessels/MiniCPM5-2B-int4-symg128-ov / agent-loop**:
  - call-restraint: FAIL (calls=[('web_search', {'query': 'API acronym'})] content='')

**HarmenWessels/granite-4.1-3b-int4-cw-code-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])

**HarmenWessels/K2-Horizon-7B-int4-symg128-ov / agent-loop**:
  - stop-done: FAIL (calls=[])

**OpenVINO/Qwen3-8B-int4-cw-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])

**OpenVINO/Qwen3-14B-int4-ov / agent-loop**:
  - chain-depth: FAIL (turn 1: repeated call run_tests)

**HarmenWessels/Ornith-1.0-9B-int4-symg128-ov / agent-loop**:
  - chain-depth: FAIL (turn 4: repeated call read_file)

**HarmenWessels/gemma-4-12B-it-qat-int4-ov / agent-loop**:
  - stop-done: FAIL (calls=['read_file'])

**HarmenWessels/gemma-4-E2B-it-qat-int4-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])

**Echo9Zulu/Ornith-1.5-9B-int4_asym-awq-ov / agent-loop**:
  - chain-depth: FAIL (turn 4: repeated call read_file)

**OpenVINO/Qwen3.5-9B-int4-ov / agent-loop**:
  - chain-depth: FAIL (turn 1: repeated call read_file)

**HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / agent-loop**:
  - no-repeat: FAIL (calls=[('read_file', "{'path': 'stats.py'}")])
  - chain-depth: FAIL (turn 1: repeated call run_tests)

**HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 1: edits=0 green-tests-seen=0)

**HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov / agent-loop**:
  - call-simple: FAIL (calls=[])
  - chain-depth: FAIL (turn 3: repeated call run_tests)

**HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov / agent-loop**:
  - call-restraint: FAIL (calls=[('web_search', {'query': 'What does the acronym API stand for?')
  - chain-depth: FAIL (turn 1: repeated call run_tests)

**HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (turn 3: repeated call read_file)

**OpenVINO/Qwen3.5-4B-int4-ov / agent-loop**:
  - stop-done: FAIL (calls=['run_tests'])
  - chain-depth: FAIL (turn 2: repeated call run_tests)

**HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 2: edits=1 green-tests-seen=0)

**HarmenWessels/Ornith-1.5-9B-int4-symg128-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (turn 7: repeated call read_file)

**HarmenWessels/granite-4.1-3b-int4-cw-ov / agent-loop**:
  - call-simple: FAIL (calls=[])
  - call-restraint: FAIL (calls=[('web_search', {'query': 'API acronym'})] content='')
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**HarmenWessels/MiniCPM5-1B-int4-g128-ov / agent-loop**:
  - call-choose: FAIL (calls=[])
  - call-restraint: FAIL (calls=[] content="I don't have access to a tool that can determine the)
  - chain-depth: FAIL (stopped at turn 1: edits=0 green-tests-seen=0)

**OpenVINO/Qwen3-0.6B-int4-ov / agent-loop**:
  - call-choose: FAIL (calls=[])
  - call-restraint: FAIL (calls=[] content='The acronym API stands for **A-P-I**. It is used in )
  - chain-depth: FAIL (turn 2: repeated call run_tests)

**HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov / agent-loop**:
  - call-simple: FAIL (calls=[('read_file', {'path': '/cedarfeather'})])
  - call-choose: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**HarmenWessels/SmolLM3-3B-int4-symg128-ov / agent-loop**:
  - call-restraint: FAIL (calls=[('web_search', {'query': 'What does the acronym API stand for?')
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 1: edits=0 green-tests-seen=0)

**HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov / agent-loop**:
  - call-simple: FAIL (calls=[])
  - no-repeat: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**Echo9Zulu/Qwen3.5-2B-int4_sym-ov / agent-loop**:
  - call-simple: FAIL (calls=[])
  - no-repeat: FAIL (calls=[('read_file', "{'path': 'stats.py'}"), ('edit_file', "{'path': )
  - chain-depth: FAIL (turn 0: repeated call run_tests)

**Echo9Zulu/OmniCoder-9B-int4_sym-ov / agent-loop**:
  - call-simple: FAIL (calls=[])
  - call-choose: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

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

**HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov / analysis**:
  - recall-deep: FAIL (content='(answer omitted from transcript for brevity) (answer omitted )

**HarmenWessels/granite-4.1-8b-int4-cw-code-ov / analysis**:
  - diagnose: FAIL (content='the test fails because the expected median value for the list)

**HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov / analysis**:
  - route: FAIL (4/6 [chat:chat, edit:edit, design:design, chat:chat, edit:None, design)

**HarmenWessels/MiniCPM5-2B-int4-symg128-ov / analysis**:
  - route: FAIL (5/6 [chat:chat, edit:edit, design:design, chat:design, edit:edit, desi)

**OpenVINO/Qwen3-1.7B-int4-ov / analysis**:
  - plan: FAIL (symbols=0 numbered=False code=False)

**HarmenWessels/granite-4.1-3b-int4-cw-code-ov / analysis**:
  - recall-deep: FAIL (content='(answer omitted from transcript for brevity) (answer omitted )

**HarmenWessels/Ministral-3-3B-Instruct-int4-symg128-ov / analysis**:
  - route: FAIL (3/6 [chat:chat, edit:edit, design:None, chat:chat, edit:chat, design:N)

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

**HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov / analysis**:
  - route: FAIL (4/6 [chat:chat, edit:None, design:None, chat:chat, edit:edit, design:d)

**OpenVINO/Qwen3-8B-int4-cw-ov / analysis**:
  - plan: FAIL (symbols=0 numbered=False code=False)

**HarmenWessels/K2-Horizon-7B-int4-symg128-ov / analysis**:
  - route: FAIL (3/6 [chat:None, edit:None, design:design, chat:None, edit:edit, design)

**OpenVINO/Qwen3.5-9B-int4-ov / analysis**:
  - route: FAIL (5/6 [chat:chat, edit:edit, design:design, chat:chat, edit:edit, design)

**OpenVINO/Qwen3.5-4B-int4-ov / analysis**:
  - route: FAIL (0/6 [chat:None, edit:None, design:None, chat:None, edit:None, design:c)

**HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov / analysis**:
  - plan: FAIL (EXC: timed out)

**Echo9Zulu/OmniCoder-9B-int4_sym-ov / analysis**:
  - route: FAIL (2/6 [chat:chat, edit:None, design:chat, chat:chat, edit:None, design:N)

**HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / analysis**:
  - route: FAIL (2/6 [chat:chat, edit:design, design:chat, chat:explain, edit:edit, des)
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
  - diagnose: FAIL (content='')
  - plan: FAIL (symbols=1 numbered=True code=False)

**HarmenWessels/Ornith-1.5-9B-int4-symg128-ov / analysis**:
  - route: FAIL (5/6 [chat:chat, edit:edit, design:design, chat:chat, edit:None, design)
  - plan: FAIL (symbols=4 numbered=False code=False)

**HarmenWessels/SmolLM3-3B-int4-symg128-ov / analysis**:
  - plan: FAIL (symbols=0 numbered=False code=False)
  - recall-deep: FAIL (content='')

**HarmenWessels/Ornith-1.0-9B-int4-symg128-ov / analysis**:
  - route: FAIL (5/6 [chat:chat, edit:edit, design:design, chat:None, edit:edit, design)
  - diagnose: FAIL (content='the expression `(s[mid] + s[mid + 1]) / 2` is wrong because `)

**HarmenWessels/Ministral-3-3B-Reasoning-int4-symg128-ov / analysis**:
  - route: FAIL (4/6 [chat:chat, edit:edit, design:None, chat:chat, edit:edit, design:c)
  - plan: FAIL (symbols=0 numbered=False code=False)

**HarmenWessels/Ornith-1.5-9B-int4-asymg128-ov / analysis**:
  - route: FAIL (0/6 [chat:None, edit:None, design:None, chat:None, edit:None, design:N)
  - diagnose: FAIL (content='{"route": "chat"}')

**Echo9Zulu/Qwen3.5-2B-int4_sym-ov / analysis**:
  - route: FAIL (4/6 [chat:None, edit:None, design:design, chat:chat, edit:edit, design)
  - plan: FAIL (symbols=0 numbered=False code=False)

**HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov / analysis**:
  - route: FAIL (5/6 [chat:None, edit:edit, design:design, chat:chat, edit:edit, design)
  - recall-deep: FAIL (content='')

**HarmenWessels/Ornith-1.5-9B-int4-symg64-ov / analysis**:
  - route: FAIL (4/6 [chat:chat, edit:None, design:design, chat:chat, edit:None, design)
  - diagnose: FAIL (content='{"route": "chat"}')
  - recall-deep: FAIL (content='{"route": "chat"}')

**HarmenWessels/MiniCPM5-1B-int4-g128-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/Qwen2.5-Coder-1.5B-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

**OpenVINO/Qwen3-0.6B-int4-ov / autocomplete-fim**:
  - merge-fim: FAIL

**Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/granite-4.1-3b-int4-cw-code-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/granite-4.1-3b-int4-cw-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

**OpenVINO/Qwen3-1.7B-int4-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/granite-4.1-8b-int4-cw-code-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/granite-4.1-8b-int4-cw-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/Ministral-3-8B-Instruct-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/Ministral-3-8B-Reasoning-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/K2-Horizon-7B-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/Ministral-3-14B-Instruct-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/Ministral-3-14B-Reasoning-int4-symg128-ov / autocomplete-fim**:
  - merge-fim: FAIL

<!--LEADERBOARD END-->
