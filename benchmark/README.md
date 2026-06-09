# Benchmark

A clean, provenance-logged benchmark that scores each locally-runnable model — and
combinations of models — per task type, to decide which single models and which
`virtual/agent` combos are worth serving. The repo's product is the server; this is the
tool that tells us what to put in it.

## Method

- **Backend:** OpenVINO **GenAI only** (the source-built engine in `.venv-genai`). One engine,
  no cross-engine confounds.
- **Task types (5 suites):** `codegen` · `edit` · `autocomplete-fim` · `agent-loop`
  · `analysis` (diagnose / plan / route / recall).
- **Scoring:** per (entry, task type) → **quality** = probe pass-rate, **runtime** = total
  wall-clock to solve the suite. Ranked **quality first, then total runtime**. tok/s is logged
  but is *never* the rank key — more tokens for the same task is not per se better.
- **Decoding:** every model runs at the operating point in **its card** (`cards/<owner>__<name>.yaml`):
  sampling for open-ended `codegen` (generative class), greedy for `edit`/`agent-loop`/`analysis`/`fim`
  (structured/fim classes — rule 0f). Best-of-N blocks on the non-deterministic VLM path.
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
| 1 | [OpenVINO/Qwen3-14B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-14B-int4-ov) | 25/26 | 1881 | 9.7 GB | data-free |
| 2 | [HarmenWessels/gemma-4-12B-it-qat-int4-ov](https://huggingface.co/HarmenWessels/gemma-4-12B-it-qat-int4-ov) | 23/25 | 1970 | 8.2 GB | qat |
| 3 | [OpenVINO/Qwen3-8B-int4-cw-ov](https://huggingface.co/OpenVINO/Qwen3-8B-int4-cw-ov) | 22/26 | 1393 | 4.7 GB | data-free |
| 4 | [HarmenWessels/gemma-4-E4B-it-qat-int4-ov](https://huggingface.co/HarmenWessels/gemma-4-E4B-it-qat-int4-ov) | 22/25 | 1421 | 6.6 GB | qat |
| 5 | [OpenVINO/Qwen3-4B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-4B-int4-ov) | 20/26 | 877 | 2.3 GB | awq |
| 6 | [HarmenWessels/granite-4.1-8b-int4-cw-ov](https://huggingface.co/HarmenWessels/granite-4.1-8b-int4-cw-ov) | 19/26 | 768 | 4.4 GB | awq+se |
| 7 | [HarmenWessels/granite-4.1-3b-int4-cw-code-ov](https://huggingface.co/HarmenWessels/granite-4.1-3b-int4-cw-code-ov) | 17/26 | 507 | 1.8 GB | awq+se |
| 8 | [HarmenWessels/gemma-4-E2B-it-qat-int4-ov](https://huggingface.co/HarmenWessels/gemma-4-E2B-it-qat-int4-ov) | 17/25 | 1510 | 4.4 GB | qat |
| 9 | [OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov](https://huggingface.co/OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov) | 16/26 | 410 | 1.8 GB | scale_estimation |
| 10 | [Echo9Zulu/OmniCoder-9B-int4_sym-ov](https://huggingface.co/Echo9Zulu/OmniCoder-9B-int4_sym-ov) | 16/25 | 3651 | 6.1 GB | data-free |
| 11 | [HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov](https://huggingface.co/HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov) | 14/26 | 188 | 0.7 GB | data-free |
| 12 | [OpenVINO/Qwen3-1.7B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-1.7B-int4-ov) | 13/26 | 541 | 1.2 GB | data-free |
| 13 | [HarmenWessels/granite-4.1-3b-int4-cw-ov](https://huggingface.co/HarmenWessels/granite-4.1-3b-int4-cw-ov) | 12/26 | 260 | 1.8 GB | awq+se |
| 14 | [Echo9Zulu/Qwen3.5-2B-int4_sym-ov](https://huggingface.co/Echo9Zulu/Qwen3.5-2B-int4_sym-ov) | 12/25 | 1688 | 2.1 GB | data-free |
| 15 | [OpenVINO/Qwen3-0.6B-int4-ov](https://huggingface.co/OpenVINO/Qwen3-0.6B-int4-ov) | 10/26 | 257 | 0.4 GB | data-free |
| 16 | [OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov](https://huggingface.co/OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov) | 10/26 | 300 | 0.9 GB | data-free |
| 17 | [Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov](https://huggingface.co/Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov) | 5/26 | 989 | 0.7 GB | data-free |

## Per-task-type leaderboard

_81 runs._

### codegen

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 12/12 | 735 | 61 | data-free | sampling | nothink | 2026.3.0.0 |
| 2 | OpenVINO/Qwen3-14B-int4-ov | single | 9.7 GB | 12/12 | 910 | 76 | data-free | sampling | nothink | 2026.3.0.0 |
| 3 | HarmenWessels/gemma-4-12B-it-qat-int4-ov | single | 8.2 GB | 12/12 | 1446 | 120 | qat | sampling | nothink | 2026.3.0.0 |
| 4 | HarmenWessels/gemma-4-E4B-it-qat-int4-ov | single | 6.6 GB | 10/12 | 1188 | 99 | qat | sampling | nothink | 2026.3.0.0 |
| 5 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 9/12 | 503 | 42 | awq | sampling | nothink | 2026.3.0.0 |
| 6 | Echo9Zulu/OmniCoder-9B-int4_sym-ov | single | 6.1 GB | 9/12 | 1789 | 149 | data-free | greedy | nothink | 2026.3.0.0 |
| 7 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 1.8 GB | 8/12 | 220 | 18 | awq+se | greedy | nothink | 2026.3.0.0 |
| 8 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 8/12 | 615 | 51 | awq+se | greedy | nothink | 2026.3.0.0 |
| 9 | HarmenWessels/gemma-4-E2B-it-qat-int4-ov | single | 4.4 GB | 8/12 | 1064 | 89 | qat | sampling | nothink | 2026.3.0.0 |
| 10 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 6/12 | 132 | 11 | data-free | sampling | nothink | 2026.3.0.0 |
| 11 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 6/12 | 324 | 27 | scale_estimation | sampling | nothink | 2026.3.0.0 |
| 12 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 1.8 GB | 5/12 | 181 | 15 | awq+se | greedy | nothink | 2026.3.0.0 |
| 13 | Echo9Zulu/Qwen3.5-2B-int4_sym-ov | single | 2.1 GB | 5/12 | 824 | 69 | data-free | greedy | nothink | 2026.3.0.0 |
| 14 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.4 GB | 4/12 | 120 | 10 | data-free | sampling | nothink | 2026.3.0.0 |
| 15 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 4/12 | 217 | 18 | data-free | sampling | nothink | 2026.3.0.0 |
| 16 | OpenVINO/Qwen3-1.7B-int4-ov | single | 1.2 GB | 4/12 | 317 | 26 | data-free | sampling | nothink | 2026.3.0.0 |
| 17 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.7 GB | 3/12 | 787 | 66 | data-free | sampling | nothink | 2026.3.0.0 |

### edit

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | OpenVINO/Qwen3-14B-int4-ov | single | 9.7 GB | 2/2 | 62 | 31 | data-free | greedy | nothink | 2026.3.0.0 |
| 2 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 1/2 | 23 | 12 | awq+se | greedy | nothink | 2026.3.0.0 |
| 3 | HarmenWessels/gemma-4-E2B-it-qat-int4-ov | single | 4.4 GB | 1/2 | 34 | 17 | qat | greedy | nothink | 2026.3.0.0 |
| 4 | HarmenWessels/gemma-4-E4B-it-qat-int4-ov | single | 6.6 GB | 1/2 | 35 | 18 | qat | greedy | nothink | 2026.3.0.0 |
| 5 | HarmenWessels/gemma-4-12B-it-qat-int4-ov | single | 8.2 GB | 1/2 | 74 | 37 | qat | greedy | nothink | 2026.3.0.0 |
| 6 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.4 GB | 0/2 | 7 | 4 | data-free | greedy | nothink | 2026.3.0.0 |
| 7 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 0/2 | 9 | 4 | data-free | greedy | nothink | 2026.3.0.0 |
| 8 | OpenVINO/Qwen3-1.7B-int4-ov | single | 1.2 GB | 0/2 | 12 | 6 | data-free | greedy | nothink | 2026.3.0.0 |
| 9 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 1.8 GB | 0/2 | 13 | 6 | awq+se | greedy | nothink | 2026.3.0.0 |
| 10 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 0/2 | 14 | 7 | scale_estimation | greedy | nothink | 2026.3.0.0 |
| 11 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 0/2 | 15 | 8 | data-free | greedy | nothink | 2026.3.0.0 |
| 12 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 0/2 | 17 | 8 | awq | greedy | nothink | 2026.3.0.0 |
| 13 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 0/2 | 20 | 10 | data-free | greedy | nothink | 2026.3.0.0 |
| 14 | Echo9Zulu/Qwen3.5-2B-int4_sym-ov | single | 2.1 GB | 0/2 | 23 | 12 | data-free | greedy | nothink | 2026.3.0.0 |
| 15 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.7 GB | 0/2 | 38 | 19 | data-free | greedy | nothink | 2026.3.0.0 |
| 16 | Echo9Zulu/OmniCoder-9B-int4_sym-ov | single | 6.1 GB | 0/2 | 78 | 39 | data-free | greedy | nothink | 2026.3.0.0 |
| 17 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 1.8 GB | 0/2 | 81 | 40 | awq+se | greedy | nothink | 2026.3.0.0 |

### agent-loop

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HarmenWessels/gemma-4-E4B-it-qat-int4-ov | single | 6.6 GB | 7/7 | 47 | 7 | qat | greedy | nothink | 2026.3.0.0 |
| 2 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 7/7 | 55 | 8 | awq+se | greedy | nothink | 2026.3.0.0 |
| 3 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 6/7 | 33 | 5 | scale_estimation | greedy | nothink | 2026.3.0.0 |
| 4 | OpenVINO/Qwen3-1.7B-int4-ov | single | 1.2 GB | 6/7 | 35 | 5 | data-free | greedy | nothink | 2026.3.0.0 |
| 5 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 6/7 | 36 | 5 | awq | greedy | nothink | 2026.3.0.0 |
| 6 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 1.8 GB | 6/7 | 40 | 6 | awq+se | greedy | nothink | 2026.3.0.0 |
| 7 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 6/7 | 60 | 9 | data-free | greedy | nothink | 2026.3.0.0 |
| 8 | OpenVINO/Qwen3-14B-int4-ov | single | 9.7 GB | 6/7 | 107 | 15 | data-free | greedy | nothink | 2026.3.0.0 |
| 9 | HarmenWessels/gemma-4-12B-it-qat-int4-ov | single | 8.2 GB | 6/7 | 120 | 17 | qat | greedy | nothink | 2026.3.0.0 |
| 10 | HarmenWessels/gemma-4-E2B-it-qat-int4-ov | single | 4.4 GB | 6/7 | 160 | 23 | qat | greedy | nothink | 2026.3.0.0 |
| 11 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 5/7 | 21 | 3 | data-free | greedy | nothink | 2026.3.0.0 |
| 12 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 1.8 GB | 4/7 | 23 | 3 | awq+se | greedy | nothink | 2026.3.0.0 |
| 13 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.4 GB | 4/7 | 27 | 4 | data-free | greedy | nothink | 2026.3.0.0 |
| 14 | Echo9Zulu/Qwen3.5-2B-int4_sym-ov | single | 2.1 GB | 4/7 | 147 | 21 | data-free | greedy | nothink | 2026.3.0.0 |
| 15 | Echo9Zulu/OmniCoder-9B-int4_sym-ov | single | 6.1 GB | 4/7 | 302 | 43 | data-free | greedy | nothink | 2026.3.0.0 |
| 16 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 2/7 | 27 | 4 | data-free | greedy | nothink | 2026.3.0.0 |
| 17 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.7 GB | 0/7 | 51 | 7 | data-free | greedy | nothink | 2026.3.0.0 |

### analysis

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HarmenWessels/gemma-4-E4B-it-qat-int4-ov | single | 6.6 GB | 4/4 | 151 | 38 | qat | greedy | think | 2026.3.0.0 |
| 2 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 4/4 | 317 | 79 | awq | greedy | think | 2026.3.0.0 |
| 3 | HarmenWessels/gemma-4-12B-it-qat-int4-ov | single | 8.2 GB | 4/4 | 330 | 82 | qat | greedy | think | 2026.3.0.0 |
| 4 | OpenVINO/Qwen3-14B-int4-ov | single | 9.7 GB | 4/4 | 788 | 197 | data-free | greedy | think | 2026.3.0.0 |
| 5 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 3/4 | 34 | 8 | data-free | greedy | think | 2026.3.0.0 |
| 6 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 3/4 | 37 | 9 | scale_estimation | greedy | think | 2026.3.0.0 |
| 7 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 1.8 GB | 3/4 | 41 | 10 | awq+se | greedy | think | 2026.3.0.0 |
| 8 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 3/4 | 68 | 17 | awq+se | greedy | think | 2026.3.0.0 |
| 9 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 1.8 GB | 3/4 | 164 | 41 | awq+se | greedy | think | 2026.3.0.0 |
| 10 | OpenVINO/Qwen3-1.7B-int4-ov | single | 1.2 GB | 3/4 | 175 | 44 | data-free | greedy | think | 2026.3.0.0 |
| 11 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 3/4 | 576 | 144 | data-free | greedy | think | 2026.3.0.0 |
| 12 | Echo9Zulu/Qwen3.5-2B-int4_sym-ov | single | 2.1 GB | 3/4 | 694 | 174 | data-free | greedy | think | 2026.3.0.0 |
| 13 | Echo9Zulu/OmniCoder-9B-int4_sym-ov | single | 6.1 GB | 3/4 | 1482 | 370 | data-free | greedy | think | 2026.3.0.0 |
| 14 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 2/4 | 25 | 6 | data-free | greedy | think | 2026.3.0.0 |
| 15 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.4 GB | 2/4 | 102 | 26 | data-free | greedy | think | 2026.3.0.0 |
| 16 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.7 GB | 2/4 | 112 | 28 | data-free | greedy | think | 2026.3.0.0 |
| 17 | HarmenWessels/gemma-4-E2B-it-qat-int4-ov | single | 4.4 GB | 2/4 | 252 | 63 | qat | greedy | think | 2026.3.0.0 |

### autocomplete-fim

| # | Entry | Kind | Size/Roles | Quality | Total s | Avg s | Recipe | Decode | Think | Engine |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | single | 0.7 GB | 1/1 | 1 | 1 | data-free | greedy | nothink | 2026.3.0.0 |
| 2 | OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov | single | 0.9 GB | 1/1 | 2 | 2 | data-free | greedy | nothink | 2026.3.0.0 |
| 3 | OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov | single | 1.8 GB | 1/1 | 2 | 2 | scale_estimation | greedy | nothink | 2026.3.0.0 |
| 4 | OpenVINO/Qwen3-4B-int4-ov | single | 2.3 GB | 1/1 | 4 | 4 | awq | greedy | nothink | 2026.3.0.0 |
| 5 | OpenVINO/Qwen3-8B-int4-cw-ov | single | 4.7 GB | 1/1 | 7 | 7 | data-free | greedy | nothink | 2026.3.0.0 |
| 6 | OpenVINO/Qwen3-14B-int4-ov | single | 9.7 GB | 1/1 | 14 | 14 | data-free | greedy | nothink | 2026.3.0.0 |
| 7 | Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov | single | 0.7 GB | 0/1 | 1 | 1 | data-free | greedy | nothink | 2026.3.0.0 |
| 8 | OpenVINO/Qwen3-0.6B-int4-ov | single | 0.4 GB | 0/1 | 1 | 1 | data-free | greedy | nothink | 2026.3.0.0 |
| 9 | HarmenWessels/granite-4.1-3b-int4-cw-code-ov | single | 1.8 GB | 0/1 | 2 | 2 | awq+se | greedy | nothink | 2026.3.0.0 |
| 10 | HarmenWessels/granite-4.1-3b-int4-cw-ov | single | 1.8 GB | 0/1 | 2 | 2 | awq+se | greedy | nothink | 2026.3.0.0 |
| 11 | OpenVINO/Qwen3-1.7B-int4-ov | single | 1.2 GB | 0/1 | 2 | 2 | data-free | greedy | nothink | 2026.3.0.0 |
| 12 | HarmenWessels/granite-4.1-8b-int4-cw-ov | single | 4.4 GB | 0/1 | 7 | 7 | awq+se | greedy | nothink | 2026.3.0.0 |

## Retest queue

_None — all entries current._

## Failures

**HarmenWessels/gemma-4-E4B-it-qat-int4-ov / codegen**:
  - group-anagrams#0: FAIL (missing definition)
  - group-anagrams#1: FAIL (missing definition)

**OpenVINO/Qwen3-4B-int4-ov / codegen**:
  - lru-cache#0: FAIL (NameError: name 'LRUCache' is not defined)
  - parse-duration#0: FAIL (NameError: name 'parse_duration' is not defined)
  - group-anagrams#0: FAIL (missing definition)

**Echo9Zulu/OmniCoder-9B-int4_sym-ov / codegen**:
  - merge-intervals#0: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [(1, 6), (8, 10)])
  - parse-duration#1: FAIL (ValueError: not enough values to unpack (expected 2, got 1))
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)

**HarmenWessels/granite-4.1-3b-int4-cw-code-ov / codegen**:
  - merge-intervals#0: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [(1, 6), [8, 10]])
  - parse-duration#0: FAIL (parse_duration('2h45m') -> 7500)
  - parse-duration#1: FAIL (parse_duration('2h45m') -> 0)
  - rle-codec#1: FAIL (ValueError: invalid literal for int() with base 10: '')

**HarmenWessels/granite-4.1-8b-int4-cw-ov / codegen**:
  - lru-cache#0: FAIL (_lru() -> [1, 2, -1, 1])
  - lru-cache#1: FAIL (AttributeError: 'NoneType' object has no attribute 'next')
  - parse-duration#1: FAIL (ValueError: invalid literal for int() with base 10: 's')
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)

**HarmenWessels/gemma-4-E2B-it-qat-int4-ov / codegen**:
  - merge-intervals#0: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [(1, 6), (8, 10)])
  - lru-cache#1: FAIL (NameError: name 'lrucache' is not defined)
  - parse-duration#1: FAIL (parse_duration('2h45m') -> 162000)
  - group-anagrams#0: FAIL (missing definition)

**HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / codegen**:
  - rate-limiter#0: FAIL (NameError: name 'window_seconds' is not defined)
  - rate-limiter#1: FAIL (_seq() -> [True, False, False, False])
  - parse-duration#0: FAIL (NameError: name 'parse_duration' is not defined)
  - parse-duration#1: FAIL (parse_duration('2h45m') -> 0)
  - rle-codec#0: FAIL (NameError: name 'rle_encode' is not defined)
  - rle-codec#1: FAIL (NameError: name 'rle_encode' is not defined)

**OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / codegen**:
  - rate-limiter#0: FAIL (_seq() -> [False, False, False, False])
  - rate-limiter#1: FAIL (_seq() -> [True, True, True, True])
  - parse-duration#0: FAIL (ValueError: invalid literal for int() with base 10: '2h45')
  - parse-duration#1: FAIL (parse_duration('2h45m') -> 0)
  - rle-codec#0: FAIL (IndexError: string index out of range)
  - rle-codec#1: FAIL (rle_decode('a3b1c3') -> 'a333b1c333')

**HarmenWessels/granite-4.1-3b-int4-cw-ov / codegen**:
  - merge-intervals#0: FAIL (merge_intervals([[1,3],[2,6],[8,10]]) -> [[1, 3], [2, 6], [8, 10]])
  - rate-limiter#0: FAIL (ImportError: cannot import name 'deque' from 'datetime' (~\AppData\Local\Programs\Python\Python312\Lib\datetime.py))
  - rate-limiter#1: FAIL (ImportError: cannot import name 'deque' from 'datetime' (~\AppData\Local\Programs\Python\Python312\Lib\datetime.py))
  - lru-cache#0: FAIL (_lru() -> [1, 2, 3, -1])
  - parse-duration#0: FAIL (ValueError: invalid literal for int() with base 10: 'h')
  - parse-duration#1: FAIL (TypeError: 'NoneType' object is not subscriptable)
  - rle-codec#1: FAIL (ValueError: invalid literal for int() with base 10: '')

**Echo9Zulu/Qwen3.5-2B-int4_sym-ov / codegen**:
  - rate-limiter#0: FAIL (ModuleNotFoundError: No module named 'SlidingWindowLimiter')
  - lru-cache#1: FAIL (TypeError: 'NoneType' object is not callable)
  - parse-duration#0: FAIL (SyntaxError: invalid syntax (<string>, line 1))
  - parse-duration#1: FAIL (SyntaxError: invalid syntax (<string>, line 1))
  - rle-codec#0: FAIL (SyntaxError: invalid syntax (<string>, line 1))
  - rle-codec#1: FAIL (missing definition)
  - group-anagrams#0: FAIL (SyntaxError: invalid syntax (<string>, line 2))

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
  - rate-limiter#0: FAIL (_seq() -> [True, True, False, False])
  - rate-limiter#1: FAIL (_seq() -> [False, False, False, False])
  - lru-cache#1: FAIL (NameError: name 'LRUCache' is not defined)
  - parse-duration#0: FAIL (ValueError: Invalid input: 2h45m)
  - parse-duration#1: FAIL (NameError: name 'parse_duration' is not defined)
  - rle-codec#0: FAIL (IndexError: string index out of range)
  - rle-codec#1: FAIL (IndexError: string index out of range)
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

**HarmenWessels/granite-4.1-8b-int4-cw-ov / edit**:
  - edit-exact: FAIL (old-match=True old='')

**HarmenWessels/gemma-4-E2B-it-qat-int4-ov / edit**:
  - write-full: FAIL (exec/assert failed: unterminated triple-quoted string literal (detecte)

**HarmenWessels/gemma-4-E4B-it-qat-int4-ov / edit**:
  - write-full: FAIL (exec/assert failed: unterminated triple-quoted string literal (detecte)

**HarmenWessels/gemma-4-12B-it-qat-int4-ov / edit**:
  - write-full: FAIL (exec/assert failed: unterminated triple-quoted string literal (detecte)

**OpenVINO/Qwen3-0.6B-int4-ov / edit**:
  - edit-exact: FAIL (old-match=True old='window must satisfy 1 <= window <= len(values)')
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / edit**:
  - edit-exact: FAIL (old-match=False old='def moving_average(values, window):\\n    if wind)
  - write-full: FAIL (exec/assert failed: 'mean')

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

**OpenVINO/Qwen3-4B-int4-ov / edit**:
  - edit-exact: FAIL (old-match=True old='return [sum(values[i:i + window]) / window\n      )
  - write-full: FAIL (writes=0 calls=[])

**OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov / edit**:
  - edit-exact: FAIL (old-match=True old='window must satisfy 1 <= window <= len(values)')
  - write-full: FAIL (writes=0 calls=[])

**Echo9Zulu/Qwen3.5-2B-int4_sym-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=[])
  - write-full: FAIL (writes=0 calls=[])

**Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=[])
  - write-full: FAIL (exec/assert failed: '(' was never closed (<string>, line 4))

**Echo9Zulu/OmniCoder-9B-int4_sym-ov / edit**:
  - edit-exact: FAIL (edits=0 calls=[])
  - write-full: FAIL (writes=0 calls=[])

**HarmenWessels/granite-4.1-3b-int4-cw-code-ov / edit**:
  - edit-exact: FAIL (old-match=True old='def moving_average(values, window):\n    """Moving)
  - write-full: FAIL (writes=0 calls=[])

**OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / agent-loop**:
  - result-use: FAIL (calls=[('read_file', {'path': 'config.yaml'})] content='')

**OpenVINO/Qwen3-1.7B-int4-ov / agent-loop**:
  - chain-depth: FAIL (stopped at turn 2: edits=0 green-tests-seen=0)

**OpenVINO/Qwen3-4B-int4-ov / agent-loop**:
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**HarmenWessels/granite-4.1-3b-int4-cw-code-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])

**OpenVINO/Qwen3-8B-int4-cw-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])

**OpenVINO/Qwen3-14B-int4-ov / agent-loop**:
  - chain-depth: FAIL (turn 1: repeated call run_tests)

**HarmenWessels/gemma-4-12B-it-qat-int4-ov / agent-loop**:
  - stop-done: FAIL (calls=['read_file'])

**HarmenWessels/gemma-4-E2B-it-qat-int4-ov / agent-loop**:
  - no-repeat: FAIL (calls=[])

**HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / agent-loop**:
  - no-repeat: FAIL (calls=[('read_file', "{'path': 'stats.py'}")])
  - chain-depth: FAIL (turn 1: repeated call run_tests)

**HarmenWessels/granite-4.1-3b-int4-cw-ov / agent-loop**:
  - call-simple: FAIL (calls=[])
  - call-restraint: FAIL (calls=[('web_search', {'query': 'API acronym'})] content='')
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**OpenVINO/Qwen3-0.6B-int4-ov / agent-loop**:
  - call-choose: FAIL (calls=[])
  - call-restraint: FAIL (calls=[] content='The acronym API stands for **A-P-I**. It is used in )
  - chain-depth: FAIL (turn 2: repeated call run_tests)

**Echo9Zulu/Qwen3.5-2B-int4_sym-ov / agent-loop**:
  - call-simple: FAIL (calls=[])
  - call-choose: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**Echo9Zulu/OmniCoder-9B-int4_sym-ov / agent-loop**:
  - call-simple: FAIL (calls=[])
  - call-choose: FAIL (calls=[])
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

**OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov / agent-loop**:
  - call-choose: FAIL (calls=[('read_file', {'path': 'https://www.python.org/downloads/'})])
  - result-use: FAIL (calls=[('read_file', {'path': 'config.yaml'})] content='')
  - no-repeat: FAIL (calls=[('read_file', "{'path': 'stats.py'}")])
  - stop-done: FAIL (calls=['run_tests'])
  - chain-depth: FAIL (stopped at turn 0: edits=0 green-tests-seen=0)

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

**OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov / analysis**:
  - diagnose: FAIL (content='the issue lies in the line `return (s[mid] + s[mid + 1]) / 2`)

**HarmenWessels/granite-4.1-3b-int4-cw-ov / analysis**:
  - recall-deep: FAIL (content='(remembered from earlier) FROBNICATE_77.')

**HarmenWessels/granite-4.1-8b-int4-cw-ov / analysis**:
  - diagnose: FAIL (content='the test fails because the expected median value is incorrect)

**HarmenWessels/granite-4.1-3b-int4-cw-code-ov / analysis**:
  - recall-deep: FAIL (content='(answer omitted from transcript for brevity) (answer omitted )

**OpenVINO/Qwen3-1.7B-int4-ov / analysis**:
  - plan: FAIL (symbols=0 numbered=False code=False)

**OpenVINO/Qwen3-8B-int4-cw-ov / analysis**:
  - plan: FAIL (symbols=4 numbered=False code=False)

**Echo9Zulu/Qwen3.5-2B-int4_sym-ov / analysis**:
  - route: FAIL (3/6 [chat:chat, edit:None, design:None, chat:chat, edit:chat, design:d)

**Echo9Zulu/OmniCoder-9B-int4_sym-ov / analysis**:
  - route: FAIL (2/6 [chat:chat, edit:None, design:chat, chat:chat, edit:None, design:N)

**HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov / analysis**:
  - route: FAIL (3/6 [chat:chat, edit:edit, design:chat, chat:edit, edit:edit, design:c)
  - diagnose: FAIL (content='the test fails because the expected median of `[4, 1, 3, 2]` )

**OpenVINO/Qwen3-0.6B-int4-ov / analysis**:
  - route: FAIL (4/6 [chat:design, edit:edit, design:design, chat:design, edit:edit, de)
  - diagnose: FAIL (content='')

**Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / analysis**:
  - diagnose: FAIL (content='')
  - plan: FAIL (symbols=1 numbered=True code=False)

**HarmenWessels/gemma-4-E2B-it-qat-int4-ov / analysis**:
  - route: FAIL (3/6 [chat:None, edit:edit, design:None, chat:chat, edit:edit, design:N)
  - recall-deep: FAIL (content='(answer omitted from transcript for brevity) (answer omitted )

**Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov / autocomplete-fim**:
  - merge-fim: FAIL

**OpenVINO/Qwen3-0.6B-int4-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/granite-4.1-3b-int4-cw-code-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/granite-4.1-3b-int4-cw-ov / autocomplete-fim**:
  - merge-fim: FAIL

**OpenVINO/Qwen3-1.7B-int4-ov / autocomplete-fim**:
  - merge-fim: FAIL

**HarmenWessels/granite-4.1-8b-int4-cw-ov / autocomplete-fim**:
  - merge-fim: FAIL

<!--LEADERBOARD END-->
