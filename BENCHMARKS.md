# Workload benchmarks

Results from [`scripts/bench_workloads.py`](scripts/bench_workloads.py) — the
workload-representative benchmark that supersedes raw decode numbers for choosing models.
Raw per-run JSON (including full output texts, so probes can be re-scored without re-running)
lives in [`bench_results/`](bench_results/).

## Method (v3, 2026-06-05)

Four profiles model the phases of software work:

| Profile | Prompt shape | Budget | Primary metric |
|---|---|---|---|
| `autocomplete` | FIM (Qwen-Coder) / raw continuation (others), **no chat template** | 96 tok | total completion latency |
| `assistant-edit` | chat: "rewrite this exact function" (echo-heavy) | 512 tok | decode tok/s |
| `assistant-explain` | chat: "explain this code" (medium echo) | 256 tok | decode tok/s |
| `architect` | chat: design/trade-off question, no code in prompt | 512 tok | decode tok/s |

Greedy decoding, 1 warm-up + 3 measured runs, medians reported. LLM-shaped models repeat the
chat profiles with **prompt-lookup** (`+PL`). Two **pass/fail probes** validate the artifact
(not the model's intelligence — see RESEARCH.md for that division of labor):

- `autocomplete`: the completed function is executed against test cases
- `assistant-edit`: the rewritten function is executed and must reproduce the original's
  behavior on branch-covering inputs (run on the plain *and* PL outputs)

Known limitations: raw-continuation autocomplete (non-FIM models) lacks a stop criterion and
can over-generate past the function → probe false-negatives; explain/architect have no
objective probe. Quality *ranking* should come from base models' official benchmarks; these
probes only validate conversion + quantization + serving integrity.

## Results — 2026-06-05 (Core Ultra 155H, OpenVINO 2026.3 nightly, driver .8724)

Decode tok/s (TTFT in parentheses); probe verdicts inline.

| Model | autocomplete | assistant-edit | +PL | assistant-explain | +PL | architect | +PL |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | **1.07 s** (0.05) probe ✓ | 58.9 ✗syntax | 67.1 (+14%) ✗ | 58.1 | 37.6 (−35%) | 58.4 | 37.3 (−36%) |
| Qwen2.5-Coder-7B | 3.64 s (0.15) probe ✓ | 16.1 **✓** | **34.4 (+114%) ✓** | 16.1 | 12.7 (−21%) | 16.1 | 15.1 (−6%) |
| Granite-4.1-3b-cw (self-converted) | 3.27 s (0.07) probe ✗¹ | 28.8 **✗ behavior changed** | 42.3 (+47%) **✗ syntax** | 29.5 | 17.1 (−42%) | 29.6 | 17.9 (−39%) |
| Qwen3.5-2B | n/a (VLM) | **42.1 ✓** | n/a | **37.2** | n/a | **43.2** | n/a |
| Gemma 4 E2B (default chat) | n/a (VLM) | 23.1 ✓ | n/a | 22.9 | n/a | 23.0 | n/a |
| Qwen2.5-Coder-3B | 1.79 s (0.10) probe ✓ | 29.2 **✓** | **63.2 (+116%) ✓** | 29.2 | 21.7 (−26%) | 29.6 | 25.8 (−13%) |
| Granite-4.1-3b **int8** (rebuilt) | 6.67 s (0.11) probe ✓ | 14.8 **✓** | 24.6 (+66%) **✓** | 14.3 | 14.4 (+0%) | 14.9 | 15.9 (+6%) |
| Qwen3.5-0.8B | n/a (VLM) | 61.4 ✓ | n/a | 61.4 | n/a | **61.4** | n/a |
| OmniCoder-9B | n/a (VLM) | 13.3 ✗² | n/a | 13.4 | n/a | 13.4 | n/a |
| Qwen3.5-4B | n/a (VLM) | 20.2 ✗² | n/a | 20.6 | n/a | 20.2 | n/a |
| Granite-4.1-3b **cw v2 (AWQ+SE)** | 1.75 s (0.09) ✗¹ | 31.3 **✓** | 30.9 (−1%) **✓** | 31.4 | 23.2 (−26%) | 31.7 | 24.8 (−22%) |
| Qwen2.5-Coder-0.5B | **0.95 s** (0.03) ✓ | 80.6 ✗³ | 130.6 (+62%) ✓³ | 87.7 | 67.9 (−23%) | 93.4 | 88.9 (−5%) |
| Qwen3-VL-8B | n/a (VLM) | 14.8 ✓ | n/a | 15.0 | n/a | 15.1 | n/a |
| Gemma 4 E2B (our conversion) | n/a (VLM) | 21.5 ✓ | n/a | 23.3 | n/a | 24.5 | n/a |
| Qwen3-0.6B (thinking) | 1.16 s (0.05) ✓ | 86.7 ✗ behavior | 56.5 (−35%) ✗ | 85.2 | 47.6 (−44%) | 79.6 | 53.1 (−33%) |
| LFM2.5-1.2B-Thinking | 1.19 s (0.05) ✗¹ | 83.9 ✗ no code | 92.7 (+10%) ✗ | 84.7 | 101.6 (+20%) | 84.7 | **137.9 (+63%)** |
| MiniCPM5-1B g128 (ours, no-think template)⁶ | 1.25 s (0.04) ✗¹ | **81.4 ✓** | 95.9 (+18%) ✗ PL-flaky | 82.6 | 59.1 (−28%) | 83.9 | 77.1 (−8%) |
| **Granite-4.1-8b cw (AWQ+SE, ours)** | 6.94 s (0.20) ✗¹ | 14.1 **✓** | **27.0 (+92%) ✓** | 14.1 | 12.2 (−14%) | 14.2 | 11.9 (−16%) |

¹ raw-continuation probe artifact (no stop criterion), not a verified failure.
² **untagged thinking preamble**: the model spends the token budget on prose reasoning before
(or instead of) the code — OmniCoder-9B produced *zero* code in 512 tokens. Not a conversion
defect, but a practical disqualifier for the edit role at these decode speeds.
³ Coder-0.5B's plain edit failed (missing import) while its PL edit passed — the two backends'
greedy paths diverge, and at 0.5B scale correctness is effectively a coin flip per path. Edits
need ≥1.5B; the 0.5B remains autocomplete-only.

## Findings

1. **PL is a per-workload switch, not a per-model one**: +14…+114% on echo-heavy edits, but
   −21…−42% on explain/architect *for the same models*. Enable it only for models dedicated to
   FIM/edit duty.
2. **PL is not always quality-neutral**: Granite's +PL edit output dropped a closing
   parenthesis (SyntaxError) — reproduced across runs. The continuous-batching backend's
   numerics can change outputs for the worse; the probe gate matters.
3. **RESOLVED: Granite-4.1-cw's edit flaws are channel-wise quantization damage.** The int8
   rebuild of the same base passes *every* probe (autocomplete, edit, edit+PL) where the cw
   build dropped logic and broke syntax under PL. The cw recipe's 2× speed costs measurable
   correctness on this model. Secondary observation: int8's PL penalties on explain/architect
   are also gone (+0%/+6% vs cw's −42%/−39%) — speculation overhead amortizes better at int8's
   slower decode.
4. **Qwen2.5-Coder-1.5B is an autocomplete specialist**: best-in-class completion latency with
   a passing probe, but it cannot complete the edit task within 512 tokens (verbose type
   annotation spam + a missing `Tuple` import).
5. **Qwen3.5-2B outperforms the current chat default (Gemma 4 E2B) on every chat profile**
   (~42 vs ~23 tok/s, lower TTFT, edit probe passing) on this suite.
6. **Qwen2.5-Coder-3B is the new edit champion**: all probes green, edit+PL at **63.2 tok/s**
   (+116%) — nearly twice the 7B+PL — while keeping clean FIM autocomplete. The Coder family's
   instruction style (code first, no musing) is exactly what the edit role rewards.
7. **Untagged thinking preambles disqualify otherwise-strong models from the edit role**:
   OmniCoder-9B and Qwen3.5-4B reason in prose before coding and blow the budget; Qwen3.5-2B
   and 0.8B don't share this behavior despite being the same family.
8. **Qwen3.5-0.8B does 61.4 tok/s across all chat profiles with a passing edit probe** — at
   0.85 GiB. Whether 0.8B-class reasoning is *good enough* for explain/architect is a quality
   question for official benchmarks, but the serving math is remarkable.
9. **AWQ + scale-estimation repairs data-free cw-int4 damage at zero size/speed cost.** The
   recalibrated Granite cw build (v2) passes every probe the data-free build failed, at the
   same 1.72 GiB and ~31 tok/s. Data-aware calibration (`--awq --scale-estimation --dataset
   wikitext2`) should be the default for int4 conversions. Both HF artifacts updated in place.
   The recipe re-validated at 8B scale: our Granite-4.1-8b conversion (first OV IR of that
   model) passes its edit probes on day one.
10. **Thinking models split on PL by echo style, refining finding 1**: LFM2.5-1.2B-Thinking
    gains from PL on *every* profile (architect +63%, the fastest chat measurement recorded:
    137.9 tok/s) because its reasoning restates the prompt heavily, while Qwen3's free-prose
    thinking loses everywhere. Echo overlap must be measured per model, not inferred from
    "thinking vs non-thinking".

## Three-axis summary: quality × speed × integrity

The decision view: vendor-reported quality (left), our measured speed (middle), our probe
verdicts (right). Caveats: vendors report different suites — MMLU-Pro is the only cross-family
anchor; coding columns are **not comparable across rows** (HumanEval ≠ LiveCodeBench);
Qwen2.5-Coder quality numbers come from the 2024 report (arXiv:2409.12186), Qwen3-VL-8B and
Qwen3-0.6B cards defer to tech reports without numbers. Speed = this suite (edit shows the
best validated mode; "+PL" where prompt-lookup wins).

| Artifact | MMLU-Pro | Coding (vendor suite) | IFEval | autocomplete | edit | chat | Edit probe |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-0.5B | — | HE 61.6 / MBPP 52.4 | — | **0.95 s** | 130.6 +PL | ~90 | flaky³ |
| Qwen2.5-Coder-1.5B | — | HE 70.7 / MBPP 69.2 | — | 1.07 s | 67.1 +PL | ~57 | ✗ (budget) |
| Qwen2.5-Coder-3B | — | HE 84.1 / MBPP 73.6 | — | 1.79 s | **63.2 +PL** | ~29 | **✓** |
| Qwen2.5-Coder-7B | — | **HE 88.4** / MBPP 83.5 | — | 3.64 s | 34.4 +PL | ~16 | **✓** |
| Qwen3.5-0.8B | 29.7 | n.r. | 52.1 | n/a | 61.4 | 61.4 | ✓ |
| Qwen3.5-2B | 55.3 | n.r. | 61.2 | n/a | 42.1 | ~40 | ✓ |
| Qwen3.5-4B (default template) | **79.1**⁴ | LCB-v6 55.8 | 89.8 | n/a | (thinking preamble) | ~20 | ✗² |
| **Qwen3.5-4B, no-think rt_info patch⁷** | <79.1 (unmeasured)⁷ | — | — | n/a | **19.8 ✓** | 19.8 | **✓** |
| Gemma 4 E2B (both PTQ conversions) | 60.0 | LCB-v6 44.0 | — | n/a | 21–23 | ~23 | ✓ |
| **Gemma 4 E2B QAT (self-converted)⁸** | 60.0 (≈bf16-preserved) | LCB-v6 44.0 | — | n/a | 21.7 ✓ | ~22 | **✓** |
| **Gemma 4 E4B QAT (self-converted)⁸** | 69.4 (≈bf16-preserved) | LCB-v6 52.0 | — | n/a | 16.7 ✓ | ~17 | **✓** |
| Gemma 4 E4B | **69.4** | LCB-v6 52.0 | — | n/a | 15.6 | ~16 | **✓** |
| Granite-4.1-3b cw v2 (self-converted) | 49.8 | HE 81.7 / MBPP 71.2 | 82.3 | — | 31.3 | ~31 | **✓** |
| Granite-4.1-3b int8 (self-converted) | 49.8 | same base | 82.3 | — | 24.6 +PL | ~15 | **✓** |
| Granite-4.1-8b cw (self-converted) | 56.0 | HE 85.4 / **MBPP 87.3** | **87.1** | — | 27.0 +PL | ~14 | **✓** |
| OmniCoder-9B (base: Qwen3.5-9B⁴: MMLU-Pro 82.5, LCB-v6 65.6, IFEval 91.5) | — | GPQA-D 83.8 / TB-2.0 23.6 | — | n/a | (zero code in budget) | ~13 | ✗² |
| **OmniCoder-9B, no-think rt_info patch⁷** | — | — | — | n/a | **12.9 ✓** | 12.8 | **✓** |
| Qwen3-VL-8B | n.r.⁵ | n.r.⁵ | n.r.⁵ | n/a | 14.8 | ~15 | ✓ |
| Qwen3-0.6B | n.r.⁵ | n.r.⁵ | n.r.⁵ | 1.16 s | (behavior changed) | ~84 | ✗ |
| LFM2.5-1.2B-Thinking | 49.7 | **card warns against programming use** | 88.4 | — | (no code block) | 84.7 (137.9 architect+PL) | ✗ |

⁴ Qwen3.5-4B/9B cards report thinking-mode-default scores — not directly comparable with
non-thinking rows (the 2B card shows the gap: 55.3 non-thinking vs 66.5 thinking on MMLU-Pro),
and the thinking preamble is precisely what fails our edit-budget probe.
⁵ official numbers exist only in the Qwen3 tech report (tables not published on the cards in
extractable form).
⁸ converted from Google's `gemma-4-E2B-it-qat-q4_0-unquantized` (quantization-aware-trained
weights) with the QAT-matched scheme (`--sym --group-size 32`). Same speed as the PTQ builds,
but int4 quality ≈ bf16 by construction — the recommended E2B artifact. Loader warnings about
"missing" k/v projections on layers 15–34 are the tied KV-shared weights (verified benign:
probe PASS, outputs equivalent to PTQ build).
⁷ **rt_info template patch**: GenAI reads the chat template from `openvino_tokenizer.xml`
rt_info and cannot pass `enable_thinking`; hardcoding the no-think prefix there flips
hybrid-thinking models into direct-answer mode (validated on MiniCPM5-1B, Qwen3.5-4B and
OmniCoder-9B — all three edit probes went FAIL→PASS). Quality caveat: the vendors' official
scores were measured in *thinking* mode; no-think quality is lower and locally unmeasured
(the Qwen3.5-2B card brackets the gap: 55.3 no-think vs 66.5 thinking MMLU-Pro).
⁶ Two artifact-level fixes were required (full story in RESEARCH playbook): (a) the cw+AWQ
recipe produced **degenerate garbage** at 1B scale — g128/int8 are coherent → granularity must
scale with model size; (b) the model "thought by default" under GenAI because the chat template
**baked into `openvino_tokenizer.xml` rt_info** leaves thinking to the model when
`enable_thinking` is unpassable — hardcoding the no-think prefix there (not in the ignored
`.jinja`) flipped the edit probe from FAIL to PASS and is the row shown. PL remains unusable at
1B (correctness coin-flip, as with Coder-0.5B).

Reading across the axes: on raw scores the **Qwen3.5-4B/9B generation leads everything**
(MMLU-Pro 79–83, LCB 56–66) — but those are thinking-mode numbers, and the thinking preamble
is exactly what disqualifies them from fast edit workloads on this hardware; they're
quality-first chat picks if you accept reasoning latency. Among **probe-validated,
non-thinking** artifacts: **Gemma E4B** has the best general scores (MMLU-Pro 69.4, probe ✓,
15.6 tok/s — but VLM-shaped, so no PL acceleration), **Granite-4.1-8b** leads instruction
following / MBPP / tool calling with the fastest validated quality-tier edits (27 tok/s +PL),
**Coder-7B** edges raw HumanEval, **Qwen3.5-2B** is the chat speed/quality balance point, and
LFM's architect speed carries its vendor's own warning against programming-domain use.

## Not benchmarkable on this hardware (separate from results — these never ran)

| Model | Why it cannot run |
|---|---|
| gpt-oss-20b (11.7 GiB) | OOM: device allocation fails at compile despite free host RAM |
| Qwen3-Coder-30B-A3B (15.2 GiB) | OOM: device allocation fails at compile |
| Gemma 4 26B A4B (14.3 GiB) | OOM: fails during weight upload, tested 3× |
| LFM2.5-8B-A1B (MoE) | GPU compile deadlocks (~5 min in, threads parked) — reproduced on a fresh own-conversion IR and two runtimes (RESEARCH.md finding 10) |
| granite-4.0-h-tiny 7B-A1B (MoE) | GPU compile grinds unboundedly (killed at 57 min) — finding 10 |
| LFM2.5-350M (official conversions) | runtime `ScatterNDUpdate` shape bug at inference |
| Ministral-3-3B, Gemma-4-12B, MiniCPM-V-4.6, Mellum2-12B | architectures unsupported by the conversion toolchain (see RESEARCH.md watch items) |

Root causes and full forensics: RESEARCH.md (findings 2–3, 10 and the screening ledger).

## Role-fitness benchmark (agent capabilities, 2026-06-06)

Tool use is a *separate capability axis* from chat/coding quality — measured with
`scripts/bench_roles.py`: 13 executable pass/fail probes distilled from live agent-frontend
failures (Continue CLI / Kilo CLI driving the server). v1 probes: schema-correct calls, tool
selection, restraint, result use, repeat avoidance, byte-exact edits, full-file writes, stop
discipline, 3-way routing. v2 probes: bug diagnosis, planning, scripted multi-turn loop
endurance, deep recall. Raw per-probe JSON in `bench_results/roles__*.json`.

Full 15-probe matrix on the shipped serving stack (end-of-day reruns,
`roles__20260606-21*.json`):

Each model is scored in its measured-best tool language (ᴺ = native adapter; others
hermes, which is the native training format for Qwen and granite):

| Model | total | avg s/probe³ | loop (chain-depth) | edit-exact | diagnose | route |
|---|---|---|---|---|---|---|
| **Gemma 4 E4B QAT ᴺ (self-converted)** | **13/15** | 16.1 | **✓ clean stop** | **✓** | ✗ nothink / **✓ think** | 6/6 |
| **granite-4.1-8b cw (self-converted)** | **12/15** | 13.3 | **✓ clean stop** | **✓** | ✗ blames the test² | 6/6 |
| Gemma 4 E2B QAT ᴺ (self-converted) | 11/15 | 7.9 | ✓ | ✓ mechanics, fix wrong | ✓ | 4/6 |
| Qwen2.5-Coder-7B | 11/15 | 10.2 | ✗ | ✗ fabricates whitespace | ✓ | 6/6 |
| Qwen3.5-2B | 10/15 | 22.1⁴ | ✗ stalls | ✗ | ✓ (fastest) | 6/6 |
| granite-4.1-3b cw (self-converted) | 10/15 | 7.1 | ✗ | ✗ | ✓ | 6/6 |
| Qwen2.5-Coder-3B | 10/15 | **6.4** | ✓ | ✗ isolated / ✓ in-loop | ✗ | 6/6 |
| LFM2.5-1.2B-Instruct ᴺ (self-converted) | 7/13 | 4.2 | ✗ | ✗ | ✗ | 3/6 |
| Qwen2.5-Coder-1.5B | 6/15 | 5.3 | ✗ | ✗ | ✗ | **6/6** |
| Qwen3.5-0.8B | 5/9 (v1 only) | — | — | ✗ | — | 4/6 |
| LFM2.5-1.2B-Thinking | 4/13 | 10.2 | ✗ | ✗ | **✓ (only 1B-class pass)** | 0/6 |

¹ emits its native `<|tool_call_start|>` Pythonic tool format regardless of instructed
format — hermes-style serving understates LFM models (RESEARCH.md finding 9).
² an anomaly, not a family trait: granite-3b and five other models diagnose the planted bug
correctly — the 8B uniquely argues with the test. Prompt-engineering consequence: treat
failing tests as ground truth in instructions to granite-8b.
³ mean wall-clock per probe (failure time included — a stalling model pays for it here);
quality first, latency as the tiebreaker between equal scores.
⁴ skewed by a 128 s chain-depth stall; its analysis probes run 6–12 s.

granite-8b and E2B tie at 12/15 with exactly complementary failures (executor core vs
analysis) — the cleanest evidence for role-split serving. Qwen2.5-Coder-3B is the surprise
budget-executor candidate: the only other model to complete the scripted fix-test-stop loop
(1.9 GiB), though it re-calls tools instead of reading results outside the loop harness.

**Native tool-language retest (2026-06-07,** `roles__20260607-090450.json`**)**: with
per-family adapters (the server renders each model's own chat template with tools — see
README), the Gemmas transform: **E4B 10/15 → 13/15, the new suite champion** — its
"tool-shyness" was pure format mismatch; in its native protocol it does byte-exact edits
AND clean loops (previously granite-exclusive skills) at 16.1 s/probe. E2B 12 → 11 with a
profile shift (executor skills appear: edit-exact ✓, chain-depth ✓; routing/recall drop).
Gemma thinking (pattern C, first measurements): **fixes E4B's diagnose verdict**
(nothink blames the test, think answers correctly at 66.7 s) — the first measured case of
thinking changing a quality outcome. LFM remains hermes-served (its template hides the
protocol from the detector) — still understated. Executor seat now contested:
granite-8b 12/15 vs E4B-native 13/15 — pending a virtual-model A/B.

**LFM2.5-1.2B variant pair** (2026-06-07, native adapter, `roles__20260607-09*.json`):
a clean natural experiment — Instruct 7/13 vs Thinking 4/13. The Thinking variant is the
only 1.2B-class artifact to pass `diagnose` (thinking-training buys analysis) but its
reasoning preamble collapses every operational probe (route 0/6, result-use/recall fail,
2.4× latency). Variant choice is a profile choice, not a quality knob; neither earns a
seat.

**Think-steering sweep** (all 13 probes × think/nothink on the hybrid-thinking models,
`roles__20260606-214914.json`): thinking is quality-neutral-to-harmful at this scale.
Qwen3.5-2B: one verdict improved, otherwise identical — at up to **15× latency** on
uncertainty-heavy probes (call-restraint 5.3 → 82 s). MiniCPM5-1B: actively destructive
(plan/diagnose collapse, reasoning budget exhausted mid-thought; full profile 7/26 — no
role seat), with one exception: *recall precision* — thinking recovered an exact identifier
that no-think hallucinated. Operational rule: never default `reasoning_effort: high` on
≤2B models; thinking earns its cost only on exact-recall tasks.

Key verdicts: **actor ≠ analyst** (granite uniquely sustains loops and byte-exact edits but
misdiagnoses; Gemma/Qwen diagnose but can't drive loops) — the empirical basis for
architect/executor role-split serving. `write-full` failed 0-for-10: coder roles must be
edit-first with server-side verification. Thinking mode (Qwen3.5-2B, both probes re-run with
`reasoning_effort: high`) changed nothing at this difficulty: 4/4 both ways, equal latency.

## Serving optimizations (measured, 2026-06-06)

| Lever | Result | Status |
|---|---|---|
| **Prefix caching + chunked prefill** (`SCHEDULER_MODELS`) | warm-prefix TTFT **63 s → 0.9 s** raw, **71.5 s → 2.6 s** through the API (27×); clears granite's 16k single-allocation wall (24k+ now prefills); KV pool is *reserved at load* — budget like weights | **shipped** |
| Prompt-lookup decoding | +92–116% on edit workloads, −14…−42% on explain/architect (finding 6) | shipped (per-model) |
| OpenAI tool calling, **per-family native languages** (gemma/lfm template rendering, hermes injection + JSON repair elsewhere) | unlocks native-tool agent frontends fairly; format mismatch cost E4B 3 points (finding 16) | **shipped** |
| u8 KV-cache hint | redundant — GPU defaults to int8 KV already; explicit hint crashes (upstream bug, finding 13) | do not use |
| **NPU autocomplete offload** (`MODEL_DEVICES`, per-device gen locks) | cw-sym 1.5B probe-certified on NPU; ~7 s completions that never queue behind GPU turns (finding 14) | **shipped** |
| `models.yaml` registry | every per-model setting above pinned in one config | **shipped** |
| Draft-model speculation, EAGLE-3 | promising, untested | next up |

Prefill cost grows superlinearly and differs per architecture (finding 11): granite-8b
43 s @ 8k vs Qwen3.5-2B 17 s @ 16k — per-model context budgets matter more than decode rank
for agent/long-context use.

## Casting tournament (virtual/agent, 2026-06-07, `castings.jsonl`)

Three castings × two design tasks × reviewer on/off, scored by executing the returned
code: **A (3-brain: 2B architect + granite executor)** — both easy-task passes at 25-40 s,
hard task fails with broken output. **B (Qwen3-8B architect + granite)** — easy passes at
~55 s; hard task fails *close* (clean running code, one boundary bug). **C (Qwen3-8B every
seat)** — same results as B at 2-4× the latency, including the *identical* boundary bug.
Executor variants (A-E4B, A-Q38B, A-C3B) completed the matrix (24 cells): granite and E4B
— the agentic champions — produce *broken* long-form code on the hard task, while
Qwen3-8B-in-any-seat produces near-misses (closest: 3/4 of the behavior sequence as
executor). Agentic discipline and long-form codegen are different muscles. Verdicts:
**A (granite) stays production** (fastest, agentic king); **A-Q38B + review is the
design-flow alternative** (best hard-task code, reviewer covers its spec slips); C/E4B/C3B
dominated. **Reviewer, measured across 12 reviewed cells: 2 catches, both spec/format
defects (tuple-vs-list, hallucinated import), zero logic bugs** — a spec-conformance
checker, opt-in (`"review": true`), worthwhile with qwen-family executors in design flows;
execution remains the only *logic* reviewer that works at this scale.

## Current role recommendations (will evolve as more models run)

| Role | Recommendation | Why |
|---|---|---|
| **Agent executor (tool loops, edits)** | **granite-4.1-8b cw, prefix-cached, PL off — contested by Gemma E4B QAT (native tool language)** | granite 12/15 (lighter, ≤8k context, treat failing tests as ground truth); E4B 13/15 native (byte-exact edits + loops, diagnose fixable with thinking) — A/B pending |
| **Agent router / classification** | Qwen2.5-Coder-1.5B (g128 build) | route 6/6 at ~2.4 s/decision, already resident for autocomplete; the cw build drops to 3/6 — quantization damage is task-selective |
| **Agent architect / analysis** | Qwen3.5-2B, prefix-cached, no-think | fastest correct diagnoser; near-flat prefill (16k in 17 s) for big planning contexts |
| Autocomplete | Qwen2.5-Coder-1.5B (+PL) | 1.07 s completions, probe ✓ |
| **Autocomplete, lock-free (NPU)** | Qwen2.5-Coder-1.5B int4-cw (self-converted) on NPU | probe-certified; ~7 s, never queues behind GPU chat/agent turns |
| Assistant (edit-heavy) | **Qwen2.5-Coder-3B with PL** | 63.2 tok/s, all probes ✓; Coder-7B+PL (34.4, probes ✓) when max quality matters |
| Assistant (explain) / Architect | Qwen3.5-2B | 37–43 tok/s, probe ✓; Qwen3.5-0.8B (61.4) as the speed option — both pending quality A/B |
| Fast edit/chat (experimental) | MiniCPM5-1B g128, no-think template (self-converted) | **fastest probe-passing edits measured (81.4 tok/s)**, ~83 tok/s chat, 128k ctx — 1B quality is the open question; no PL (flaky at this scale) |
| Chat/assistant (quality tier) | **Qwen3.5-4B with the no-think rt_info patch** | probe ✓ at 19.8 tok/s; thinking-mode card scores (MMLU-Pro 79.1) overstate no-think quality — A/B vs Gemma E4B (69.4, probe ✓, 15.6) to pick |
| Max coding quality | OmniCoder-9B, no-think patch | probe ✓ at 12.9 tok/s; GPQA-D 83.8 / Terminal-Bench champion base |
| Assistant (edit/tool quality tier) | Granite-4.1-8b cw (self-converted) **+PL** | IFEval 87 / MBPP 87 / BFCL 68: 27 tok/s edits / 14 chat, probes ✓, 128k context. Enable PL only for edit-heavy use (−14% on explain) |
| Assistant (edit, non-Coder option) | Granite-4.1-3b cw **v2** | 31.3 tok/s, probes ✓, 128k context — no PL needed |
| Avoid for edits | OmniCoder-9B, Qwen3.5-4B, Granite-4.1-cw **v1** | thinking preambles (former two); quantization damage (v1, fixed in v2) |
