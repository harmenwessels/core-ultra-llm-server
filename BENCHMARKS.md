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
| Granite-4.1-3b-cw (ours) | 3.27 s (0.07) probe ✗¹ | 28.8 **✗ behavior changed** | 42.3 (+47%) **✗ syntax** | 29.5 | 17.1 (−42%) | 29.6 | 17.9 (−39%) |
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
| **Gemma 4 E2B QAT (ours)⁸** | 60.0 (≈bf16-preserved) | LCB-v6 44.0 | — | n/a | 21.7 ✓ | ~22 | **✓** |
| **Gemma 4 E4B QAT (ours)⁸** | 69.4 (≈bf16-preserved) | LCB-v6 52.0 | — | n/a | 16.7 ✓ | ~17 | **✓** |
| Gemma 4 E4B | **69.4** | LCB-v6 52.0 | — | n/a | 15.6 | ~16 | **✓** |
| Granite-4.1-3b cw v2 (ours) | 49.8 | HE 81.7 / MBPP 71.2 | 82.3 | — | 31.3 | ~31 | **✓** |
| Granite-4.1-3b int8 (ours) | 49.8 | same base | 82.3 | — | 24.6 +PL | ~15 | **✓** |
| Granite-4.1-8b cw (ours) | 56.0 | HE 85.4 / **MBPP 87.3** | **87.1** | — | 27.0 +PL | ~14 | **✓** |
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

## Current role recommendations (will evolve as more models run)

| Role | Recommendation | Why |
|---|---|---|
| Autocomplete | Qwen2.5-Coder-1.5B (+PL) | 1.07 s completions, probe ✓ |
| Assistant (edit-heavy) | **Qwen2.5-Coder-3B with PL** | 63.2 tok/s, all probes ✓; Coder-7B+PL (34.4, probes ✓) when max quality matters |
| Assistant (explain) / Architect | Qwen3.5-2B | 37–43 tok/s, probe ✓; Qwen3.5-0.8B (61.4) as the speed option — both pending quality A/B |
| Architect (experimental) | LFM2.5-1.2B-Thinking **+PL** | 137.9 tok/s with design-aligned reasoning — but LiquidAI's own card advises against knowledge-intensive/programming use; try-and-judge with low expectations |
| Fast edit/chat (experimental) | MiniCPM5-1B g128, no-think template (ours) | **fastest probe-passing edits measured (81.4 tok/s)**, ~83 tok/s chat, 128k ctx — 1B quality is the open question; no PL (flaky at this scale) |
| Chat/assistant (quality tier) | **Qwen3.5-4B with the no-think rt_info patch** | probe ✓ at 19.8 tok/s; thinking-mode card scores (MMLU-Pro 79.1) overstate no-think quality — A/B vs Gemma E4B (69.4, probe ✓, 15.6) to pick |
| Max coding quality | OmniCoder-9B, no-think patch | probe ✓ at 12.9 tok/s; GPQA-D 83.8 / Terminal-Bench champion base |
| Assistant (edit/tool quality tier) | Granite-4.1-8b cw (ours) **+PL** | IFEval 87 / MBPP 87 / BFCL 68: 27 tok/s edits / 14 chat, probes ✓, 128k context. Enable PL only for edit-heavy use (−14% on explain) |
| Assistant (edit, non-Coder option) | Granite-4.1-3b cw **v2** | 31.3 tok/s, probes ✓, 128k context — no PL needed |
| Avoid for edits | OmniCoder-9B, Qwen3.5-4B, Granite-4.1-cw **v1** | thinking preambles (former two); quantization damage (v1, fixed in v2) |

## Appendix: full model overview (raw decode method)

Single-prompt decode/TTFT measurements (`scripts/bench.py`) with modalities,
context windows and base-release dates — including models that predate the
workload-profile method above, and the memory/architecture failure cases.

| Model | Base releasedᵃ | Weights | Modalities | Max context | Decode | TTFT | PL edits³ | Verdict |
|---|---|---|---|---|---|---|---|---|
| [Qwen2.5-Coder-0.5B INT4](https://huggingface.co/OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov) | 2024-11 | 0.3 GB | text | 32k | 87.6 tok/s | 0.06 s | 131.0 | fastest; quality floor for autocomplete |
| [LFM2.5-1.2B-Thinking INT4](https://huggingface.co/Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov) | 2026-01 | 0.6 GB | text | 128k | 87.6 tok/s | 0.08 s | — | hybrid conv/attention; reasoning model (thinking tokens add latency); community conversion |
| [Qwen3.5-0.8B INT4](https://huggingface.co/yangsu0423/Qwen3.5-0.8B-int4-ov) | 2026-02 | 0.9 GB | text, imageᵇ | 256k | 72.7 tok/s | 0.08 s | — | newest gen at near-0.5B speed; community conversion |
| [Qwen3-0.6B INT4](https://huggingface.co/OpenVINO/Qwen3-0.6B-int4-ov) | 2025-04 | 0.4 GB | text | 40k | 62.7 tok/s | 0.10 s | 53.5 ↓ | slower than the newer, similar-size Qwen3.5-0.8B |
| [Qwen2.5-Coder-1.5B INT4](https://huggingface.co/OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov) (default autocomplete) | 2024-09 | 0.9 GB | text | 32k | 57.0 tok/s | 0.06 s | 70.0 | autocomplete sweet spot: FIM-trained, 2.4× faster than the 3B |
| [Ministral-3b-instruct INT4](https://huggingface.co/Echo9Zulu/Ministral-3b-instruct-int4_asym-ov) | 2024-03 | 1.7 GB | text | 128k | 36.0 tok/s | 0.07 s | — | community Mistral derivative (not official Mistral AI); 2024-era quality |
| [Qwen3.5-2B INT4](https://huggingface.co/Echo9Zulu/Qwen3.5-2B-int4_sym-ov) | 2026-02 | 2.0 GB | text, imageᵇ | 256k | 34.6 tok/s | 0.17 s | — | fastest chat-quality model; community conversion |
| [Gemma 4 E2B INT4](https://huggingface.co/gregor160300/gemma-4-E2B-it-int4-ov) (default chat) | 2026-03 | 4.1 GB | text, image, audioᵇ | 128k | 29.9 tok/s | 0.23 s | — | very responsive in Continue |
| [Granite-4.1-3b INT4-cw](https://huggingface.co/HarmenWessels/granite-4.1-3b-int4-cw-ov) (our conversion) | 2026-04 | 1.7 GB | text | 128k | 27.4 tok/s | 0.13 s | 47.1 | newest Granite; first OV IR of 4.1; channel-wise recipe is 2.1× faster than the int4 default here (RESEARCH.md) |
| [Qwen3-4B INT4](https://huggingface.co/OpenVINO/Qwen3-4B-int4-ov) | 2025-04 | 2.1 GB | text | 40k | 24.9 tok/s | 0.10 s | 17.3 ↓ | same speed as Coder-3B with a newer base |
| [Granite-4.0-micro INT4](https://huggingface.co/llmware/granite-4-micro-ov) | 2025-09 | 2.2 GB | text | 128k | 24.6 tok/s | 0.16 s | — | IBM; 128k context at 3B-class speed; community conversion (llmware) |
| [Qwen2.5-Coder-3B INT4](https://huggingface.co/OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov) | 2024-11 | 2.1 GB | text | 32k | 24.0 tok/s | 0.15 s | 57.2 | strong FIM quality |
| [Qwen3.5-4B INT4](https://huggingface.co/yangsu0423/Qwen3.5-4B-int4-ov) | 2026-02 | 3.3 GB | text, imageᵇ | 256k | 19.9 tok/s | 0.31 s | — | newest gen; faster than the 9B at similar quality class; community conversion |
| [Gemma 4 E4B INT4](https://huggingface.co/OpenVINO/gemma-4-E4B-it-int4-ov) | 2026-03 | 6.0 GB | text, image, audioᵇ | 128k | 15.7 tok/s | 0.52 s | — | mid |
| [Qwen2.5-Coder-7B INT4](https://huggingface.co/OpenVINO/Qwen2.5-Coder-7B-Instruct-int4-ov) | 2024-09 | 4.2 GB | text | 32k | 15.0 tok/s | 0.20 s | **41.8** | best chat quality that fits; with prompt-lookup, the strongest edit-workload model |
| [Qwen3-8B INT4](https://huggingface.co/OpenVINO/Qwen3-8B-int4-ov) | 2025-04 | 4.6 GB | text | 40k | 15.0 tok/s | 0.13 s | 12.4 ↓ | Coder-7B speed with a newer base |
| [Qwen3-VL-8B INT4](https://huggingface.co/OpenVINO/Qwen3-VL-8B-Instruct-int4-ov) | 2025-10 | 5.5 GB | text, image, videoᵇ | 256k | 14.5 tok/s | 0.15 s | — | chat-class speed; vision+video capable |
| [Qwen3.5-9B INT4-asym](https://huggingface.co/droans/qwen3.5-9B-int4-asym-ov) | 2026-02 | 5.7 GB | text, imageᵇ | 256k | ≈13 tok/s | 0.46 s | — | newest model generation; community conversion (droans) |
| [OmniCoder-9B INT4](https://huggingface.co/Echo9Zulu/OmniCoder-9B-int4_sym-ov) | 2026-03 | 5.7 GB | text, imageᵇ | 256k | ≈13 tok/s | 0.50 s | — | coding finetune of Qwen3.5-9B — strongest coding model that fits; community conversion |
| ~~[LFM2.5-350M INT8/FP16](https://huggingface.co/OpenVINO/LFM2.5-350M-int8-ov)~~ | ~~2026-03~~ | ~~0.4 GB~~ | ~~text~~ | — | — | — | — | **runtime bug** (`ScatterNDUpdate` shape validation, both official variants) |
| ~~[LFM2.5-8B-A1B INT4](https://huggingface.co/Echo9Zulu/LFM2.5-8B-A1B-int4_sym-awq-ov)~~ | ~~2026-05~~ | ~~4.5 GB~~ | ~~text~~ | — | — | — | — | **GPU compile never completes** (MoE expert graph; the dense-hybrid 1.2B works fine) |
| ~~[gpt-oss-20b INT4](https://huggingface.co/OpenVINO/gpt-oss-20b-int4-ov)~~ | ~~2025-08~~ | ~~11.7 GiB~~ | ~~text~~ | ~~128k~~ | — | — | — | **OOM on 32 GB RAM**: device allocation fails at compile despite 18 GB free host RAM |
| ~~[Qwen3-Coder-30B-A3B INT4](https://huggingface.co/OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov)~~ | ~~2025-07~~ | ~~15.2 GiB~~ | ~~text~~ | ~~256k~~ | — | — | — | **OOM on 32 GB RAM**: device allocation fails at compile |
| ~~[Gemma 4 26B A4B INT4](https://huggingface.co/Morteza89/gemma-4-26b-a4b-it-int4-ov)~~ | ~~2026-03~~ | ~~14.3 GiB~~ | ~~text, image, audioᵇ~~ | ~~256k~~ | — | — | — | **OOM on 32 GB RAM** (tested 3×): fails during weight upload even with 24 GB free RAM |

³ "PL edits" = decode with **prompt-lookup speculative decoding** on an echo-heavy code-edit
prompt. Measured with a *different prompt* than the Decode column — compare PL values with each
other, not against Decode. ↓ = slower than plain decoding on the same prompt (thinking-mode and
low-echo models lose; see RESEARCH.md Finding 6). "—" = not measured or unsupported (VLM-shaped).

ᵃ "Base released" is the Hugging Face creation date of the *original base model* repo (e.g.
`google/gemma-4-E2B-it`, `Qwen/Qwen2.5-Coder-1.5B-Instruct`), not the OpenVINO conversion date.

ᵇ Modalities and max context are the *model's* capabilities (from each model's `config.json`).
The server currently exposes a **text-only** API and keeps practical context well below the
maximum — KV-cache grows with context and competes with weights for the same shared iGPU
memory. Multimodal IRs run fine text-only through `VLMPipeline`.

The short version of *why* the table looks like this: decode speed on this iGPU is
memory-bandwidth-bound (smaller weights = proportionally faster), the usable model size is
capped well below the driver's ≈50%-of-RAM memory ceiling by compile-time overhead, and
quantization recipe / speculative decoding gains are architecture- and workload-specific.
The full methodology, measurements and conversion playbook live in [RESEARCH.md](RESEARCH.md).
