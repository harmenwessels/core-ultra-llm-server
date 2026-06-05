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
| Qwen3.5-4B | **79.1**⁴ | LCB-v6 55.8 | 89.8 | n/a | (thinking preamble) | ~20 | ✗² |
| Gemma 4 E2B (both conversions) | 60.0 | LCB-v6 44.0 | — | n/a | 21–23 | ~23 | ✓ |
| Gemma 4 E4B | **69.4** | LCB-v6 52.0 | — | n/a | 15.6 | ~16 | **✓** |
| Granite-4.1-3b cw v2 (ours) | 49.8 | HE 81.7 / MBPP 71.2 | 82.3 | — | 31.3 | ~31 | **✓** |
| Granite-4.1-3b int8 (ours) | 49.8 | same base | 82.3 | — | 24.6 +PL | ~15 | **✓** |
| Granite-4.1-8b cw (ours) | 56.0 | HE 85.4 / **MBPP 87.3** | **87.1** | — | 27.0 +PL | ~14 | **✓** |
| OmniCoder-9B (base: Qwen3.5-9B⁴: MMLU-Pro 82.5, LCB-v6 65.6, IFEval 91.5) | — | GPQA-D 83.8 / TB-2.0 23.6 | — | n/a | (zero code in budget) | ~13 | ✗² |
| Qwen3-VL-8B | n.r.⁵ | n.r.⁵ | n.r.⁵ | n/a | 14.8 | ~15 | ✓ |
| Qwen3-0.6B | n.r.⁵ | n.r.⁵ | n.r.⁵ | 1.16 s | (behavior changed) | ~84 | ✗ |
| LFM2.5-1.2B-Thinking | 49.7 | **card warns against programming use** | 88.4 | — | (no code block) | 84.7 (137.9 architect+PL) | ✗ |

⁴ Qwen3.5-4B/9B cards report thinking-mode-default scores — not directly comparable with
non-thinking rows (the 2B card shows the gap: 55.3 non-thinking vs 66.5 thinking on MMLU-Pro),
and the thinking preamble is precisely what fails our edit-budget probe.
⁵ official numbers exist only in the Qwen3 tech report (tables not published on the cards in
extractable form).

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
| Chat (quality tier) | Gemma 4 E4B | best validated general scores (MMLU-Pro 69.4), 15.6 tok/s, probe ✓ — when answer quality beats pace |
| Assistant (edit/tool quality tier) | Granite-4.1-8b cw (ours) **+PL** | IFEval 87 / MBPP 87 / BFCL 68: 27 tok/s edits / 14 chat, probes ✓, 128k context. Enable PL only for edit-heavy use (−14% on explain) |
| Assistant (edit, non-Coder option) | Granite-4.1-3b cw **v2** | 31.3 tok/s, probes ✓, 128k context — no PL needed |
| Avoid for edits | OmniCoder-9B, Qwen3.5-4B, Granite-4.1-cw **v1** | thinking preambles (former two); quantization damage (v1, fixed in v2) |
