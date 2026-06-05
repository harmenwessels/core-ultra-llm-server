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

¹ raw-continuation probe artifact (no stop criterion), not a verified failure.

## Findings

1. **PL is a per-workload switch, not a per-model one**: +14…+114% on echo-heavy edits, but
   −21…−42% on explain/architect *for the same models*. Enable it only for models dedicated to
   FIM/edit duty.
2. **PL is not always quality-neutral**: Granite's +PL edit output dropped a closing
   parenthesis (SyntaxError) — reproduced across runs. The continuous-batching backend's
   numerics can change outputs for the worse; the probe gate matters.
3. **Granite-4.1-cw silently dropped a line of logic** in the edit task (the backordered
   remainder of partial shipments) — reproduced. Whether this is channel-wise quantization
   damage or the base model is unresolved (re-test against an int8 build to isolate).
4. **Qwen2.5-Coder-1.5B is an autocomplete specialist**: best-in-class completion latency with
   a passing probe, but it cannot complete the edit task within 512 tokens (verbose type
   annotation spam + a missing `Tuple` import).
5. **Qwen3.5-2B outperforms the current chat default (Gemma 4 E2B) on every chat profile**
   (~42 vs ~23 tok/s, lower TTFT, edit probe passing) on this suite.

## Current role recommendations (will evolve as more models run)

| Role | Recommendation | Why |
|---|---|---|
| Autocomplete | Qwen2.5-Coder-1.5B (+PL) | 1.07 s completions, probe ✓ |
| Assistant (edit-heavy) | Qwen2.5-Coder-7B **with PL** | only fast *and* correct edit model (34.4 tok/s, probes ✓) |
| Assistant (explain) / Architect | Qwen3.5-2B | 37–43 tok/s, probe ✓ — pending a chat-quality A/B vs Gemma E2B |
