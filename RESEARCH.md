# Research log: LLM inference on the Intel Core Ultra iGPU

Methods and findings from benchmarking 19 models and converting several ourselves
(June 2026). Everything here was measured on one machine; treat absolute numbers
as machine-specific and the *rules* as the transferable result.

**Test rig:** Dell XPS 13, Intel Core Ultra 155H (Meteor Lake), Arc iGPU (Xe-LPG, 128 EU),
32 GB LPDDR5x, Windows 11, Intel driver 32.0.101.8724, OpenVINO GenAI 2026.3 nightly
(`dev20260603`), Python 3.12. Current per-model results: [BENCHMARKS.md](BENCHMARKS.md)
(workload method); the superseded raw-decode overview is archived in the appendix below.

---

## Methodology

- **Benchmark protocol** ([`scripts/bench.py`](scripts/bench.py)): load pipeline with compile
  cache, one warm-up generation (discarded), then 3 measured greedy generations of ~256 tokens.
  Report median decode tok/s (= generated tokens ÷ time after first token) and TTFT.
- **A/B comparisons must run back-to-back in one session.** Thermal state moves absolute
  numbers by up to ~20% (we measured the same model at 29.9 and 22.6 tok/s hours apart).
  Cross-session comparisons of small differences are meaningless.
- **Differences under ~1 tok/s are noise** at this run count. We verified this by A/B-ing a
  finetune against its base model (OmniCoder vs Qwen3.5-9B): the ranking flipped between
  sessions, medians pooled to identical.
- **Speculative decoding needs `perf_metrics`** ([`scripts/bench_prompt_lookup.py`](scripts/bench_prompt_lookup.py)):
  streamer callbacks deliver token *batches* under speculation, so counting callbacks
  undercounts tokens and fabricates a slowdown. Use the pipeline's own metrics.
- **Download progress bars count files, not bytes** — a stalled multi-GB download can show
  "53%" forever. Check that the `.incomplete` temp file under
  `<target>/.cache/huggingface/download/` is actually growing.

---

## Finding 1 — Decode speed is memory-bandwidth-bound

Decode throughput is a near-pure function of **bytes read per token**, across two orders of
magnitude and four model generations (2024-09 → 2026-03):

| Weights read/token | Measured decode | Example |
|---|---|---|
| 0.3 GB | 87.6 tok/s | Qwen2.5-Coder-0.5B |
| 0.9 GB | 57–73 tok/s | Coder-1.5B / Qwen3.5-0.8B |
| ~2 GB | 24–35 tok/s | Coder-3B / Qwen3-4B / Qwen3.5-2B |
| ~4.5 GB | 15 tok/s | Coder-7B / Qwen3-8B |
| ~5.7 GB | ≈13 tok/s | Qwen3.5-9B / OmniCoder-9B |

Consequences:
- **Newer architecture generations buy quality, not speed**, at equal size (Qwen2.5 → Qwen3 →
  Qwen3.5 all land on the same curve). Exception: Qwen3.5 runs slightly *above* its size class.
- A coding **finetune is exactly as fast as its base model** (A/B verified).
- Runtime tuning hints (`PERFORMANCE_HINT=LATENCY`, `INFERENCE_PRECISION_HINT=f16`,
  `KV_CACHE_PRECISION=u8`) changed nothing measurable — the bottleneck is physics, not config.

## Finding 2 — The memory ceiling is lower than it looks

- The Windows driver exposes **≈ 50% of installed RAM** as iGPU-addressable memory
  (`GPU_DEVICE_TOTAL_MEM_SIZE`; 16.4 GiB here). Mind GiB-vs-GB when comparing tools.
- **First-time compile transiently needs ~1.4× the weight bytes on the device** (original +
  kernel-reordered copies coexist). Practical model limit on 32 GB RAM: largest verified load
  is 6.0 GiB of weights; 11.7 GiB (gpt-oss-20b) fails. The 16.4 GiB ceiling is *not* the
  usable weight budget.
- Three distinct failure modes, all observed:
  - `USM Device` allocation failure → device ceiling (more system RAM won't help)
  - `USM Host` allocation failure → first compile also wants ~weights-sized *free system RAM*
    (close apps / reboot fixes this one)
  - async `CL_EXEC_STATUS_ERROR` mid-upload → device ceiling surfacing late
- The `.ovcache` compiled blob removes most of the peak on later loads — but it can only be
  produced by surviving the peak once, and blobs are device+driver specific (not shareable,
  not cross-device, CPU and GPU blobs are unrelated artifacts).

## Finding 3 — Three independent support gates

A model runs only if it clears all three; we hit failures at each level:

1. **transformers knows the config class** (`ministral3` didn't exist in 4.x)
2. **optimum-intel has an export config** for the architecture (`mistral3` missing even on git
   master under transformers 5.x — a catch-22 that currently makes Ministral-3 unconvertible)
3. **The intel_gpu plugin compiles and runs the graph**: LFM2.5's *dense-hybrid* 1.2B compiles
   in 13 s and runs great (87.6 tok/s); the same family's MoE 8B-A1B grinds indefinitely
   (killed at 27 min, 15 GB RSS); the official 350M conversions hit a `ScatterNDUpdate`
   runtime bug. Support is **per-model, not per-family**.

Check gate 2 from the installed toolchain:
`TasksManager._SUPPORTED_MODEL_TYPE` (166 types with OpenVINO export as of June 2026).

## Finding 4 — Only selective-read architectures beat the bandwidth law

- **Gemma 4 E-series (PLE/MatFormer)** stores ~4 GB but *streams* only ~1.4 GB per token (the
  per-layer embedding tables are gathered, not read wholesale) → 29.9 tok/s at a 4.1 GB disk
  size that "should" do ~15. The only curve-breaker that actually runs on this machine.
- **MoE** has the same property in theory; in practice every interesting MoE either exceeded
  the memory ceiling (gpt-oss-20b, Qwen3-30B-A3B, Gemma-26B-A4B) or failed to compile
  (LFM2.5-8B-A1B). As of June 2026: no working MoE on this hardware.
- The PLE path is also why E2B's TTFT and per-read-byte efficiency are slightly worse than
  dense peers — gather overhead. A channel-wise requantization gained it nothing (see next).

## Finding 5 — Quantization recipe can halve or double speed, per-architecture

Same model (granite-4.1-3b), same toolchain, three recipes, same session:

| Recipe | Size | Decode |
|---|---|---|
| int4 sym **channel-wise** (`--sym --group-size -1`) | 1.72 GiB | **27.4 tok/s** |
| int8 per-channel | 3.19 GiB | 17.4 tok/s |
| int4 asym group-128 (the common default) | 1.78 GiB | **13.0 tok/s** |

Group-wise dequantization is expensive on Arc kernels — the *smaller* g128 file ran at half
the cw speed. But the sensitivity is **architecture-specific**: the identical cw change on
Gemma 4 E2B (whose g128 build already rides the curve) gained 0%. And group-wise int4 is fine
for every Qwen we tested. **Rule: when converting, benchmark recipes; never assume.**
(Quality caveat: cw quantizes more coarsely than g128; we measured speed, not perplexity.)

## Finding 6 — Prompt-lookup gain is predicted by output/prompt n-gram overlap

Speculative decoding without a draft model (drafts from prompt n-grams, batched verification).
Three results, in the order we learned them:

**(a) The gain scales with model size within one family** (same code-edit prompt, same session;
every accepted draft token saves a weight-read, and weight-reads are what big models pay for):

| Qwen2.5-Coder | Plain | PL | Δ |
|---|---|---|---|
| 0.5B | 80.1 | 131.0 | +64% |
| 1.5B | 62.1 | 70.0 | +13% |
| 3B | 30.2 | 57.2 | +89% |
| 7B | 17.1 | **41.8** | **+144%** |

The 7B at 41.8 tok/s on edit workloads rewrites the speed/quality trade-off — 7B quality at
3B-class effective speed for echo-heavy tasks.

**(b) Why other models *lose* with PL** — measured via the draft-acceptance proxy: the fraction
of generated 3-grams already present in the prompt
([`scripts/research_pl_overlap.py`](scripts/research_pl_overlap.py)):

| Model | Output/prompt overlap | Plain | PL | Δ |
|---|---|---|---|---|
| Qwen3-0.6B (thinking) | 27.4% | 78.0 | 53.5 | −31% |
| Qwen2.5-Coder-1.5B | 44.4% | 61.6 | 68.5 | +11% |
| Granite-4.1-3b (general instruct) | 71.6% | 29.9 | 47.1 | **+58%** |

**(c) The rule** (this *corrects* our first hypothesis "FIM-trained vs general"): PL gain tracks
**output/prompt n-gram overlap**, break-even ≈ 35–40% here. The two real drivers:
- **Free-prose thinking is PL's worst case** — Qwen3's `<think>` preamble is hundreds of
  free-form tokens with ~zero prompt overlap; every draft is rejected. This, not "general vs
  coder", is why Qwen3-4B/8B regressed (−33%/−20%). But thinking per se isn't the variable:
  LFM2.5-1.2B-Thinking *gains* +63% on architect prompts because its reasoning restates the
  prompt heavily. Echo overlap must be measured, not inferred from model category.
- **Instruction-faithful echoing is the best case regardless of family** — Granite-4.1 (not
  FIM-trained) hit 71.6% overlap by following "keep the logic identical" verbatim and gained
  +58%, *beating* the Coder, which rewrote more creatively (44.4%).

The server enables PL per model via `PROMPT_LOOKUP_MODELS` (default: the autocomplete coder).
Caveats: LLMPipeline only (not VLM-shaped IRs); switches to the continuous-batching backend
whose numerics differ slightly — outputs are quality-equivalent but not bit-identical; all
gains are for echo-heavy prompts — free-form generation runs at plain speed or below.

## Finding 7 — Self-converted models reach parity with community artifacts

We replicated an existing community conversion (Gemma 4 E2B, matched recipe) and benched at
parity same-session (24.6 vs 22.6 tok/s, overlapping ranges). The conversion pipeline below is
therefore trusted for publication-grade artifacts. First published result:
[HarmenWessels/granite-4.1-3b-int4-cw-ov](https://huggingface.co/HarmenWessels/granite-4.1-3b-int4-cw-ov)
— the first OpenVINO IR of Granite 4.1.

---

## Conversion playbook (Route B)

Separate venv (`.venv-convert/`, gitignored) with: `optimum` + `optimum-onnx` + `optimum-intel`
from git master, torch CPU wheels, `nncf`, `compressed-tensors`.

```powershell
# typical text-only model, speed-first recipe WITH data-aware calibration
# (data-free cw-int4 measurably damaged quality on granite — AWQ+SE repaired it
#  at zero size/speed cost; see BENCHMARKS.md finding 9)
optimum-cli export openvino -m <org>/<model> --weight-format int4 --sym --group-size -1 `
  --awq --scale-estimation --dataset wikitext2 models\<owner>\<name>-int4-cw-ov

# multimodal (Gemma 4, Qwen-VL...): the supported task must be explicit
optimum-cli export openvino -m google/gemma-4-E2B-it --task image-text-to-text `
  --weight-format int4 models\<owner>\<name>
```

Hard-won rules:
0. **Quantization granularity must scale with model size**: cw-sym int4 + AWQ is the speed
   recipe for ~3B–8B (validated on Granite 4.1), but at ≤1B it produces *degenerate output*
   (MiniCPM5-1B: repetition loops; int8 and g128 of the same model are coherent). For tiny
   models use g128 or int8 and always run a coherence probe before benchmarking speed.
0b. **Hybrid-thinking models are controlled via the tokenizer IR's rt_info template.** GenAI
   cannot pass `enable_thinking`, and it reads the chat template from `openvino_tokenizer.xml`
   **rt_info** — not from `chat_template.jinja` (patching that file is a no-op). Hardcoding the
   no-think prefix (`<think>\n\n</think>\n\n` after the assistant header) in rt_info switched
   MiniCPM5-1B from preamble-failing to the fastest probe-passing edit model measured
   (81.4 tok/s). Corollary: "thinks by default" verdicts on other models (Qwen3 family) reflect
   their conversions' baked templates and may be flippable the same way — re-test before
   excluding a thinking-capable model.
1. **transformers version must match the target architecture** — and the requirements differ
   per model: granite wants 4.57.x; gemma4 wants exactly 5.5.0 (5.10 renamed an attention
   attribute and breaks the trace); Qwen3.5 wants 5.x. Swap per export; pip's dependency
   warnings against optimum's pins are expected and harmless.
2. **Every working conversion ships its recipe**: `openvino_config.json` in any HF conversion
   records the exact transformers version and quantization parameters used. Read it before
   reinventing.
3. **Never pass `--task text-generation`** — it exports without KV-cache (→ `beam_idx` error
   in GenAI). Omit `--task` for text models (auto-infers `-with-past`); pass the explicit
   multimodal task for VLMs.
4. **Bench every artifact before publishing** (Finding 5). Verify stateful export:
   the IR should have 4 inputs (`input_ids`, `attention_mask`, `position_ids`, `beam_idx`).

## Candidate screening ledger (sweep of 2026-06-06)

Exhaustive sweep of public models against our gates (dense or GPU-runnable, supported
architecture, ≤ ~6 GiB int4, permissive license, quality above incumbents). Screened out:

| Candidate | Reason |
|---|---|
| Gemma-4-12B (MMLU-Pro 77.2, LCB 72.0) | new `gemma4_unified` arch — unknown to transformers ≤5.5 and the export registry |
| GLM-4.7-Flash | MoE (`glm4_moe_lite`), unsupported type, too big |
| Qwen3.6-27B / Mistral-Small-4 / Gemma-4-31B / EXAONE-4.5-33B / Codestral | over the memory ceiling |
| Qwen3.6 small dense / Qwen3.5-Coder / EXAONE-4.5 ≤8B | not released yet |
| Seed-Coder-8B | 2025-05 vintage — matched/beaten by granite-4.1-8b (already published) |
| EXAONE-4.0 family | gated + restrictive license |
| GLM-4-9B-0414, Phi-4-mini, Falcon-3, OLMo-3, Hunyuan-7B | dated or dominated by incumbents at equal size |
| DeepSeek-R1 distills | thinking-default (edit-budget failures) |
| MiniCPM5-1B | converted & tested: coherent at g128 (~82–87 tok/s) but thinks by default under the OV chat template → no role won vs Qwen3.5-0.8B; also produced the granularity-vs-scale finding (playbook rule 0) |

Conclusion: as of 2026-06-06 the served lineup is at the practical optimum for this hardware —
every higher-quality candidate is upstream-blocked or unreleased, not effort-blocked.

## Open items (as of 2026-06)

- **Qwen3.5-Coder**: not yet released — would likely obsolete the Qwen2.5-Coder autocomplete
  default the moment a small FIM-trained variant ships.
- **Qwen3.6 small dense** (2B/4B/8B class): the 27B dense (2026-04, SWE-bench 77.2) suggests
  smaller siblings are coming — would likely supersede the Qwen3.5 tier.
- **EXAONE-4.5 small sizes**: 33B-only so far; STEM avg 77.3 beats GPT-5-mini — but mind the
  restrictive EXAONE license before investing.
- **Ministral-3 / `mistral3` export support** in optimum-intel: blocked upstream (gate-2
  catch-22 above).
- **MiniCPM-V-4.6 / `minicpmv4_6`**: the "best open model under 2B" (vision-capable, Apache-2.0,
  `qwen3_5_text` backbone) is blocked at both gates — confirmed empirically 2026-06-06
  (transformers ≤5.5 doesn't know the type; export registry has only the older `minicpmv`).
  Would fill the sub-2B vision niche nothing in our table covers.
- **Gemma-4-12B / `gemma4_unified`**: the quality standout of the fitting size class
  (MMLU-Pro 77.2, LiveCodeBench 72.0 at 11.95B) is a new encoder-free architecture absent
  from both transformers ≤5.5 and optimum-intel's export registry — unconvertible today,
  which explains why no OV IR of it exists. High-value re-check when support lands
  (~6.7 GiB int4 would also probe the load ceiling).
- **Gemma 4 E2B coding finetunes**: exist only as GGUF (e.g. `Gemma-4-e2bxOpus-4.7-turbo`);
  a safetensors release would enable converting the only curve-breaking architecture with
  coding tuning — the most valuable potential artifact for this hardware.
- **LFM2.5 fixes**: 350M `ScatterNDUpdate` runtime bug and the MoE compile hang may resolve
  in newer OpenVINO nightlies.
- **Linux**: the ~50%-of-RAM ceiling is Windows driver policy; the same laptop under native
  Ubuntu might load the 12–16 GiB models that OOM here. Untested.
- ~~Server enhancement — per-request thinking mode~~ **SHIPPED 2026-06-06**: the server derives
  think/nothink template variants at load and swaps them per request via
  `set_chat_template` under the generation lock (`reasoning_effort` / `enable_thinking`
  request fields; reasoning returned as `message.reasoning_content`). Observation from
  testing: 1B-scale thinking can *loop* on trivial problems (MiniCPM5 spent 500 tokens
  re-adding 460+161 and never finished, while no-think answered instantly and correctly) —
  thinking is not a free quality knob at small scale.
- **Server enhancement — per-request prompt-lookup**: PL is per-workload (+92% edits, −14%
  explain for the same model), but `PROMPT_LOOKUP_MODELS` toggles per model. A finer policy —
  enable PL only on `/v1/completions`, or when the chat prompt contains a code block — would
  capture the edit gains without the explain/architect penalty. Requires two pipeline
  instances or the CB pipeline's per-request config.

## Appendix: raw-decode model overview (legacy method, superseded by BENCHMARKS.md)

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

