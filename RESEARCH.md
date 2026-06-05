# Research log: LLM inference on the Intel Core Ultra iGPU

Methods and findings from benchmarking 19 models and converting several ourselves
(June 2026). Everything here was measured on one machine; treat absolute numbers
as machine-specific and the *rules* as the transferable result.

**Test rig:** Dell XPS 13, Intel Core Ultra 155H (Meteor Lake), Arc iGPU (Xe-LPG, 128 EU),
32 GB LPDDR5x, Windows 11, Intel driver 32.0.101.8724, OpenVINO GenAI 2026.3 nightly
(`dev20260603`), Python 3.12. Full per-model results: see the table in [README.md](README.md).

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

## Finding 6 — Prompt-lookup decoding is a per-workload bet

Speculative decoding without a draft model (drafts from prompt n-grams, batched verification).
Same code-edit prompt, three models:

| Model | Plain | Prompt-lookup | Δ |
|---|---|---|---|
| Qwen2.5-Coder-1.5B (FIM-trained) | 53.4 | 66.6 | **+25%** |
| Qwen3-4B (general) | 25.9 | 17.3 | −33% |
| Qwen3-8B (general) | 15.6 | 12.4 | −20% |

The dividing line is **echo fidelity**: FIM-trained coders reproduce prompt text near-verbatim
(high draft acceptance → batched bandwidth wins); general chat models reformulate (drafts miss
→ pure verification overhead). The server enables PL only for the autocomplete model
(`PROMPT_LOOKUP_MODELS`). Caveats: LLMPipeline only (not VLM-shaped IRs); switches to the
continuous-batching backend whose numerics differ slightly — outputs are quality-equivalent
but not bit-identical.

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
# typical text-only model, speed-first recipe
optimum-cli export openvino -m <org>/<model> --weight-format int4 --sym --group-size -1 `
  models\<owner>\<name>-int4-cw-ov

# multimodal (Gemma 4, Qwen-VL...): the supported task must be explicit
optimum-cli export openvino -m google/gemma-4-E2B-it --task image-text-to-text `
  --weight-format int4 models\<owner>\<name>
```

Hard-won rules:
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

## Open items (as of 2026-06)

- **Qwen3.5-Coder**: not yet released — would likely obsolete the Qwen2.5-Coder autocomplete
  default the moment a small FIM-trained variant ships.
- **Ministral-3 / `mistral3` export support** in optimum-intel: blocked upstream (gate-2
  catch-22 above).
- **Gemma 4 E2B coding finetunes**: exist only as GGUF (e.g. `Gemma-4-e2bxOpus-4.7-turbo`);
  a safetensors release would enable converting the only curve-breaking architecture with
  coding tuning — the most valuable potential artifact for this hardware.
- **LFM2.5 fixes**: 350M `ScatterNDUpdate` runtime bug and the MoE compile hang may resolve
  in newer OpenVINO nightlies.
- **Linux**: the ~50%-of-RAM ceiling is Windows driver policy; the same laptop under native
  Ubuntu might load the 12–16 GiB models that OOM here. Untested.
