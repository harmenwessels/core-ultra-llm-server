# Core Ultra LLM Server

Local, OpenAI-compatible LLM inference server for **Intel Core Ultra laptops** — runs chat and
code autocomplete entirely on the integrated **Arc iGPU** using [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai).
Built as a lean backend for [Continue.dev](https://continue.dev) in VS Code: ~30 tok/s chat and
0.15 s time-to-first-token autocomplete on a Core Ultra 155H, with no cloud, no GPU passthrough
tricks, and no heavyweight serving framework.

- **OpenAI API surface**: `/v1/models`, `/v1/chat/completions` (SSE streaming), `/v1/completions` (FIM autocomplete)
- **Multi-model**: serves several models from one port, routed by the request's `model` field
- **iGPU-first**: INT4 OpenVINO IR models, compile caching, single-flight scheduling tuned for shared-memory iGPUs
- **Lean**: one ~300-line FastAPI file plus three scripts; dependencies are `openvino-genai`, `fastapi`, `uvicorn`, `huggingface_hub`

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together.

## Requirements

- Intel Core Ultra CPU with integrated Arc graphics (tested: Core Ultra 155H / Meteor Lake)
- Windows 11 with a current Intel graphics driver (tested: 32.0.101.8724)
- Python 3.12 (3.13 is not yet supported by the pinned OpenVINO nightly toolchain)
- A Hugging Face account for model downloads (`hf auth login` — anonymous downloads of
  multi-GB files stall)

Older Iris Xe iGPUs (11th–13th gen) should work through the same OpenVINO GPU plugin at roughly
half the throughput (untested). Linux should also work with the Intel compute runtime installed
(untested; the code is platform-neutral apart from two Windows-only diagnostics that degrade
gracefully).

## Quick start

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1. Verify the Arc iGPU is visible to OpenVINO
.\.venv\Scripts\python.exe scripts\check_gpu.py

# 2. Download the default models (~5 GB total; requires `hf auth login`)
.\.venv\Scripts\python.exe scripts\download_model.py --repo gregor160300/gemma-4-E2B-it-int4-ov
.\.venv\Scripts\python.exe scripts\download_model.py --repo OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov

# 3. Optional: benchmark a model (TTFT + decode tok/s)
.\.venv\Scripts\python.exe scripts\bench.py --model-dir models\gemma-4-E2B-it-int4-ov

# 4. Start the server on http://127.0.0.1:8000/v1
.\.venv\Scripts\python.exe server.py
```

The server loads **two models by default** and routes by the request's `model` field:
`gemma-4-E2B-it-int4-ov` (chat) and `Qwen2.5-Coder-1.5B-Instruct-int4-ov` (autocomplete via
`/v1/completions` with FIM). The first launch per model pays a one-time compile cost (~30–70 s);
subsequent launches load from the cache in seconds.

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_DIRS` | the two models above | `;`-separated list of model directories to serve |
| `MODEL_DIR` | — | single-model override |
| `DEVICE` | `GPU` | OpenVINO device (`CPU` as debug fallback) |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | bind address |
| `CACHE_DIR` | `./.ovcache` | compiled-blob cache location |

Example — serve one bigger chat model instead:

```powershell
$env:MODEL_DIR = "models\Qwen2.5-Coder-7B-Instruct-int4-ov"
.\.venv\Scripts\python.exe server.py
```

### Continue.dev config (`~/.continue/config.yaml`)

```yaml
models:
  - name: Gemma 4 E2B (local OpenVINO)
    provider: openai
    model: gemma-4-E2B-it-int4-ov
    apiBase: http://127.0.0.1:8000/v1
    apiKey: dummy
    roles:
      - chat
      - edit
  - name: Qwen2.5 Coder 1.5B (local autocomplete)
    provider: openai
    model: Qwen2.5-Coder-1.5B-Instruct-int4-ov
    apiBase: http://127.0.0.1:8000/v1
    apiKey: dummy
    roles:
      - autocomplete
```

`apiBase` must include `/v1`. Continue streams by default; the server implements SSE on both
`/v1/chat/completions` and `/v1/completions` (the latter serves raw/FIM prompts with the chat
template disabled — required for autocomplete to work).

## Measured results

Core Ultra 155H (Meteor Lake), Arc iGPU, 32 GB LPDDR5x, driver 32.0.101.8724,
OpenVINO 2026.3 nightly:

| Model | Base released² | Weights | Modalities | Max context | Decode | TTFT | Verdict |
|---|---|---|---|---|---|---|---|
| [Qwen2.5-Coder-0.5B INT4](https://huggingface.co/OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov) | 2024-11 | 0.3 GB | text | 32k | **87.6 tok/s** | 0.06 s | fastest; quality floor for autocomplete |
| [LFM2.5-1.2B-Thinking INT4](https://huggingface.co/Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov) | 2026-01 | 0.6 GB | text | 128k | **87.6 tok/s** | 0.08 s | hybrid conv/attention; reasoning model (thinking tokens add latency); community conversion |
| [Qwen3.5-0.8B INT4](https://huggingface.co/yangsu0423/Qwen3.5-0.8B-int4-ov) | 2026-02 | 0.9 GB | text, image¹ | 256k | **72.7 tok/s** | 0.08 s | newest gen at near-0.5B speed; community conversion |
| [Qwen3-0.6B INT4](https://huggingface.co/OpenVINO/Qwen3-0.6B-int4-ov) | 2025-04 | 0.4 GB | text | 40k | **62.7 tok/s** | 0.10 s | slower than the newer, similar-size Qwen3.5-0.8B |
| [Qwen2.5-Coder-1.5B INT4](https://huggingface.co/OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov) (default autocomplete) | 2024-09 | 0.9 GB | text | 32k | **57.0 tok/s** | 0.06 s | autocomplete sweet spot: FIM-trained, 2.4× faster than the 3B |
| **[Gemma 4 E2B INT4](https://huggingface.co/gregor160300/gemma-4-E2B-it-int4-ov)** (default chat) | 2026-03 | 4.1 GB | text, image, audio¹ | 128k | 29.9 tok/s | 0.23 s | fastest chat-quality model; very responsive in Continue |
| [Qwen3-4B INT4](https://huggingface.co/OpenVINO/Qwen3-4B-int4-ov) | 2025-04 | 2.1 GB | text | 40k | 24.9 tok/s | 0.10 s | same speed as Coder-3B with a newer base |
| [Granite-4.0-micro INT4](https://huggingface.co/llmware/granite-4-micro-ov) | 2025-09 | 2.2 GB | text | 128k | 24.6 tok/s | 0.16 s | IBM; 128k context at 3B-class speed; community conversion (llmware) |
| [Qwen2.5-Coder-3B INT4](https://huggingface.co/OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov) | 2024-11 | 2.1 GB | text | 32k | 24.0 tok/s | 0.15 s | strong FIM quality |
| [Qwen3.5-4B INT4](https://huggingface.co/yangsu0423/Qwen3.5-4B-int4-ov) | 2026-02 | 3.3 GB | text, image¹ | 256k | 19.9 tok/s | 0.31 s | newest gen; faster than the 9B at similar quality class; community conversion |
| [Gemma 4 E4B INT4](https://huggingface.co/OpenVINO/gemma-4-E4B-it-int4-ov) | 2026-03 | 6.0 GB | text, image, audio¹ | 128k | 15.7 tok/s | 0.52 s | mid |
| [Qwen2.5-Coder-7B INT4](https://huggingface.co/OpenVINO/Qwen2.5-Coder-7B-Instruct-int4-ov) | 2024-09 | 4.2 GB | text | 32k | 15.0 tok/s | 0.20 s | best chat quality that fits |
| [Qwen3-8B INT4](https://huggingface.co/OpenVINO/Qwen3-8B-int4-ov) | 2025-04 | 4.6 GB | text | 40k | 15.0 tok/s | 0.13 s | Coder-7B speed with a newer base |
| [Qwen3-VL-8B INT4](https://huggingface.co/OpenVINO/Qwen3-VL-8B-Instruct-int4-ov) | 2025-10 | 5.5 GB | text, image, video¹ | 256k | 14.5 tok/s | 0.15 s | chat-class speed; vision+video capable |
| [Qwen3.5-9B INT4-asym](https://huggingface.co/droans/qwen3.5-9B-int4-asym-ov) | 2026-02 | 5.7 GB | text, image¹ | 256k | ≈13 tok/s | 0.46 s | newest model generation; community conversion (droans) |
| [OmniCoder-9B INT4](https://huggingface.co/Echo9Zulu/OmniCoder-9B-int4_sym-ov) | 2026-03 | 5.7 GB | text, image¹ | 256k | ≈13 tok/s | 0.50 s | coding finetune of Qwen3.5-9B — strongest coding model that fits; identical speed to its base (A/B verified); community conversion |
| ~~[LFM2.5-350M INT8/FP16](https://huggingface.co/OpenVINO/LFM2.5-350M-int8-ov)~~ | ~~2026-03~~ | ~~0.4 GB~~ | ~~text~~ | — | — | — | **runtime bug**: compiles, but inference fails on `ScatterNDUpdate` shape validation (both official variants; OpenVINO nightly dev20260603) |
| ~~[LFM2.5-8B-A1B INT4](https://huggingface.co/Echo9Zulu/LFM2.5-8B-A1B-int4_sym-awq-ov)~~ | ~~2026-05~~ | ~~4.5 GB~~ | ~~text~~ | — | — | — | **GPU compile never completes** (killed after 27 min, 15 GB RSS) — likely its MoE expert graph: the dense-hybrid LFM2.5-1.2B compiles in 13 s |
| ~~[gpt-oss-20b INT4](https://huggingface.co/OpenVINO/gpt-oss-20b-int4-ov)~~ | ~~2025-08~~ | ~~11.7 GiB~~ | ~~text~~ | ~~128k~~ | — | — | **OOM on 32 GB RAM**: device allocation fails at compile despite 18 GB free host RAM |
| ~~[Qwen3-Coder-30B-A3B INT4](https://huggingface.co/OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov)~~ | ~~2025-07~~ | ~~15.2 GiB~~ | ~~text~~ | ~~256k~~ | — | — | **OOM on 32 GB RAM**: device allocation fails at compile |
| ~~[Gemma 4 26B A4B INT4](https://huggingface.co/Morteza89/gemma-4-26b-a4b-it-int4-ov)~~ | ~~2026-03~~ | ~~14.3 GiB~~ | ~~text, image, audio¹~~ | ~~256k~~ | — | — | **OOM on 32 GB RAM** (tested 3×): fails during weight upload even with 24 GB free RAM |

² "Base released" is the Hugging Face creation date of the *original base model* repo (e.g.
`google/gemma-4-E2B-it`, `Qwen/Qwen2.5-Coder-1.5B-Instruct`), not the OpenVINO conversion date.

¹ Modalities and max context are the *model's* capabilities (from each model's `config.json`).
The server currently exposes a **text-only** API and keeps practical context well below the
maximum — KV-cache grows with context and competes with weights for the same shared iGPU
memory. Multimodal IRs run fine text-only through `VLMPipeline`.

### What we learned about the hardware

- **The iGPU memory ceiling is ≈ 50% of installed system RAM**, enforced by the Windows
  driver — on this 32 GB machine that is 16.4 GiB (= 17.6 decimal GB, the figure Task Manager
  shows as "~18 GB" shared GPU memory; query it via `GPU_DEVICE_TOTAL_MEM_SIZE`). A 16 GB
  laptop gets ~8 GiB; a 64 GB machine ~32 GiB. Weights *plus* upload/compile buffers must fit
  this budget, and the compile-time overhead is substantial: on 32 GB RAM the largest model
  that loads is 6.0 GiB of weights, while 11.7 GiB (gpt-oss-20b) already fails — so the
  practical limit lies somewhere between, well below the 16.4 GiB ceiling itself. All three
  failed MoE models would likely fit with 64 GB of RAM.
  Unit note: "32 GB" RAM is binary (32 GiB = 34.4 decimal GB), so dividing a decimal vRAM
  figure (e.g. "17.9 GB") by it overstates the ratio — in consistent units the measured
  ceiling is 52% of RAM (17,626,103,808 bytes vs 33,777,467,392 bytes).
- **Host RAM matters separately**: the first compile of a large model also needs roughly
  weights-sized free *system* RAM, failing with a `USM Host` allocation error otherwise. Freeing
  RAM fixes that failure mode — but not the device ceiling.
- **Decode is memory-bandwidth-bound**, not compute-bound: throughput scales almost exactly
  inversely with weight size across two orders of magnitude — 0.3 GB → 88 tok/s,
  0.9 GB → 57, 2.1 GB → 24, 4.2 GB → 15 (tok/s × GB ≈ 26–60, tightening as models grow).
- **Gemma 4 IRs are VLM-shaped** (MatFormer per-layer embeddings + vision tower) and require
  `VLMPipeline` even for text-only use; Qwen IRs are plain `LLMPipeline`. The server and bench
  auto-detect the IR shape per model directory.
- GPU latency tuning hints (`PERFORMANCE_HINT=LATENCY`, `INFERENCE_PRECISION_HINT=f16`,
  `KV_CACHE_PRECISION=u8`) made no measurable difference on this hardware.

## Repository layout

```
server.py                  OpenAI-compatible FastAPI server (multi-model, SSE, FIM)
scripts/check_gpu.py       verify OpenVINO sees the Arc iGPU
scripts/download_model.py  fetch OpenVINO IR models from Hugging Face
scripts/bench.py           TTFT + decode-throughput benchmark (3 measured runs)
requirements.txt           pinned dependency versions (incl. OpenVINO nightly index)
models/                    downloaded models (gitignored)
.ovcache/                  compiled-blob cache (gitignored)
```

## Alternatives / related projects

This server is deliberately minimal. If you need more than chat + autocomplete, look at:

- [OpenVINO Model Server (OVMS)](https://github.com/openvinotoolkit/model_server) — Intel's official
  serving solution, speaks the OpenAI API, production-grade.
- [OpenArc](https://github.com/SearchSavior/OpenArc) — community inference server for Intel devices
  (CPU/GPU/NPU): LLMs, VLMs, Whisper, TTS, embeddings and rerankers over OpenAI endpoints, with
  multi-model concurrency and speculative decoding. Note: model conversions published by its author
  (e.g. the LFM2.5 family) are typically validated on Arc B-series discrete GPUs — support on a
  Core Ultra iGPU may differ (see the LFM2.5 row in the results table).

## License

This project is licensed under [Apache-2.0](LICENSE). Model weights are **not** distributed with
this repository and are subject to their own licenses — e.g. Gemma models are governed by
[Google's Gemma Terms of Use](https://ai.google.dev/gemma/terms), Qwen2.5-Coder models are
Apache-2.0. Review the license on each model's Hugging Face page before use.
