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

# 2. Download the default models (~6 GB total; requires `hf auth login`)
.\.venv\Scripts\python.exe scripts\download_model.py --repo gregor160300/gemma-4-E2B-it-int4-ov
.\.venv\Scripts\python.exe scripts\download_model.py --repo OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov

# 3. Optional: benchmark a model (TTFT + decode tok/s)
.\.venv\Scripts\python.exe scripts\bench.py --model-dir models\gemma-4-E2B-it-int4-ov

# 4. Start the server on http://127.0.0.1:8000/v1
.\.venv\Scripts\python.exe server.py
```

The server loads **two models by default** and routes by the request's `model` field:
`gemma-4-E2B-it-int4-ov` (chat) and `Qwen2.5-Coder-3B-Instruct-int4-ov` (autocomplete via
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
  - name: Qwen2.5 Coder 3B (local autocomplete)
    provider: openai
    model: Qwen2.5-Coder-3B-Instruct-int4-ov
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

| Model | Weights | Decode | TTFT | Verdict |
|---|---|---|---|---|
| **Gemma 4 E2B INT4** (default chat) | 4.1 GB | **29.9 tok/s** | 0.23 s | fastest; very responsive in Continue |
| Qwen2.5-Coder-3B INT4 (default autocomplete) | 2.1 GB | 24.0 tok/s | 0.15 s | lowest TTFT |
| Gemma 4 E4B INT4 | 6.0 GB | 15.7 tok/s | 0.52 s | mid |
| Qwen2.5-Coder-7B INT4 | 4.2 GB | 15.0 tok/s | 0.20 s | best chat quality that fits |
| Qwen3-Coder-30B-A3B INT4 | 15.2 GiB | — | — | **OOM**: device allocation fails at compile |
| Gemma 4 26B A4B INT4 | 14.3 GiB | — | — | **OOM** (tested 3×): fails during weight upload even with 24 GB free RAM |

### What we learned about the hardware

- **The iGPU memory ceiling is 16.4 GiB (= 17.6 decimal GB)**, enforced by the Windows driver
  (`GPU_DEVICE_TOTAL_MEM_SIZE`) — the same figure Task Manager shows as "~18 GB" shared GPU
  memory. Weights *plus* upload/compile buffers must fit this budget, so the practical model
  limit is ≈ 12–13 GiB of weights. Both ~14–15 GiB MoE models fail at load.
- **Host RAM matters separately**: the first compile of a large model also needs roughly
  weights-sized free *system* RAM, failing with a `USM Host` allocation error otherwise. Freeing
  RAM fixes that failure mode — but not the device ceiling.
- **Decode is memory-bandwidth-bound**, not compute-bound: ~15 tok/s at ~4 GB of weights,
  ~24–30 tok/s at ~2–4 GB. Smaller models are faster in direct proportion to bytes read per token.
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

## License

This project is licensed under [Apache-2.0](LICENSE). Model weights are **not** distributed with
this repository and are subject to their own licenses — e.g. Gemma models are governed by
[Google's Gemma Terms of Use](https://ai.google.dev/gemma/terms), Qwen2.5-Coder models are
Apache-2.0. Review the license on each model's Hugging Face page before use.
