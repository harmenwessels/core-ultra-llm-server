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

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together,
[RESEARCH.md](RESEARCH.md) for the performance findings (bandwidth scaling, memory ceiling,
quantization recipes, prompt-lookup decoding) and the model-conversion playbook, and
[benchmark/](benchmark/README.md) for the per-task-type leaderboard.

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

# 2. Download the registry's models (~7 GB; requires `hf auth login`)
.\.venv\Scripts\python.exe scripts\download_model.py --repo HarmenWessels/granite-4.1-8b-int4-cw-ov
.\.venv\Scripts\python.exe scripts\download_model.py --repo Echo9Zulu/Qwen3.5-2B-int4_sym-ov
.\.venv\Scripts\python.exe scripts\download_model.py --repo OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov

# 3. Optional: benchmark a model (TTFT + decode tok/s)
.\.venv\Scripts\python.exe scripts\bench.py --model-dir models\HarmenWessels\granite-4.1-8b-int4-cw-ov

# 4. Start the server on http://127.0.0.1:8000/v1 (config: models.yaml)
.\.venv\Scripts\python.exe server.py
```

Models live under `models/<owner>/<name>`, mirroring their Hugging Face repo ids.
`models.yaml` defines what gets served: by default **granite-8b** (chat/edit + agent
executor, prefix-cached), **qwen3.5-2b** (fast chat + analyst, prefix-cached) and
**coder-1.5b** (autocomplete via `/v1/completions` with FIM, prompt-lookup), plus the
**`virtual/agent`** orchestrating model id. Registry entries whose directories are missing
are skipped with a warning (the NPU autocomplete entry requires a self-converted cw-sym
artifact — one `optimum-cli` command, see [RESEARCH.md](RESEARCH.md)). The first launch
per model pays a one-time compile cost (~30–70 s); subsequent launches load from the
cache in seconds.

### Configuration

**`models.yaml` — the model registry (preferred).** Every model's measured operating
manual in one file: served alias, device (GPU/NPU), prefix-caching pool, prompt-lookup,
**tool language** (native gemma/lfm protocols vs hermes), thinking policy, context
budget, and the virtual model's role casting. The shipped file documents the production
stack; `MODELS_CONFIG` points elsewhere. Explicit `MODEL_DIRS` env overrides the file
for quick experiments.

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_DIRS` | the two models above | `;`-separated list of model directories to serve |
| `MODEL_DIR` | — | single-model override |
| `DEVICE` | `GPU` | OpenVINO device (`CPU` as debug fallback) |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | bind address |
| `CACHE_DIR` | `./.ovcache` | compiled-blob cache location |
| `PROMPT_LOOKUP_MODELS` | the autocomplete model | `;`-separated model ids that use prompt-lookup speculative decoding (helps FIM models, hurts general chat — see [RESEARCH.md](RESEARCH.md)) |
| `SCHEDULER_MODELS` | `granite-4.1-8b…=4` | `model_id=GB` pairs: prefix caching + chunked prefill. Multi-turn TTFT collapses ~27× (71 s → 2.6 s on an 8k history); the GB value is a permanently reserved KV pool — budget it against iGPU memory |
| `MAX_NEW_TOKENS_CAP` | `8192` | hard cap on client `max_tokens` (agent frontends ask for context-sized budgets) |
| `MODEL_DEVICES` | — | `model_id=NPU` pairs: per-model device with **per-device generation locks** — NPU autocomplete keeps answering (~7 s) while the GPU runs chat/agent turns. NPU needs channel-wise-sym int4 IRs, probe-certified per device ([RESEARCH.md](RESEARCH.md) finding 14) |
| `VIRTUAL_ROLES` | the measured casting | role=model_id triples for the **virtual model** (below) |

### The virtual model (`virtual/agent`)

One model id that routes every turn to the best measured brain (see [benchmark/](benchmark/README.md)):
a router classifies the request, the **architect** (Qwen3.5-2B) investigates and plans with
read-only tools, the **executor** (granite-8b) implements with edit→test→verify loops —
with server-side guards for the failure modes the role-fitness suite measured (tool-call
loops, non-matching edits). Design requests return the plan as `reasoning_content` (a
collapsible thinking block in Continue) with the implementation as the answer. Works
identically from Continue chat, Continue CLI (`cn`), or any OpenAI-compatible frontend —
the orchestration lives server-side.

The server also implements **OpenAI tool calling** for local models — speaking each
model family's **native function-calling language** (Gemma's declaration/`call:` protocol
and LFM's Pythonic calls via server-side rendering of the model's own chat template;
hermes-style injection for families trained on it, e.g. Qwen and granite). This makes
agentic frontends that require native function calling — Continue agent mode/CLI,
Kilo CLI — work against any served model, fairly. Model-by-model agent fitness is
measured in [BENCHMARKS.md](benchmark/README.md) (role-fitness suite); format mismatch
measurably suppresses scores, so the language is pinned per model in `models.yaml`.

Example — serve one bigger chat model instead (env overrides the registry):

```powershell
$env:MODEL_DIR = "models\OpenVINO\Qwen2.5-Coder-7B-Instruct-int4-ov"
.\.venv\Scripts\python.exe server.py
```

### Continue.dev config (`~/.continue/config.yaml`)

Model ids match the registry aliases (or `owner/name` when no alias is set):

```yaml
models:
  - name: virtual agent (router/architect/executor)
    provider: openai
    model: virtual/agent
    apiBase: http://127.0.0.1:8000/v1
    apiKey: dummy
    roles:
      - chat
    capabilities:
      - tool_use
  - name: granite-8b (local chat/edit)
    provider: openai
    model: granite-8b
    apiBase: http://127.0.0.1:8000/v1
    apiKey: dummy
    roles:
      - chat
      - edit
  - name: Qwen Coder 1.5B (local autocomplete)
    provider: openai
    model: coder-1.5b
    apiBase: http://127.0.0.1:8000/v1
    apiKey: dummy
    roles:
      - autocomplete
```

`apiBase` must include `/v1`. Continue streams by default; the server implements SSE on both
`/v1/chat/completions` and `/v1/completions` (the latter serves raw/FIM prompts with the chat
template disabled — required for autocomplete to work).

### Thinking mode (hybrid-reasoning models)

For hybrid-thinking models (Qwen3.5, MiniCPM5, …) the server detects the thinking control in
the model's chat template at load and enables a **per-request switch** — no reload needed:

- default: thinking **off** (direct answers; best for coding workloads)
- `"reasoning_effort": "low" | "medium" | "high"` or `"enable_thinking": true` → thinking on;
  the reasoning is returned separately as `message.reasoning_content` (DeepSeek-style)

Responses are **tag-clean in both modes**: reasoning never appears as raw `<think>` text — it
streams as `delta.reasoning_content` chunks (DeepSeek convention, understood by Continue) and
arrives as `message.reasoning_content` on non-streamed responses.

To make the switch a dropdown in Continue, define two entries for the same served model:

```yaml
  - name: Qwen3.5 4B (fast)
    provider: openai
    model: yangsu0423/Qwen3.5-4B-int4-ov
    apiBase: http://127.0.0.1:8000/v1
    apiKey: dummy
    roles: [chat, edit]
  - name: Qwen3.5 4B (thinking)
    provider: openai
    model: yangsu0423/Qwen3.5-4B-int4-ov   # same model — same loaded instance
    apiBase: http://127.0.0.1:8000/v1
    apiKey: dummy
    roles: [chat]
    requestOptions:
      extraBodyProperties:
        reasoning_effort: high
```

Background: GenAI cannot pass `enable_thinking` template kwargs, so the server swaps between
two derived template variants at generation time (single-flight makes this safe). See
RESEARCH.md for the template mechanics.

## Benchmark results

Best model per task type on a Core Ultra 155H Arc iGPU:

<!--BEST-SETUP START-->
| Task type | Best single | Q | Best combo | Q |
|---|---|---|---|---|
| codegen | OpenVINO/Qwen3-8B-int4-cw-ov | 12/12 | — | — |
| edit | OpenVINO/Qwen3-14B-int4-ov | 2/2 | — | — |
| agent-loop | HarmenWessels/gemma-4-E4B-it-qat-int4-ov | 7/7 | — | — |
| analysis | HarmenWessels/gemma-4-E4B-it-qat-int4-ov | 4/4 | — | — |
| autocomplete-fim | HarmenWessels/LFM2.5-1.2B-Instruct-int4-ov | 1/1 | — | — |

<!--BEST-SETUP END-->

Full per-task leaderboard, every model, and the run records: [benchmark/](benchmark/README.md).

## Repository layout

```
server.py                       OpenAI-compatible FastAPI server (multi-model, SSE, FIM)
scripts/check_gpu.py            verify OpenVINO sees the Arc iGPU
scripts/download_model.py       fetch OpenVINO IR models from Hugging Face
scripts/bench.py                TTFT + decode-throughput benchmark (3 measured runs)
scripts/bench_prompt_lookup.py  A/B harness for prompt-lookup speculative decoding
requirements.txt                pinned dependency versions (incl. OpenVINO nightly index)
models/<owner>/<name>/          downloaded models, mirroring HF repo ids (gitignored)
.ovcache/                       compiled-blob cache (gitignored)
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
