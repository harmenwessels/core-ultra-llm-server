r"""Phase 3/5: lean OpenAI-compatible API server for OpenVINO GenAI on Intel Arc iGPU.

Serves one or more models behind the OpenAI surface Continue.dev needs:
  GET  /v1/models           -> all loaded model ids
  POST /v1/chat/completions -> chat (routed by the request's "model" field)
  POST /v1/completions      -> legacy/raw completions (Continue autocomplete FIM)

Design choices (see README):
  - Pipelines loaded ONCE at startup, compile cache enabled (CACHE_DIR).
  - Single-flight across ALL models: one generation at a time, serialized with
    a lock (single-user box; the iGPU is bandwidth-bound — concurrency only
    thrashes).
  - VLM-shaped IRs (Gemma 4) are auto-detected and served text-only through
    VLMPipeline; plain LLM IRs (Qwen) go through LLMPipeline.

Config via environment variables:
  MODEL_DIRS (";"-separated list; default: models/gemma-4-E2B-it-int4-ov;models/Qwen2.5-Coder-3B-Instruct-int4-ov)
  MODEL_DIR  (single-model override, kept for backward compat)
  DEVICE     (default: GPU)
  HOST       (default: 127.0.0.1)
  PORT       (default: 8000)
  CACHE_DIR  (default: ./.ovcache)

Run:
    .\.venv\Scripts\python.exe server.py
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import queue
import threading
import time
import uuid

import openvino_genai as ov_genai
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = pathlib.Path(__file__).resolve().parent

_default_dirs = (
    f"{ROOT / 'models' / 'gemma-4-E2B-it-int4-ov'};"
    f"{ROOT / 'models' / 'Qwen2.5-Coder-3B-Instruct-int4-ov'}"
)
if os.environ.get("MODEL_DIR"):  # single-model override
    _default_dirs = os.environ["MODEL_DIR"]
MODEL_DIRS = [
    pathlib.Path(p) for p in os.environ.get("MODEL_DIRS", _default_dirs).split(";") if p
]
DEVICE = os.environ.get("DEVICE", "GPU")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
CACHE_DIR = pathlib.Path(os.environ.get("CACHE_DIR", ROOT / ".ovcache"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("server")

app = FastAPI(title="openvino-windows-openai-api")

_pipes: dict[str, object] = {}  # model id -> pipeline
_gen_lock = threading.Lock()  # single-flight generation across all models


class QueueStreamer:
    """Decodes streamed tokens via ov TextStreamer into a thread-safe queue.

    StreamerBase.write receives raw token ids from VLMPipeline (and str pieces
    from LLMPipeline) — TextStreamer normalizes both to decoded text.
    """

    _DONE = object()

    def __init__(self, tokenizer: ov_genai.Tokenizer) -> None:
        self.q: queue.Queue = queue.Queue()
        self.token_count = 0
        self.first_token_time: float | None = None
        self.text_streamer = ov_genai.TextStreamer(tokenizer, self._on_text)

    def _on_text(self, text: str) -> ov_genai.StreamingStatus:
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self.token_count += 1
        if text:
            self.q.put(text)
        return ov_genai.StreamingStatus.RUNNING

    def finish(self) -> None:
        self.q.put(self._DONE)


def _load_pipelines() -> None:
    for model_dir in MODEL_DIRS:
        if not model_dir.exists():
            raise SystemExit(f"model dir not found: {model_dir}")
        is_vlm = (model_dir / "openvino_vision_embeddings_model.xml").exists()
        pipe_cls = ov_genai.VLMPipeline if is_vlm else ov_genai.LLMPipeline
        log.info("Loading %s from %s on %s (cache: %s)",
                 pipe_cls.__name__, model_dir, DEVICE, CACHE_DIR)
        t0 = time.perf_counter()
        _pipes[model_dir.name] = pipe_cls(str(model_dir), DEVICE, CACHE_DIR=str(CACHE_DIR))
        log.info("%s ready in %.1fs", model_dir.name, time.perf_counter() - t0)


def _resolve_pipe(body: dict):
    """Pick the pipeline for the request's model field (default: first loaded)."""
    model_id = body.get("model")
    if not model_id:
        model_id = next(iter(_pipes))
    pipe = _pipes.get(model_id)
    if pipe is None:
        raise HTTPException(
            status_code=404,
            detail=f"model '{model_id}' not loaded; available: {list(_pipes)}",
        )
    return model_id, pipe


def _build_generation_config(pipe, body: dict, default_max: int = 1024
                             ) -> ov_genai.GenerationConfig:
    cfg = pipe.get_generation_config()
    cfg.max_new_tokens = int(
        body.get("max_completion_tokens") or body.get("max_tokens") or default_max
    )
    temperature = body.get("temperature")
    top_p = body.get("top_p")
    if temperature is not None and temperature > 0:
        cfg.do_sample = True
        cfg.temperature = float(temperature)
        if top_p is not None:
            cfg.top_p = float(top_p)
    else:
        cfg.do_sample = False
    stop = body.get("stop")
    if stop:
        cfg.stop_strings = set([stop] if isinstance(stop, str) else stop)
    return cfg


def _to_chat_history(messages: list) -> ov_genai.ChatHistory:
    """Map OpenAI messages to ov ChatHistory; flatten content-part lists to text."""
    history = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):  # OpenAI content-parts form
            content = "".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
        history.append({"role": m["role"], "content": content})
    return ov_genai.ChatHistory(history)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def _run_streaming(pipe, model_id: str, inputs, gen_cfg) -> QueueStreamer:
    """Start a generation thread that feeds a QueueStreamer; returns the streamer."""
    streamer = QueueStreamer(pipe.get_tokenizer())

    def _generate() -> None:
        with _gen_lock:
            t0 = time.perf_counter()
            try:
                pipe.generate(inputs, generation_config=gen_cfg,
                              streamer=streamer.text_streamer)
            except Exception:  # noqa: BLE001 — surface in log; SSE just ends
                log.exception("generation failed (%s)", model_id)
                return
            finally:
                streamer.finish()
            dt = time.perf_counter() - t0
            ttft = (streamer.first_token_time - t0) if streamer.first_token_time else 0
            decode = streamer.token_count - 1
            tps = decode / (dt - ttft) if dt > ttft and decode > 0 else 0
            log.info("[%s] stream done: %d tokens, TTFT %.2fs, %.1f tok/s on %s",
                     model_id, streamer.token_count, ttft, tps, DEVICE)

    threading.Thread(target=_generate, daemon=True).start()
    return streamer


@app.get("/v1/models")
def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": 0, "owned_by": "local"}
            for mid in _pipes
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages")
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    model_id, pipe = _resolve_pipe(body)
    history = _to_chat_history(messages)
    gen_cfg = _build_generation_config(pipe, body)
    comp_id = _new_id("chatcmpl")
    created = int(time.time())

    if not body.get("stream", False):
        with _gen_lock:
            t0 = time.perf_counter()
            result = pipe.generate(history, generation_config=gen_cfg)
            dt = time.perf_counter() - t0
        text = result.texts[0] if hasattr(result, "texts") else str(result)
        log.info("[%s] non-stream done in %.2fs", model_id, dt)
        return JSONResponse({
            "id": comp_id,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    streamer = _run_streaming(pipe, model_id, history, gen_cfg)

    def _sse():
        def chunk(delta: dict, finish: str | None = None) -> str:
            return "data: " + json.dumps({
                "id": comp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }) + "\n\n"

        yield chunk({"role": "assistant", "content": ""})
        while True:
            piece = streamer.q.get()
            if piece is QueueStreamer._DONE:
                break
            yield chunk({"content": piece})
        yield chunk({}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")


@app.post("/v1/completions")
async def completions(request: Request):
    """Legacy completions: raw prompt continuation (Continue.dev autocomplete/FIM)."""
    body = await request.json()
    prompt = body.get("prompt")
    if prompt is None:
        raise HTTPException(status_code=400, detail="prompt is required")
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""

    model_id, pipe = _resolve_pipe(body)
    gen_cfg = _build_generation_config(pipe, body, default_max=256)
    # Raw continuation: do NOT wrap the prompt in the chat template (defaults to
    # True in openvino_genai and would break FIM autocomplete prompts).
    gen_cfg.apply_chat_template = False
    comp_id = _new_id("cmpl")
    created = int(time.time())

    if not body.get("stream", False):
        with _gen_lock:
            result = pipe.generate(prompt, generation_config=gen_cfg)
        text = result.texts[0] if hasattr(result, "texts") else str(result)
        return JSONResponse({
            "id": comp_id,
            "object": "text_completion",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    streamer = _run_streaming(pipe, model_id, prompt, gen_cfg)

    def _sse():
        def chunk(text: str, finish: str | None = None) -> str:
            return "data: " + json.dumps({
                "id": comp_id,
                "object": "text_completion",
                "created": created,
                "model": model_id,
                "choices": [{"index": 0, "text": text, "finish_reason": finish}],
            }) + "\n\n"

        while True:
            piece = streamer.q.get()
            if piece is QueueStreamer._DONE:
                break
            yield chunk(piece)
        yield chunk("", finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream")


if __name__ == "__main__":
    _load_pipelines()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
