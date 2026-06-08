r"""Phase 2 / Gate 2: minimal smoke test + decode-throughput benchmark for an
openvino_genai pipeline on the Intel Arc iGPU.

The script auto-picks LLMPipeline vs VLMPipeline based on the model directory
contents: a `openvino_vision_embeddings_model.xml` means the IR is shaped as a
VLM (e.g. Gemma 4) and VLMPipeline is required even for text-only generation;
otherwise LLMPipeline (the fast, plain-LLM path) is used.

This is the real go/no-go for the project. We measure:
  - TTFT  : time to first generated token  (seconds)
  - decode: tokens / second during decode  (excludes TTFT)
  - peak  : peak working-set bytes for this process (Windows-only nicety)

Runs warm-up first (first run pays compile + first-token cost), then 3 measured
runs to show variance. Compile cache lives in --cache-dir.

Usage:
    .\.venv\Scripts\python.exe scripts\bench.py
    .\.venv\Scripts\python.exe scripts\bench.py --device CPU       # debug fallback
    .\.venv\Scripts\python.exe scripts\bench.py --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import pathlib
import time

import openvino_genai as ov_genai


DEFAULT_MODEL_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "models"
    / "gemma-4-E4B-it-int4-ov"
)
DEFAULT_CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / ".ovcache"

# ~100-token coding prompt; expanded as a Gemma chat turn by the pipeline.
PROMPT = (
    "Write a Python function `merge_intervals(intervals)` that takes a list of "
    "[start, end] pairs and returns the list of merged, non-overlapping intervals "
    "in sorted order. Include a short docstring and handle the empty input case. "
    "Then explain the time and space complexity in two short sentences."
)


class TokenCountingStreamer(ov_genai.StreamerBase):
    """Counts streamed pieces and records when the first one arrives."""

    def __init__(self) -> None:
        super().__init__()
        self.first_token_time: float | None = None
        self.token_count = 0
        self._start: float = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()
        self.first_token_time = None
        self.token_count = 0

    def write(self, token):  # noqa: ANN001 — openvino_genai callback signature
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self.token_count += 1
        return ov_genai.StreamingStatus.RUNNING

    def end(self) -> None:
        pass


def peak_working_set_mb() -> float | None:
    """Windows: peak working set in MiB. Returns None on non-Windows / failure."""
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:  # noqa: BLE001
        return None

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    psapi = ctypes.WinDLL("psapi.dll")
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return None
    return counters.PeakWorkingSetSize / (1024 * 1024)


def run_once(pipe, gen_cfg: ov_genai.GenerationConfig, label: str) -> dict:
    streamer = TokenCountingStreamer()
    streamer.start()
    t0 = time.perf_counter()
    pipe.generate(PROMPT, generation_config=gen_cfg, streamer=streamer)
    t1 = time.perf_counter()

    total = t1 - t0
    ttft = (streamer.first_token_time - t0) if streamer.first_token_time else float("nan")
    decode_time = total - ttft
    decode_tps = (streamer.token_count - 1) / decode_time if decode_time > 0 else 0.0
    print(
        f"{label}: tokens={streamer.token_count:>4d}  total={total:6.2f}s  "
        f"TTFT={ttft:6.2f}s  decode={decode_tps:6.1f} tok/s"
    )
    return {
        "tokens": streamer.token_count,
        "total": total,
        "ttft": ttft,
        "decode_tps": decode_tps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=pathlib.Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cache-dir", type=pathlib.Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", default="GPU", help="GPU (default) or CPU")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--tuned",
        action="store_true",
        help="Apply latency-oriented hints: PERFORMANCE_HINT=LATENCY, "
             "INFERENCE_PRECISION_HINT=f16, KV_CACHE_PRECISION=u8. "
             "Recompiles on first run so allow extra startup time.",
    )
    args = parser.parse_args()

    if not args.model_dir.exists():
        print(f"ERROR: model dir not found: {args.model_dir}")
        return 2

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model dir : {args.model_dir}")
    print(f"Cache dir : {args.cache_dir}")
    print(f"Device    : {args.device}")
    print(f"max_new   : {args.max_new_tokens}")

    pipe_kwargs: dict = {"CACHE_DIR": str(args.cache_dir)}
    if args.tuned:
        pipe_kwargs.update({
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "f16",
            "KV_CACHE_PRECISION": "u8",
        })
        print(f"Tuning hints : {pipe_kwargs}")

    is_vlm = (args.model_dir / "openvino_vision_embeddings_model.xml").exists()
    pipe_cls = ov_genai.VLMPipeline if is_vlm else ov_genai.LLMPipeline
    print(f"Pipeline    : {pipe_cls.__name__} "
          f"({'VLM-shaped IR detected' if is_vlm else 'plain LLM IR'})")

    t_load_0 = time.perf_counter()
    pipe = pipe_cls(str(args.model_dir), args.device, **pipe_kwargs)
    print(f"Pipeline ready in {time.perf_counter() - t_load_0:5.1f}s "
          f"(includes any compile-cache load/build).")

    gen_cfg = pipe.get_generation_config()
    gen_cfg.max_new_tokens = args.max_new_tokens
    # Deterministic for benchmarking; we care about throughput, not quality variance.
    gen_cfg.do_sample = False

    print("\n--- warm-up (ignored) ---")
    run_once(pipe, gen_cfg, "warm")

    print("\n--- measured runs ---")
    results = [run_once(pipe, gen_cfg, f"run{i+1}") for i in range(args.runs)]

    decodes = [r["decode_tps"] for r in results]
    ttfts = [r["ttft"] for r in results]
    print(
        f"\nSummary: decode median={sorted(decodes)[len(decodes)//2]:.1f} tok/s  "
        f"min={min(decodes):.1f}  max={max(decodes):.1f}  "
        f"TTFT median={sorted(ttfts)[len(ttfts)//2]:.2f}s"
    )

    peak = peak_working_set_mb()
    if peak is not None:
        print(f"Peak working set: {peak:.0f} MiB")

    median_decode = sorted(decodes)[len(decodes) // 2]
    if args.device == "GPU" and median_decode < 15:
        print("\nGATE 2 FAIL: median decode < 15 tok/s on GPU. Investigate before "
              "building the server (silent CPU fallback? wrong device? thermal?).")
        return 1
    print("\nGATE 2 PASS." if args.device == "GPU" else
          "\n(Device != GPU; threshold not enforced.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
