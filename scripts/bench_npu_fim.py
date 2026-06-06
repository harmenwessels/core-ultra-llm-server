"""NPU autocomplete pilot: FIM latency on NPU vs GPU, solo and under GPU load.

Phases:
  1. load Coder-1.5B and Coder-0.5B on NPU (existing g128 artifacts; NPU
     prefers cw-sym — a load failure or slow result here is itself data)
  2. FIM completion latency x3 per model per device (NPU vs GPU baseline)
  3. concurrency: granite-8b generating 256 tokens on GPU while the NPU
     serves FIM completions — the single-gen-lock breaker test

Run: .venv/Scripts/python.exe scripts/bench_npu_fim.py
"""

import json
import pathlib
import threading
import time

import openvino_genai as ov_genai

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "bench_results"
M = ROOT / "models"

FIM = ('def merge_sorted(a, b):\n    """Merge two sorted lists into one '
       'sorted list."""\n')

CODERS = {
    "Coder-1.5B": M / "OpenVINO" / "Qwen2.5-Coder-1.5B-Instruct-int4-ov",
    "Coder-0.5B": M / "OpenVINO" / "Qwen2.5-Coder-0.5B-Instruct-int4-ov",
}


def fim_times(pipe, n=3, max_new=48):
    cfg = pipe.get_generation_config()
    cfg.max_new_tokens = max_new
    cfg.do_sample = False
    cfg.apply_chat_template = False
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        pipe.generate(FIM, generation_config=cfg)
        out.append(round(time.perf_counter() - t0, 2))
    return out


results: dict = {}

# --- phase 1+2: per-device FIM latency --------------------------------------
for name, path in CODERS.items():
    results[name] = {}
    for device in ("NPU", "GPU"):
        kwargs = {}
        if device == "NPU":
            kwargs = {"MAX_PROMPT_LEN": 2048}
        else:
            kwargs = {"CACHE_DIR": str(ROOT / ".ovcache")}
        try:
            t0 = time.perf_counter()
            pipe = ov_genai.LLMPipeline(str(path), device, **kwargs)
            load_s = round(time.perf_counter() - t0, 0)
            lat = fim_times(pipe)
            results[name][device] = {"load_s": load_s, "fim_x3": lat}
            print(f"{name} on {device}: load {load_s:.0f}s, FIM {lat}",
                  flush=True)
            del pipe
        except Exception as e:  # noqa: BLE001
            msg = str(e).strip().splitlines()[-1][:110]
            results[name][device] = {"error": msg}
            print(f"{name} on {device}: FAILED ({msg})", flush=True)

# --- phase 3: concurrency (best NPU coder + granite on GPU) -----------------
npu_pick = next((n for n in CODERS if "fim_x3" in results[n].get("NPU", {})),
                None)
if npu_pick:
    print(f"\nconcurrency: granite-8b on GPU + {npu_pick} on NPU", flush=True)
    granite = ov_genai.LLMPipeline(
        str(M / "HarmenWessels" / "granite-4.1-8b-int4-cw-ov"), "GPU",
        CACHE_DIR=str(ROOT / ".ovcache"))
    npu = ov_genai.LLMPipeline(str(CODERS[npu_pick]), "NPU",
                               MAX_PROMPT_LEN=2048)
    gcfg = granite.get_generation_config()
    gcfg.max_new_tokens = 256
    gcfg.do_sample = False
    g_done = {}

    def gpu_job():
        t0 = time.perf_counter()
        granite.generate("Explain the actor model in distributed systems.",
                         generation_config=gcfg)
        g_done["seconds"] = round(time.perf_counter() - t0, 1)

    th = threading.Thread(target=gpu_job)
    th.start()
    time.sleep(2)  # let the GPU generation get going
    lat_busy = fim_times(npu, n=3)
    th.join()
    results["concurrency"] = {
        "gpu_model": "granite-8b 256 tok", "gpu_seconds": g_done["seconds"],
        "npu_model": npu_pick, "npu_fim_x3_during_gpu_gen": lat_busy,
        "npu_fim_x3_solo": results[npu_pick]["NPU"]["fim_x3"],
    }
    print(f"GPU job: {g_done['seconds']}s | NPU FIM during GPU gen: "
          f"{lat_busy} (solo was {results[npu_pick]['NPU']['fim_x3']})",
          flush=True)
else:
    print("\nno coder loaded on NPU — concurrency phase skipped", flush=True)

OUT_DIR.mkdir(exist_ok=True)
out = OUT_DIR / f"npu_fim__{time.strftime('%Y%m%d-%H%M%S')}.json"
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"saved: {out}")
