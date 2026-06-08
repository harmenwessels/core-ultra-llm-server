"""Quality characterization of the int4 Gemma-4-12B at f32, run *directly*
against optimum's OVModelForVisualCausalLM — the serving GenAI can't load
gemma4_unified ("Unsupported VLM model type"). Reuses the exact casting tasks
and pass/fail probes so the score is comparable to the leaderboard
(Qwen3-14B 10/12, OmniCoder-9B 9/12, both greedy/card).

Greedy decoding (deterministic, comparable to the greedy leaderboard rows).
Slow (~1.4 tok/s at the f32 the model requires) — minutes per cell.

Run (convert venv has the PR optimum-intel + transformers 5.10):
  .venv-convert/Scripts/python.exe scripts/bench_12b_local.py [CPU|GPU]
"""
import json
import pathlib
import sys
import time

from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoProcessor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench_castings import TASKS, probe  # noqa: E402 — reuse exact tasks+probe

PATH = r"C:\git\GitHub\openvino-windows-openai-api\models\HarmenWessels\gemma-4-12B-it-qat-int4-ov"
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "GPU"
F16 = "--f16" in sys.argv  # GPU f16 path (needs the baked ACTIVATIONS_SCALE_FACTOR fix)
MAX_NEW = 1024  # bound wall-time; enough for a function/class + brief preamble
OUT = pathlib.Path(__file__).resolve().parent.parent / "bench_results" / (
    "gemma12b_gpu_f16_castings.jsonl" if F16 else "gemma12b_f32_castings.jsonl")
# f16 on GPU works once the IR's ACTIVATIONS_SCALE_FACTOR is raised 8->64 (the 8.0
# default overflows the 12B's larger activations); else CPU f32 is the only coherent path.
CFG = {} if F16 else {"INFERENCE_PRECISION_HINT": "f32", "KV_CACHE_PRECISION": "f32",
                      "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0}

print(f"loading gemma-4-12B int4 on {DEVICE} ({'f16' if F16 else 'f32'})...", flush=True)
model = OVModelForVisualCausalLM.from_pretrained(PATH, device=DEVICE, ov_config=CFG)
try:
    proc = AutoProcessor.from_pretrained(PATH)
except Exception:
    from transformers import AutoTokenizer
    proc = AutoTokenizer.from_pretrained(PATH)


def generate(prompt: str) -> str:
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True,
                                      tokenize=True, return_dict=True,
                                      return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    return proc.decode(out[0][inputs["input_ids"].shape[1]:],
                       skip_special_tokens=True)


if __name__ == "__main__":
    passes = cells = 0
    for tname, task in TASKS.items():
        for pi, ask in enumerate(task["asks"]):
            t0 = time.perf_counter()
            content = generate(ask)
            dt = round(time.perf_counter() - t0, 1)
            verdict = probe(task, content)
            cells += 1
            passes += verdict == "PASS"
            row = {"model": "gemma-4-12B-int4-f32", "task": tname,
                   "phrasing": pi, "probe": verdict, "seconds": dt}
            print(json.dumps(row), flush=True)
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    print(f"TOTAL gemma-4-12B int4 f32: {passes}/{cells}", flush=True)
