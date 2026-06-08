"""Coherence test for the int4 Gemma-4-12B build at the PR-recommended f32
inference precision (the model is numerically sensitive: bf16/f16 -> garbage).
Isolates whether the QAT-grid-matched int4 *weights* are usable."""
import sys
import time
from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoProcessor

PATH = r"C:\git\GitHub\openvino-windows-openai-api\models\HarmenWessels\gemma-4-12B-it-qat-int4-ov"
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "CPU"
PREC = "f16" if "--f16" in sys.argv else "f32"  # f16 = the server-default path
CFG = ({} if PREC == "f16" else
       {"INFERENCE_PRECISION_HINT": "f32", "KV_CACHE_PRECISION": "f32",
        "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0})

print(f"loading on {DEVICE} with {PREC} config...", flush=True)
t0 = time.perf_counter()
model = OVModelForVisualCausalLM.from_pretrained(PATH, ov_config=CFG)
proc = AutoProcessor.from_pretrained(PATH)
print(f"loaded in {time.perf_counter()-t0:.0f}s", flush=True)

for prompt in ["Write a Python function that reverses a string.",
               "What is the capital of France? Answer in one word."]:
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True,
                                      tokenize=True, return_dict=True,
                                      return_tensors="pt")
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    txt = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n=== PROMPT: {prompt}\n{txt}\n[{time.perf_counter()-t0:.0f}s]", flush=True)
