"""Sweep the BAKED rt_info ACTIVATIONS_SCALE_FACTOR in the 12B LM IR and test
f16 coherence via optimum. The baked value (8.0) likely overrides ov_config —
so we edit the IR metadata directly. Hypothesis: the 12B's larger activations
need a bigger divisor than E4B's working 8.0. Restores the original at the end."""
import re
import time

from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoTokenizer

XML = r"C:\git\GitHub\openvino-windows-openai-api\models\HarmenWessels\gemma-4-12B-it-qat-int4-ov\openvino_language_model.xml"
PATH = r"C:\git\GitHub\openvino-windows-openai-api\models\HarmenWessels\gemma-4-12B-it-qat-int4-ov"
PROMPT = "Write a Python function that reverses a string."

orig = open(XML, encoding="utf-8").read()
m = re.search(r'<ACTIVATIONS_SCALE_FACTOR value="([\d.]+)" />', orig)
print("original scale factor:", m.group(1), flush=True)
tok = AutoTokenizer.from_pretrained(PATH)


def run(scale):
    patched = re.sub(r'<ACTIVATIONS_SCALE_FACTOR value="[\d.]+" />',
                     f'<ACTIVATIONS_SCALE_FACTOR value="{scale}" />', orig)
    open(XML, "w", encoding="utf-8").write(patched)
    model = OVModelForVisualCausalLM.from_pretrained(PATH, device="GPU", ov_config={})
    msgs = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                     return_dict=True, return_tensors="pt")
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    txt = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    dt = time.perf_counter() - t0
    snippet = txt[:90].replace("\n", " ")
    print(f"  scale={scale:<6} [{dt:.0f}s]  {snippet!r}", flush=True)


try:
    for s in (16.0, 32.0, 64.0, 128.0, 512.0):
        run(s)
finally:
    open(XML, "w", encoding="utf-8").write(orig)  # restore 8.0
    print("restored original scale factor", flush=True)
