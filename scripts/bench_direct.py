"""Direct-path casting bench — runs ANY model the same way Gemma-4-12B was run
(optimum OVModel, raw prompt -> chat template -> greedy generate), so the
comparison is on an equal harness (no server / no virtual-agent / no GenAI).

Same tasks + probes as the leaderboard. Greedy, MAX_NEW=3072, GPU f16, nothink
where the tokenizer supports it (to match the leaderboard's nothink condition).

Run (convert venv): .venv-convert/Scripts/python.exe scripts/bench_direct.py <models/owner/name>
"""
import json
import os
import pathlib
import sys
import time

from transformers import AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bench_castings import TASKS, probe  # noqa: E402 — identical tasks+probe

ROOT = pathlib.Path(__file__).resolve().parent.parent
REL = sys.argv[1]
PATH = str(ROOT / REL)
MAX_NEW = int(os.environ.get("MAX_NEW", "3072"))
OUT = ROOT / "bench_results" / "direct_castings.jsonl"
IS_VLM = (pathlib.Path(PATH) / "openvino_vision_embeddings_model.xml").exists()
LABEL = REL.split("/")[-1] + "-direct-greedy-3072"

if IS_VLM:
    from optimum.intel import OVModelForVisualCausalLM as Model
else:
    from optimum.intel import OVModelForCausalLM as Model

print(f"loading {REL} ({'VLM' if IS_VLM else 'text'}) on GPU f16...", flush=True)
model = Model.from_pretrained(PATH, device="GPU")
tok = AutoTokenizer.from_pretrained(PATH)


# --temp=/--top-p=/--top-k= : per-model card operating point (else greedy)
TEMP = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--temp=")), None)
TOP_P = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--top-p=")), None)
TOP_K = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--top-k=")), None)
if TEMP is not None:
    LABEL = REL.split("/")[-1] + f"-direct-card{TEMP}-3072"
    _GEN = dict(do_sample=True, temperature=TEMP, top_p=TOP_P, top_k=TOP_K)
else:
    _GEN = dict(do_sample=False)


SYS = os.environ.get("SYS")  # optional system prompt (equalize "clean output")


def generate(prompt: str) -> str:
    content = [{"type": "text", "text": prompt}] if IS_VLM else prompt
    msgs = ([{"role": "system", "content": SYS}] if SYS else []) + \
        [{"role": "user", "content": content}]
    kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True,
              return_tensors="pt")
    try:
        inputs = tok.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError:
        inputs = tok.apply_chat_template(msgs, **kw)
    out = model.generate(**inputs, max_new_tokens=MAX_NEW, **_GEN)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


if __name__ == "__main__":
    passes = cells = 0
    for tname, task in TASKS.items():
        for pi, ask in enumerate(task["asks"]):
            t0 = time.perf_counter()
            try:
                content = generate(ask)
                verdict = probe(task, content)
            except Exception as e:  # noqa: BLE001
                verdict = f"FAIL (EXC: {type(e).__name__})"
            dt = round(time.perf_counter() - t0, 1)
            cells += 1
            passes += verdict == "PASS"
            row = {"model": LABEL, "task": tname, "phrasing": pi,
                   "probe": verdict, "seconds": dt}
            print(json.dumps(row), flush=True)
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    print(f"TOTAL {LABEL}: {passes}/{cells}", flush=True)
