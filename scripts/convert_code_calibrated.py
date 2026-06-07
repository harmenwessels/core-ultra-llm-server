"""Code-calibrated AWQ+SE int4 cw conversion (granite experiment).

Identical recipe to the published cw artifacts except the calibration
dataset: real Python code sampled from this machine instead of wikitext2
prose. Hypothesis: domain-matched activation statistics protect the
channels that matter for code editing (the executor workload).

Run (in .venv-convert, transformers 4.57.x for granite):
  .venv-convert/Scripts/python.exe scripts/convert_code_calibrated.py \
      ibm-granite/granite-4.1-3b models/HarmenWessels/granite-4.1-3b-int4-cw-code-ov
"""

import pathlib
import random
import sys

from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig
from transformers import AutoTokenizer

MODEL_ID = sys.argv[1]
OUT_DIR = sys.argv[2]
ROOT = pathlib.Path(__file__).resolve().parent.parent


def build_code_corpus(n_samples: int = 128, chunk_chars: int = 2400) -> list[str]:
    """Sample real Python from this machine: our scripts + site-packages."""
    pools = [ROOT / "scripts", ROOT,
             ROOT / ".venv" / "Lib" / "site-packages"]
    files: list[pathlib.Path] = []
    for pool in pools:
        files += list(pool.rglob("*.py"))[:4000]
    random.seed(42)  # reproducible calibration set
    random.shuffle(files)
    corpus: list[str] = []
    for f in files:
        if len(corpus) >= n_samples:
            break
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if len(text) < 400:
            continue
        start = random.randint(0, max(0, len(text) - chunk_chars))
        corpus.append(text[start:start + chunk_chars])
    print(f"calibration corpus: {len(corpus)} code chunks "
          f"(~{sum(map(len, corpus)) // 1000}k chars)", flush=True)
    return corpus


if __name__ == "__main__":
    corpus = build_code_corpus()
    qcfg = OVWeightQuantizationConfig(
        bits=4, sym=True, group_size=-1, ratio=1.0,
        dataset=corpus, awq=True, scale_estimation=True)
    print(f"converting {MODEL_ID} (int4 cw-sym, AWQ+SE, code-calibrated)…",
          flush=True)
    model = OVModelForCausalLM.from_pretrained(
        MODEL_ID, export=True, quantization_config=qcfg, compile=False)
    model.save_pretrained(OUT_DIR)
    hf_tok = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_tok.save_pretrained(OUT_DIR)
    # GenAI needs the OpenVINO tokenizer IRs (the CLI exports these
    # automatically; the Python API does not)
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer
    ov_tok, ov_detok = convert_tokenizer(hf_tok, with_detokenizer=True)
    ov.save_model(ov_tok, f"{OUT_DIR}/openvino_tokenizer.xml")
    ov.save_model(ov_detok, f"{OUT_DIR}/openvino_detokenizer.xml")
    print(f"saved: {OUT_DIR}", flush=True)
