"""Hand-rolled AWQ+SE int4-cw quantization of OmniCoder-9B's language model,
calibrated on Python code — bypassing optimum-intel's VLM dataset gate.

Why this exists: optimum-intel's visual-LM quantization path accepts only
`--dataset contextual` (image-instruction pairs); calibrating a *coding*
model on image-chat activations measured 3/12 (worse than the data-free
build's 9/12). This drives nncf.compress_weights directly on the fp16
language-model IR with code-derived calibration inputs, so the recipe matches
the granite-repair recipe exactly (cw INT4_SYM + AWQ + scale-estimation,
group_size=-1) and changes only the calibration domain.

Pipeline:
  1. optimum-cli export the bf16 source at --weight-format fp16 (separate step)
  2. this script: code corpus -> text_embeddings IR -> inputs_embeds; build the
     LM's other inputs (attention_mask, 4-row mrope position_ids, beam_idx);
     nncf.compress_weights(fp16_lm, INT4_SYM cw, awq+SE, dataset)
  3. reassemble: compressed LM + the data-free build's vision/tokenizer IRs +
     configs + the no-think rt_info template

Run (in .venv-convert, transformers 5.2.x for qwen3_5):
  .venv-convert/Scripts/python.exe scripts/convert_omni_awqse_codecalib.py \
      <fp16_export_dir> <donor_data_free_dir> <out_dir>
"""

import pathlib
import random
import shutil
import sys

import numpy as np
import nncf
import openvino as ov
from transformers import AutoTokenizer

FP16_DIR = pathlib.Path(sys.argv[1])    # fp16 export (source of the LM to compress)
DONOR_DIR = pathlib.Path(sys.argv[2])   # data-free build (vision/tokenizer/config donor)
OUT_DIR = pathlib.Path(sys.argv[3])
ROOT = pathlib.Path(__file__).resolve().parent.parent

N_SAMPLES = 48
MAX_TOKENS = 512


def build_code_corpus(n_samples: int, chunk_chars: int = 2400) -> list[str]:
    """Sample real Python from this machine (seed-pinned, reproducible)."""
    pools = [ROOT / "scripts", ROOT, ROOT / ".venv" / "Lib" / "site-packages"]
    files: list[pathlib.Path] = []
    for pool in pools:
        files += list(pool.rglob("*.py"))[:4000]
    random.seed(42)
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
    print(f"corpus: {len(corpus)} code chunks", flush=True)
    return corpus


def build_calibration_items(corpus: list[str]) -> list[dict]:
    """Tokenize -> text_embeddings IR -> full LM input dicts."""
    tok = AutoTokenizer.from_pretrained(DONOR_DIR)
    core = ov.Core()
    te = core.compile_model(
        core.read_model(DONOR_DIR / "openvino_text_embeddings_model.xml"), "CPU")
    items: list[dict] = []
    for text in corpus:
        ids = tok(text, return_tensors="np").input_ids[:, :MAX_TOKENS].astype(np.int64)
        seq = ids.shape[1]
        if seq < 8:
            continue
        embeds = te(ids)[0]                                   # [1, seq, 4096] f32
        attn = np.ones((1, seq), dtype=np.int64)
        # Qwen3.5 mrope: 4 position rows; for text-only tokens each row is the
        # plain sequence position (h/w/t sections collapse to the text index).
        pos = np.tile(np.arange(seq, dtype=np.int64), (4, 1, 1))   # [4, 1, seq]
        items.append({
            "inputs_embeds": embeds,
            "attention_mask": attn,
            "position_ids": pos,
            "beam_idx": np.zeros((1,), dtype=np.int32),
        })
    print(f"calibration items: {len(items)} (<= {MAX_TOKENS} tok each)", flush=True)
    return items


if __name__ == "__main__":
    items = build_calibration_items(build_code_corpus(N_SAMPLES))
    ds = nncf.Dataset(items)            # items already in model-input form

    core = ov.Core()
    lm = core.read_model(FP16_DIR / "openvino_language_model.xml")
    print("compressing LM (int4_sym cw, AWQ+SE, code-calibrated)…", flush=True)
    compressed = nncf.compress_weights(
        lm,
        mode=nncf.CompressWeightsMode.INT4_SYM,
        group_size=-1,
        ratio=1.0,
        dataset=ds,
        awq=True,
        scale_estimation=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ov.save_model(compressed, OUT_DIR / "openvino_language_model.xml")
    print("LM saved; copying donor components…", flush=True)

    # Everything except the LM comes from the data-free build (vision IRs,
    # text_embeddings, tokenizer IRs, configs, chat_template). The donor's
    # tokenizer IR already carries the no-think rt_info patch.
    for f in DONOR_DIR.iterdir():
        if f.is_file() and not f.name.startswith("openvino_language_model"):
            shutil.copy2(f, OUT_DIR / f.name)
    print(f"saved: {OUT_DIR}", flush=True)
