"""Fleet tokenizer audit: does each IR's OpenVINO tokenizer agree with its HF one?

    python audit_tokenizers.py [models_root]        (run in .venv-convert: needs transformers)

For every model dir with tokenizer.json + openvino_tokenizer.xml, compares:
  1. every added/special token, encoded alone           -> must be ONE id, the same id
  2. a rendered 2-turn chat (template applied by HF)     -> id sequences must match
  3. a plain code snippet                                 -> id sequences must match

Exists because openvino_tokenizers 2026.3 silently dropped added token id 0
(DeepSeek-style BOS) — Seed-Coder-8B ran every benchmark with 11 junk tokens
prepended to each turn and scored 22/26; fixed it scores 25/26. A mismatch here
is a hidden score depression, not a cosmetic issue. Exit code = number of
models with any mismatch.
"""
import json
import pathlib
import sys

import numpy as np
import openvino as ov
import openvino_tokenizers  # noqa: F401  registers the tokenizer ops with the Core
from tokenizers import Tokenizer  # Rust reference: honours tokenizer.json exactly
from transformers import AutoTokenizer

ROOT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[1] / "models"
CODE = "def merge(a, b):\n    return sorted(a + b)  # O(n log n)\n"
CHAT = [{"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
        {"role": "user", "content": "And 3+3?"}]

core = ov.Core()


def audit(d: pathlib.Path) -> list[str]:
    problems: list[str] = []
    # Ground truth is the Rust `tokenizers` object built from tokenizer.json. NOT
    # AutoTokenizer: for granite it resolves to the legacy GPT2Tokenizer class, whose
    # built-in regex splits "(a" that the JSON's pre_tokenizer merges — the audit then
    # blames the IR for the loader's mistake (2026-09-17).
    ref = Tokenizer.from_file(str(d / "tokenizer.json"))
    hf = AutoTokenizer.from_pretrained(str(d), trust_remote_code=True)  # templates only
    ovt = core.compile_model(str(d / "openvino_tokenizer.xml"), "CPU")

    def ref_ids(text: str) -> list[int]:
        return ref.encode(text, add_special_tokens=False).ids

    def ov_raw(text: str) -> list[int]:
        out = ovt([text])
        ids = out["input_ids"] if "input_ids" in [o.get_any_name() for o in ovt.outputs] else out[0]
        return [int(x) for x in np.asarray(ids)[0]]

    # The OV tokenizer may prepend BOS (LFM, gemma) on its own; HF is compared
    # with add_special_tokens=False, so strip whatever OV emits for "" first.
    prefix = ov_raw("")

    def ov_ids(text: str) -> list[int]:
        ids = ov_raw(text)
        return ids[len(prefix):] if ids[:len(prefix)] == prefix else ids

    tj = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
    for tok in tj.get("added_tokens", []):
        s, want = tok["content"], tok["id"]
        got = ov_ids(s)
        if got != [want]:
            problems.append(f"special {s!r} id {want}: OV -> {got[:8]}{'…' if len(got) > 8 else ''}")

    hf_code = ref_ids(CODE)
    if ov_ids(CODE) != hf_code:
        problems.append(f"plain code: OV {ov_ids(CODE)[:10]} vs HF {hf_code[:10]}")

    try:
        rendered = hf.apply_chat_template(CHAT, tokenize=False, add_generation_prompt=True)
        hf_chat = ref_ids(rendered)
        ov_chat = ov_ids(rendered)
        if ov_chat != hf_chat:
            i = next((k for k, (a, b) in enumerate(zip(ov_chat, hf_chat)) if a != b), min(len(ov_chat), len(hf_chat)))
            problems.append(f"chat render: diverges at token {i} (OV {len(ov_chat)} ids vs HF {len(hf_chat)}): "
                            f"OV {ov_chat[i:i+6]} vs HF {hf_chat[i:i+6]}")
    except Exception as e:  # no HF template (vendor template served server-side) or jinja rejects
        if not (d / "chat_template.vendor.jinja").exists():
            problems.append(f"chat render skipped: {type(e).__name__}: {str(e)[:80]}")
    return problems


bad = 0
dirs = sorted(p.parent for p in ROOT.glob("*/*/openvino_tokenizer.xml") if (p.parent / "tokenizer.json").exists())
print(f"auditing {len(dirs)} model dirs under {ROOT}", flush=True)
for d in dirs:
    name = d.relative_to(ROOT).as_posix()
    try:
        probs = audit(d)
    except Exception as e:  # noqa: BLE001
        probs = [f"AUDIT ERROR {type(e).__name__}: {str(e)[:100]}"]
    flag = "OK  " if not probs else "BAD "
    bad += bool(probs)
    print(f"[{flag}] {name}", flush=True)
    for p in probs[:6]:
        print(f"        - {p}", flush=True)
    if len(probs) > 6:
        print(f"        - … {len(probs) - 6} more", flush=True)
print(f"\n{bad} of {len(dirs)} models have tokenizer mismatches", flush=True)
sys.exit(bad)
