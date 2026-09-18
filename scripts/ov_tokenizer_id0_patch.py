r"""Work around two openvino_tokenizers (2026.3.0) conversion bugs that silently
degrade a model's tokenization. Both are invisible at load time and only show
as depressed benchmark scores; `scripts/audit_tokenizers.py` detects them.

1. Added token id 0 is dropped (2026-09-15, Spark-X2.5 / Seed-Coder).
   `tokenizer_pipeline.BPETokenizationStep.from_hf_json` builds the added-token
   vocab with `... if token["id"]`, falsy for id 0. The special-token splitter
   still isolates the text but the BPE step then byte-encodes it. DeepSeek-style
   tokenizers put BOS at id 0 and emit it as text at every turn -> 11-12 junk
   tokens per turn. Seed-Coder-8B went 22/26 -> 25/26 once fixed.
   Symptom: `encode(bos_token)` returns a dozen ids instead of `[0]`.

2. `\uXXXX` escapes in a Split regex break the `\s+(?!\S)` alternative
   (2026-09-17, K2-Horizon: `(?:\p{L}|\p{M}|\u200C|\u200D)+`). PCRE2 does not
   accept `\u` (it wants `\x{XXXX}`); the op evidently falls back to an engine
   without lookahead, so every Python indent tokenizes as an N-space token plus a
   bare word instead of (N-1) spaces + " word" -- a distribution shift on all
   indented code. The Rust `tokenizers` library accepts both spellings, so the
   rewrite keeps the JSON valid for HF too.

Import and call `apply()` before `convert_tokenizer(...)`, or run as a script to
regenerate a model dir's tokenizer IRs in place with both fixes:

    python ov_tokenizer_id0_patch.py <model_dir> [<model_dir> ...]
"""
import re

from openvino_tokenizers import hf_parser as _hp
from openvino_tokenizers import tokenizer_pipeline as _tp

_orig_from_hf_json = _tp.BPETokenizationStep.from_hf_json.__func__
_orig_parse_split = _hp.parse_split_step
_applied = False


def _from_hf_json_keep_id0(cls, tokenizer_json):
    step = _orig_from_hf_json(cls, tokenizer_json)
    for token in tokenizer_json.get("added_tokens", []):
        if token["id"] == 0 and token["content"] not in step.added_tokens:
            step.added_tokens[token["content"]] = 0
            print(f"[ov-tok-patch] restored dropped added token id 0: {token['content']!r}", flush=True)
    return step


def pcre2_escapes(pattern: str) -> str:
    r"""`\uXXXX` -> `\x{XXXX}`; PCRE2 rejects the former."""
    return re.sub(r"\\u([0-9A-Fa-f]{4})", r"\\x{\1}", pattern)


def _parse_split_pcre2(pretokenizer_dict):
    pat = pretokenizer_dict.get("pattern", {})
    if isinstance(pat, dict) and pat.get("Regex") and "\\u" in pat["Regex"]:
        fixed = pcre2_escapes(pat["Regex"])
        if fixed != pat["Regex"]:
            pretokenizer_dict = {**pretokenizer_dict, "pattern": {**pat, "Regex": fixed}}
            print("[ov-tok-patch] rewrote \\uXXXX escapes in a Split regex for PCRE2", flush=True)
    return _orig_parse_split(pretokenizer_dict)


def apply():
    global _applied
    if not _applied:
        _tp.BPETokenizationStep.from_hf_json = classmethod(_from_hf_json_keep_id0)
        # the dispatch table bound the original at class-definition time
        _hp.TransformersTokenizerPipelineParser.pre_tokenization_map["Split"] = _parse_split_pcre2
        _hp.parse_split_step = _parse_split_pcre2
        _applied = True


def regenerate(model_dir: str) -> None:
    """Rebuild openvino_tokenizer/detokenizer.xml for an IR dir with both fixes."""
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer
    from transformers import AutoTokenizer

    apply()
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    t, d = convert_tokenizer(tok, with_detokenizer=True)
    ov.save_model(t, f"{model_dir}/openvino_tokenizer.xml")
    ov.save_model(d, f"{model_dir}/openvino_detokenizer.xml")
    print(f"[ov-tok-patch] regenerated tokenizer IRs in {model_dir}", flush=True)


if __name__ == "__main__":
    import sys

    for d in sys.argv[1:]:
        regenerate(d)
