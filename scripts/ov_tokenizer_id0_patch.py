"""Work around an openvino_tokenizers bug that drops an added token with id 0.

`tokenizer_pipeline.BPETokenizationStep.from_hf_json` (openvino_tokenizers
2026.3.0.0, present in every env here) builds the BPE added-token vocab with

    {token["content"]: token["id"] for token in tokenizer_json["added_tokens"] if token["id"]}

`if token["id"]` is falsy for id 0, so that token is silently omitted. The
special-token splitter still isolates its text, but with no vocab entry the BPE
step byte-encodes it: DeepSeek-style tokenizers put BOS at id 0
(`<｜start▁of▁sentence｜>` on Spark-X2.5, `<｜begin▁of▁sentence｜>` on DeepSeek), and
their chat templates emit BOS as literal text at the start of EVERY turn -- so the
model reads its own turn openers as 12 tokens of junk and even remarks on it
("the user's message is a bit messy with start/end markers"). Symptom to look for:
`tokenizer.encode(bos_token)` returns a dozen ids instead of `[0]`, while EOS is fine.

Import and call `apply()` before `convert_tokenizer(...)`.
"""
from openvino_tokenizers import tokenizer_pipeline as _tp

_orig_from_hf_json = _tp.BPETokenizationStep.from_hf_json.__func__
_applied = False


def _from_hf_json_keep_id0(cls, tokenizer_json):
    step = _orig_from_hf_json(cls, tokenizer_json)
    for token in tokenizer_json.get("added_tokens", []):
        if token["id"] == 0 and token["content"] not in step.added_tokens:
            step.added_tokens[token["content"]] = 0
            print(f"[ov-tok-patch] restored dropped added token id 0: {token['content']!r}", flush=True)
    return step


def apply():
    global _applied
    if not _applied:
        _tp.BPETokenizationStep.from_hf_json = classmethod(_from_hf_json_keep_id0)
        _applied = True
