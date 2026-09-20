# Upstream issue drafts

Bugs found in Intel's stack while building this fleet, written up ready to file.
Each has a minimal reproduction that needs no model weights beyond a public
tokenizer. Status column tracks filing.

| # | Repo | Title | Status |
|---|------|-------|--------|
| 1 | openvinotoolkit/openvino_tokenizers | Added token with id 0 is dropped from the BPE vocab | fixed upstream in 2026.4.0.0 — not filed |
| 2 | openvinotoolkit/openvino_tokenizers | `\uXXXX` in a Split regex disables the `\s+(?!\S)` lookahead | draft — still present in 2026.4.0.0 |
| 3 | openvinotoolkit/openvino | GPU: fused int4 MatMul pair sharing an input produces non-deterministic garbage (hybrid-attention gate) | draft |

---

## 1. openvino_tokenizers — added token with id 0 is silently dropped

**Version:** openvino_tokenizers 2026.3.0.0 (683-6d28e696dfa), OpenVINO 2026.3.1

**Summary.** `tokenizer_pipeline.BPETokenizationStep.from_hf_json` builds the
added-token vocabulary with

```python
added_tokens = {token["content"]: token["id"] for token in tokenizer_json["added_tokens"] if token["id"]}
```

`if token["id"]` is falsy for id 0, so that token never reaches the BPE step. The
`SpecialTokensSplit` step still isolates its text, but with no vocab entry the BPE
step byte-encodes it. Tokenizers that put BOS at id 0 (DeepSeek V2/V3/R1 and
derivatives, ByteDance Seed-Coder, XHToken Spark-X2.5) then encode their own BOS
as 11–12 byte tokens. Their chat templates emit BOS as literal text at the start of
every turn, so every prompt is corrupted; nothing errors.

**Reproduction** (any DeepSeek-style tokenizer, e.g. `ByteDance-Seed/Seed-Coder-8B-Instruct`):

```python
from transformers import AutoTokenizer
import openvino as ov, openvino_tokenizers as ot, numpy as np
hf = AutoTokenizer.from_pretrained("ByteDance-Seed/Seed-Coder-8B-Instruct")
bos = hf.bos_token                      # '<[begin▁of▁sentence]>', id 0
t = ov.Core().compile_model(ot.convert_tokenizer(hf), "CPU")
print(hf.encode(bos, add_special_tokens=False))      # [0]
print(np.asarray(t([bos])[0])[0].tolist())            # [155, 186, 13082, 6645, ...]  (11 ids)
```

**Fix.** `if token["id"] is not None` (one line). We work around it by
re-inserting the id-0 entry after `from_hf_json`.

**Impact seen.** An 8B coding model's agent-task pass rate went 5/7 → 7/7 and its
analysis task 3/4 → 4/4 with no change but a regenerated tokenizer IR.

---

## 2. openvino_tokenizers — `\uXXXX` in a Split regex disables the `\s+(?!\S)` alternative

**Version:** as above.

**Summary.** A pre-tokenizer `Split` regex containing `\u200C`/`\u200D` (IFM
K2-Horizon: `[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+|...|\s+(?!\S)|\s+`)
converts without any warning, but the resulting `RegexSplit` op no longer honours
the `\s+(?!\S)` alternative: runs of whitespace before a word are consumed whole
instead of leaving one space attached to the word. Every Python indent therefore
tokenizes as an N-space token plus a bare identifier, where the reference gives
N−1 spaces plus `Ġword`. PCRE2 does not accept `\u` (it wants `\x{200C}`); the
behaviour is consistent with the pattern failing to compile under PCRE2 and the
op falling back to an engine without lookahead support. Rewriting `\uXXXX` →
`\x{XXXX}` in the pattern before conversion gives an exact match with the Rust
`tokenizers` reference on every case we tried — and the Rust library accepts both
spellings, so the JSON stays valid for HF.

**Reproduction:**

```python
from tokenizers import Tokenizer
from transformers import AutoTokenizer
import openvino as ov, openvino_tokenizers as ot, numpy as np
mid = "IFM/K2-Horizon-3.7B"
ref = Tokenizer.from_file(hf_hub_download(mid, "tokenizer.json"))
t = ov.Core().compile_model(ot.convert_tokenizer(AutoTokenizer.from_pretrained(mid)), "CPU")
s = "if a:\n        return b\n"
print([ref.id_to_token(i) for i in ref.encode(s, add_special_tokens=False).ids])
#  ['if', 'Ġa', ':Ċ', 'ĠĠĠĠĠĠĠ', 'Ġreturn', 'Ġb', 'Ċ']
print([ref.id_to_token(int(i)) for i in np.asarray(t([s])[0])[0][1:]])
#  ['if', 'Ġa', ':Ċ', 'ĠĠĠĠĠĠĠĠ', 'return', 'Ġb', 'Ċ']
```

**Suggested fix.** Normalise `\uXXXX` → `\x{XXXX}` when building the `RegexSplit`
pattern, and surface a warning (or error) when a Split pattern fails to compile
under PCRE2 instead of degrading silently.

---

## 3. OpenVINO GPU plugin — fused pair of int4 MatMuls sharing an input yields non-deterministic garbage

**Version:** OpenVINO 2026.3.1 (22476-759c5a6ab8c), Intel Arc Graphics (Core Ultra 155H,
Xe-LPG), driver 32.0.101.8991, OpenVINO GenAI 2026.3.1.0.

**Summary.** XHToken/Spark-X2.5 (model_type `spark2_5`) has, per attention layer, a
fused `q_k_v_proj` [6144×2560] and a per-head sigmoid output gate `g_proj` [16×2560]
that consume the same normalized hidden state. With both weights compressed to
int4 (sym, group 128 — asym and no-AWQ variants behave the same), the model is
fully coherent on CPU but on the iGPU emits one or two correct tokens and then
collapses into repeated-token garbage. The point of collapse varies between
otherwise identical processes under greedy decoding. Keeping *either* of the two
weights in fp16 makes the GPU output correct; keeping `out_proj` fp16 instead does
not. `ACTIVATIONS_SCALE_FACTOR` from 0.5 to 64, `DYNAMIC_QUANTIZATION_GROUP_SIZE=0`,
`KV_CACHE_PRECISION=f16` and `GPU_ENABLE_SDPA_OPTIMIZATION=NO` change nothing.
fp16 and int8 IRs of the same model are correct on GPU (4/4 and 4/4 fresh
processes). The pattern is consistent with the plugin's horizontal fusion of the
two int4 MatMuls into one kernel that mishandles the [16×2560] partner.

**Reproduction.** Export `XHToken/Spark-X2.5-1.7B` (smaller, same behaviour) with
optimum-intel to int4 sym g128, then run 4 fresh-process greedy generations of a
~200-token coding prompt on GPU; compare to CPU. Our export needs a small patch
for the model's remote code under transformers 5 — happy to share the wrapper and
the two IRs (with and without the `g_proj` exclusion).

**Workaround.** `OVWeightQuantizationConfig(ignored_scope={"patterns": [r".*g_proj.*"]})`.
