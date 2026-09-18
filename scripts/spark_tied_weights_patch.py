"""Make XHToken/Spark-X2.5's remote code loadable under transformers >= 5.2.

`modeling_spark.py` declares the pre-5.x form `_tied_weights_keys = ["lm_head.weight"]`
(a list). transformers >= 5.2 expects a {tied: source} MAPPING and calls .keys() on
it inside post_init() -> `AttributeError: 'list' object has no attribute 'keys'`.
Both convert envs (5.2.0 and 5.10.2) are past that change, so this cannot be
sidestepped by picking a different env.

Coerce the list into the mapping the new API wants, resolving the source parameter
from the model's actual input embedding -- spark names it `model.embedding.weight`,
not the usual `model.embed_tokens.weight`, so the name must be discovered, not assumed.

A second shim covers the mask API. `modeling_spark.py` calls
`create_causal_mask(input_embeds=..., cache_position=...)`, but every recent
transformers spells that argument `inputs_embeds` (vendor typo), and 5.10 dropped
`cache_position` from the signature entirely. The shim renames the argument and
drops any kwarg the installed signature does not accept, so the same code runs on
5.2.0 (where `cache_position` still exists) and on 5.10.2 (where it does not).
Both must be applied before the remote module is imported, i.e. before
`from_pretrained` -- the remote code binds these functions at its own import time.

Import `apply()` before the first `from_pretrained` call.
"""
import functools
import inspect

import torch
import transformers.masking_utils as _masking_utils
from transformers.modeling_utils import PreTrainedModel

_orig_expanded = PreTrainedModel.get_expanded_tied_weights_keys
_applied = False


def _coerce_tied_weights_keys(self, *args, **kwargs):
    twk = getattr(self, "_tied_weights_keys", None)
    if isinstance(twk, (list, tuple)):
        src = None
        emb = self.get_input_embeddings()
        if emb is not None:
            for name, mod in self.named_modules():
                if mod is emb:
                    src = f"{name}.weight" if name else "weight"
                    break
        if src is None:
            raise RuntimeError("cannot resolve input-embedding parameter for tied weights")
        self._tied_weights_keys = {k: src for k in twk}
        print(f"[spark-patch] coerced _tied_weights_keys -> {self._tied_weights_keys}", flush=True)
    return _orig_expanded(self, *args, **kwargs)


def _adapt_mask_fn(fn):
    """Rename spark's `input_embeds` and drop kwargs this transformers cannot take."""
    accepted = set(inspect.signature(fn).parameters)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        dropped = [k for k in kwargs if k not in accepted]
        for k in dropped:
            kwargs.pop(k)
        if dropped:
            print(f"[spark-patch] {fn.__name__}: dropped unsupported kwargs {dropped}", flush=True)
        return fn(*args, **kwargs)

    return wrapper


_orig_from_pretrained = PreTrainedModel.from_pretrained.__func__


def _from_pretrained_fp32(cls, *args, **kwargs):
    kwargs.pop("torch_dtype", None)
    kwargs["dtype"] = torch.float32
    return _orig_from_pretrained(cls, *args, **kwargs)


def force_float32_load():
    """Load the torch model in fp32 so the traced OV graph is type-consistent.

    `modeling_spark.py` runs its trunk in fp32 (`hidden_states = inputs_embeds.float()`),
    but transformers 5 changed `from_pretrained` to default to the CHECKPOINT dtype
    (bf16 here) instead of fp32. That leaves f32 activations meeting a bf16 tied
    embedding at the lm_head matmul, and OV rejects the mixed types:
        Arguments do not have the same element type (arg0: f32, arg1: bf16)
    Forcing an fp32 load makes weights and activations agree. Costs ~16 GB RAM
    during export; NNCF still emits the int4 IR.
    """
    PreTrainedModel.from_pretrained = classmethod(_from_pretrained_fp32)


def apply():
    global _applied
    if not _applied:
        PreTrainedModel.get_expanded_tied_weights_keys = _coerce_tied_weights_keys
        for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
            setattr(_masking_utils, name, _adapt_mask_fn(getattr(_masking_utils, name)))
        _applied = True
