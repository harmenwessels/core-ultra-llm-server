"""Export XHToken/Spark-X2.5 (model_type `spark2_5`, trust-remote-code) to an
OpenVINO IR for this fleet.

    python spark_export_remote.py <hf_id> <out_dir> [--fp16]

Default: int4 sym g128, ratio 1.0, AWQ + scale-estimation on wikitext2 -- the
fleet recipe -- with ONE exclusion (below). `--fp16` writes an unquantized IR
(diagnostic / base for NNCF variant experiments).

Why this is not a plain optimum-cli call
----------------------------------------
* optimum-intel has no `spark2_5` entry. The remote code targets the modern
  transformers APIs (DynamicCache, masking_utils, stock forward signature), so
  only the *export config* is missing. We register Gemma2's: spark is hybrid
  attention (3 sliding-512 : 1 full, `layer_types`) and builds its cache as
  `DynamicCache(config=...)`, which under transformers 5.x gives sliding layers a
  *windowed* cache. `Gemma2ModelPatcher` re-wraps past_key_values as a config-less
  `DynamicCache`, so every layer traces with a uniform full KV. Both Llama's and
  Gemma2's configs derive the same (b, 4, seq, 256) x36 shapes; the patcher is the
  whole reason to prefer Gemma2.
* Three transformers-5 skews in the vendor code, fixed by `spark_tied_weights_patch`
  (list-form `_tied_weights_keys`, the `input_embeds` typo, and the checkpoint-dtype
  default that leaves f32 activations meeting a bf16 tied embedding).
* Run in `.venv-convert` (transformers 5.2.0): 5.10 dropped `cache_position` from
  the mask API, which the remote code still passes.

The exclusion: `g_proj` stays fp16
----------------------------------
`g_proj` is spark's per-head sigmoid attention-output gate, a [16 x 2560] weight
that shares its input with the fused [6144 x 2560] `q_k_v_proj`. With BOTH at int4
the Arc iGPU produces garbage (per-process non-deterministic, first 1-2 tokens
right, then repeated-token collapse); CPU runs the same IR perfectly, and fp16 /
int8 IRs run perfectly on GPU. Bisection (N=3-4 fresh processes per variant,
2026-09-15): keeping EITHER q_k_v_proj OR g_proj at fp16 fixes it, keeping
out_proj does not -- i.e. the GPU plugin's horizontal fusion of the two parallel
int4 MatMuls is what breaks. Excluding g_proj costs ~0.01 GB. This is NOT an
ACTIVATIONS_SCALE_FACTOR case (swept 0.5-64, all garbage).

Loose kwargs to `main_export` are silently ignored (learned on Ornith), so the
quantization goes through `OVWeightQuantizationConfig`, which the CLI cannot
express an `ignored_scope` for.
"""
import sys

from optimum.exporters.tasks import TasksManager
from optimum.exporters.openvino.model_configs import Gemma2OpenVINOConfig

import spark_tied_weights_patch  # noqa: E402  (same dir; must precede from_pretrained)

spark_tied_weights_patch.apply()
spark_tied_weights_patch.force_float32_load()

register = TasksManager.create_register("openvino", overwrite_existing=True)
register("spark2_5", *[
    "text-generation", "text-generation-with-past",
    "feature-extraction", "feature-extraction-with-past",
])(Gemma2OpenVINOConfig)

from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

hf_id, out = sys.argv[1], sys.argv[2]
fp16 = "--fp16" in sys.argv[3:]

if fp16:
    qcfg = None
else:
    qcfg = OVWeightQuantizationConfig(
        bits=4, sym=True, group_size=128, ratio=1.0,
        awq=True, scale_estimation=True, dataset="wikitext2",
        ignored_scope={"patterns": [r".*g_proj.*"]},
    )
print(f"[spark] export {hf_id} -> {out}  quant={'fp16' if fp16 else 'int4 sym g128 awq+se, g_proj excluded'}", flush=True)

model = OVModelForCausalLM.from_pretrained(
    hf_id, export=True, trust_remote_code=True,
    quantization_config=qcfg, load_in_8bit=False,
)
model.save_pretrained(out)

# GenAI needs the OV tokenizer/detokenizer IRs next to the model; the CLI writes
# them, the Python API path does not carry them through save_pretrained.
import openvino as ov  # noqa: E402
from openvino_tokenizers import convert_tokenizer  # noqa: E402

import ov_tokenizer_id0_patch  # noqa: E402  (BOS is id 0 here; upstream drops it -- see that file)

ov_tokenizer_id0_patch.apply()
tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
tok.save_pretrained(out)
ov_tok, ov_detok = convert_tokenizer(tok, with_detokenizer=True)
ov.save_model(ov_tok, f"{out}/openvino_tokenizer.xml")
ov.save_model(ov_detok, f"{out}/openvino_detokenizer.xml")
print("[spark] done", flush=True)
