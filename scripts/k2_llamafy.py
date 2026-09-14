"""Llamafy IFM/K2-Horizon-0.9B for a plain optimum-intel llama export.

Why this is exact and not an approximation: in the shipped 0.9B config every
K2-specific switch is off -- no MoVA attention (mova_num_experts unset), no MoE
(all 28 layers in mlp_only_layers), no q/k norm, rope_head_dim == head_dim,
no sliding window, and layernorm_num_groups == 1, which makes K2HorizonRMSNorm
bit-identical to LlamaRMSNorm. Weight keys already follow the llama layout
(model.layers.N.self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj, ...), the
MLP is SiLU-gated, and rope is rotate_half with YaRN -- all of which llama's
HF implementation reproduces from config alone.

NOT valid for 3.7B/7B: those ship layernorm_num_groups == 2 (grouped RMSNorm),
which llama cannot express; they need the trust-remote-code export instead.

Usage: python k2_llamafy.py <hf-snapshot-dir> <out-dir>
"""
import json
import os
import shutil
import sys

src, dst = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(src, "config.json"), encoding="utf-8"))

# Refuse anything that isn't provably llama-equivalent.
n_layers = cfg["num_hidden_layers"]
assert cfg.get("layernorm_num_groups", 1) == 1, "grouped RMSNorm -> not llamafiable"
assert not cfg.get("query_key_norm"), "qk-norm on -> not llamafiable"
assert not cfg.get("mova_num_experts"), "MoVA attention on -> not llamafiable"
assert not cfg.get("use_sliding_window") and cfg.get("sliding_window") in (None, 0)
assert sorted(cfg.get("mlp_only_layers", [])) == list(range(n_layers)), "MoE layers present"
assert cfg.get("rope_head_dim") in (None, cfg.get("head_dim")), "partial rope -> not llamafiable"
assert cfg.get("attention_gate_func") is None

keep = ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings",
        "rms_norm_eps", "hidden_act", "attention_bias", "attention_dropout",
        "tie_word_embeddings", "bos_token_id", "eos_token_id", "pad_token_id",
        "initializer_range", "use_cache"]
llama = {k: cfg[k] for k in keep if k in cfg}
llama.update({
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "dtype": cfg.get("dtype") or cfg.get("torch_dtype", "bfloat16"),
    "mlp_bias": False,
})
# transformers 5.x llama reads rope from `rope_parameters` (rope_theta inside it).
rp = dict(cfg.get("rope_parameters") or {})
rp.setdefault("rope_theta", cfg.get("rope_theta"))
rp.setdefault("rope_type", "default")
llama["rope_parameters"] = rp
llama["rope_theta"] = rp["rope_theta"]

os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f.endswith((".safetensors", ".json", ".jinja", ".txt", ".model")) and f not in (
            "config.json",) and not f.startswith("chat_template_"):
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
json.dump(llama, open(os.path.join(dst, "config.json"), "w", encoding="utf-8"), indent=2)
print("llamafied ->", dst)
print(json.dumps(llama, indent=1))
