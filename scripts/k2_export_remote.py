"""Export IFM/K2-Horizon-3.7B / -7B (model_type `k2_horizon`, trust-remote-code)
to an OpenVINO int4 IR through optimum-cli, in-process.

optimum-intel has no `k2_horizon` entry, so the CLI refuses the model. The
shipped configs run the arch as a dense llama-style decoder (no MoVA, no MoE,
no qk-norm, full rope) with ONE non-llama piece: grouped RMSNorm
(layernorm_num_groups == 2). That lives inside the traced forward -- reshape /
pow / mean / rsqrt -- so it exports fine; only the *export config* (input and
KV-cache signatures) is needed, and llama's is the right shape. Register
llama's OV config under `k2_horizon` and run the real CLI so the int4 /
AWQ / scale-estimation flags are honoured (loose kwargs to main_export are
silently ignored -- learned on Ornith).

Usage: python k2_export_remote.py <hf_id> <out_dir> [extra optimum-cli flags...]
"""
import sys

from optimum.exporters.tasks import TasksManager
from optimum.exporters.openvino.model_configs import LlamaOpenVINOConfig

register = TasksManager.create_register("openvino", overwrite_existing=True)
register("k2_horizon", *[
    "text-generation", "text-generation-with-past",
    "feature-extraction", "feature-extraction-with-past",
])(LlamaOpenVINOConfig)

hf_id, out = sys.argv[1], sys.argv[2]
extra = sys.argv[3:]
sys.argv = [
    "optimum-cli", "export", "openvino",
    "--model", hf_id, "--task", "text-generation-with-past", "--trust-remote-code",
    "--weight-format", "int4", "--sym", "--group-size", "128", "--ratio", "1.0",
    "--awq", "--scale-estimation", "--dataset", "wikitext2",
    *extra, out,
]
print("[k2] argv:", " ".join(sys.argv[1:]), flush=True)
from optimum.commands.optimum_cli import main  # noqa: E402

main()
