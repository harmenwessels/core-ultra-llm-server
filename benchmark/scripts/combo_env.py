"""Print the MODEL_DIRS and VIRTUAL_ROLES env strings for a named combo, with
role aliases resolved to HF ids via the cards. Used by run_combos.ps1.

  python combo_env.py small-trio
  -> MODEL_DIRS=models/OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov;...
     VIRTUAL_ROLES=router=OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov;...
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bench_meta as bm  # noqa: E402

name = sys.argv[1]
roles = (bm.combo_info(name).get("roles") or {})
hf = {role: bm.alias_to_hf(alias) for role, alias in roles.items()}
dirs = sorted({f"models/{h}" for h in hf.values()})
print("MODEL_DIRS=" + ";".join(dirs))
print("VIRTUAL_ROLES=" + ";".join(f"{r}={h}" for r, h in hf.items()))
