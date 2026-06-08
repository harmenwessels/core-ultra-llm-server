"""Card-temperature re-bench sweep orchestrator.

Today's finding: the greedy leaderboard understated every sampling-tuned model
(Omni 7-8->9, Qwen3-14B 9->10). This re-benches the fleet at each model's own
card-advised operating point, 2 blocks per condition for variance, and keeps
the existing greedy results for comparison.

Per model it: loads the model solo (server subprocess), runs the role suite at
card sampling x2, and for "brain" models the castings breadth block at card
sampling x2. Resumable: skips a (model, run) whose output already exists.
granite is omitted — its cards specify greedy, already how it is benched/served.

Run (server must NOT be already running):
  .venv/Scripts/python.exe scripts/run_card_sweep.py [tier1|tier2|tier3|all]
"""

import json
import pathlib
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
RESULTS = ROOT / "bench_results"
BASE = "http://127.0.0.1:8000"

# family -> card operating point (temp, top_p, top_k); rep_penalty inherited
CARD = {
    "qwen2.5": (0.7, 0.8, 20),
    "qwen3":   (0.7, 0.8, 20),
    "qwen3.5": (0.6, 0.95, 20),
    "gemma":   (1.0, 0.95, 64),
}

# dir (under models/), family, brain? (brain => also run castings)
PLAN = {
    "tier1": [  # brains: castings + roles
        ("OpenVINO/Qwen3-8B-int4-cw-ov", "qwen3", True),
        ("OpenVINO/Qwen2.5-Coder-7B-Instruct-int4-ov", "qwen2.5", True),
    ],
    "tier2": [  # generative seats: roles + castings
        ("Echo9Zulu/Qwen3.5-2B-int4_sym-ov", "qwen3.5", True),
        ("OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov", "qwen2.5", True),
        ("OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov", "qwen2.5", False),
        ("yangsu0423/Qwen3.5-4B-int4-ov", "qwen3.5", True),
    ],
    "tier3": [  # smaller / missed: roles only
        ("yangsu0423/Qwen3.5-0.8B-int4-ov", "qwen3.5", False),
        ("HarmenWessels/Qwen3-1.7B-int4-cw-ov", "qwen3", False),
        ("OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov", "qwen2.5", False),
        ("HarmenWessels/gemma-4-E2B-it-qat-int4-ov", "gemma", False),
        ("HarmenWessels/gemma-4-E4B-it-qat-int4-ov", "gemma", False),
    ],
}


def wait_ready(proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{BASE}/v1/models", timeout=5) as r:
                if json.load(r)["data"]:
                    return True
        except Exception:
            time.sleep(3)
    return False


def run(cmd):
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=False)


GREEDY = False  # set by --greedy: role suite once at greedy, no castings —
# the matched baseline for card deltas (deterministic probes need greedy)


def bench_model(model_dir, family, brain):
    mid = model_dir  # served id == dir path under models/ (owner/name)
    t, p, k = CARD[family]
    env_ok = (ROOT / "models" / model_dir / "openvino_language_model.xml").exists() \
        or (ROOT / "models" / model_dir / "openvino_model.xml").exists()
    if not env_ok:
        print(f"!! SKIP {model_dir}: IR not found on disk", flush=True)
        return
    import os
    env = dict(os.environ)
    env["MODEL_DIRS"] = f"models/{model_dir}"
    env["VIRTUAL_ROLES"] = (f"router={mid};architect={mid};executor={mid}")
    print(f"\n##### {model_dir}  (family={family} card={t}/{p}/{k} brain={brain})",
          flush=True)
    proc = subprocess.Popen([str(PY), "server.py"], cwd=ROOT, env=env)
    try:
        if not wait_ready(proc):
            print(f"!! {model_dir}: server failed to start", flush=True)
            return
        if GREEDY:  # matched greedy baseline on the current suite (1 run, det.)
            run([str(PY), "scripts/bench_roles.py", mid])
            return
        for r in (1, 2):  # 2 role-suite runs (sampling variance)
            run([str(PY), "scripts/bench_roles.py", mid,
                 "--sample", f"{t},{p},{k}"])
        if brain:
            for r in (1, 2):
                label = f"{model_dir.split('/')[-1]}-card-r{r}"
                run([str(PY), "scripts/bench_castings.py", label,
                     f"--temp={t}", f"--top-p={p}", f"--top-k={k}"])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(3)


if __name__ == "__main__":
    if "--greedy" in sys.argv:
        GREEDY = True
        sys.argv.remove("--greedy")
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    tiers = [which] if which in PLAN else list(PLAN)
    for tier in tiers:
        print(f"\n========== {tier} ==========", flush=True)
        for model_dir, family, brain in PLAN[tier]:
            bench_model(model_dir, family, brain)
    print("\n===== SWEEP DONE =====", flush=True)
