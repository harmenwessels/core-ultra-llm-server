"""Re-run the casting leaderboard contenders at GREEDY + MAX_TOKENS=3072 — the
same condition Gemma-4-12B's 12/12 was measured at — so the comparison is fair
(the original leaderboard used a 2048 cap, which truncated verbose models).

Loads each model solo (server subprocess, all virtual roles = the model), runs
the 12-cell exec-probed casting suite greedy. Label: <name>-greedy-3072.

Run (server must NOT be running):  .venv/Scripts/python.exe scripts/rerun_castings_equal.py
"""
import json
import os
import pathlib
import subprocess
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
BASE = "http://127.0.0.1:8000"

MODELS = [
    "OpenVINO/Qwen3-14B-int4-ov",
    "Echo9Zulu/OmniCoder-9B-int4_sym-ov",
    "OpenVINO/Qwen3-8B-int4-cw-ov",
    "OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov",
]


def wait_ready(proc, timeout=200):
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


for mid in MODELS:
    env = dict(os.environ)
    env["MODEL_DIRS"] = f"models/{mid}"
    env["VIRTUAL_ROLES"] = f"router={mid};architect={mid};executor={mid}"
    env["MAX_TOKENS"] = "3072"
    label = f"{mid.split('/')[-1]}-greedy-3072"
    print(f"\n##### {label}", flush=True)
    proc = subprocess.Popen([str(PY), "server.py"], cwd=ROOT, env=env)
    try:
        if not wait_ready(proc):
            print(f"!! {mid}: server failed to start", flush=True)
            continue
        subprocess.run([str(PY), "scripts/bench_castings.py", label], cwd=ROOT,
                       env=env, check=False)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(3)
print("\n===== EQUAL RE-RUN DONE =====", flush=True)
