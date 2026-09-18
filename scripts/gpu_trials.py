"""Fresh-process GPU coherence trials for a text-LM OpenVINO IR.

    python gpu_trials.py <model_dir> [n=4] [device=GPU]

Runs N generations, each in its own process, and exits non-zero if any is
degenerate. Exists because of Spark-X2.5 (2026-09-15): the iGPU's fused
int4 kernel failure is NON-deterministic per process — one run passed by luck
before 8/8 failed — so a single smoke test proves nothing. Use N >= 3.
Verdict is repetition-based (unique-word ratio / top-word frequency); a plain
ASCII-ratio score rated "[[ [[ [[" as 97% fine.
"""
import subprocess
import sys

PROMPT = ("Write a Python class `LRUCache` with get and put in O(1). "
          "Explain the approach briefly, then give the full code.")

CHILD = r'''
import sys, time, openvino_genai as g
p = g.LLMPipeline(sys.argv[1], sys.argv[2]); t = p.get_tokenizer()
c = g.GenerationConfig(); c.max_new_tokens = 200; c.do_sample = False
t0 = time.time()
out = str(p.generate(t.apply_chat_template([{"role": "user", "content": sys.argv[3]}],
                                           add_generation_prompt=True), c))
w = out.split()
uniq = len(set(w)) / max(len(w), 1)
top = max((w.count(x) for x in set(w)), default=0) / max(len(w), 1)
bad = len(w) < 20 or uniq < 0.35 or top > 0.15
print(f"{'GARBAGE' if bad else 'OK'} uniq={uniq:.2f} top={top:.2f} {time.time()-t0:.0f}s :: {out[:90]!r}")
sys.exit(1 if bad else 0)
'''


def main() -> int:
    model_dir = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    device = sys.argv[3] if len(sys.argv) > 3 else "GPU"
    fails = 0
    for i in range(1, n + 1):
        r = subprocess.run([sys.executable, "-c", CHILD, model_dir, device, PROMPT],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = (r.stdout.strip().splitlines() or [r.stderr.strip()[-200:] or "no output"])[-1]
        print(f"[{device} trial {i}/{n}] {line}", flush=True)
        fails += r.returncode != 0
    print(f"[{device}] {n - fails}/{n} OK", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
