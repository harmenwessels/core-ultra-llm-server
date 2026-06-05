r"""Why does prompt-lookup help some models and hurt others?

Measures the *draft-acceptance proxy*: the fraction of generated 3-grams that
already occur in the prompt. Prompt-lookup drafts from prompt n-grams, so this
overlap directly bounds how often speculation can win. Also shows the first
output lines (to spot thinking-mode preambles) and runs the PL timing A/B.

Usage:
    .\.venv\Scripts\python.exe scripts\research_pl_overlap.py --model-dir models\OpenVINO\Qwen2.5-Coder-1.5B-Instruct-int4-ov
"""

from __future__ import annotations

import argparse
import pathlib

import openvino_genai as ov_genai

from bench_prompt_lookup import PROMPT, DEFAULT_CACHE, run


def ngram_overlap(prompt: str, output: str, n: int = 3) -> float:
    """Fraction of output word n-grams that appear in the prompt."""
    def grams(text: str) -> list[tuple[str, ...]]:
        words = text.split()
        return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

    prompt_grams = set(grams(prompt))
    out_grams = grams(output)
    if not out_grams:
        return 0.0
    return sum(1 for g in out_grams if g in prompt_grams) / len(out_grams)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--skip-pl", action="store_true",
                        help="only measure overlap, skip the PL timing A/B")
    args = parser.parse_args()

    print("--- plain decoding ---")
    pipe = ov_genai.LLMPipeline(str(args.model_dir), "GPU",
                                CACHE_DIR=str(DEFAULT_CACHE))
    cfg = pipe.get_generation_config()
    cfg.max_new_tokens = args.max_new_tokens
    cfg.do_sample = False
    text = run(pipe, cfg, "plain", runs=1)

    print("--- output head (first 300 chars) ---")
    print(text[:300].replace("\n", "\\n"))
    print()
    print(f"3-gram overlap with prompt: {ngram_overlap(PROMPT, text):.1%}")
    print(f"output length: {len(text.split())} words")

    if args.skip_pl:
        return 0
    del pipe

    print("\n--- prompt-lookup decoding ---")
    pipe = ov_genai.LLMPipeline(str(args.model_dir), "GPU", prompt_lookup=True,
                                CACHE_DIR=str(DEFAULT_CACHE))
    cfg = pipe.get_generation_config()
    cfg.max_new_tokens = args.max_new_tokens
    cfg.do_sample = False
    cfg.num_assistant_tokens = 5
    cfg.max_ngram_size = 3
    run(pipe, cfg, "PL   ", runs=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
