"""Apply the model-card policy to every IR README under models/HarmenWessels:

  * no benchmark numbers on the card — a fixed "## Benchmarks" section links the
    GitHub leaderboard and the model's cards/*.yaml instead (numbers go stale the
    moment the engine or server changes; six cards had to be rewritten after the
    2026-09-17 tokenizer fixes);
  * the tokenizer-fix notes on the republished K2 / Seed-Coder cards survive as an
    engineering section, minus the before/after scores;
  * one consistent frontmatter tag set: openvino, nncf, int4|int8, intel, plus
    use-case tags (code, tool-calling, reasoning, vision) and `npu` only where a
    model is verified on the NPU. Hardware tags (core-ultra, arc, igpu) go — the
    IRs run on any OpenVINO device; the benchmark hardware belongs on GitHub.

Idempotent. Prints a one-line summary per card.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1] / "models" / "HarmenWessels"
REPO = "https://github.com/harmenwessels/core-ultra-llm-server"
NPU_VERIFIED = {"Qwen2.5-Coder-1.5B-int4-symg128-ov"}


def benchmarks_section(model: str) -> str:
    return (
        "## Benchmarks\n\n"
        f"This model is measured in [core-ultra-llm-server]({REPO}) — per-task pass/fail "
        "(code generation, editing, agentic tool use, analysis, autocomplete) on an Intel Core Ultra "
        "155H iGPU, re-run whenever the engine or server changes. The "
        f"[leaderboard]({REPO}/blob/main/benchmark/README.md) there is authoritative; this card "
        "carries no numbers so it cannot go stale. Its benchmark card (decoding, think budget) is "
        f"[`cards/HarmenWessels__{model}.yaml`]({REPO}/blob/main/cards/HarmenWessels__{model}.yaml).\n"
    )


K2_TOKENIZER_NOTE = (
    "## Tokenizer IR (regenerated 2026-09-17)\n\n"
    "The original upload's `openvino_tokenizer.xml` was built by openvino_tokenizers 2026.3, whose "
    "regex handling drops the `\\s+(?!\\S)` whitespace rule when the pre-tokenizer pattern contains "
    "`\\uXXXX` escapes (K2's does): every Python indent tokenized as an N-space token plus a bare "
    "word instead of N−1 spaces + \" word\", on every prompt. The tokenizer IRs here are rebuilt with "
    "the escapes rewritten to PCRE2's `\\x{XXXX}` and match the reference `tokenizers` library "
    "exactly. If you downloaded this model before 2026-09-18, re-download the two tokenizer files.\n"
)
SEED_TOKENIZER_NOTE = (
    "## Tokenizer IR (regenerated 2026-09-17)\n\n"
    "The original upload's `openvino_tokenizer.xml` was built by openvino_tokenizers 2026.3, which "
    "silently drops the added token with **id 0** from the BPE vocabulary — and Seed-Coder's BOS "
    "`<[begin▁of▁sentence]>` is id 0. The chat template emits BOS as text at the start of every "
    "turn, so every prompt began with 11 byte-fallback junk tokens instead of one BOS; nothing "
    "errored, the model just read corrupted input. The tokenizer IRs here are regenerated with the "
    "id-0 entry restored (`encode(bos) == [0]`). If you downloaded this model before 2026-09-18, "
    "re-download the two tokenizer files.\n"
)

# prose that quoted scores — replaced with the same statement, numbers-free
PROSE = {
    "granite-4.1-8b-int4-cw-code-ov": [(
        r"On this repo's per-task-type\nbenchmark it scored \*\*20/26 \(codegen 9/12\)\*\* versus the "
        r"wikitext2-calibrated sibling's\n\*\*19/26 \(codegen 8/12\)\*\* — a small but real gain on codegen, "
        r"and slightly faster, with the\nother task types unchanged\.",
        "On this repo's per-task-type\nbenchmark it scores slightly above the wikitext2-calibrated sibling — "
        "the gain lands on\ncode generation, and it is slightly faster, with the other task types unchanged.",
    )],
    "Spark-X2.5-1.7B-int4-symg128-ov": [(
        r"At that setting this build passed 1 of 12 codegen cases, failing on syntax \(mismatched brackets,\n"
        r"undefined names\); at temperature 0\.7 / top_p 0\.8 / top_k 20 it passes 5 — and gains on edit and\n"
        r"analysis too\. Use the cooler setting\.",
        "At that setting this build failed most code-generation cases on syntax (mismatched brackets,\n"
        "undefined names); at temperature 0.7 / top_p 0.8 / top_k 20 it does markedly better on code\n"
        "generation, editing and analysis alike. Use the cooler setting.",
    )],
}

USECASE_TAGS = {"code", "tool-calling", "reasoning", "vision", "conversational", "image-text-to-text"}
DROP_TAGS = {"intel", "arc", "igpu", "core-ultra", "en", "zh", "awq", "qat", "int8", "int4", "openvino",
             "nncf", "npu", "gemma", "llama", "mistral", "custom_code", "gpu"}


def normalise_tags(front: str, model: str) -> str:
    m = re.search(r"^tags:\n((?:- .*\n)+)", front, flags=re.M)
    old = [l[2:].strip() for l in m.group(1).splitlines()] if m else []
    keep = [t for t in old if t in USECASE_TAGS]
    if "pipeline_tag: image-text-to-text" in front:
        keep += [t for t in ("vision", "image-text-to-text") if t not in keep]
    if "conversational" not in keep:  # every IR here is a chat model
        keep.append("conversational")
    prec = "int8" if "int8" in model else "int4"
    tags = ["openvino", "nncf", prec, "intel"] + [t for t in keep if t not in ("conversational", "image-text-to-text")]
    if model in NPU_VERIFIED:
        tags.append("npu")
    tags += [t for t in ("conversational", "image-text-to-text") if t in keep]
    block = "tags:\n" + "".join(f"- {t}\n" for t in tags)
    return re.sub(r"^tags:\n(?:- .*\n)+", block, front, count=1, flags=re.M) if m else front.rstrip("\n") + "\n" + block


def rewrite(path: pathlib.Path) -> str:
    model = path.parent.name
    s = path.read_text(encoding="utf-8")
    assert s.startswith("---\n"), path
    _, front, body = s.split("---\n", 2)
    front = normalise_tags(front, model)

    for pat, rep in PROSE.get(model, []):
        body, n = re.subn(pat, rep, body)
        assert n == 1, (model, pat[:40])

    # cut everything from the measured / old benchmarks section to the end
    cut = re.search(r"^## (?:Measured behaviour|Tokenizer IR fix \(2026-09-17\) and measured behaviour|"
                    r"Benchmarks|Tokenizer IR \(regenerated 2026-09-17\))\s*$", body, flags=re.M)
    kept = body[:cut.start()].rstrip("\n") + "\n" if cut else body.rstrip("\n") + "\n"
    note = K2_TOKENIZER_NOTE if model.startswith("K2-Horizon") else SEED_TOKENIZER_NOTE if model.startswith("Seed-Coder") else ""
    body = kept + "\n" + (note + "\n" if note else "") + benchmarks_section(model)

    out = f"---\n{front}---\n{body}"
    leftover = re.findall(r"\b\d{1,2}/(?:2[56]|12)\b|\brank #\d+|#\d+ overall", body)
    path.write_text(out, encoding="utf-8")
    tag_block = re.search(r"^tags:\n((?:- .*\n)+)", front, flags=re.M).group(1)
    tags = ",".join(re.findall(r"^- (.*)$", tag_block, flags=re.M))
    return (f"{model:44} {'cut+' if cut else 'append'} {'note ' if note else ''}tags={tags}"
            + (f"  LEFTOVER {leftover}" if leftover else ""))


if __name__ == "__main__":
    targets = sorted(ROOT.glob("*/README.md"))
    if len(sys.argv) > 1:
        targets = [ROOT / a / "README.md" for a in sys.argv[1:]]
    for p in targets:
        print(rewrite(p))
