"""Generate per-model cards (cards/<owner>__<name>.yaml) for every model IR on
disk, seeding serving config from the existing models.yaml entry where present
and detecting the family for decoding defaults. Reusable: run after downloading
a new model to get a preconfigured card. Won't overwrite an existing card unless
--force (so hand-tuning persists).

Run: .venv-genai/Scripts/python.exe benchmark/scripts/scaffold_cards.py [--force]
"""
import sys

import yaml

import bench_meta as bm

FORCE = "--force" in sys.argv

FAMILY_BY_NAME = [
    ("qwen3.5", "qwen3.5"), ("qwen3-", "qwen3"), ("qwen3_", "qwen3"),
    ("qwen2.5", "qwen2.5"), ("gemma-4", "gemma4"), ("gemma4", "gemma4"),
    ("granite", "granite"), ("omnicoder", "omnicoder"),
    ("lfm2", "lfm2"), ("minicpm", "minicpm"),
]


def detect_family(hf_id: str, quant: dict) -> str | None:
    low = hf_id.lower()
    for needle, fam in FAMILY_BY_NAME:
        if needle in low:
            return fam
    return (quant.get("model_type") or None)


def load_models_yaml() -> dict:
    f = bm.REPO_ROOT / "models.yaml"
    if not f.exists():
        return {}
    cfg = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    by_dir = {}
    for m in cfg.get("models") or []:
        d = str(m["dir"]).replace("\\", "/").rstrip("/")
        key = d.split("models/", 1)[-1]
        by_dir[key] = m
    return by_dir


SERVING_KEYS = ("device", "tool_format", "thinking", "context_budget",
                "prompt_lookup", "scheduler", "max_prompt_len", "roles")


def main() -> None:
    bm.CARDS_DIR.mkdir(parents=True, exist_ok=True)
    served = load_models_yaml()
    made = skipped = 0
    for ir_dir in sorted(bm.MODELS_DIR.glob("*/*")):
        if bm._ir_path(ir_dir) is None:
            continue
        hf_id = ir_dir.relative_to(bm.MODELS_DIR).as_posix()
        out = bm.card_path(hf_id)
        if out.exists() and not FORCE:
            skipped += 1
            continue
        quant = bm.read_quant(ir_dir)
        m = served.get(hf_id, {})
        card = {"hf_id": hf_id,
                "alias": m.get("alias") or ir_dir.name.replace("-int4", "").lower(),
                "family": detect_family(hf_id, quant)}
        for k in SERVING_KEYS:
            if k in m:
                card[k] = m[k]
        card.setdefault("device", "GPU")
        # decoding/think left to family defaults; add an empty hook to edit
        card["decoding"] = {}
        out.write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=True),
                       encoding="utf-8")
        print(f"card: {out.name}  (family={card['family']}, recipe={quant['recipe']})")
        made += 1
    print(f"\n{made} cards written, {skipped} kept (use --force to overwrite)")


if __name__ == "__main__":
    main()
