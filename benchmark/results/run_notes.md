# Fleet run notes (2026-06-09 run, started 01:13)

## Findings

- **The small "Qwen3.5" fleet entries are VL (vision) variants, not text models.**
  Both `yangsu0423/Qwen3.5-0.8B-int4-ov` and `Echo9Zulu/Qwen3.5-2B-int4_sym-ov` have vision-embedding
  IRs (`vlm=True`). Consequence: they can't serve FIM autocomplete (auto-skipped) and are weak at
  **codegen** specifically (2B-VL: codegen 3/12 but analysis 3/4 — reasoning is fine, code-gen is the
  gap). Decide: keep the VL variants, or benchmark the text Qwen3.5-2B/4B instead.

- **agent-loop is a per-probe think trade-off, not a regression.** The probes are unchanged
  (`bench_roles.py` last content-changed in the original suite commit `0d38dc7`; the rebuild was a pure
  `git mv`). The old single "/13" role score is the SAME 13 probes the new benchmark splits into
  edit(2)+agent-loop(7)+analysis(4); a 1-probe miss once diluted in /13 now shows as 6/7 in its bucket.
  Forcing think ON (via `BENCH_FORCE_THINK`) fixes the originally-failing probe but breaks a different
  one (14B: chain-depth↔no-repeat; over-reasoning yields `calls=[]`), so net stays 6/7. Policy: keep
  **nothink** for agent-loop (act-don't-ruminate).

- **Gemma thinking works** — via the native-render path (`_render_native(enable_thinking=...)`), which
  is silent in the `thinking mode -> think` swap-log (that log only covers hermes-format models). Verified
  directly: reasoning_effort=high → 110s/1910-char `thought…` trace; nothink → 46s/725-char direct. So the
  Gemma `think=think` analysis records are correct. Gemma reasoning stays inline in `content` (prefixed
  `thought`) rather than `reasoning_content` — left as-is (cosmetic; the `<channel|>` delimiter is stripped
  by the detokenizer and there's no token-id access to split on without a streamer change).

## VLM codegen wedges (iGPU)

VLM IRs running long codegen generations can wedge the iGPU pipeline (server CPU flatlines, reproducible;
server-kill recovers it and `run_fleet` auto-advances, so no polluted records). Outcome differs by model:

- **Echo9Zulu/OmniCoder-9B-int4_sym-ov** — wedged under **sampling**; **greedy is stable**. Card pinned
  greedy for codegen; the greedy re-run completed clean → **codegen 9/12** (full record landed). The
  decoding was the cause, not the model.
- **yangsu0423/Qwen3.5-VL-0.8B-int4-ov** — wedges under **both** sampling and greedy. Under greedy its
  gens ran fast and stable (~8–11s) then one still wedged the GPU ~10 min in. Fundamentally unstable on
  this iGPU regardless of decoding → **cannot be benchmarked here; recommend DROP from the fleet.** No record.
