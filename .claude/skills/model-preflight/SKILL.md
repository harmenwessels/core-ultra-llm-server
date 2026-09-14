---
name: model-preflight
description: >
  Run BEFORE benchmarking a newly converted OpenVINO model. Produces its best
  per-model card (decoding, think policy, budget) and verifies the server is
  compatible with it via live probes — chat template, reasoning split, and
  tool-call parsing — so the benchmark runs on a known-good setup instead of
  discovering format mismatches mid-run. Use when adding any new model to
  cards/ + benchmark/fleet.txt, especially a new architecture or vendor
  (Mistral/Ministral, a reasoning model, a tool-calling model). Catches the
  traps this project has hit: per-model advised temps, [THINK]/[TOOL_CALLS]
  special-token stripping by the detokenizer, Hermes-vs-Mistral tool formats,
  and reasoning token budgets.
---

# Model preflight — card + server-compat verification

Goal: a new model should be **proven compatible** (card correct, server parses
its reasoning + tool calls, output coherent) *before* it costs a full benchmark
run. Every check below maps to a real failure we hit and fixed.

Prereqs: the model is converted to an OV IR at `models/<owner>/<name>/`
(`openvino_model.xml` present). `.venv-genai` serves; `.venv-convert` has
transformers/HF for static inspection.

Work through the phases in order. **A–C are static (no GPU); D is the live
probe.** Only add the model to `benchmark/fleet.txt` after D passes.

## A. Static inspection (no server)

1. **Architecture / convertibility** — read `models/<id>/config.json`:
   `architectures`, `model_type`, presence of `vision_config`.
   - Standard text arch (llama/qwen2/qwen3/mistral/granite…) → OV-native, fine.
   - VLM (`vision_config` present, e.g. `mistral3`/pixtral) → VLM path; verify
     optimum-intel registers it (`mistral3` does NOT — that path is blocked).
   - Not registered for OV → see [[ministral3-mistral-reasoning-watch]] for the
     llamafied-text-only escape hatch + provenance caveat.

2. **Chat template** — is there a `chat_template.jinja` in the IR dir (or a
   `chat_template` field in `tokenizer_config.json`)?
   - Missing (Mistral ships none) → **borrow** the matching `unsloth/<same-model>`
     template (`hf_hub_download(repo,'tokenizer_config.json')['chat_template']`),
     write it to the IR dir as `chat_template.jinja`. Confirm the tokenizer
     encodes the template's special tokens as **single** tokens
     (`[INST]`,`[THINK]`,`[/THINK]`,`[TOOL_CALLS]`,`[ARGS]`,…) — if it does, the
     borrow is vocab-compatible.

3. **Advised decoding (PER MODEL — they differ!)** — fetch the *official* vendor
   model card and quote the recommended `temperature`/`top_p`. Do NOT assume.
   Example from this repo: Ministral-3 Instruct = **0.1**, Reasoning 3B/8B =
   **0.7**, Reasoning 14B = **1.0**. Put these in the card (see B).

4. **Classify** the model for the server:
   - **Reasoning?** Does the template/system-prompt drive a thinking trace?
     Which delimiters — `<think>…</think>` (Qwen/SmolLM/LFM) or Mistral
     `[THINK]…[/THINK]`? Are they **special tokens** (check the tokenizer)?
   - **Tool format?** Inspect the template: `<tool_call>` (hermes),
     `[TOOL_CALLS]`+`[AVAILABLE_TOOLS]` (mistral), gemma `<|tool…`, lfm
     `<|tool_call_start|>`. Are the tool tokens **special**?

## B. Draft the card

Write `cards/<owner>__<name>.yaml`. Minimum: `hf_id`, `alias`, `family`, `device`,
`decoding`. Apply A.3 + A.4:
- **decoding**: set `generative` (and `structured` for a reasoner — greedy loops,
  so reasoners must sample on BOTH classes; override the default greedy with
  `{greedy: false, temp, top_p, top_k}`).
- **think_max_tokens**: reasoners need a big budget (default 3072 is for
  instruct). Set ~6144 (≤3B) / 8192 (8B–14B); refine from the measured budget in D.
- `family` falls back to global defaults if unknown — harmless.

## C. Server-compat check (read server.py — fix gaps BEFORE probing)

The server must recognize the model's formats. Verify each hook; if a gap, add
support (these are the exact fixes made this session):

- **Tool format** — `_detect_tool_format()` must classify the template correctly
  and `_NATIVE_PARSERS` must have a matching parser. Mistral `[TOOL_CALLS]` is
  NOT Hermes; it has its own `_parse_mistral_calls`. A new vendor format needs a
  new branch + parser.
- **Tool/think tokens are special → stripped by the detokenizer.** Any model
  whose tool/think delimiters are special tokens MUST decode with
  `skip_special_tokens=False` (the `_REASONING_SYSPROMPT`-or-`mistral` branch in
  `_blocking_generate`). Otherwise the parser/splitter never sees them — the
  symptom is "model reasons/calls tools but content has no markers".
- **Reasoning split** — `_split_reasoning` must handle the delimiter pair
  (`_THINK_DELIMS`). System-prompt-driven reasoners (Mistral) also need
  `_register_reasoning` (neuter the auto-inject; inject on think only).
- **Special-token markers are now auto-detected** (`_needs_raw_decode` reads
  `tokenizer.json` at load) — but still *check the probe's tool-call output*: if
  content reads like `name="read_file"> name="path">…` the markers were eaten and
  the marker set in `_MARKER_TOKENS` needs the new token.
- **Vendor templates GenAI cannot parse** (K2: `is sameas`, HF `{% generation %}`):
  the pipeline refuses to CONSTRUCT with them in `chat_template.jinja`. Keep the
  vendor file as `chat_template.vendor.jinja` (server-side render only, generation
  tags stripped) — the server prefers it when present.
- **Tool history shape**: templates iterate `tool_call.arguments` as a dict (the
  server hands them a dict copy of OpenAI's JSON string) and some (K2 3.7B/7B)
  demand a thinking field on every assistant turn (`_render_native` supplies the
  effort-matched empty tag). **Probe a tool-RESULT continuation turn too** — the
  preflight probe's single-turn tool call did not catch either of these; the
  agent-loop suite did, as HTTP 500s.
- **Hybrid thinkers loop under greedy think-mode** (MiniCPM5-2B: 25–36k chars of
  "The user says… So we need…"). Sample on the structured class for them, as for
  reasoners. Sampling is seeded (`seed` = block index since 2026-09-14): one run
  is one trajectory, so a knife-edge behaviour (Ornith route/diagnose) can flip
  between passes without anything in the stack changing.

## D. Live probes (start server, run the probe script)

Start the server on JUST this model, then run the probe:
```
# (PowerShell) start server detached on the model, wait for /v1/models, then:
.venv-genai/Scripts/python.exe .claude/skills/model-preflight/preflight_probe.py \
    --model <hf_id> [--reasoning] [--tools]
# stop the server afterward (leave server OFF — standing rule).
```
The probe reports PASS/FAIL for:
- **coherence** — non-empty, sensible output (catches broken conversions, e.g. a
  llamafication that emits garbage).
- **nothink bounded** — a codegen prompt with no `reasoning_effort` returns a
  bounded answer with NO leaked `<think>`/`[THINK]` markers in content.
- **think split** (`--reasoning`) — `reasoning_effort=high` + big budget →
  `reasoning_content` populated, `content` clean (no markers), and it reports the
  observed think-token length so you can set `think_max_tokens`.
- **tool call** (`--tools`) — a tool-using prompt returns `finish_reason=tool_calls`
  with a parsed `tool_calls[]` and NO raw `[TOOL_CALLS]`/`<tool_call>` left in
  content. This is the check that would have caught the agent-loop bug pre-benchmark.

## E. Finalize

- Fix the card from D (e.g. `think_max_tokens` = measured budget + margin).
- Only now add the `hf_id` to `benchmark/fleet.txt` and run the benchmark.
- If you changed `server.py`, note it — it's a general improvement (it helped
  every future model of that family).

## Reference — the server/bench hooks this touches
`server.py`: `_detect_tool_format`, `_NATIVE_PARSERS` (+ `_parse_mistral_calls`),
`_split_reasoning`/`_THINK_DELIMS`, `_register_reasoning`/`_REASONING_SYSPROMPT`,
`_blocking_generate` (special-token-preserve decode). `bench_meta.py`:
`card_for` (decoding + `think_max_tokens` per task). Related memory:
[[ministral3-mistral-reasoning-watch]], [[conversion-playbook-and-watch-items]].
