# Research log: LLM inference on the Intel Core Ultra iGPU

Methods and findings from benchmarking 19 models and converting several ourselves
(June 2026). Everything here was measured on one machine; treat absolute numbers
as machine-specific and the *rules* as the transferable result.

**Test rig:** Dell XPS 13, Intel Core Ultra 155H (Meteor Lake), Arc iGPU (Xe-LPG, 128 EU),
32 GB LPDDR5x, Windows 11, Intel driver 32.0.101.8724, OpenVINO GenAI 2026.3 nightly
(`dev20260603`), Python 3.12. Current per-model results: [BENCHMARKS.md](BENCHMARKS.md)
(workload method); the superseded raw-decode overview is archived in the appendix below.

---

## Methodology

- **Benchmark protocol** ([`scripts/bench.py`](scripts/bench.py)): load pipeline with compile
  cache, one warm-up generation (discarded), then 3 measured greedy generations of ~256 tokens.
  Report median decode tok/s (= generated tokens ÷ time after first token) and TTFT.
- **A/B comparisons must run back-to-back in one session.** Thermal state moves absolute
  numbers by up to ~20% (we measured the same model at 29.9 and 22.6 tok/s hours apart).
  Cross-session comparisons of small differences are meaningless.
- **Differences under ~1 tok/s are noise** at this run count. We verified this by A/B-ing a
  finetune against its base model (OmniCoder vs Qwen3.5-9B): the ranking flipped between
  sessions, medians pooled to identical.
- **Speculative decoding needs `perf_metrics`** ([`scripts/bench_prompt_lookup.py`](scripts/bench_prompt_lookup.py)):
  streamer callbacks deliver token *batches* under speculation, so counting callbacks
  undercounts tokens and fabricates a slowdown. Use the pipeline's own metrics.
- **Download progress bars count files, not bytes** — a stalled multi-GB download can show
  "53%" forever. Check that the `.incomplete` temp file under
  `<target>/.cache/huggingface/download/` is actually growing.

---

## Finding 1 — Decode speed is memory-bandwidth-bound

Decode throughput is a near-pure function of **bytes read per token**, across two orders of
magnitude and four model generations (2024-09 → 2026-03):

| Weights read/token | Measured decode | Example |
|---|---|---|
| 0.3 GB | 87.6 tok/s | Qwen2.5-Coder-0.5B |
| 0.9 GB | 57–73 tok/s | Coder-1.5B / Qwen3.5-0.8B |
| ~2 GB | 24–35 tok/s | Coder-3B / Qwen3-4B / Qwen3.5-2B |
| ~4.5 GB | 15 tok/s | Coder-7B / Qwen3-8B |
| ~5.7 GB | ≈13 tok/s | Qwen3.5-9B / OmniCoder-9B |

Consequences:
- **Newer architecture generations buy quality, not speed**, at equal size (Qwen2.5 → Qwen3 →
  Qwen3.5 all land on the same curve). Exception: Qwen3.5 runs slightly *above* its size class.
- A coding **finetune is exactly as fast as its base model** (A/B verified).
- Runtime tuning hints (`PERFORMANCE_HINT=LATENCY`, `INFERENCE_PRECISION_HINT=f16`,
  `KV_CACHE_PRECISION=u8`) changed nothing measurable — the bottleneck is physics, not config.

## Finding 2 — The memory ceiling is lower than it looks

- The Windows driver exposes **≈ 50% of installed RAM** as iGPU-addressable memory
  (`GPU_DEVICE_TOTAL_MEM_SIZE`; 16.4 GiB here). Mind GiB-vs-GB when comparing tools.
- **First-time compile transiently needs ~1.4× the weight bytes on the device** (original +
  kernel-reordered copies coexist). Practical model limit on 32 GB RAM: largest verified load
  is 6.0 GiB of weights; 11.7 GiB (gpt-oss-20b) fails. The 16.4 GiB ceiling is *not* the
  usable weight budget.
- Three distinct failure modes, all observed:
  - `USM Device` allocation failure → device ceiling (more system RAM won't help)
  - `USM Host` allocation failure → first compile also wants ~weights-sized *free system RAM*
    (close apps / reboot fixes this one)
  - async `CL_EXEC_STATUS_ERROR` mid-upload → device ceiling surfacing late
- The `.ovcache` compiled blob removes most of the peak on later loads — but it can only be
  produced by surviving the peak once, and blobs are device+driver specific (not shareable,
  not cross-device, CPU and GPU blobs are unrelated artifacts).

## Finding 3 — Three independent support gates

A model runs only if it clears all three; we hit failures at each level:

1. **transformers knows the config class** (`ministral3` didn't exist in 4.x)
2. **optimum-intel has an export config** for the architecture (`mistral3` missing even on git
   master under transformers 5.x — a catch-22 that currently makes Ministral-3 unconvertible)
3. **The intel_gpu plugin compiles and runs the graph**: LFM2.5's *dense-hybrid* 1.2B compiles
   in 13 s and runs great (87.6 tok/s); the same family's MoE 8B-A1B grinds indefinitely
   (killed at 27 min, 15 GB RSS); the official 350M conversions hit a `ScatterNDUpdate`
   runtime bug. Support is **per-model, not per-family**.

Check gate 2 from the installed toolchain:
`TasksManager._SUPPORTED_MODEL_TYPE` (166 types with OpenVINO export as of June 2026).

## Finding 4 — Only selective-read architectures beat the bandwidth law

- **Gemma 4 E-series (PLE/MatFormer)** stores ~4 GB but *streams* only ~1.4 GB per token (the
  per-layer embedding tables are gathered, not read wholesale) → 29.9 tok/s at a 4.1 GB disk
  size that "should" do ~15. The only curve-breaker that actually runs on this machine.
- **MoE** has the same property in theory; in practice every interesting MoE either exceeded
  the memory ceiling (gpt-oss-20b, Qwen3-30B-A3B, Gemma-26B-A4B) or failed to compile
  (LFM2.5-8B-A1B). As of June 2026: no working MoE on this hardware.
- The PLE path is also why E2B's TTFT and per-read-byte efficiency are slightly worse than
  dense peers — gather overhead. A channel-wise requantization gained it nothing (see next).

## Finding 5 — Quantization recipe can halve or double speed, per-architecture

Same model (granite-4.1-3b), same toolchain, three recipes, same session:

| Recipe | Size | Decode |
|---|---|---|
| int4 sym **channel-wise** (`--sym --group-size -1`) | 1.72 GiB | **27.4 tok/s** |
| int8 per-channel | 3.19 GiB | 17.4 tok/s |
| int4 asym group-128 (the common default) | 1.78 GiB | **13.0 tok/s** |

Group-wise dequantization is expensive on Arc kernels — the *smaller* g128 file ran at half
the cw speed. But the sensitivity is **architecture-specific**: the identical cw change on
Gemma 4 E2B (whose g128 build already rides the curve) gained 0%. And group-wise int4 is fine
for every Qwen we tested. **Rule: when converting, benchmark recipes; never assume.**
(Quality caveat: cw quantizes more coarsely than g128; we measured speed, not perplexity.)

## Finding 6 — Prompt-lookup gain is predicted by output/prompt n-gram overlap

Speculative decoding without a draft model (drafts from prompt n-grams, batched verification).
Three results, in the order we learned them:

**(a) The gain scales with model size within one family** (same code-edit prompt, same session;
every accepted draft token saves a weight-read, and weight-reads are what big models pay for):

| Qwen2.5-Coder | Plain | PL | Δ |
|---|---|---|---|
| 0.5B | 80.1 | 131.0 | +64% |
| 1.5B | 62.1 | 70.0 | +13% |
| 3B | 30.2 | 57.2 | +89% |
| 7B | 17.1 | **41.8** | **+144%** |

The 7B at 41.8 tok/s on edit workloads rewrites the speed/quality trade-off — 7B quality at
3B-class effective speed for echo-heavy tasks.

**(b) Why other models *lose* with PL** — measured via the draft-acceptance proxy: the fraction
of generated 3-grams already present in the prompt
([`scripts/research_pl_overlap.py`](scripts/research_pl_overlap.py)):

| Model | Output/prompt overlap | Plain | PL | Δ |
|---|---|---|---|---|
| Qwen3-0.6B (thinking) | 27.4% | 78.0 | 53.5 | −31% |
| Qwen2.5-Coder-1.5B | 44.4% | 61.6 | 68.5 | +11% |
| Granite-4.1-3b (general instruct) | 71.6% | 29.9 | 47.1 | **+58%** |

**(c) The rule** (this *corrects* our first hypothesis "FIM-trained vs general"): PL gain tracks
**output/prompt n-gram overlap**, break-even ≈ 35–40% here. The two real drivers:
- **Free-prose thinking is PL's worst case** — Qwen3's `<think>` preamble is hundreds of
  free-form tokens with ~zero prompt overlap; every draft is rejected. This, not "general vs
  coder", is why Qwen3-4B/8B regressed (−33%/−20%). But thinking per se isn't the variable:
  LFM2.5-1.2B-Thinking *gains* +63% on architect prompts because its reasoning restates the
  prompt heavily. Echo overlap must be measured, not inferred from model category.
- **Instruction-faithful echoing is the best case regardless of family** — Granite-4.1 (not
  FIM-trained) hit 71.6% overlap by following "keep the logic identical" verbatim and gained
  +58%, *beating* the Coder, which rewrote more creatively (44.4%).

The server enables PL per model via `PROMPT_LOOKUP_MODELS` (default: the autocomplete coder).
Caveats: LLMPipeline only (not VLM-shaped IRs); switches to the continuous-batching backend
whose numerics differ slightly — outputs are quality-equivalent but not bit-identical; all
gains are for echo-heavy prompts — free-form generation runs at plain speed or below.

## Finding 7 — Self-converted models reach parity with community artifacts

We replicated an existing community conversion (Gemma 4 E2B, matched recipe) and benched at
parity same-session (24.6 vs 22.6 tok/s, overlapping ranges). The conversion pipeline below is
therefore trusted for publication-grade artifacts. First published result:
[HarmenWessels/granite-4.1-3b-int4-cw-ov](https://huggingface.co/HarmenWessels/granite-4.1-3b-int4-cw-ov)
— the first OpenVINO IR of Granite 4.1.

---

## Finding 8 — Agent harnesses are priced in prefill, and harness choice dominates

Live experiments driving agentic CLI frontends against the server (2026-06-06) reduced to
prompt-weight economics. Kilo CLI (OpenCode engine) sends ~67k chars (~17k tokens) of system
prompt + 13 tool schemas before the user's request; Continue CLI sends ~8.4k chars (~2k tokens)
for the same assignment — an 8× difference that decides usability by itself on prefill-bound
hardware. Since the OpenAI surface is stateless, every agent turn re-prefills the whole
conversation: we measured 13–23 s TTFT per turn for ≤10-token tool-call outputs (TTFT is ~95%
of agent turn time). Secondary finding: native-tool engines (OpenCode) require server-side
`tools` support — implemented hermes-style (schema injection + `<tool_call>` parsing with
small-model JSON repair) in `server.py`; Roo-lineage extensions use text-protocol tools and
need nothing server-side.

## Finding 9 — Tool discipline is its own capability axis (and granite-8b owns it)

The role-fitness suite (`scripts/bench_roles.py`, 13 executable probes: tool-call validity,
selection, restraint, result-use, repeat-avoidance, byte-exact editing, full-file writes,
stop discipline, routing, diagnosis, planning, scripted multi-turn loops, deep recall) over
10 artifacts shows tool discipline correlates with *instruction-following*, not size or
coding score:

- granite-4.1-8b: 8/9 v1 probes — the **only** model in the stable that emits byte-exact
  `edit_file` old_strings, and the only one that survives a scripted 6-turn fix-test-verify
  loop (clean stop on green). Its BFCL/IFEval scores predicted this.
- Qwen2.5-Coder-7B (the PL edit star) *fabricates whitespace* in edit calls — coding skill
  ≠ tool discipline.
- Bigger Gemma is worse: E4B answers in prose instead of calling tools at all ("tool-shy"),
  scoring below E2B (5/9 vs 7/9).
- Actor ≠ analyst: granite alone sustains loops but misdiagnoses a planted bug (blames the
  test); E2B/Qwen3.5-2B diagnose correctly but cannot drive loops. No single small model does
  both — the strongest empirical argument for role-split serving (architect/executor).
- `write-full` is 0-for-10: no local model emits whole files inside tool-call JSON. Coder
  roles must be edit-first with server-side old-string verification.
- Routing (3-way classification) is easy: 8/10 pass 6/6 — the *cheapest* passing model
  (Qwen2.5-Coder-1.5B) takes the router seat.
- Models with strong native tool formats (LFM2.5: `<|tool_call_start|>` Pythonic) ignore
  injected hermes instructions — the suite (and any hermes-style server) understates them;
  a per-model tool-format adapter would be needed for a fair reading.

## Finding 10 — MoE expert graphs do not build on this iGPU (two failure signatures)

Despite OpenVINO 2026.0 "MoE GA" notes (validated on gpt-oss-20b / Qwen3-30B-A3B), every MoE
we built fails GPU compile on this machine, each with a distinct, reproducible signature:

| MoE | signature |
|---|---|
| LFM2.5-8B-A1B (own IR, fresh patcher) | thread deadlock ~5 min in: ~290 CPU-s then 60 threads parked, GPU idle, RAM paged out; reproduces on pinned and newest nightly |
| granite-4.0-h-tiny 7B-A1B (`granitemoehybrid`) | unbounded phased grind: steady ~1-core compile with 4→20 GB RAM alloc/release cycles, no convergence after 57 min (killed) |

Dense models from the same families compile in seconds (LFM2.5-1.2B: 4 s). Conversion is NOT
the blocker — both IRs export cleanly. Verdict: MoE-on-this-iGPU is closed until an OpenVINO
release demonstrably fixes it; this pre-judges JetBrains Mellum2 (12B-A2.5B, `mellum` arch,
not yet in the export registry) even after gate 2 lands.

## Finding 11 — Prefill scales superlinearly and sets per-model context budgets

TTFT-vs-prompt-size sweeps (`scripts/bench_prefill.py`) diverge wildly by architecture —
decode-rank does not predict prefill-rank:

| ~tokens | granite-8b | Gemma E2B (VLM) | Qwen3.5-2B (VLM) | Qwen2.5-Coder-1.5B |
|---|---|---|---|---|
| 1k | 6 s | 3.7 s | 3.0 s | — |
| 8k | 43 s | 19 s | 7.3 s | 14 s (16k: 22 s) |
| 16k | OOM¹ | 67 s | **17 s** | |

¹ not total-memory exhaustion: a **single-allocation cap** (one ~4.1 GiB buffer request) —
chunked prefill clears it (see Finding 12). Qwen3.5-2B's near-flat curve makes it the
long-context seat of the lineup regardless of its mid-pack decode speed.

## Finding 12 — Prefix caching collapses warm TTFT ~27–60× (shipped)

`SchedulerConfig(enable_prefix_caching=True, max_num_batched_tokens=2048)` on the GenAI
pipeline (works through both LLMPipeline and VLMPipeline CB paths):

- granite-8b, 8k shared prefix: 63 s cold → **0.9 s warm** standalone; 71.5 s → **2.6 s**
  through the full server API (27×). Qwen3.5-2B: 9.2 s → 0.6 s.
- Chunked prefill clears the 16k single-allocation wall: granite now prefills 24k+ (at ~2×
  cold-prefill overhead; the 2B pays ~25% and is *faster* at 16k than unchunked).
- Cost model: the KV block pool (`cache_size` GB) is **reserved at load, permanently** —
  budget it against the ceiling like model weights. Validated co-resident: granite+4GB pool,
  2B+2GB pool, coder+PL ≈ 12.6 GiB standing, all serving, warm hits intact.
- Shipped as `SCHEDULER_MODELS` env in `server.py`. Every multi-turn shape (chat history,
  agent loops) now pays full prefill once per conversation, not once per turn.

## Finding 13 — The GPU already quantizes KV cache to int8; the explicit hint is broken

Compiled-model introspection shows `KV_CACHE_PRECISION: int8_t` *by default* (plugin
"dynamic" mode) — the memory saving usually sought via u8-KV hints already exists on this
stack. Setting `KV_CACHE_PRECISION=u8` explicitly is both redundant and broken: it flips the
paged-attention kernel into BY_CHANNEL quant mode expecting metadata-extended KV blocks
(`block_size + block_size/16×4` = 20) while the GenAI allocator hands it plain 16-token
blocks → `Incorrect block size ... Expected 20, but got 12`. Reproduces through nightly
build 22103; found by us, not publicly reported. Action: never set the hint.

## Finding 14 — The NPU is a 1–2B express lane, and quantization damage is task-selective

Overnight NPU campaign (2026-06-06/07), after the cw-sym discovery unblocked compilation:

- **Size/latency law** (96-token FIM, warm): 0.5B → 2.9 s, 1.5B → 5.6 s, 3B → 7.9 s,
  granite-3b → 11.9 s. The autocomplete-usable band ends at ~1.5B.
- **Quantization damage is task-selective**: the cw-sym Coder-1.5B passes the executable
  FIM probe but drops routing from 6/6 (g128) to 3/6 — *on both devices*, so it is the
  quantization, not NPU numerics. Certify per role, not per artifact. (And data-free cw
  broke the Coder-3B's FIM outright — the granite AWQ lesson, reproduced on qwen.)
- **NPU optimization knobs are null on this stack**: GENERATE_HINT/PYRAMID/NPUW prefix
  caching moved nothing (±1%); NPUW_LLM_ENABLE_PREFIX_CACHING shows zero warm-prefix
  benefit (watch item). The one real lever: the `CACHE_DIR` blob cache (load 16 s → 3 s).
- **Concurrency is real and shipped**: per-device generation locks (`MODEL_DEVICES`) +
  moving non-stream generates off the event loop (`asyncio.to_thread` — a long non-stream
  generate used to freeze the whole HTTP server) give lock-free NPU autocomplete at
  ~7 s while the GPU runs multi-stage agent turns.
- **Correction (2026-06-07, via a community-shared doc note): the Series-1 NPU constraint
  is SYMMETRY, not channel-wise layout** — sym cw *and* sym group-wise int4 are supported;
  asym is what trips the `vpux` verifier (all our failing g128 artifacts were asym). The
  untested **sym-g128** recipe threads the needle: Coder-1.5B-symg128 compiles on NPU,
  routes **6/6 on both devices** (3.0 s/dec NPU) AND passes the FIM probe — one artifact
  now holds both NPU seats (autocomplete + router), superseding the cw build that had
  sacrificed routing. The virtual model's router runs on NPU: classification costs zero
  GPU contention. Earlier verdict, superseded: "exactly one seat — cw-1.5B autocomplete;
  routing candidates all failed" (that ladder tested cw and asym builds only).
- **There is no numerics-safe size threshold** (Qwen3-1.7B paired test, 2026-06-07):
  the same cw IR routes 4/6 on GPU and 5/6 on NPU *with different errors*, and recall
  returns empty on NPU — device numerics shift near-threshold behaviors in both
  directions, exactly like calibration domain (playbook 0c). Per-device, per-artifact
  probe certification is mandatory at any size. NPU long-form generation is ~16× slower
  (plan: 3.8 s GPU → 61 s NPU) — short-output roles only.

## Finding 15 — The virtual model: measured role-split serving, shipped

`virtual/agent` (server.py) routes each turn to the best measured brain from the
role-fitness suite: router (Coder-1.5B g128, 6/6) classifies fresh requests; architect
(Qwen3.5-2B, prefix-cached, no-think) analyzes and plans with read-only tools; executor
(granite-4.1-8b, prefix-cached, PL off) runs edit→test→verify loops with full tools.
Stateless across requests: tool continuations route via role-encoded call ids
(`call_arch_…`/`call_exec_…`); a plan marker in history switches the conversation to
execution phase. Server-side guards encode the measured failure modes: a loop-breaker
(identical-call hash → corrective note + one retry) and edit old_string verification
against file content seen in-conversation. No-tools design requests run plan→implement
in one response, with the architect's plan streamed as `reasoning_content` (renders as a
thinking block in Continue). Validated end-to-end with NPU autocomplete serving
concurrently throughout.

## Finding 16 — Speak each model's tool language: format mismatch costs more than size

Per-family tool adapters (server-side jinja2 rendering of the model's OWN chat template
with `tools`/`enable_thinking`/tool-role turns — everything GenAI's template application
cannot pass — plus per-family emission parsers) re-scored the Gemmas dramatically
(2026-06-07): **E4B 10/15 → 13/15, the new role-fitness champion**, gaining byte-exact
edits and clean loop endurance — skills the hermes-era matrix called granite-exclusive.
The "tool-shy" verdict measured our protocol, not the model. Corollaries: (a) any
agentic score on a non-native protocol is a lower bound; (b) Gemma thinking (pattern C)
recorded its first measured quality win — it flips E4B's diagnose verdict from
test-blaming to correct (66.7 s vs 9.4 s) — thinking earns selective architect-style use
on Gemma; (c) parser care matters: Gemma's brace-delimited args break on code content
with nested braces (v1 limitation); (d) LFM's template hides its protocol from
literal-string detection — still served hermes, still understated. Registry
(`models.yaml`) carries `tool_format` per model so the language is pinned, not guessed.

Full-fleet template survey (2026-06-07, round 2): granite's "native" format **is**
hermes (its template builds the `<tools>`/`<tool_call>` block verbatim) — injection was
correct all along, and faithful native rendering measured 1pt *worse* (empty old_string
emission under native framing) → granite pinned `tool_format: hermes` in the registry.
LFM's declaration mechanism found (`List of tools: [...]` in the system prompt) and the
full adapter landed: **LFM2.5-1.2B 4/13 → 7/13** — its honest score (call probes pass;
remaining failures are 1.2B capability limits, not protocol). Qwen/MiniCPM/OmniCoder
templates are natively hermes. Net language map: gemma → native adapter (big win),
lfm → native adapter (fair reading), everything else → hermes (correct by training).

## Finding 17 — Three engines, one memory: place roles by contention, not just speed

All three engines (CPU, iGPU, NPU) share one physical RAM pool and its bandwidth —
"VRAM" is a driver carve-out, the NPU maps the same memory. What differs per engine is
*compute ownership*: GPU cycles belong to the big brains, CPU cycles to the user's
applications, NPU cycles to nobody. Measured on the router workload (Coder-1.5B, 6-case
classification): **CPU 0.69 s/dec solo and 1.84 s under full GPU load — 2-4× faster than
NPU (3.0/4.05 s) both ways**, with reference numerics and no quantization-layout
constraints. CPU degrades more under load (2.7× vs NPU's 1.35×) but its worst case beats
the NPU's best. Final auxiliary placement: **router on CPU** (fast, exact), **lock-free
autocomplete on NPU** (typing-time is when the CPU belongs to the IDE), big brains on GPU.
NPU long-form remains ~16× slower than GPU — short-output roles only on both auxiliaries.

## Conversion playbook (Route B)

Separate venv (`.venv-convert/`, gitignored) with: `optimum` + `optimum-onnx` + `optimum-intel`
from git master, torch CPU wheels, `nncf`, `compressed-tensors`.

```powershell
# typical text-only model, speed-first recipe WITH data-aware calibration
# (data-free cw-int4 measurably damaged quality on granite — AWQ+SE repaired it
#  at zero size/speed cost; see BENCHMARKS.md finding 9)
optimum-cli export openvino -m <org>/<model> --weight-format int4 --sym --group-size -1 `
  --awq --scale-estimation --dataset wikitext2 models\<owner>\<name>-int4-cw-ov

# multimodal (Gemma 4, Qwen-VL...): the supported task must be explicit
optimum-cli export openvino -m google/gemma-4-E2B-it --task image-text-to-text `
  --weight-format int4 models\<owner>\<name>
```

Hard-won rules:
0. **Quantization granularity must scale with model size**: cw-sym int4 + AWQ is the speed
   recipe for ~3B–8B (validated on Granite 4.1), but at ≤1B it produces *degenerate output*
   (MiniCPM5-1B: repetition loops; int8 and g128 of the same model are coherent). For tiny
   models use g128 or int8 and always run a coherence probe before benchmarking speed.
0b. **Hybrid-thinking models are controlled via the tokenizer IR's rt_info template.** GenAI
   cannot pass `enable_thinking`, and it reads the chat template from `openvino_tokenizer.xml`
   **rt_info** — not from `chat_template.jinja` (patching that file is a no-op). Hardcoding the
   no-think prefix (`<think>\n\n</think>\n\n` after the assistant header) in rt_info switched
   MiniCPM5-1B from preamble-failing to the fastest probe-passing edit model measured
   (81.4 tok/s). Corollary: "thinks by default" verdicts on other models (Qwen3 family) reflect
   their conversions' baked templates and may be flippable the same way — re-test before
   excluding a thinking-capable model.
1. **transformers version must match the target architecture** — and the requirements differ
   per model: granite wants 4.57.x; gemma4 wants exactly 5.5.0 (5.10 renamed an attention
   attribute and breaks the trace); Qwen3.5 wants 5.x; **lfm2/lfm2_moe wants 5.0.x exactly**
   (the OV patcher imports `Lfm2HybridConvCache`, removed in 5.5's cache refactor, while
   ≤4.57.6 predates `lfm2_moe`; 5.4 has both symbols but a drifted sdpa-mask signature breaks
   tracing). Swap per export; pip's dependency warnings against optimum's pins are expected
   and harmless.
0c. **Calibration domain moves near-threshold behaviors — in either direction.** Same
   recipe, only the AWQ/scale-estimation dataset changed (wikitext2 prose → 128 chunks of
   real Python, seed-pinned; `scripts/convert_code_calibrated.py`): granite-4.1-**3b**
   *gained* loop endurance (`chain-depth` flipped to a clean edit→test→stop loop,
   8/13 → 9/13), but granite-4.1-**8b** *lost* it (11/13 → 10/13, stalls at turn 2) with
   `diagnose` unchanged. Both greedy/deterministic per build. Coarse probes saw nothing
   either way — the effect lives in the agentic margins, and it is a lottery, not a lever.
   Rule: calibration dataset is a per-build hyperparameter — convert, run the seat-critical
   probes, keep the winner. (wikitext2-8b keeps the executor seat; the code-3b is the
   better 3b artifact.)
1b. **Believe the declared pin first.** optimum-intel master declares `transformers<5.1` —
   that pointer would have found the lfm2 window immediately; symbol-probing across versions
   found it the slow way. Read the installed package's requirements before bisecting.
2. **Every working conversion ships its recipe**: `openvino_config.json` in any HF conversion
   records the exact transformers version and quantization parameters used. Read it before
   reinventing.
3. **Never pass `--task text-generation`** — it exports without KV-cache (→ `beam_idx` error
   in GenAI). Omit `--task` for text models (auto-infers `-with-past`); pass the explicit
   multimodal task for VLMs.
4. **Bench every artifact before publishing** (Finding 5). Verify stateful export:
   the IR should have 4 inputs (`input_ids`, `attention_mask`, `position_ids`, `beam_idx`).

## Candidate screening ledger (sweep of 2026-06-06)

Exhaustive sweep of public models against our gates (dense or GPU-runnable, supported
architecture, ≤ ~6 GiB int4, permissive license, quality above incumbents). Screened out:

| Candidate | Reason |
|---|---|
| Gemma-4-12B (MMLU-Pro 77.2, LCB 72.0) | new `gemma4_unified` arch — unknown to transformers ≤5.5 and the export registry |
| GLM-4.7-Flash | MoE (`glm4_moe_lite`), unsupported type, too big |
| Qwen3.6-27B / Mistral-Small-4 / Gemma-4-31B / EXAONE-4.5-33B / Codestral | over the memory ceiling |
| Qwen3.6 small dense / Qwen3.5-Coder / EXAONE-4.5 ≤8B | not released yet |
| Seed-Coder-8B | 2025-05 vintage — matched/beaten by granite-4.1-8b (already published) |
| EXAONE-4.0 family | gated + restrictive license |
| GLM-4-9B-0414, Phi-4-mini, Falcon-3, OLMo-3, Hunyuan-7B | dated or dominated by incumbents at equal size |
| DeepSeek-R1 distills | thinking-default (edit-budget failures) |
| MiniCPM5-1B | converted & tested: coherent at g128 (~82–87 tok/s) but thinks by default under the OV chat template → no role won vs Qwen3.5-0.8B; also produced the granularity-vs-scale finding (playbook rule 0) |
| LFM2.5-8B-A1B (agentic flagship: IFEval 91.8, Tau²-Telecom 88.1, 1.5B active) | own IR converts cleanly (transformers 5.0.x window) but MoE GPU compile deadlocks — Finding 10; top retest candidate |
| granite-4.0-h-tiny 7B-A1B | converted as the MoE-discriminator experiment; compile grinds unboundedly — Finding 10 |
| LFM2.5-1.2B-Instruct | converted & role-tested: 4/13 — emits its native `<|tool_call_start|>` Pythonic format over instructed hermes (Finding 9 caveat), route 3/6, no seat; ~90 tok/s chat is its only niche |

Conclusion: as of 2026-06-06 the served lineup is at the practical optimum for this hardware —
every higher-quality candidate is upstream-blocked or unreleased, not effort-blocked.

## Open items (as of 2026-06)

- **Qwen3.5-Coder**: not yet released — would likely obsolete the Qwen2.5-Coder autocomplete
  default the moment a small FIM-trained variant ships.
- **Qwen3.6 small dense** (2B/4B/8B class): the 27B dense (2026-04, SWE-bench 77.2) suggests
  smaller siblings are coming — would likely supersede the Qwen3.5 tier.
- **EXAONE-4.5 small sizes**: 33B-only so far; STEM avg 77.3 beats GPT-5-mini — but mind the
  restrictive EXAONE license before investing.
- **Ministral-3 / `mistral3` export support** in optimum-intel: blocked upstream (gate-2
  catch-22 above).
- **MiniCPM-V-4.6 / `minicpmv4_6`**: the "best open model under 2B" (vision-capable, Apache-2.0,
  `qwen3_5_text` backbone) is blocked at both gates — confirmed empirically 2026-06-06
  (transformers ≤5.5 doesn't know the type; export registry has only the older `minicpmv`).
  Would fill the sub-2B vision niche nothing in our table covers.
- **Gemma-4-12B / `gemma4_unified`**: the quality standout of the fitting size class
  (MMLU-Pro 77.2, LiveCodeBench 72.0 at 11.95B). **Gate 1 opened 2026-06-07** (transformers
  5.10 knows the arch); optimum-intel export config still missing. The load-ceiling
  concern is retired: Qwen3-14B int4 (~8 GiB) compiles and runs (6.4 tok/s) — the
  single-model wall sits between 8 and 11.7 GiB. Re-check optimum-intel monthly.
- **OmniCoder-9B AWQ+SE re-quantization — highest-value open quality experiment**: the
  breadth-tournament leader (8/12 solo, analyst++ role profile) runs on a data-free
  int4_sym artifact — the recipe class that measurably damaged granite-3b until AWQ+SE
  repaired it. A calibrated own conversion directly targets its failure margin
  (syntax/format slips). Needs the original safetensors repo (~18 GB) + ~2 h conversion.
- **Qwen3-14B breadth block**: compiles, generalist, thinking-default (pattern B
  switchable), 6.4 tok/s — its monolith audition is queued but slow (~2 h).
- **Gemma 4 E2B coding finetunes**: exist only as GGUF (e.g. `Gemma-4-e2bxOpus-4.7-turbo`);
  a safetensors release would enable converting the only curve-breaking architecture with
  coding tuning — the most valuable potential artifact for this hardware.
- **MoE-on-iGPU (Finding 10)**: both blocked IRs are kept on disk for one-command retests per
  OpenVINO release (`LFM2.5-8B-A1B-int4-ov` — deadlock; `granite-4.0-h-tiny-int4-ov` —
  unbounded grind). 350M `ScatterNDUpdate` runtime bug unchanged. Retest monthly.
- **JetBrains Mellum2 12B-A2.5B** (`mellum`, Apache-2.0, LCB 69.9, FIM lineage, explicit
  "focal model for routing/sub-agent tasks"): transformers ≥5.10 knows the arch; optimum-intel
  export config missing (released 2026-06-02); and it is MoE — all three gates must clear.
  The single most interesting watch item for the agent-serving direction.
- **u8 KV hint bug (Finding 13)**: `KV_CACHE_PRECISION=u8` → paged-attention BY_CHANNEL
  block-size assertion; reproduces through nightly 22103; candidate upstream issue (clean
  one-line reproducer + source diagnosis available). No local impact — defaults already int8.
- **NPU offload for autocomplete — VALIDATED (2026-06-06): the single-gen-lock is breakable.**
  The earlier "driver compiler blocked" hypothesis was wrong: official cw artifacts compile
  fine; the `vpux StopLocationVerifierPass: duplicated names` failure is triggered by
  **group-wise (g128) quantization layout — the NPU requires channel-wise-sym int4**
  (playbook addendum). A fresh `--sym --group-size -1` Coder-0.5B export compiles on NPU and
  emits correct FIM code. Measured: NPU FIM 1.8 s solo, **2.1–2.2 s while granite-8b
  generates on the GPU** (GPU job pays ~20%, DRAM sharing) — vs queueing tens of seconds
  behind the gen lock today. GPU baselines: same model 0.49–0.53 s when the GPU is free.
  Probe certification (the gate that matters): **NPU numerics flip tokens** — the cw-0.5B
  passes the executable autocomplete probe on GPU but FAILS the identical greedy run on NPU
  (completions start identical, diverge mid-stream); the **cw-1.5B passes on both devices**
  (5.6 s/96-tok NPU, 1.25 s GPU). Same-IR-different-device probe runs are mandatory before
  trusting any NPU artifact — this also explains official Phi-3-mini-cw's degenerate NPU
  output. Serving trade-off, measured: GPU ~0.9 s but queues 30 s+ behind the gen lock;
  NPU ~3–5.6 s, never queues, certified. Remaining build: per-model device targeting in
  server.py (`MODEL_DEVICES`) with the NPU path outside the GPU lock; hybrid GPU-idle/NPU-busy
  routing as the endgame.
- **Draft-model speculative decoding**: untested. granite-4.1-3b drafting for granite-8b
  could accelerate executor decode on low-overlap outputs where prompt-lookup fails
  (agent/architect turns) — complements Finding 6's PL boundary.
- ~~Per-model tool-format adapters~~ **SHIPPED 2026-06-07 (Finding 16)**: native template
  rendering + per-family parsers (gemma, lfm, hermes); fleet-wide language survey done;
  formats pinned per model in `models.yaml`. Remaining refinement: Gemma's brace-delimited
  arg parser breaks on code content with nested braces, and Gemma `reasoning_content`
  separation needs token-level boundary handling.
- **Gemma 4 thinking — switchable, historically never engaged (2026-06-07)**: the template
  gates thinking on an `enable_thinking` kwarg GenAI cannot pass → all our Gemma numbers
  are nothink. Pattern C in `_derive_think_variants` (force the gate true/false) now
  switches it per request: validated on E2B QAT (think = structured `thought` deliberation,
  36 s vs 13 s on the diagnose task, both correct). Caveat: Gemma's reasoning has no
  textual end-delimiter in decoded output (the boundary token is consumed by the
  detokenizer) — `reasoning_content` separation needs token-level handling; served default
  remains nothink.
- **FluidInference/helenai NPU catalogs screened (2026-06-07)**: qwen3-1.7b-int4-ov-npu
  compiles and runs on NPU but routes 1/6 (thinking-default burns the budget; 3/6 with
  `/no_think`) — no router seat; NPU remains autocomplete-only. The catalogs stay relevant
  as ready-made NPU artifacts for future candidates.
- **Linux**: the ~50%-of-RAM ceiling is Windows driver policy; the same laptop under native
  Ubuntu might load the 12–16 GiB models that OOM here. Untested.
- ~~Server enhancement — per-request thinking mode~~ **SHIPPED 2026-06-06**: the server derives
  think/nothink template variants at load and swaps them per request via
  `set_chat_template` under the generation lock (`reasoning_effort` / `enable_thinking`
  request fields; reasoning returned as `message.reasoning_content`). Observation from
  testing: 1B-scale thinking can *loop* on trivial problems (MiniCPM5 spent 500 tokens
  re-adding 460+161 and never finished, while no-think answered instantly and correctly) —
  thinking is not a free quality knob at small scale.
- **Server enhancement — per-request prompt-lookup**: PL is per-workload (+92% edits, −14%
  explain for the same model), but `PROMPT_LOOKUP_MODELS` toggles per model. A finer policy —
  enable PL only on `/v1/completions`, or when the chat prompt contains a code block — would
  capture the edit gains without the explain/architect penalty. Requires two pipeline
  instances or the CB pipeline's per-request config.

## Appendix: raw-decode model overview (legacy method, superseded by BENCHMARKS.md)

Single-prompt decode/TTFT measurements (`scripts/bench.py`) with modalities,
context windows and base-release dates — including models that predate the
workload-profile method above, and the memory/architecture failure cases.

| Model | Base releasedᵃ | Weights | Modalities | Max context | Decode | TTFT | PL edits³ | Verdict |
|---|---|---|---|---|---|---|---|---|
| [Qwen2.5-Coder-0.5B INT4](https://huggingface.co/OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int4-ov) | 2024-11 | 0.3 GB | text | 32k | 87.6 tok/s | 0.06 s | 131.0 | fastest; quality floor for autocomplete |
| [LFM2.5-1.2B-Thinking INT4](https://huggingface.co/Echo9Zulu/LFM2.5-1.2B-Thinking-int4_asym-ov) | 2026-01 | 0.6 GB | text | 128k | 87.6 tok/s | 0.08 s | — | hybrid conv/attention; reasoning model (thinking tokens add latency); community conversion |
| [Qwen3.5-0.8B INT4](https://huggingface.co/yangsu0423/Qwen3.5-0.8B-int4-ov) | 2026-02 | 0.9 GB | text, imageᵇ | 256k | 72.7 tok/s | 0.08 s | — | newest gen at near-0.5B speed; community conversion |
| [Qwen3-0.6B INT4](https://huggingface.co/OpenVINO/Qwen3-0.6B-int4-ov) | 2025-04 | 0.4 GB | text | 40k | 62.7 tok/s | 0.10 s | 53.5 ↓ | slower than the newer, similar-size Qwen3.5-0.8B |
| [Qwen2.5-Coder-1.5B INT4](https://huggingface.co/OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov) (default autocomplete) | 2024-09 | 0.9 GB | text | 32k | 57.0 tok/s | 0.06 s | 70.0 | autocomplete sweet spot: FIM-trained, 2.4× faster than the 3B |
| [Ministral-3b-instruct INT4](https://huggingface.co/Echo9Zulu/Ministral-3b-instruct-int4_asym-ov) | 2024-03 | 1.7 GB | text | 128k | 36.0 tok/s | 0.07 s | — | community Mistral derivative (not official Mistral AI); 2024-era quality |
| [Qwen3.5-2B INT4](https://huggingface.co/Echo9Zulu/Qwen3.5-2B-int4_sym-ov) | 2026-02 | 2.0 GB | text, imageᵇ | 256k | 34.6 tok/s | 0.17 s | — | fastest chat-quality model; community conversion |
| [Gemma 4 E2B INT4](https://huggingface.co/gregor160300/gemma-4-E2B-it-int4-ov) (default chat) | 2026-03 | 4.1 GB | text, image, audioᵇ | 128k | 29.9 tok/s | 0.23 s | — | very responsive in Continue |
| [Granite-4.1-3b INT4-cw](https://huggingface.co/HarmenWessels/granite-4.1-3b-int4-cw-ov) (our conversion) | 2026-04 | 1.7 GB | text | 128k | 27.4 tok/s | 0.13 s | 47.1 | newest Granite; first OV IR of 4.1; channel-wise recipe is 2.1× faster than the int4 default here (RESEARCH.md) |
| [Qwen3-4B INT4](https://huggingface.co/OpenVINO/Qwen3-4B-int4-ov) | 2025-04 | 2.1 GB | text | 40k | 24.9 tok/s | 0.10 s | 17.3 ↓ | same speed as Coder-3B with a newer base |
| [Granite-4.0-micro INT4](https://huggingface.co/llmware/granite-4-micro-ov) | 2025-09 | 2.2 GB | text | 128k | 24.6 tok/s | 0.16 s | — | IBM; 128k context at 3B-class speed; community conversion (llmware) |
| [Qwen2.5-Coder-3B INT4](https://huggingface.co/OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov) | 2024-11 | 2.1 GB | text | 32k | 24.0 tok/s | 0.15 s | 57.2 | strong FIM quality |
| [Qwen3.5-4B INT4](https://huggingface.co/yangsu0423/Qwen3.5-4B-int4-ov) | 2026-02 | 3.3 GB | text, imageᵇ | 256k | 19.9 tok/s | 0.31 s | — | newest gen; faster than the 9B at similar quality class; community conversion |
| [Gemma 4 E4B INT4](https://huggingface.co/OpenVINO/gemma-4-E4B-it-int4-ov) | 2026-03 | 6.0 GB | text, image, audioᵇ | 128k | 15.7 tok/s | 0.52 s | — | mid |
| [Qwen2.5-Coder-7B INT4](https://huggingface.co/OpenVINO/Qwen2.5-Coder-7B-Instruct-int4-ov) | 2024-09 | 4.2 GB | text | 32k | 15.0 tok/s | 0.20 s | **41.8** | best chat quality that fits; with prompt-lookup, the strongest edit-workload model |
| [Qwen3-8B INT4](https://huggingface.co/OpenVINO/Qwen3-8B-int4-ov) | 2025-04 | 4.6 GB | text | 40k | 15.0 tok/s | 0.13 s | 12.4 ↓ | Coder-7B speed with a newer base |
| [Qwen3-VL-8B INT4](https://huggingface.co/OpenVINO/Qwen3-VL-8B-Instruct-int4-ov) | 2025-10 | 5.5 GB | text, image, videoᵇ | 256k | 14.5 tok/s | 0.15 s | — | chat-class speed; vision+video capable |
| [Qwen3.5-9B INT4-asym](https://huggingface.co/droans/qwen3.5-9B-int4-asym-ov) | 2026-02 | 5.7 GB | text, imageᵇ | 256k | ≈13 tok/s | 0.46 s | — | newest model generation; community conversion (droans) |
| [OmniCoder-9B INT4](https://huggingface.co/Echo9Zulu/OmniCoder-9B-int4_sym-ov) | 2026-03 | 5.7 GB | text, imageᵇ | 256k | ≈13 tok/s | 0.50 s | — | coding finetune of Qwen3.5-9B — strongest coding model that fits; community conversion |
| ~~[LFM2.5-350M INT8/FP16](https://huggingface.co/OpenVINO/LFM2.5-350M-int8-ov)~~ | ~~2026-03~~ | ~~0.4 GB~~ | ~~text~~ | — | — | — | — | **runtime bug** (`ScatterNDUpdate` shape validation, both official variants) |
| ~~[LFM2.5-8B-A1B INT4](https://huggingface.co/Echo9Zulu/LFM2.5-8B-A1B-int4_sym-awq-ov)~~ | ~~2026-05~~ | ~~4.5 GB~~ | ~~text~~ | — | — | — | — | **GPU compile never completes** (MoE expert graph; the dense-hybrid 1.2B works fine) |
| ~~[gpt-oss-20b INT4](https://huggingface.co/OpenVINO/gpt-oss-20b-int4-ov)~~ | ~~2025-08~~ | ~~11.7 GiB~~ | ~~text~~ | ~~128k~~ | — | — | — | **OOM on 32 GB RAM**: device allocation fails at compile despite 18 GB free host RAM |
| ~~[Qwen3-Coder-30B-A3B INT4](https://huggingface.co/OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov)~~ | ~~2025-07~~ | ~~15.2 GiB~~ | ~~text~~ | ~~256k~~ | — | — | — | **OOM on 32 GB RAM**: device allocation fails at compile |
| ~~[Gemma 4 26B A4B INT4](https://huggingface.co/Morteza89/gemma-4-26b-a4b-it-int4-ov)~~ | ~~2026-03~~ | ~~14.3 GiB~~ | ~~text, image, audioᵇ~~ | ~~256k~~ | — | — | — | **OOM on 32 GB RAM** (tested 3×): fails during weight upload even with 24 GB free RAM |

³ "PL edits" = decode with **prompt-lookup speculative decoding** on an echo-heavy code-edit
prompt. Measured with a *different prompt* than the Decode column — compare PL values with each
other, not against Decode. ↓ = slower than plain decoding on the same prompt (thinking-mode and
low-echo models lose; see RESEARCH.md Finding 6). "—" = not measured or unsupported (VLM-shaped).

ᵃ "Base released" is the Hugging Face creation date of the *original base model* repo (e.g.
`google/gemma-4-E2B-it`, `Qwen/Qwen2.5-Coder-1.5B-Instruct`), not the OpenVINO conversion date.

ᵇ Modalities and max context are the *model's* capabilities (from each model's `config.json`).
The server currently exposes a **text-only** API and keeps practical context well below the
maximum — KV-cache grows with context and competes with weights for the same shared iGPU
memory. Multimodal IRs run fine text-only through `VLMPipeline`.

The short version of *why* the table looks like this: decode speed on this iGPU is
memory-bandwidth-bound (smaller weights = proportionally faster), the usable model size is
capped well below the driver's ≈50%-of-RAM memory ceiling by compile-time overhead, and
quantization recipe / speculative decoding gains are architecture- and workload-specific.
The full methodology, measurements and conversion playbook live in [RESEARCH.md](RESEARCH.md).

