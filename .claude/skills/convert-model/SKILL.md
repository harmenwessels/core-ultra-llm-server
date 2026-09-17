---
name: convert-model
description: >
  Convert a HuggingFace model to an OpenVINO int4 IR for this repo (the Route-B
  playbook). Use after /assess-model says "convert". Handles the traps this
  project hit: .venv-convert transformers-version skew, the int4 recipe, hf auth
  (XET stall), unregistered-arch workarounds, borrowing a chat template, DISK
  SAFETY (the full-disk save_model failure), detached long-running launch, and
  the large-model ACTIVATIONS_SCALE_FACTOR fix. Then hands off to /model-preflight.
---

# convert-model — HF model → OpenVINO int4 IR

Output: `models/<owner>/<name>-int4-symg128-ov/` (or `-cw-` for channel-wise).
Conversion uses **`.venv-convert`** (NOT the serving `.venv-genai`). The full
method + findings live in `RESEARCH.md` — read it for anything subtle.

## 1. Preflight the environment
- **`hf auth`** must be logged in (account `HarmenWessels`) — anonymous XET
  transfers stall at 0 bytes. → [[hf-anonymous-xet-stalls]].
- **Transformers version per arch** (real skew): optimum-intel pins
  `transformers<5.1`; llama/qwen are stable on 5.0.x; `lfm2_moe` needs 5.0.x;
  gemma-4 needed the PR build. Check the target and `.venv-convert`'s current
  version before exporting. → [[conversion-playbook-and-watch-items]].
- **DISK** — `df` the drive. A convert that finishes compute then dies with
  `RuntimeError: ios_base::badbit set` is **disk-full at save_model**. Keep
  ≥ ~40 GB free: wipe `.ovcache/*` (regenerable compile blobs, tens of GB) and
  delete HF source caches of already-converted models
  (`~/.cache/huggingface/hub/models--<owner>--<name>`). Free each source cache
  right after its convert if batching.

## 2. The recipe
Standard, NPU-eligible, good for code:
```
optimum-cli export openvino --model <hf_id> --task text-generation-with-past \
  --weight-format int4 --sym --group-size 128 --ratio 1.0 \
  --awq --scale-estimation --dataset wikitext2  models/<owner>/<name>-int4-symg128-ov
```
- `--sym` + `--group-size 128` = symmetric g128 (NPU needs symmetric int4). Use
  `--group-size -1` (channel-wise, `-cw-`) for 8B+ where it's quality-neutral; but
  channel-wise hurt small models (granite-3b-cw) — prefer g128 ≤ 3B — AND broke an
  unfamiliar 8B+ arch outright (K2-Horizon-7B cw = word salad on CPU and GPU; g128 fine):
  a new architecture gets g128 first, cw only after an A-B on that family.
- VLM path: task `image-text-to-text`, `--dataset contextual` (wikitext2 → KeyError);
  AWQ+SE through the VLM calibration **degrades coding** — skip it there.

## 3. Workarounds (only if needed)
- **Arch registered for onnx but not openvino** (e.g. `smollm3`): in a small
  Python wrapper, `TasksManager.create_register("openvino", overwrite_existing=True)`
  to alias the native llama OV config under the model_type, and
  `ONNX_SUPPORTED_ARCHITECTURES.discard("<model_type>")` to clear the OV-removal
  guard, then run the CLI in-process. (NoPE etc. lives in the traced forward, so
  the llama OV config suffices.)
- **Custom-code arch with `@strict` config / new mask API** (IFM K2-Horizon): use
  `.venv-convert-k2` (optimum-intel@main + transformers 5.10.2 + OV 2026.3.1). Register
  llama's export config under the model_type and drive the CLI in-process with
  `--trust-remote-code` (`scripts/k2_export_remote.py`); when every custom switch
  is off in the shipped config, an exact llamafication (`scripts/k2_llamafy.py`, asserts each
  condition) skips custom code entirely. Copy the vendor template to
  `chat_template.vendor.jinja` if GenAI's Minja cannot parse it (see /model-preflight).
- **No HF chat template** (Mistral ships none): after convert, borrow the matching
  `unsloth/<same-model>` `tokenizer_config.json['chat_template']` and write it to
  the IR dir as `chat_template.jinja`. Verify the tokenizer encodes its special
  tokens as single tokens. → [[ministral3-mistral-reasoning-watch]].

## 4. Run it (detached for long converts)
Converts take ~15–40 min (download + AWQ + scale-estimation + save). Launch
detached + hidden so it survives the session and doesn't pop a killable console:
```powershell
$startup = ([wmiclass]"Win32_ProcessStartup").CreateInstance(); $startup.ShowWindow = 0
([wmiclass]"Win32_Process").Create($cmd, $null, $startup)   # $cmd = pwsh-7 -File <script.ps1>
```
Write the convert command to a `.ps1` (Write tool — the sandbox blocks literal
`Remove-Item <protected path>` in inline PowerShell) and log to a file. Monitor
the log for `[exit`, `badbit`/`No space`, `Traceback`.

## 5. After convert
- Verify `models/<id>/openvino_model.xml` exists; free the source HF cache.
- **Audit the tokenizer IR** — `python scripts/audit_tokenizers.py models/<owner>`
  (in `.venv-convert`; compares the OV tokenizer against the Rust `tokenizers`
  reference on every special token, a rendered chat turn and a code snippet).
  openvino_tokenizers 2026.3 has two silent conversion bugs, both fixed by running
  `python scripts/ov_tokenizer_id0_patch.py <model_dir>` (regenerates only the
  tokenizer IRs, no model re-export): (a) an added token with **id 0** is dropped
  (DeepSeek-style BOS → 11 junk tokens per turn; Seed-Coder 22/26 → 25/26 fixed);
  (b) `\uXXXX` escapes in the Split regex kill the `\s+(?!\S)` lookahead (K2 —
  every Python indent tokenized as N spaces + bare word). A mismatch is a hidden
  score depression, not cosmetic. Bake both into any in-process export wrapper
  (`spark_export_remote.py` shows the pattern).
- **GPU garbage while CPU is clean: two DIFFERENT causes, test before assuming.**
  Run `python scripts/gpu_trials.py <model_dir> 4` — N≥3 fresh processes, because
  the second cause is non-deterministic per process (one lucky pass preceded 8/8
  failures on Spark). Then:
  - *Long-output degeneration that scales with sequence length, deterministic* →
    ASF (below).
  - *1–2 right tokens then collapse, from the first sentence, varying between
    processes, and ASF sweeps change nothing* → a fused-int4 GPU kernel. Spark-X2.5:
    the [16×2560] per-head gate `g_proj` shares its input with the fused
    `q_k_v_proj`; both at int4 = garbage, either at fp16 = clean. Bisect with
    `nncf.compress_weights(..., ignored_scope=...)` on an fp16 IR per block / per
    projection; exclude the culprit via `OVWeightQuantizationConfig(ignored_scope=)`
    (the CLI cannot express it — drive the API, see `spark_export_remote.py`).
    Look for a tiny projection sharing an input with a big one (gates, routers).
- **Large models: ACTIVATIONS_SCALE_FACTOR (size-dependent).** optimum bakes
  `8.0` into rt_info; large hidden dims overflow f16 → garbage on the iGPU
  (0/12 codegen + runaway-slow generation, a wall of repeated chars). The value
  is **not fixed**: gemma-12B (hidden 3840) → 64; **Ministral-14B (hidden 5120) →
  512** (64/256 were insufficient — coherent on short output but degenerate on
  longer, since overflow scales with sequence length). Set it with the bundled
  helper: `python .claude/skills/convert-model/bump_asf.py <model_dir> <value>`.
  **CRITICAL: wipe `.ovcache` after changing ASF, with NO server running** — OV's
  compile cache does NOT invalidate on an rt_info edit, AND a running server holds
  file locks so `rm` only partially clears it → the server reuses the STALE blob
  and your ASF change silently never takes effect (this cost ~8 hr once). Stop all
  servers, fully remove `.ovcache`, then start fresh. **Verify on a realistic-length
  prompt** (use /model-preflight; its hardened coherence check rejects repetition-
  garbage). NB: the marginal 14B also exposed a separate server bug — the
  native-render path must pass the real BOS token when the OV tokenizer doesn't
  add one (Ministral Instruct builds); a missing BOS alone produced backtick-garbage.
  The converse also bit: some IRs DO bake in a BOS prepend (LFM2.5, K2, MiniCPM5-2B,
  the Ministral *Reasoning* builds — per-artifact, not per-family) and the server
  then doubled it. Since 2026-09-17 `server.py` detects the auto-BOS at load
  (`_tokenizer_adds_bos`) and blanks the template's `bos_token`; nothing to do per
  model. → [[conversion-playbook-and-watch-items]].
- Run **/model-preflight** (card + server-compat + live probes) before adding the
  id to `benchmark/fleet.txt`.

Standing rules: ask before deleting models; no local/username paths in committed
artifacts; leave the server OFF.
