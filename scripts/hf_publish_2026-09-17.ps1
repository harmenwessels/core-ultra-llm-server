# Publish / republish six IRs to the HarmenWessels hub account (2026-09-17).
# RUN THIS FROM YOUR OWN SHELL, not from the agent: large uploads stall at 0 bytes
# from the agent's process environment (hf-anonymous-xet-stalls). Resumable — if it
# stops, re-run; hf re-hashes and skips files already pushed.
#
#   pwsh -File scripts\hf_publish_2026-09-17.ps1            # all six
#   pwsh -File scripts\hf_publish_2026-09-17.ps1 -Only spark # just the two new Sparks
#
param([ValidateSet('all', 'spark', 'republish')] [string]$Only = 'all')
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$hf   = Join-Path $root '.venv-convert\Scripts\hf.exe'

# The hf CLI prints "✓" and other non-cp1252 glyphs; on a Windows console with the
# default codepage Python dies with "'charmap' codec can't encode character" before
# printing anything useful — so force UTF-8 for every hf call in this script. It also
# colours its output interactively (ANSI escapes between "User=" and the name).
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:NO_COLOR = '1'
$raw = (& $hf auth whoami 2>&1 | ForEach-Object { "$_" }) -join "`n"
$clean = $raw -replace "`e\[[0-9;]*[A-Za-z]", ''
$who = if ($clean -match '(?im)^\s*user\s*[=:]\s*(\S+)') { $Matches[1] } else { '' }
if ($who -ne 'HarmenWessels') {
  Write-Host "could not confirm the hf account. Raw output:`n$clean" -ForegroundColor Yellow
  throw "hf auth is '$who', expected HarmenWessels (hf auth switch)"
}

$new = @(
  @{ id = 'HarmenWessels/Spark-X2.5-4B-int4-symg128-ov';   msg = 'Spark-X2.5-4B int4 sym g128 AWQ+SE, g_proj fp16 (iGPU fused-int4 kernel fix), id-0 BOS tokenizer fix' },
  @{ id = 'HarmenWessels/Spark-X2.5-1.7B-int4-symg128-ov'; msg = 'Spark-X2.5-1.7B int4 sym g128 AWQ+SE, g_proj fp16, id-0 BOS tokenizer fix' }
)
$republish = @(
  @{ id = 'HarmenWessels/Seed-Coder-8B-Instruct-int4-cw-ov'; msg = 'Regenerate tokenizer IR: restore dropped id-0 BOS (openvino_tokenizers 2026.3 bug); README with re-benchmark 25/26' },
  @{ id = 'HarmenWessels/K2-Horizon-0.9B-int4-symg128-ov';  msg = 'Regenerate tokenizer IR: \uXXXX -> \x{XXXX} in the Split regex restores the whitespace lookahead; README with re-benchmark 11/26' },
  @{ id = 'HarmenWessels/K2-Horizon-3.7B-int4-symg128-ov';  msg = 'Regenerate tokenizer IR: \uXXXX -> \x{XXXX} in the Split regex restores the whitespace lookahead; README with re-benchmark 24/26' },
  @{ id = 'HarmenWessels/K2-Horizon-7B-int4-symg128-ov';    msg = 'Regenerate tokenizer IR: \uXXXX -> \x{XXXX} in the Split regex restores the whitespace lookahead; README with re-benchmark 24/26' }
)
$jobs = switch ($Only) { 'spark' { $new } 'republish' { $republish } default { $new + $republish } }

foreach ($j in $jobs) {
  $dir = Join-Path $root ('models\' + $j.id.Replace('/', '\'))
  if (-not (Test-Path (Join-Path $dir 'README.md'))) { throw "no README.md in $dir" }
  if (Get-ChildItem $dir -Recurse -Directory -Filter model_cache) { throw "model_cache/ present in $dir" }
  Write-Host "`n=== $($j.id)  ($([math]::Round((Get-ChildItem $dir -Recurse -File | Measure-Object Length -Sum).Sum/1GB,2)) GB) ===" -ForegroundColor Cyan
  & $hf upload $j.id $dir . --repo-type model --commit-message $j.msg
  if ($LASTEXITCODE -ne 0) { throw "upload failed for $($j.id) (exit $LASTEXITCODE) — re-run to resume" }
}
Write-Host "`nall uploads done. Next: flip 'published: true' in the two Spark cards and re-assemble the leaderboard." -ForegroundColor Green
