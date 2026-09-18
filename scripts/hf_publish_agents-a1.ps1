# Publish Agents-A1-4B (rejected-model IR kept for reproducibility) — 2026-09-18.
# RUN THIS FROM YOUR OWN SHELL, not from the agent: large uploads stall at 0 bytes
# from the agent's process environment. Resumable — if it stops, re-run.
#
#   pwsh -File scripts\hf_publish_agents-a1.ps1
#
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$hf   = Join-Path $root '.venv-convert\Scripts\hf.exe'

# The hf CLI prints "✓" and dies under the console codepage without UTF-8; it also
# colours its output interactively.
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

$jobs = @(
  @{ id = 'HarmenWessels/Agents-A1-4B-int4-asymg128-ov'
     msg = 'Agents-A1-4B int4 asym g128 (base recipe); kept for reproducibility, not recommended for coding' }
)

foreach ($j in $jobs) {
  $dir = Join-Path $root ('models\' + $j.id.Replace('/', '\'))
  if (-not (Test-Path (Join-Path $dir 'README.md'))) { throw "no README.md in $dir" }
  if (Get-ChildItem $dir -Recurse -Directory -Filter model_cache) { throw "model_cache/ present in $dir" }
  Write-Host "`n=== $($j.id)  ($([math]::Round((Get-ChildItem $dir -Recurse -File | Measure-Object Length -Sum).Sum/1GB,2)) GB) ===" -ForegroundColor Cyan
  & $hf upload $j.id $dir . --repo-type model --commit-message $j.msg
  if ($LASTEXITCODE -ne 0) { throw "upload failed for $($j.id) (exit $LASTEXITCODE) — re-run to resume" }
}
Write-Host "`ndone. Next: flip 'published: true' in the Agents-A1 card and re-assemble." -ForegroundColor Green
