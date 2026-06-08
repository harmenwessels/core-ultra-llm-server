# Full unattended benchmark run (for an overnight slot). Step 1 single-model
# fleet, then (optionally) Step 2 combos, then regenerate the leaderboard tables.
# Leaves servers OFF. Each sub-sweep already stops servers between models.
#
#   benchmark/scripts/run_all.ps1            # Step 1 + assemble
#   benchmark/scripts/run_all.ps1 -Combos    # Step 1 + Step 2 + assemble
param([switch]$Combos)
$ErrorActionPreference = "Continue"   # one model failing must not abort the night
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$py   = "$root\.venv-genai\Scripts\python.exe"
$log  = "$root\benchmark\results\run_all.log"
function Log($m) { Add-Content -Path $log -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m) }

Log "===== run_all START (combos=$Combos) ====="
& "$root\benchmark\scripts\run_fleet.ps1"
Log "Step 1 fleet done"
if ($Combos) {
  & "$root\benchmark\scripts\run_combos.ps1"
  Log "Step 2 combos done"
}
& $py "$root\benchmark\scripts\assemble_leaderboard.py"
Log "assembled leaderboard -> benchmark/README.md + root README"
# safety: ensure nothing is left serving
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'server\.py' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Log "===== run_all COMPLETE — servers off ====="
