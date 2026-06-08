# Step 1 — single-model role-fitness. For each fleet model: start the .venv-genai
# server SOLO (MODEL_DIRS), wait until /v1/models is ready, run bench_run.py over
# the task types, stop. Servers off at end. Provenance run-records land in
# benchmark/results/runs/.
#
#   benchmark/scripts/run_fleet.ps1                 # full fleet, all tasks
#   benchmark/scripts/run_fleet.ps1 -Tasks codegen  # one task type
#   benchmark/scripts/run_fleet.ps1 -Models "OpenVINO/Qwen3-14B-int4-ov"
param(
  [string]$Tasks = "all",
  [string[]]$Models = @()
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$py   = "$root\.venv-genai\Scripts\python.exe"
$log  = "$root\benchmark\results\fleet.log"
New-Item -ItemType Directory -Force -Path "$root\benchmark\results" | Out-Null

function Log($m) { Add-Content -Path $log -Value ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m) }
function Stop-Servers {
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'server\.py' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 2
}

$fleet = if ($Models.Count) { $Models } else { @(
  "HarmenWessels/gemma-4-E2B-it-qat-int4-ov",
  "HarmenWessels/gemma-4-E4B-it-qat-int4-ov",
  "HarmenWessels/gemma-4-12B-it-qat-int4-ov",
  "OpenVINO/Qwen3-8B-int4-cw-ov",
  "OpenVINO/Qwen3-14B-int4-ov",
  "yangsu0423/Qwen3.5-0.8B-int4-ov",
  "Echo9Zulu/Qwen3.5-2B-int4_sym-ov",
  "Echo9Zulu/OmniCoder-9B-int4_sym-ov",
  "OpenVINO/Qwen2.5-Coder-1.5B-Instruct-int4-ov",
  "OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov",
  "HarmenWessels/granite-4.1-3b-int4-cw-ov",
  "HarmenWessels/granite-4.1-3b-int4-cw-code-ov",   # code-calibrated AWQ+SE — head-to-head vs the wikitext-calibrated cw
  "HarmenWessels/granite-4.1-8b-int4-cw-ov"
) }

Log "=== fleet sweep: $($fleet.Count) models, tasks=$Tasks ==="
foreach ($m in $fleet) {
  Stop-Servers
  $dir = "models/$m"
  if (-not (Test-Path "$root\$dir\openvino_language_model.xml") -and
      -not (Test-Path "$root\$dir\openvino_model.xml")) { Log "SKIP $m (IR not found)"; continue }
  Log "--- $m : starting server ---"
  $env:MODEL_DIRS = $dir; $env:DEVICE = "GPU"
  $slog = "$root\benchmark\results\server_$($m -replace '[\\/]','_').log"
  $srv = Start-Process -FilePath $py -ArgumentList "server.py" -WorkingDirectory $root `
           -PassThru -WindowStyle Hidden -RedirectStandardOutput $slog -RedirectStandardError "$slog.err"
  $ready = $false
  for ($i=0; $i -lt 180; $i++) {
    if ($srv.HasExited) { Log "SERVER EXITED early (code $($srv.ExitCode)) for $m"; break }
    try { $r = Invoke-WebRequest "http://127.0.0.1:8000/v1/models" -TimeoutSec 5 -UseBasicParsing
          if ($r.Content -match [regex]::Escape($m)) { $ready = $true; break } } catch {}
    Start-Sleep -Seconds 2
  }
  if (-not $ready) { Log "NOT READY $m"; Stop-Servers; continue }
  Log "--- $m : running bench_run ($Tasks) ---"
  & $py "$root\benchmark\scripts\bench_run.py" $m --tasks $Tasks 2>&1 | ForEach-Object { Log "  $_" }
  Log "--- $m : done ---"
  Stop-Servers
}
Log "=== fleet sweep complete — servers off ==="
