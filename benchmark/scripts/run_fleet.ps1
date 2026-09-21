# Step 1 — single-model role-fitness. For each fleet model: start the .venv-genai
# server SOLO (MODEL_DIRS), wait until /v1/models is ready, run bench_run.py over
# the task types, stop. Servers off at end. Provenance run-records land in
# benchmark/results/runs/.
#
#   benchmark/scripts/run_fleet.ps1                 # full fleet, all tasks
#   benchmark/scripts/run_fleet.ps1 -Tasks codegen  # one task type
#   benchmark/scripts/run_fleet.ps1 -Models "OpenVINO/Qwen3-14B-int4-ov"
#   benchmark/scripts/run_fleet.ps1 -Device NPU -Models ...   # NPU tables (sym-int4 IRs only)
param(
  [string]$Tasks = "all",
  [string[]]$Models = @(),
  [string]$Venv = ".venv-genai",  # e.g. .venv-genai-231 for an engine A-B; the record's engine field says which ran
  [string]$Device = "GPU"          # GPU (the leaderboard) or NPU (its own tables); recorded in each run's header
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$py   = "$root\$Venv\Scripts\python.exe"
$log  = "$root\benchmark\results\fleet.log"
New-Item -ItemType Directory -Force -Path "$root\benchmark\results" | Out-Null

function Log($m) { Add-Content -Path $log -Value ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m) }
function Stop-Servers {
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'server\.py' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 2
}

# Default fleet from benchmark/fleet.txt (single source of truth, shared with the
# assembler); -Models overrides for targeted re-runs.
$fleet = if ($Models.Count) { $Models } else {
  # strip inline comments too — a line like "owner/model  # note" must yield the id, not
  # a path with the note glued on (24 fleet entries were silently SKIPped on 2026-09-19)
  Get-Content "$root\benchmark\fleet.txt" |
    ForEach-Object { ($_ -split '#', 2)[0].Trim() } |
    Where-Object { $_ }
}

$Device = $Device.ToUpper()
$env:BENCH_DEVICE = $Device
Log "=== fleet sweep: $($fleet.Count) models, tasks=$Tasks, device=$Device ==="
foreach ($m in $fleet) {
  Stop-Servers
  $dir = "models/$m"
  if (-not (Test-Path "$root\$dir\openvino_language_model.xml") -and
      -not (Test-Path "$root\$dir\openvino_model.xml")) { Log "SKIP $m (IR not found)"; continue }
  Log "--- $m : starting server ---"
  $env:MODEL_DIRS = $dir; $env:DEVICE = $Device
  $slog = "$root\benchmark\results\server_$($m -replace '[\\/]','_')$(if ($Device -ne 'GPU') { "_$Device" }).log"
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
