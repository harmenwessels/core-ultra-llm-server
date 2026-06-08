# Step 2 — combinations. For each combo in combos.yaml: load its role models
# co-resident under the .venv-genai server, set VIRTUAL_ROLES, run bench_run.py
# against virtual/agent over the design-pipeline task types, stop. Servers off
# at end. A combo is benchmarked as a peer entry (provenance records the role map).
#
# NB: combo runs send the task-class decoding to virtual/agent; the routed role
# applies it. Per-role card decoding would need the server to read cards per
# role-call (a deeper change) — documented in benchmark/README.md.
#
#   benchmark/scripts/run_combos.ps1
#   benchmark/scripts/run_combos.ps1 -Combos small-trio -Tasks codegen
param(
  [string]$Tasks = "codegen,edit,agent-loop,analysis",
  [string[]]$Combos = @()
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$py   = "$root\.venv-genai\Scripts\python.exe"
$log  = "$root\benchmark\results\combos.log"
New-Item -ItemType Directory -Force -Path "$root\benchmark\results" | Out-Null

function Log($m) { Add-Content -Path $log -Value ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m) }
function Stop-Servers {
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'server\.py' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 2
}

# combo names: from -Combos, else all keys in combos.yaml
$names = if ($Combos.Count) { $Combos } else {
  & $py -c "import sys,pathlib; sys.path.insert(0, r'$root\benchmark\scripts'); import bench_meta as bm, yaml; print('\n'.join((yaml.safe_load(bm.COMBOS_FILE.read_text())['combos'] or {}).keys()))"
}

Log "=== combos sweep: tasks=$Tasks ==="
foreach ($name in $names) {
  if (-not $name) { continue }
  Stop-Servers
  $env:MODEL_DIRS = ""; $env:VIRTUAL_ROLES = ""
  foreach ($line in (& $py "$root\benchmark\scripts\combo_env.py" $name)) {
    if ($line -match '^MODEL_DIRS=(.*)$')    { $env:MODEL_DIRS = $Matches[1] }
    if ($line -match '^VIRTUAL_ROLES=(.*)$') { $env:VIRTUAL_ROLES = $Matches[1] }
  }
  if (-not $env:MODEL_DIRS) { Log "SKIP combo $name (no roles resolved)"; continue }
  $env:DEVICE = "GPU"
  Log "--- combo $name : MODEL_DIRS=$($env:MODEL_DIRS) ---"
  Log "--- combo $name : VIRTUAL_ROLES=$($env:VIRTUAL_ROLES) ---"
  $slog = "$root\benchmark\results\server_combo_$name.log"
  $srv = Start-Process -FilePath $py -ArgumentList "server.py" -WorkingDirectory $root `
           -PassThru -WindowStyle Hidden -RedirectStandardOutput $slog -RedirectStandardError "$slog.err"
  $ready = $false
  for ($i=0; $i -lt 240; $i++) {
    if ($srv.HasExited) { Log "SERVER EXITED early (code $($srv.ExitCode)) combo $name"; break }
    try { $r = Invoke-WebRequest "http://127.0.0.1:8000/v1/models" -TimeoutSec 5 -UseBasicParsing
          if ($r.Content -match 'virtual/agent') { $ready = $true; break } } catch {}
    Start-Sleep -Seconds 2
  }
  if (-not $ready) { Log "NOT READY combo $name"; Stop-Servers; continue }
  Log "--- combo $name : running bench_run virtual/agent ($Tasks) ---"
  & $py "$root\benchmark\scripts\bench_run.py" "virtual/agent" --combo $name --tasks $Tasks 2>&1 |
    ForEach-Object { Log "  $_" }
  Log "--- combo $name : done ---"
  Stop-Servers
}
Log "=== combos sweep complete — servers off ==="
