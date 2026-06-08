# Fair leaderboard sweep — every model through ONE engine (the source-built
# gemma4_unified GenAI in .venv-genai), served solo, benched single-shot by id.
# For each model: tear down servers -> start clean server -> wait ready ->
# run bench_server.py -> tear down. Ends with all servers OFF.
#
# Run (background):  .venv-genai\Scripts\python.exe is the engine python.
$ErrorActionPreference = "Stop"
$root = "C:\git\GitHub\openvino-windows-openai-api"
$py   = "$root\.venv-genai\Scripts\python.exe"
$log  = "$root\bench_results\sweep_driver.log"

function Log($m) {
  $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m
  Add-Content -Path $log -Value $line
}

function Stop-Servers {
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'server\.py' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 2
}

# Served-id == path under models/ . Order: leader first, then by size.
$models = @(
  "HarmenWessels/gemma-4-12B-it-qat-int4-ov",
  "OpenVINO/Qwen3-14B-int4-ov",
  "Echo9Zulu/OmniCoder-9B-int4_sym-ov",
  "OpenVINO/Qwen3-8B-int4-cw-ov",
  "OpenVINO/Qwen2.5-Coder-3B-Instruct-int4-ov"
)

Log "=== sweep start: $($models.Count) models, engine=$py ==="

foreach ($m in $models) {
  Stop-Servers
  $dir = "models/$m"
  if (-not (Test-Path "$root\$dir\openvino_language_model.xml") -and
      -not (Test-Path "$root\$dir\openvino_model.xml")) {
    Log "SKIP $m — IR not found at $dir"; continue
  }

  Log "--- $m : starting server ---"
  $env:MODEL_DIRS = $dir
  $env:DEVICE = "GPU"
  $slog = "$root\bench_results\server_$($m -replace '[\\/]','_').log"
  $srv = Start-Process -FilePath $py -ArgumentList "server.py" `
           -WorkingDirectory $root -PassThru -WindowStyle Hidden `
           -RedirectStandardOutput $slog -RedirectStandardError "$slog.err"

  # wait until /v1/models serves this id (or the server dies / 5 min timeout)
  $ready = $false
  for ($i = 0; $i -lt 150; $i++) {
    if ($srv.HasExited) { Log "SERVER EXITED early (code $($srv.ExitCode)) for $m"; break }
    try {
      $r = Invoke-WebRequest "http://127.0.0.1:8000/v1/models" -TimeoutSec 5 -UseBasicParsing
      if ($r.Content -match [regex]::Escape($m)) { $ready = $true; break }
    } catch {}
    Start-Sleep -Seconds 2
  }
  if (-not $ready) { Log "NOT READY $m — skipping bench"; Stop-Servers; continue }

  Log "--- $m : server ready, running bench ---"
  & $py "$root\scripts\bench_server.py" $m 2>&1 | ForEach-Object { Log "  $_" }
  Log "--- $m : bench done ---"
  Stop-Servers
}

Log "=== sweep complete — servers off ==="
