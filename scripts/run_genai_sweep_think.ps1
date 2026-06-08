# Think-mode variant — re-runs the smaller THINK-CAPABLE models with reasoning
# ON (reasoning_effort=medium) at the SAME 3072 budget, to measure thinking's
# effect on the score vs their nothink rows. Coder-3B and OmniCoder-9B are
# excluded (no thinking machinery). Appends to genai_server_castings.jsonl with
# a "-genai-think-3072" label so think/nothink rank as distinct rows.
$ErrorActionPreference = "Stop"
$root = "C:\git\GitHub\openvino-windows-openai-api"
$py   = "$root\.venv-genai\Scripts\python.exe"
$log  = "$root\bench_results\sweep_driver.log"

function Log($m) {
  Add-Content -Path $log -Value ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m)
}
function Stop-Servers {
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'server\.py' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Seconds 2
}

$models = @(
  "HarmenWessels/gemma-4-E2B-it-qat-int4-ov",
  "HarmenWessels/gemma-4-E4B-it-qat-int4-ov",
  "OpenVINO/Qwen3-8B-int4-cw-ov"
)

Log "=== THINK-mode sweep (reasoning_effort=medium, 3072): $($models.Count) models ==="

foreach ($m in $models) {
  Stop-Servers
  $dir = "models/$m"
  if (-not (Test-Path "$root\$dir\openvino_language_model.xml") -and
      -not (Test-Path "$root\$dir\openvino_model.xml")) {
    Log "SKIP $m — IR not found at $dir"; continue
  }
  Log "--- THINK $m : starting server ---"
  $env:MODEL_DIRS = $dir
  $env:DEVICE = "GPU"
  $env:THINK = "1"
  $slog = "$root\bench_results\server_$($m -replace '[\\/]','_')_think.log"
  $srv = Start-Process -FilePath $py -ArgumentList "server.py" `
           -WorkingDirectory $root -PassThru -WindowStyle Hidden `
           -RedirectStandardOutput $slog -RedirectStandardError "$slog.err"
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
  Log "--- THINK $m : server ready, running bench ---"
  & $py "$root\scripts\bench_server.py" $m 2>&1 | ForEach-Object { Log "  $_" }
  Log "--- THINK $m : bench done ---"
  Stop-Servers
}
Remove-Item Env:\THINK -ErrorAction SilentlyContinue
Log "=== THINK-mode sweep complete — servers off ==="
