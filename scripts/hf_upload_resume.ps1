# Resumable, self-retrying HuggingFace upload for large OpenVINO IRs.
#
# Why this exists: on this machine a large `hf upload` stops moving bytes after
# roughly half an hour. The transfer IS resumable — every relaunch re-hashes and
# skips what already landed — so the fix is to kill a stalled attempt and start
# another until the repo matches the local directory. This script does that, and
# checks completion against the hub rather than trusting an exit code.
#
# It must be run from YOUR shell: uploads launched from the agent's process
# environment send 0 bytes (repo creation and small files still work, so it is
# not auth). See the "HF XET stalls" memory note.
#
# Usage:
#   .\scripts\hf_upload_resume.ps1                       # all pending Ministral builds
#   .\scripts\hf_upload_resume.ps1 -Models 8B-Reasoning  # a subset
#   .\scripts\hf_upload_resume.ps1 -StallMinutes 20      # kill sooner

param(
    [string[]] $Models = @("3B-Instruct", "3B-Reasoning", "8B-Instruct",
                           "8B-Reasoning", "14B-Reasoning"),
    [int] $StallMinutes = 35,
    [int] $MaxAttempts  = 25
)

$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
$hf  = Join-Path $repoRoot ".venv-convert\Scripts\hf.exe"
$py  = Join-Path $repoRoot ".venv-convert\Scripts\python.exe"
$log = Join-Path $repoRoot "benchmark\results\hf_upload_resume.log"

function Write-Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line
    $line | Out-File $log -Append -Encoding utf8
}

# Ask the hub whether every local file is present at the right size.
function Test-UploadComplete($repoId, $dir) {
    $checker = @"
import json, os, sys, urllib.request
repo, base = sys.argv[1], sys.argv[2]
try:
    d = json.load(urllib.request.urlopen(
        f'https://huggingface.co/api/models/{repo}?blobs=true', timeout=30))
except Exception:
    print('INCOMPLETE'); sys.exit(0)
remote = {s['rfilename']: (s.get('size') or 0) for s in d.get('siblings', [])}
local = {f: os.path.getsize(os.path.join(base, f))
         for f in os.listdir(base) if os.path.isfile(os.path.join(base, f))}
bad = [f for f, sz in local.items()
       if f not in remote or (remote[f] and remote[f] != sz)]
print('COMPLETE' if not bad else 'INCOMPLETE ' + ','.join(sorted(bad)[:3]))
"@
    $tmp = Join-Path $env:TEMP "hf_check_$PID.py"
    $checker | Out-File $tmp -Encoding utf8
    $result = & $py $tmp $repoId $dir 2>$null
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    return ($result -match '^COMPLETE')
}

foreach ($m in $Models) {
    $repoId = "HarmenWessels/Ministral-3-$m-int4-symg128-ov"
    $dir    = Join-Path $repoRoot "models\HarmenWessels\Ministral-3-$m-int4-symg128-ov"
    if (-not (Test-Path $dir)) { Write-Log "SKIP $m - no local dir"; continue }

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        if (Test-UploadComplete $repoId $dir) {
            Write-Log "DONE $m (verified against the hub)"
            break
        }
        Write-Log "$m - attempt $attempt/$MaxAttempts (resumes where it stopped)"
        # Start-Process joins -ArgumentList with spaces WITHOUT quoting, so any
        # argument containing a space (the commit message, a path under
        # "Program Files") must carry its own quotes or hf sees it as extra
        # positional arguments.
        $msg = "Ministral-3-$m int4 sym g128 AWQ+SE for OpenVINO GenAI 2026.3+"
        $argList = @("upload", $repoId, "`"$dir`"", ".", "--repo-type", "model",
                     "--commit-message", "`"$msg`"")
        $started = Get-Date
        $p = Start-Process -FilePath $hf -PassThru -NoNewWindow -ArgumentList $argList
        if (-not $p.WaitForExit($StallMinutes * 60 * 1000)) {
            Write-Log "$m - no exit after $StallMinutes min, killing and retrying"
            try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
            Start-Sleep -Seconds 10
        } else {
            $secs = [int]((Get-Date) - $started).TotalSeconds
            Write-Log "$m - exited after ${secs}s (code $($p.ExitCode)); verifying"
            # A stall takes minutes; an instant non-zero exit is a real error
            # (bad arguments, auth, missing repo). Retrying that 25 times just
            # hides the message hf already printed above.
            if ($p.ExitCode -ne 0 -and $secs -lt 60) {
                Write-Log "$m - failed immediately, not a stall. See the hf error above; skipping."
                break
            }
            Start-Sleep -Seconds 5
        }
    }
    if (-not (Test-UploadComplete $repoId $dir)) {
        Write-Log "GAVE UP on $m after $MaxAttempts attempts - rerun the script to continue"
    }
}
Write-Log "run finished"
