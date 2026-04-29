# PowerShell launcher for the VoxCPM TTS Tool (Windows-native equivalent of run.sh).
#
# Strategy:
#   1. If `uv` is on PATH (recommended): use uv to provision Python 3.12 + venv + deps.
#      uv will download a standalone Python 3.12 if you don't have one -- no admin needed.
#   2. Else: fall back to a system-installed Python 3.10/3.11/3.12 + stdlib venv + pip.
#   3. Else: print install instructions for both options and exit.
#
# Models are resolved and downloaded inside app.py at startup.
#
# All arguments are forwarded to app.py:
#   .\run.ps1                       # default: --port 8808 --host 127.0.0.1
#   .\run.ps1 --port 9000           # custom port
#   .\run.ps1 --share               # public Gradio tunnel
#   .\run.ps1 --root C:\some\dir    # alt project root
#
# Re-run anytime: idempotent.
#
# Note: this is the APP launcher. The separate start.ps1 in this directory
# is the project's Mutagen sync script -- different purpose, do not confuse.

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$TargetPy = '3.12'
$VenvDir = '.venv'
$VenvPy = Join-Path $VenvDir 'Scripts\python.exe'

# ---- ffmpeg auto-detect + (opt-in) install ------------------------------
# voxcpm uses torchaudio to load reference audio. torchaudio prefers ffmpeg
# as backend; without it some formats fail or fall back to a slower path.
# ffmpeg is OPTIONAL — we ask the user, default to skip after 3 seconds.

function Read-HostWithTimeout {
    param([string]$Prompt, [int]$Seconds = 3)
    Write-Host $Prompt -NoNewline
    $deadline = [DateTime]::UtcNow.AddSeconds($Seconds)
    $buffer = ''
    while ([DateTime]::UtcNow -lt $deadline) {
        if ([Console]::KeyAvailable) {
            $k = [Console]::ReadKey($true)
            if ($k.Key -eq [ConsoleKey]::Enter) { Write-Host ''; return $buffer }
            $buffer += $k.KeyChar
            Write-Host -NoNewline $k.KeyChar
        }
        Start-Sleep -Milliseconds 50
    }
    Write-Host '   [timeout]'
    return $null
}

function Install-FFmpeg {
    $ok = $false
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host '==> Trying: winget install Gyan.FFmpeg'
        & winget install --id=Gyan.FFmpeg -e `
            --silent --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) { $ok = $true }
    }
    if (-not $ok -and (Get-Command choco -ErrorAction SilentlyContinue)) {
        Write-Host '==> Trying: choco install ffmpeg-full'
        & choco install ffmpeg-full -y
        if ($LASTEXITCODE -eq 0) { $ok = $true }
    }
    if ($ok) {
        # winget/choco modify HKLM/HKCU PATH but the current shell is stale.
        $env:Path = `
            [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + `
            [Environment]::GetEnvironmentVariable('Path', 'User')
        Write-Host '==> ffmpeg installed.'
    } else {
        Write-Warning @'
ffmpeg install failed (winget/choco unavailable). To install manually:
  winget install Gyan.FFmpeg
  choco install ffmpeg-full
  or download from https://ffmpeg.org/download.html
'@
    }
}

function Prompt-FFmpegInstall {
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { return }
    Write-Host ''
    Write-Host '==> ffmpeg not detected. It is optional but recommended for faster'
    Write-Host '    audio loading. Install now? [y/N] (default N in 3s)'
    $ans = Read-HostWithTimeout '    > ' 3
    if ($ans -and $ans -match '^[Yy]') {
        Install-FFmpeg
    } else {
        Write-Host '    Skipped. App will use torchaudio''s built-in backend.'
    }
}

Prompt-FFmpegInstall

# ---- Path A: uv (recommended) -------------------------------------------

if (Get-Command uv -ErrorAction SilentlyContinue) {
    # Silence "Failed to hardlink" warnings when uv's cache and the venv
    # live on different filesystems (common on Windows with C:/E: split).
    # uv already falls back to copy in that case; this just makes it explicit.
    $env:UV_LINK_MODE = 'copy'
    $uvVersion = (& uv --version) -join ' '
    Write-Host "==> Using uv ($uvVersion)"

    if (-not (Test-Path $VenvDir)) {
        Write-Host "==> Creating $VenvDir with Python $TargetPy (uv will download it if missing) ..."
        & uv venv --python $TargetPy $VenvDir
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    if (-not (Test-Path $VenvPy)) {
        Write-Error "$VenvDir was created but no python.exe found at $VenvPy"
        exit 1
    }

    Write-Host '==> Installing dependencies from requirements.txt via uv ...'
    $env:VIRTUAL_ENV = (Resolve-Path $VenvDir).Path
    & uv pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host '==> Installing local package (no deps) ...'
    & uv pip install -e . --no-deps
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host '==> Launching VoxCPM TTS Tool ...'
    Write-Host '    First launch will download ~6 GB of model files into pretrained_models\.'
    Write-Host ''
    & $VenvPy app.py @args
    exit $LASTEXITCODE
}

# ---- Path B: system Python 3.10–3.12 ------------------------------------

function Find-Python {
    $candidates = @('python3.12', 'python3.11', 'python3.10', 'py', 'python3', 'python')
    foreach ($cmd in $candidates) {
        $exe = Get-Command $cmd -ErrorAction SilentlyContinue
        if (-not $exe) { continue }

        if ($cmd -eq 'py') {
            foreach ($v in @('3.12', '3.11', '3.10')) {
                try {
                    $version = & py "-$v" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>$null
                    if ($LASTEXITCODE -eq 0 -and $version -match '^(3\.10|3\.11|3\.12)$') {
                        return @{ Cmd = 'py'; Args = @("-$v") }
                    }
                } catch {}
            }
            continue
        }

        try {
            $version = & $cmd -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>$null
            if ($LASTEXITCODE -eq 0 -and $version -match '^(3\.10|3\.11|3\.12)$') {
                return @{ Cmd = $cmd; Args = @() }
            }
        } catch {}
    }
    return $null
}

$python = Find-Python
if ($python) {
    $pyDisplay = (& $python.Cmd @($python.Args + '--version')) -join ' '
    Write-Host "==> Using $($python.Cmd) $($python.Args -join ' ') ($pyDisplay) -- uv not found, falling back"

    if (-not (Test-Path $VenvDir)) {
        Write-Host "==> Creating $VenvDir ..."
        & $python.Cmd @($python.Args + '-m', 'venv', $VenvDir)
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    if (-not (Test-Path $VenvPy)) {
        Write-Error "$VenvDir was created but no python.exe found at $VenvPy"
        exit 1
    }

    Write-Host '==> Upgrading pip ...'
    & $VenvPy -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host '==> Installing dependencies from requirements.txt ...'
    & $VenvPy -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host '==> Installing local package (no deps) ...'
    & $VenvPy -m pip install -e . --no-deps
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host '==> Launching VoxCPM TTS Tool ...'
    Write-Host '    First launch will download ~6 GB of model files into pretrained_models\.'
    Write-Host ''
    & $VenvPy app.py @args
    exit $LASTEXITCODE
}

# ---- Path C: nothing usable ---------------------------------------------

Write-Error @'
No compatible Python found.

This project needs Python 3.10, 3.11, or 3.12. Python 3.13 is not yet
supported (voxcpm's transitive dep 'editdistance' has no 3.13 wheel).

Please install uv -- a single binary that will auto-download a standalone
Python 3.12 for this project without touching your system Python:

  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

After installing uv, re-run this script.
'@
exit 1
