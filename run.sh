#!/usr/bin/env bash
# Launcher for the VoxCPM TTS Tool.
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
#   ./run.sh                       # default: --port 8808 --host 0.0.0.0
#   ./run.sh --port 9000           # custom port
#   ./run.sh --share               # public Gradio tunnel
#   ./run.sh --root /some/dir      # alt project root
#
# Re-run anytime: idempotent.

set -euo pipefail
cd "$(dirname "$0")"

TARGET_PY="3.12"
VENV_DIR=".venv"

# ---- ffmpeg auto-detect + (opt-in) install ------------------------------
# voxcpm uses torchaudio to load reference audio. torchaudio prefers ffmpeg
# as backend; without it some formats fail or fall back to a slower path.
# ffmpeg is OPTIONAL — we ask the user, default to skip after 3 seconds.

install_ffmpeg() {
    if [[ "${OSTYPE:-}" == "darwin"* ]] && command -v brew >/dev/null 2>&1; then
        brew install ffmpeg && return 0
    fi
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq && sudo apt-get install -y ffmpeg && return 0
    fi
    if command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y ffmpeg && return 0
    fi
    if command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --noconfirm ffmpeg && return 0
    fi
    if command -v zypper >/dev/null 2>&1; then
        sudo zypper install -y ffmpeg && return 0
    fi
    cat >&2 <<'EOF'
ffmpeg auto-install: no supported package manager found. To install manually:
  Linux (apt):    sudo apt-get install ffmpeg
  Linux (dnf):    sudo dnf install ffmpeg
  Linux (pacman): sudo pacman -S ffmpeg
  macOS:          brew install ffmpeg
EOF
    return 1
}

prompt_ffmpeg() {
    if command -v ffmpeg >/dev/null 2>&1; then
        return  # already installed, nothing to do
    fi
    echo
    echo "==> ffmpeg not detected. It's optional but recommended for faster"
    echo "    audio loading. Install now? [y/N] (default N in 3s)"
    if read -t 3 -r -p "    > " ans; then
        case "${ans:-n}" in
            [yY]*) install_ffmpeg || true ;;
            *)     echo "    Skipped. App will use torchaudio's built-in backend." ;;
        esac
    else
        echo
        echo "    Timeout — skipping ffmpeg install."
    fi
}

prompt_ffmpeg

venv_python() {
    if [[ -f "$VENV_DIR/Scripts/python.exe" ]]; then
        echo "$VENV_DIR/Scripts/python.exe"
    elif [[ -f "$VENV_DIR/bin/python" ]]; then
        echo "$VENV_DIR/bin/python"
    else
        echo ""
    fi
}

# ---- Path A: uv (recommended) -------------------------------------------

if command -v uv >/dev/null 2>&1; then
    # Silence the "Failed to hardlink" warning when uv's cache and the venv
    # live on different filesystems (common on Windows with E:/C: split, or
    # cross-mount Linux). uv already falls back to copy in that case; this
    # just makes the choice explicit.
    export UV_LINK_MODE=copy
    echo "==> Using uv ($(uv --version))"

    if [[ ! -d "$VENV_DIR" ]]; then
        echo "==> Creating $VENV_DIR with Python $TARGET_PY (uv will download it if missing) ..."
        uv venv --python "$TARGET_PY" "$VENV_DIR"
    fi

    VENV_PY=$(venv_python)
    if [[ -z "$VENV_PY" ]]; then
        echo "ERROR: $VENV_DIR was created but no python executable found." >&2
        exit 1
    fi

    echo "==> Installing dependencies from requirements.txt via uv ..."
    VIRTUAL_ENV="$(pwd)/$VENV_DIR" uv pip install -r requirements.txt
    echo "==> Installing local package (no deps) ..."
    VIRTUAL_ENV="$(pwd)/$VENV_DIR" uv pip install -e . --no-deps

    echo "==> Launching VoxCPM TTS Tool ..."
    echo "    First launch will download ~6 GB of model files into pretrained_models/."
    echo
    exec "$VENV_PY" app.py "$@"
fi

# ---- Path B: system Python 3.10-3.12 ------------------------------------

find_python() {
    for cmd in python3.12 python3.11 python3.10 py python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            local version
            version=$("$cmd" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null) || continue
            case "$version" in
                3.10|3.11|3.12)
                    echo "$cmd"
                    return 0
                    ;;
            esac
        fi
    done
    return 1
}

if PYTHON=$(find_python); then
    echo "==> Using $PYTHON ($("$PYTHON" --version)) -- uv not found, falling back"

    if [[ ! -d "$VENV_DIR" ]]; then
        echo "==> Creating $VENV_DIR ..."
        "$PYTHON" -m venv "$VENV_DIR"
    fi

    VENV_PY=$(venv_python)
    if [[ -z "$VENV_PY" ]]; then
        echo "ERROR: $VENV_DIR was created but no python executable found." >&2
        exit 1
    fi

    echo "==> Upgrading pip ..."
    "$VENV_PY" -m pip install --upgrade pip

    echo "==> Installing dependencies from requirements.txt ..."
    "$VENV_PY" -m pip install -r requirements.txt
    echo "==> Installing local package (no deps) ..."
    "$VENV_PY" -m pip install -e . --no-deps

    echo "==> Launching VoxCPM TTS Tool ..."
    echo "    First launch will download ~6 GB of model files into pretrained_models/."
    echo
    exec "$VENV_PY" app.py "$@"
fi

# ---- Path C: nothing usable ---------------------------------------------

cat >&2 <<'EOF'
ERROR: No compatible Python found.

This project needs Python 3.10, 3.11, or 3.12. Python 3.13 is not yet
supported (voxcpm's transitive dep 'editdistance' has no 3.13 wheel).

Please install uv -- a single binary that will auto-download a standalone
Python 3.12 for this project without touching your system Python:

  Linux / macOS:
    curl -LsSf https://astral.sh/uv/install.sh | sh

  Windows (PowerShell):
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

After installing uv, re-run this script.
EOF
exit 1
