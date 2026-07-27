#!/usr/bin/env bash
# Reproducible setup for soundmon. Idempotent — safe to re-run.
# Does NOT download models (run ./download-models.sh after).
#
# soundmon deliberately SHARES ComfyUI with pixelmon: same engine, same venv,
# same launch script, same port. It only adds its own node + models. So if you
# already run pixelmon on this box (or on a render-farm box), this is nearly a
# no-op — link the files, fetch the audio models, done.
#
# Force a vendor with  SOUNDMON_GPU=nvidia|amd|cpu ./install.sh  if needed.
# Pick a specific interpreter with  PYTHON=python3.11 ./install.sh
set -euo pipefail

REPO="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
COMFY="${COMFYUI_DIR:-$HOME/ComfyUI}"

detect_gpu() {
    case "${SOUNDMON_GPU:-}" in nvidia|amd|cpu|mps) echo "$SOUNDMON_GPU"; return;; esac
    if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
        echo mps
    elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo nvidia
    elif [ -x /usr/lib/wsl/lib/nvidia-smi ] && /usr/lib/wsl/lib/nvidia-smi -L >/dev/null 2>&1; then
        echo nvidia
    elif [ -e /dev/kfd ] || command -v rocminfo >/dev/null 2>&1; then
        echo amd
    else
        echo cpu
    fi
}
GPU="$(detect_gpu)"

echo "==> repo:    $REPO"
echo "==> ComfyUI: $COMFY"
echo "==> GPU:     $GPU"

# 1. Engine. If pixelmon already set ComfyUI up, reuse it wholesale.
if [ -x "$COMFY/.venv/bin/python" ]; then
    echo "==> found an existing ComfyUI + venv — reusing it (nothing to build)"
else
    echo "==> no ComfyUI here yet."
    if [ -x "$HOME/pixelmon/install.sh" ]; then
        echo "    pixelmon's install.sh is present and builds exactly the engine soundmon"
        echo "    needs (ComfyUI + venv + the right torch). Run that first:"
        echo "        ~/pixelmon/install.sh"
    else
        echo "    Clone and build ComfyUI first — the quickest path is pixelmon's installer:"
        echo "        git clone https://github.com/grymmjack/pixelmon.git ~/pixelmon"
        echo "        ~/pixelmon/install.sh"
    fi
    echo "    Then re-run this script."
    exit 1
fi

# 2. Link our files into place (this repo stays the source of truth)
link() { ln -sfn "$1" "$2"; echo "   linked $2 -> $1"; }
mkdir -p "$HOME/.local/bin" "$COMFY/custom_nodes"
chmod +x "$REPO/bin/soundmon" "$REPO/download-models.sh"
link "$REPO/soundmon.py"               "$COMFY/soundmon.py"
link "$REPO/custom_nodes/retro_sfx"    "$COMFY/custom_nodes/retro_sfx"
link "$REPO/bin/soundmon"              "$HOME/.local/bin/soundmon"

# 3. Reuse pixelmon's server aliases if you have them and haven't made your own.
if [ ! -f "$REPO/servers.json" ] && [ -f "$HOME/pixelmon/servers.json" ]; then
    cp "$HOME/pixelmon/servers.json" "$REPO/servers.json"
    echo "   copied pixelmon's servers.json (same boxes, same port — the farm is shared)"
fi

# 4. render group — AMD/ROCm only (compute needs /dev/kfd, gated behind this group)
if [ "$GPU" = amd ] && ! id -nG | tr ' ' '\n' | grep -qx render; then
    echo "==> adding $USER to the 'render' group (ROCm GPU access)"
    sudo usermod -aG render "$USER"
    echo "   ⚠  LOG OUT AND BACK IN for this to take effect."
fi

cat <<EOF

✅ install done ($GPU).
   1) ./download-models.sh        # ~5.3 GB (Stable Audio Open 1.0 + T5)
   2) restart ComfyUI so it picks up the retro_sfx node
   3) soundmon "a heavy wooden door creaking open"

   Note: make sure ~/.local/bin is on your PATH.
EOF
