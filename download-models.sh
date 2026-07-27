#!/usr/bin/env bash
# Download the models soundmon needs into ~/ComfyUI/models/.
# Public mirror on Hugging Face — no login/token required. ~5.3 GB total.
#
# NOTE: the *official* repo (stabilityai/stable-audio-open-1.0) is gated — it
# returns 401 without an HF token + accepting the license in a browser. We pull
# from an ungated byte-identical mirror so this stays one-command like pixelmon.
# To use the official repo instead:  HF_TOKEN=hf_xxx ./download-models.sh --official
set -euo pipefail

COMFY="${COMFYUI_DIR:-$HOME/ComfyUI}"
CKPT="$COMFY/models/checkpoints"
TENC="$COMFY/models/text_encoders"
mkdir -p "$CKPT" "$TENC"

REPO="RedbeardNZ/stable-audio-open-1.0"          # ungated mirror
AUTH=()
if [ "${1:-}" = "--official" ]; then
    REPO="stabilityai/stable-audio-open-1.0"
    : "${HF_TOKEN:?--official needs HF_TOKEN (and accept the license on the model page)}"
    AUTH=(-H "Authorization: Bearer $HF_TOKEN")
fi

get() {  # url  dest
    local url="$1" dest="$2"
    if [ -f "$dest" ]; then echo "✓ already have $(basename "$dest")"; return; fi
    echo "↓ downloading $(basename "$dest") ..."
    # Download to .part and rename only on success, so an interrupted transfer
    # (killed shell, full disk, dropped WSL) can never leave a truncated file
    # sitting at the real path — where the check above would treat it as
    # complete on the next run and hand ComfyUI a corrupt checkpoint. `-C -`
    # resumes an existing .part instead of restarting from zero.
    #
    # ${AUTH[@]+"${AUTH[@]}"} not "${AUTH[@]}": macOS ships bash 3.2, where
    # expanding an EMPTY array under `set -u` is an "unbound variable" fatal
    # error. This idiom expands to nothing when unset and is safe in 3.2 and 4+.
    curl -L --fail -C - ${AUTH[@]+"${AUTH[@]}"} -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
}

# Stable Audio Open 1.0 — the DiT + VAE (4.85 GB).
# Trained on Freesound + Free Music Archive: built for SFX / foley / production
# elements, NOT full songs. 44.1 kHz stereo, up to 47 s. This is the direct
# analog of pixelmon's "SDXL + Pixel Art XL" — the model IS the whole ballgame.
get "https://huggingface.co/$REPO/resolve/main/model.safetensors" \
    "$CKPT/stable-audio-open-1.0.safetensors"

# T5-base text encoder (438 MB) — what turns your description into conditioning.
# ComfyUI loads it via CLIPLoader with type=stable_audio.
get "https://huggingface.co/$REPO/resolve/main/text_encoder/model.safetensors" \
    "$TENC/t5_base.safetensors"

# --- Optional: ACE-Step 1.5 for --song (full songs with vocals + lyrics) ---
# ~9.3 GB. Skipped unless you ask:  ./download-models.sh --song  (or SONG_MODEL=1)
#
# We take the ALL-IN-ONE turbo build rather than the leaner split files (4.46 +
# 1.11 + 0.31 GB). ComfyUI's ACEStep15 class declares vae_key_prefix and
# text_encoder_key_prefix, so the aio bundles the VAE and the Qwen text encoder
# and loads through a single CheckpointLoaderSimple — same shape as Stable
# Audio above, and no risk of mismatched component versions.
if [ "${1:-}" = "--song" ] || [ "${SONG_MODEL:-0}" = 1 ]; then
    get "https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/resolve/main/checkpoints/ace_step_1.5_turbo_aio.safetensors" \
        "$CKPT/ace_step_1.5_turbo_aio.safetensors"
    echo "✅ ACE-Step 1.5 ready (for --song)."
else
    echo "ℹ️  --song model skipped (~9.3 GB). Fetch it with:  ./download-models.sh --song"
fi

echo "✅ models ready in $COMFY/models/"
echo "   checkpoint:   $CKPT/stable-audio-open-1.0.safetensors"
echo "   text encoder: $TENC/t5_base.safetensors"
