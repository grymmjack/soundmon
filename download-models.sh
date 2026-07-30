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

# --- Optional: Stable Audio 3 for --sa3 (music + sfx, full-band) ------------
# ~14 GB. Skipped unless you ask:  ./download-models.sh --sa3  (or SA3_MODEL=1)
#
# Why bother when ACE-Step already makes music: ACE's audio VAE hard-cuts at
# 16 kHz. Measured on raw model output, before any encoding — 12-16k at -23.9 dB,
# 16-18k at -48.8 dB. Everything above 16k is gone before soundmon sees the file,
# so no encoder setting or step count recovers it. That is the "spectrally
# processed, frequencies just missing" artifact.
#
# Unlike stable-audio-open-1.0 these are UNGATED via the Comfy-Org repackage,
# so no HF token is needed. Medium covers music and sfx; the two smalls are
# purpose-trained specialists worth A/B-ing against it.
if [ "${1:-}" = "--sa3" ] || [ "${SA3_MODEL:-0}" = 1 ]; then
    SA3="https://huggingface.co/Comfy-Org/stable-audio-3/resolve/main"
    get "$SA3/text_encoders/t5gemma_b_b_ul2.safetensors" "$TENC/t5gemma_b_b_ul2.safetensors"
    get "$SA3/checkpoints/stable_audio_3_medium.safetensors"      "$CKPT/stable_audio_3_medium.safetensors"
    get "$SA3/checkpoints/stable_audio_3_small_music.safetensors" "$CKPT/stable_audio_3_small_music.safetensors"
    get "$SA3/checkpoints/stable_audio_3_small_sfx.safetensors"   "$CKPT/stable_audio_3_small_sfx.safetensors"
    get "$SA3/checkpoints/stable_audio_3_medium_base.safetensors"  "$CKPT/stable_audio_3_medium_base.safetensors"
    # The Qwen REPROMPT encoder. Not optional: ComfyUI's official SA3 medium
    # workflow runs an LLM over your description first, with distinct system
    # prompts for Music / Instrument / SFX / One-shot, and only the rewritten
    # text reaches the audio model. Skipping it and feeding raw prompts is
    # exactly what produced a batch of unusable audio.
    get "https://huggingface.co/Comfy-Org/Qwen3.5/resolve/main/text_encoders/qwen3.5_2b_bf16.safetensors" \
        "$TENC/qwen3.5_2b_bf16.safetensors"
    echo "✅ Stable Audio 3 ready (medium, medium_base, specialists, qwen reprompt)."
fi

# --- Optional: Nuked OPL3 for --opl (real AdLib / Sound Blaster FM) ----------
# ~200 KB of C. Skipped unless you ask:  ./download-models.sh --opl
#
# WHY THIS IS FETCHED AND BUILT RATHER THAN COMMITTED. Nuked OPL3 is LGPL-2.1
# and soundmon is MIT. Building it locally keeps the two licences separate and
# properly attributed instead of quietly relicensing someone's work — the same
# reason the models aren't in git.
#
# It is a cycle-accurate emulation of the Yamaha YMF262, the chip in a Sound
# Blaster Pro 2 / 16, and it is what DOSBox uses. --opl drives it directly, so
# the output is not "FM-like": it is what an OPL3 does with those registers.
if [ "${1:-}" = "--opl" ] || [ "${OPL_CORE:-0}" = 1 ]; then
    VEND="$(cd "$(dirname "$0")" && pwd)/vendor"
    mkdir -p "$VEND"
    RAW="https://raw.githubusercontent.com/nukeykt/Nuked-OPL3/master"
    get "$RAW/opl3.c" "$VEND/opl3.c"
    get "$RAW/opl3.h" "$VEND/opl3.h"
    # LGPL-2.1 requires the licence travel with the source. Keep it adjacent so
    # nobody has to go looking for what covers vendor/.
    get "https://raw.githubusercontent.com/nukeykt/Nuked-OPL3/master/LICENSE" \
        "$VEND/LICENSE.Nuked-OPL3" 2>/dev/null || true

    CC_BIN="${CC:-cc}"
    command -v "$CC_BIN" >/dev/null 2>&1 || {
        echo "✗ need a C compiler ($CC_BIN not found). Install build tools, or set \$CC." >&2
        exit 1; }
    case "$(uname -s)" in
        Darwin) LIB="libopl3.dylib" ;;
        MINGW*|MSYS*|CYGWIN*) LIB="opl3.dll" ;;
        *) LIB="libopl3.so" ;;
    esac
    echo "⚙ compiling $LIB ..."
    ( cd "$VEND" && "$CC_BIN" -O2 -fPIC -shared opl3.c -o "$LIB" -lm )
    # The real General MIDI patch bank for OPL: 128 instruments, the Creative
    # Labs PLAY.EXE assignments, as carried by Schism Tracker. Fetched and NOT
    # vendored for the same reason as the core — Schism Tracker is GPL-2+ and
    # this repo is MIT. Without it, --from-midi plays every program on a
    # hand-authored 12-patch bank and a piano sounds like a brass stab.
    get "https://raw.githubusercontent.com/schismtracker/schismtracker/master/player/fmpatches.c" \
        "$VEND/fmpatches.c"
    # DMXOPL — the Doom OPL3 bank, voiced for a YMF262 rather than adapted from
    # OPL2, and MIT licensed like this repo. Strictly better than the Creative
    # PLAY.EXE set above, which stays as a fallback.
    get "https://raw.githubusercontent.com/sneakernets/DMXOPL/DMXOPL3/GENMIDI.wopl" \
        "$VEND/GENMIDI.wopl"
    get "https://raw.githubusercontent.com/Wohlstand/libADLMIDI/master/fm_banks/LICENSE-DMXOPL.txt" \
        "$VEND/LICENSE-DMXOPL.txt" 2>/dev/null || true
    echo "✅ GM OPL bank ready -> $VEND/fmpatches.c   (GPL-2+, Schism Tracker)"
    echo "✅ Nuked OPL3 ready -> $VEND/$LIB   (LGPL-2.1, see LICENSE.Nuked-OPL3)"
    echo "   try:  soundmon \"dungeon theme\" --opl --key \"D minor\" --bpm 110"
fi
