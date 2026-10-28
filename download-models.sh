#!/usr/bin/env bash
# Download the models soundmon uses into ~/ComfyUI/models/ (or $COMFYUI_DIR/models).
# All public on Hugging Face, no login. Files you already have are skipped, so re-running is safe;
# an interrupted download resumes where it stopped.
#
#   ./download-models.sh --sa3      the DEFAULT engine: Stable Audio 3 + its Qwen prompt rewriter (~14 GB)
#   ./download-models.sh --song     + ACE-Step 1.5 for --song (full songs with vocals)            (~9.3 GB)
#   ./download-models.sh --sao      + Stable Audio Open 1.0, the legacy --engine sao             (~5.3 GB)
#   ./download-models.sh --opl      + build the Nuked OPL3 core for --opl/--oplfx (needs a C compiler; tiny)
#   ./download-models.sh --all      --sa3 --song --sao (not --opl, which compiles C)             (~29 GB)
#   ./download-models.sh --list     show what's installed and what's missing; download nothing
#   ./download-models.sh            no flag: the legacy --sao set, as it always did (new boxes want --sa3)
#
# Flags combine:  ./download-models.sh --sa3 --song
# The official stabilityai SAO repo is gated (401 without a token + a license click); --sao uses an
# ungated byte-identical mirror. To use the official one:  HF_TOKEN=hf_xxx ./download-models.sh --official
# The old environment switches still work: SA3_MODEL=1, SONG_MODEL=1, OPL_CORE=1.
set -euo pipefail

COMFY="${COMFYUI_DIR:-$HOME/ComfyUI}"
CKPT="$COMFY/models/checkpoints"
TENC="$COMFY/models/text_encoders"
VEND="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)/vendor"

WANT_SAO=0 WANT_SA3=0 WANT_SONG=0 WANT_OPL=0 LIST=0 OFFICIAL=0 ANY=0
for a in "$@"; do
    case "$a" in
        --sa3) WANT_SA3=1; ANY=1 ;;
        --song) WANT_SONG=1; ANY=1 ;;
        --sao) WANT_SAO=1; ANY=1 ;;
        --official) WANT_SAO=1; OFFICIAL=1; ANY=1 ;;
        --opl) WANT_OPL=1; ANY=1 ;;
        --all) WANT_SA3=1; WANT_SONG=1; WANT_SAO=1; ANY=1 ;;
        --list) LIST=1 ;;
        -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $a (see --help)" >&2; exit 2 ;;
    esac
done
[ "${SA3_MODEL:-0}" = 1 ] && WANT_SA3=1 && ANY=1
[ "${SONG_MODEL:-0}" = 1 ] && WANT_SONG=1 && ANY=1
[ "${OPL_CORE:-0}" = 1 ] && WANT_OPL=1 && ANY=1
if [ "$ANY" = 0 ] && [ "$LIST" = 0 ]; then
    WANT_SAO=1          # the bare command has always meant this set; scripts depend on it
    echo "ℹ  no group given: fetching the LEGACY Stable Audio Open set (--sao)."
    echo "   The default engine is Stable Audio 3 — fetch it with:  ./download-models.sh --sa3"
fi

SAO_REPO="RedbeardNZ/stable-audio-open-1.0"          # ungated mirror
AUTH=()
if [ "$OFFICIAL" = 1 ]; then
    SAO_REPO="stabilityai/stable-audio-open-1.0"
    : "${HF_TOKEN:?--official needs HF_TOKEN (and accept the license on the model page)}"
    AUTH=(-H "Authorization: Bearer $HF_TOKEN")
fi

MISSING=()
get() {  # url  dest  size-label  what-for
    local url="$1" dest="$2" size="$3" why="$4" code rc
    # -s, not -f: a 0-byte file (a disk that filled up, a killed `mv`) is not a model.
    if [ -s "$dest" ]; then echo "✓ $(basename "$dest")"; return 0; fi
    if [ "$LIST" = 1 ]; then echo "✗ $(basename "$dest")  ($size) — $why"; MISSING+=("$(basename "$dest")"); return 0; fi
    mkdir -p "$(dirname "$dest")"
    echo "↓ $(basename "$dest")  ($size) — $why"
    # Download to .part and rename only on success, so an interrupted transfer
    # (killed shell, full disk, dropped WSL) can never leave a truncated file
    # sitting at the real path — where the check above would treat it as
    # complete on the next run and hand ComfyUI a corrupt checkpoint. `-C -`
    # resumes an existing .part instead of restarting from zero.
    #
    # ${AUTH[@]+"${AUTH[@]}"} not "${AUTH[@]}": macOS ships bash 3.2, where
    # expanding an EMPTY array under `set -u` is an "unbound variable" fatal
    # error. This idiom expands to nothing when unset and is safe in 3.2 and 4+.
    set +e
    code="$(curl -L --fail -C - --progress-bar ${AUTH[@]+"${AUTH[@]}"} -o "$dest.part" -w '%{http_code}' "$url")"
    rc=$?
    if [ "$rc" != 0 ] && [ "$code" = 416 ]; then
        # A .part the server won't resume (a different file version under the same URL): start over.
        rm -f "$dest.part"
        code="$(curl -L --fail --progress-bar ${AUTH[@]+"${AUTH[@]}"} -o "$dest.part" -w '%{http_code}' "$url")"
        rc=$?
    fi
    set -e
    if [ "$rc" = 0 ] && [ -s "$dest.part" ]; then
        mv "$dest.part" "$dest"
    else
        if [ -s "$dest.part" ]; then
            echo "  ✗ download failed (HTTP ${code:-?}, curl $rc): $url — the .part is kept; re-run to resume"
        else
            rm -f "$dest.part"
            echo "  ✗ download failed (HTTP ${code:-?}, curl $rc): $url"
        fi
        MISSING+=("$(basename "$dest")")
    fi
}

if [ "$WANT_SA3" = 1 ] || [ "$LIST" = 1 ]; then
    echo "== Stable Audio 3 — the default engine (SFX and --music)"
    # Why SA3 and not ACE-Step for music: ACE's audio VAE hard-cuts at 16 kHz.
    # Measured on raw model output, before any encoding — 12-16k at -23.9 dB,
    # 16-18k at -48.8 dB. Everything above 16k is gone before soundmon sees the
    # file. SA3 is full-band. UNGATED via the Comfy-Org repackage.
    SA3="https://huggingface.co/Comfy-Org/stable-audio-3/resolve/main"
    get "$SA3/text_encoders/t5gemma_b_b_ul2.safetensors" "$TENC/t5gemma_b_b_ul2.safetensors" "of ~14 GB" "SA3's text encoder"
    get "$SA3/checkpoints/stable_audio_3_medium.safetensors" "$CKPT/stable_audio_3_medium.safetensors" "of ~14 GB" \
        "the model (medium is the only one verified to sound right)"
    # The Qwen REPROMPT encoder. Not optional: ComfyUI's official SA3 medium
    # workflow runs an LLM over your description first, with distinct system
    # prompts for Music / Instrument / SFX / One-shot, and only the rewritten
    # text reaches the audio model. Skipping it and feeding raw prompts is
    # exactly what produced a batch of unusable audio.
    get "https://huggingface.co/Comfy-Org/Qwen3.5/resolve/main/text_encoders/qwen3.5_2b_bf16.safetensors" \
        "$TENC/qwen3.5_2b_bf16.safetensors" "of ~14 GB" "the Qwen prompt rewriter (part of the SA3 pipeline)"
    # The specialists and the base model, for A/B work (--sa3-size small / --base). They produced
    # scrambled audio here (AGENTS gotcha 12) but stay in the set so a comparison is one flag away.
    get "$SA3/checkpoints/stable_audio_3_small_music.safetensors" "$CKPT/stable_audio_3_small_music.safetensors" "of ~14 GB" "A/B only (--sa3-size small --music)"
    get "$SA3/checkpoints/stable_audio_3_small_sfx.safetensors" "$CKPT/stable_audio_3_small_sfx.safetensors" "of ~14 GB" "A/B only (--sa3-size small)"
    get "$SA3/checkpoints/stable_audio_3_medium_base.safetensors" "$CKPT/stable_audio_3_medium_base.safetensors" "of ~14 GB" "A/B only (--base)"
fi

if [ "$WANT_SONG" = 1 ] || [ "$LIST" = 1 ]; then
    echo "== ACE-Step 1.5 — --song (full songs with vocals and lyrics)"
    # We take the ALL-IN-ONE turbo build rather than the leaner split files (4.46 +
    # 1.11 + 0.31 GB). ComfyUI's ACEStep15 class declares vae_key_prefix and
    # text_encoder_key_prefix, so the aio bundles the VAE and the Qwen text encoder
    # and loads through a single CheckpointLoaderSimple — and no risk of mismatched
    # component versions.
    get "https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/resolve/main/checkpoints/ace_step_1.5_turbo_aio.safetensors" \
        "$CKPT/ace_step_1.5_turbo_aio.safetensors" "9.3 GB" "--song"
fi

if [ "$WANT_SAO" = 1 ] || [ "$LIST" = 1 ]; then
    echo "== Stable Audio Open 1.0 — legacy (--engine sao)"
    # Trained on Freesound + Free Music Archive: SFX / foley, not songs. 44.1 kHz
    # stereo, up to 47 s. Its VAE hard-cuts at 16 kHz.
    get "https://huggingface.co/$SAO_REPO/resolve/main/model.safetensors" \
        "$CKPT/stable-audio-open-1.0.safetensors" "4.85 GB" "--engine sao"
    # T5-base (438 MB) — SAO's text encoder, loaded via CLIPLoader type=stable_audio
    # (the checkpoint itself has NO text encoder — AGENTS gotcha 1).
    get "https://huggingface.co/$SAO_REPO/resolve/main/text_encoder/model.safetensors" \
        "$TENC/t5_base.safetensors" "438 MB" "SAO's text encoder"
fi

# --- Nuked OPL3 for --opl (real AdLib / Sound Blaster FM) ----------------------
# WHY THIS IS FETCHED AND BUILT RATHER THAN COMMITTED. Nuked OPL3 is LGPL-2.1
# and soundmon is MIT. Building it locally keeps the two licences separate and
# properly attributed instead of quietly relicensing someone's work — the same
# reason the models aren't in git.
#
# It is a cycle-accurate emulation of the Yamaha YMF262, the chip in a Sound
# Blaster Pro 2 / 16, and it is what DOSBox uses. --opl drives it directly, so
# the output is not "FM-like": it is what an OPL3 does with those registers.
case "$(uname -s)" in
    Darwin) LIB="libopl3.dylib" ;;
    MINGW*|MSYS*|CYGWIN*) LIB="opl3.dll" ;;
    *) LIB="libopl3.so" ;;
esac
if [ "$LIST" = 1 ]; then
    echo "== Nuked OPL3 core — --opl / --oplfx (built, not downloaded)"
    if [ -s "$VEND/$LIB" ]; then echo "✓ vendor/$LIB"; else echo "✗ vendor/$LIB — build it with --opl"; MISSING+=("$LIB"); fi
    for f in GENMIDI.wopl fmpatches.c; do
        if [ -s "$VEND/$f" ]; then echo "✓ vendor/$f"; else echo "✗ vendor/$f — fetched by --opl"; MISSING+=("$f"); fi
    done
elif [ "$WANT_OPL" = 1 ]; then
    echo "== Nuked OPL3 core — --opl / --oplfx"
    mkdir -p "$VEND"
    RAW="https://raw.githubusercontent.com/nukeykt/Nuked-OPL3/master"
    get "$RAW/opl3.c" "$VEND/opl3.c" "small" "the OPL3 emulator source (LGPL-2.1)"
    get "$RAW/opl3.h" "$VEND/opl3.h" "small" "its header"
    # LGPL-2.1 requires the licence travel with the source. Keep it adjacent so
    # nobody has to go looking for what covers vendor/.
    get "$RAW/LICENSE" "$VEND/LICENSE.Nuked-OPL3" "small" "its licence" || true

    CC_BIN="${CC:-cc}"
    if ! command -v "$CC_BIN" >/dev/null 2>&1; then
        echo "✗ need a C compiler ($CC_BIN not found). Install build tools, or set \$CC." >&2
        MISSING+=("$LIB")
    elif [ -s "$VEND/opl3.c" ]; then
        echo "⚙ compiling $LIB ..."
        if ( cd "$VEND" && "$CC_BIN" -O2 -fPIC -shared opl3.c -o "$LIB" -lm ); then
            echo "✓ vendor/$LIB"
        else
            MISSING+=("$LIB")
        fi
    fi
    # The real General MIDI patch bank for OPL: 128 instruments, the Creative
    # Labs PLAY.EXE assignments, as carried by Schism Tracker. Fetched and NOT
    # vendored for the same reason as the core — Schism Tracker is GPL-2+ and
    # this repo is MIT. Without it, --from-midi plays every program on a
    # hand-authored 12-patch bank and a piano sounds like a brass stab.
    get "https://raw.githubusercontent.com/schismtracker/schismtracker/master/player/fmpatches.c" \
        "$VEND/fmpatches.c" "small" "the GM OPL bank (GPL-2+, Schism Tracker)"
    # DMXOPL — the Doom OPL3 bank, voiced for a YMF262 rather than adapted from
    # OPL2, and MIT licensed like this repo. Strictly better than the Creative
    # PLAY.EXE set above, which stays as a fallback.
    get "https://raw.githubusercontent.com/sneakernets/DMXOPL/DMXOPL3/GENMIDI.wopl" \
        "$VEND/GENMIDI.wopl" "small" "the DMXOPL bank (MIT)"
    get "https://raw.githubusercontent.com/Wohlstand/libADLMIDI/master/fm_banks/LICENSE-DMXOPL.txt" \
        "$VEND/LICENSE-DMXOPL.txt" "small" "its licence" || true
    echo "   try:  soundmon \"dungeon theme\" --opl --key \"D minor\" --bpm 110"
fi

echo
if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "✗ missing: ${MISSING[*]}"
    if [ "$LIST" = 1 ]; then
        echo "  fetch a group with --sa3 / --song / --sao / --opl, or the models with --all"
        exit 0
    fi
    exit 1
fi
echo "✅ models ready in $COMFY/models/"
echo "   ComfyUI must run with --force-fp32 for Stable Audio 3 (soundmon starts it that way itself)."
