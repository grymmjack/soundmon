"""
retro_sfx — soundmon's finishing node.

This is the direct analog of pixelmon's `pixelart_palette` node, and it exists
for the same reason. Raw SDXL output downscaled is a *crushed photo*, not a
sprite; raw Stable Audio output is a *47-second stereo ambience bed*, not a game
sound effect. The model gets you the raw material — this node makes it an asset.

pixelmon's chain was:  smooth -> downscale -> palette -> transparent
soundmon's chain is:   trim  -> normalize -> format-lock -> fade

The key mapping, and the whole idea behind this node:

    a PALETTE locks an image to a fixed set of COLORS
    a FORMAT  locks audio  to a fixed SAMPLE RATE + BIT DEPTH + CHANNELS

Amiga MOD, SoundBlaster, SPC700 and Game Boy audio don't sound "old" because of
the notes — they sound old because of their format ceiling. That ceiling is a
palette. It's the same trick, one dimension over.

ComfyUI AUDIO contract:  {"waveform": Tensor[B, C, N] float, "sample_rate": int}
"""

import torch

# ---------------------------------------------------------------------------
# The format registry — soundmon's palettes.py.
# Add your own here; the key becomes the --format name on the CLI.
# ---------------------------------------------------------------------------
FORMATS = {
    "none":    None,                                          # model's own output
    "amiga":   dict(rate=8363,  bits=8,  mono=True),          # Paula / MOD
    "sb":      dict(rate=11025, bits=8,  mono=True),          # SoundBlaster, DOS
    "sb22":    dict(rate=22050, bits=8,  mono=True),          # later SB / VOC
    "gameboy": dict(rate=8192,  bits=4,  mono=True),          # DMG wave channel
    "nes":     dict(rate=4096,  bits=7,  mono=True),          # DPCM sample channel
    "snes":    dict(rate=32000, bits=16, mono=False),         # SPC700 / BRR
    "psx":     dict(rate=22050, bits=16, mono=False),         # PS1 ADPCM-ish
    "cd":      dict(rate=44100, bits=16, mono=False),         # full quality
    # --- MOD / tracker era ---------------------------------------------------
    "mod8":    dict(rate=11025, bits=8,  mono=True),          # 8-bit 11k mono
    "mod8s":   dict(rate=22050, bits=8,  mono=False),         # 8-bit 22k stereo
    "mod6":    dict(rate=11025, bits=6,  mono=True),          # 6-bit crunch
    "mod6s":   dict(rate=22050, bits=6,  mono=False),
    "crush":   dict(rate=8000,  bits=6,  mono=True),          # maximum grit
}


def _trim_silence(wave: torch.Tensor, sr: int, threshold_db: float) -> torch.Tensor:
    """Drop leading/trailing near-silence. Stable Audio pads short SFX with dead
    air out to the requested duration, so almost every render needs this."""
    amp = wave.abs().amax(dim=1)[0]                  # [N], loudest channel per sample
    thresh = 10.0 ** (threshold_db / 20.0)
    loud = (amp > thresh).nonzero()
    if loud.numel() == 0:
        return wave
    return wave[:, :, int(loud[0]) : int(loud[-1]) + 1]


def _normalize(wave: torch.Tensor, peak_db: float) -> torch.Tensor:
    """Peak-normalize so every generated SFX sits at a predictable level."""
    peak = wave.abs().amax()
    if peak < 1e-8:
        return wave
    return wave * ((10.0 ** (peak_db / 20.0)) / peak)


def _fade(wave: torch.Tensor, sr: int, fade_ms: int) -> torch.Tensor:
    """Short fade on both ends. Trimming at a non-zero crossing leaves a click;
    a few ms of ramp kills it without audibly softening the transient."""
    n = int(sr * fade_ms / 1000)
    if n <= 0 or wave.shape[-1] < 2 * n:
        return wave
    ramp = torch.linspace(0.0, 1.0, n, device=wave.device, dtype=wave.dtype)
    wave = wave.clone()
    wave[:, :, :n] *= ramp
    wave[:, :, -n:] *= ramp.flip(0)
    return wave


# ---------------------------------------------------------------------------
# TODO(grymmjack): the format lock — the heart of the node.
#
# Given a waveform at `sr`, produce one that has actually *passed through* a
# rate/bits/mono ceiling. Roughly 8-10 lines. Three real decisions live here:
#
#   1. RESAMPLING. Naive stride-decimation (wave[..., ::factor]) aliases — and
#      aliasing is exactly what an Amiga sounds like. A proper anti-aliased
#      resample sounds "correct" and loses the character. This is the same call
#      pixelmon made with `--filter nearest (crisp) / box (soft)`, where nearest
#      won because correct-looking wasn't the goal.
#
#   2. QUANTIZATION. Plain rounding to 2**bits levels is harsh and authentic.
#      Adding dither before rounding trades that grit for a smoother noise floor
#      — the audio twin of pixelmon's `--dither` Floyd-Steinberg flag.
#
#   3. WHETHER TO UPSAMPLE BACK. Leaving the file at 8363 Hz is truly authentic
#      but awkward to load in a modern engine; upsampling back to 44.1 kHz keeps
#      the crunch you just baked in while staying a normal WAV. (Note the return
#      signature lets you pick — return whatever rate you decide.)
#
# There is no neutral default here; the choice IS the sound of the tool.
# ---------------------------------------------------------------------------
def _format_lock(wave: torch.Tensor, sr: int, spec: dict) -> tuple:
    """Force `wave` through spec's rate/bits/mono ceiling.

    Args:
        wave: Tensor[B, C, N], float, nominally in [-1, 1]
        sr:   current sample rate
        spec: {"rate": int, "bits": int, "mono": bool} from FORMATS

    Returns:
        (wave, sample_rate) — the crushed waveform and the rate it's now at.

    Two decisions define the character, and both go the "authentic" way:

    NO ANTI-ALIASING on the decimation. Aliasing is the Amiga sound; filtering it
    out leaves a dull quiet recording that happens to be 8 kHz.
    NO DITHER on the quantization. Dither smooths the noise floor, which is
    exactly what a hardware ceiling exists not to do.

    Swap either if you want the polite version — they are one line each.
    """
    if not spec:
        return wave, sr

    # --- channels ------------------------------------------------------------
    # Averaging, not dropping a channel: a hard-panned MOD would otherwise lose
    # half its arrangement rather than fold down.
    if spec.get("mono") and wave.shape[1] > 1:
        wave = wave.mean(dim=1, keepdim=True)

    # --- sample rate: NAIVE stride decimation --------------------------------
    # Deliberately no anti-aliasing filter. The aliasing IS the sound: Paula had
    # no reconstruction filter worth the name, so everything above Nyquist folded
    # back as that characteristic gritty shimmer. A clean resample gives you a
    # quiet, dull 8 kHz recording — technically better and completely wrong.
    target = int(spec.get("rate") or sr)
    if 0 < target < sr:
        step = sr / float(target)
        n_out = int(wave.shape[-1] / step)
        if n_out > 1:
            idx = (torch.arange(n_out, device=wave.device) * step).long()
            idx = idx.clamp(max=wave.shape[-1] - 1)
            wave = wave.index_select(-1, idx)
            sr = target

    # --- bit depth: plain quantize, no dither --------------------------------
    # Dithering trades the grit for a smoother noise floor, which is the opposite
    # of what a hardware ceiling is for. Quantizing to `levels` steps mirrors what
    # an 8-bit DAC actually does to the signal.
    #
    # Note this is amplitude quantization only — SaveSFX still decides the stored
    # PCM width, and 8-bit WAV's unsigned offset is handled there.
    bits = int(spec.get("bits") or 0)
    if 0 < bits < 24:
        levels = float(2 ** bits)
        peak = levels / 2.0 - 1.0
        wave = torch.round(wave.clamp(-1.0, 1.0) * peak) / peak

    return wave, sr


class RetroSFX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "format": (list(FORMATS), {"default": "none"}),
                "trim_silence": ("BOOLEAN", {"default": True}),
                "max_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                                          "tooltip": "Hard length cap; 0 = no cap. Different from "
                                                     "trimming, which only removes silence."}),
                "threshold_db": ("FLOAT", {"default": -45.0, "min": -90.0, "max": 0.0, "step": 1.0}),
                "normalize_db": ("FLOAT", {"default": -1.0, "min": -30.0, "max": 0.0, "step": 0.5}),
                "fade_ms": ("INT", {"default": 5, "min": 0, "max": 500}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "audio/soundmon"

    def process(self, audio, format, trim_silence, max_seconds, threshold_db,
                normalize_db, fade_ms):
        wave, sr = audio["waveform"], audio["sample_rate"]

        if trim_silence:
            wave = _trim_silence(wave, sr, threshold_db)

        # Hard cap, applied AFTER trimming so the budget is spent on real audio
        # rather than the model's leading silence. Trimming removes quiet; this
        # removes length — a UI blip that plays once per glyph needs the latter,
        # and no amount of silence-trimming will give it.
        if max_seconds > 0:
            keep = int(sr * max_seconds)
            if 0 < keep < wave.shape[-1]:
                wave = wave[:, :, :keep]

        spec = FORMATS.get(format)
        if spec is not None:
            wave, sr = _format_lock(wave, sr, spec)

        # Order matters: normalize LAST. Stable Audio Open frequently starts at
        # full amplitude — the loudest sample of a render is often sample 0 —
        # so the de-click fade lands right on the peak and scales it toward zero.
        # Normalizing first meant the fade then silently undid it (measured
        # -5.94 dBFS on a -1.0 dBFS request). Any gain-changing stage after
        # normalize invalidates it, so normalize is the final word on level.
        wave = _fade(wave, sr, fade_ms)
        wave = _normalize(wave, normalize_db)

        return ({"waveform": wave, "sample_rate": sr},)


class SaveSFX:
    """Write a real WAV. ComfyUI core only ships flac/mp3/opus savers, none of
    which a game engine, tracker, or DOS-era toolchain will load. Bit depth is
    explicit here so an 8-bit `--format` render actually lands as an 8-bit file
    rather than a 16-bit file containing 8 bits of information."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "soundmon/sfx"}),
                "bit_depth": (["16", "8", "24"], {"default": "16"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "audio/soundmon"

    def save(self, audio, filename_prefix, bit_depth):
        import os
        import wave as wavemod
        import folder_paths

        wave_t, sr = audio["waveform"], int(audio["sample_rate"])
        out_dir = folder_paths.get_output_directory()
        full_dir, base, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, out_dir
        )
        os.makedirs(full_dir, exist_ok=True)

        bits = int(bit_depth)
        results = []
        for i in range(wave_t.shape[0]):
            buf = wave_t[i].transpose(0, 1).clamp(-1.0, 1.0).cpu()  # [N, C] interleaved
            name = f"{base}_{counter:05}_.wav"
            path = os.path.join(full_dir, name)

            if bits == 8:
                # 8-bit WAV is UNSIGNED with a 128 offset — the one PCM width that
                # breaks the signed pattern. Getting this wrong yields loud static.
                data = ((buf * 127.0).round() + 128).clamp(0, 255).to(torch.uint8).numpy().tobytes()
            elif bits == 24:
                ints = (buf * 8388607.0).round().clamp(-8388608, 8388607).to(torch.int32).numpy()
                data = b"".join(int(v).to_bytes(3, "little", signed=True) for v in ints.flatten())
            else:
                data = (buf * 32767.0).round().clamp(-32768, 32767).to(torch.int16).numpy().tobytes()

            with wavemod.open(path, "wb") as w:
                w.setnchannels(buf.shape[1])
                w.setsampwidth(bits // 8)
                w.setframerate(sr)
                w.writeframes(data)

            results.append({"filename": name, "subfolder": subfolder, "type": "output"})
            counter += 1

        return {"ui": {"audio": results}}


NODE_CLASS_MAPPINGS = {"RetroSFX": RetroSFX, "SaveSFX": SaveSFX}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RetroSFX": "Retro SFX (soundmon)",
    "SaveSFX": "Save SFX as WAV (soundmon)",
}
