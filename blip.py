#!/usr/bin/env python3
"""--blip: JRPG-style blippy narration (Undertale / Animal Crossing / Earthbound).

WHY THIS IS NOT --narrate WITH A KNOB ON IT

The blippy voice in a JRPG is not speech that has been processed. It is a
*different thing entirely*: a short instrument note fired once per character as
the text box types, with no phonetic content at all. Undertale, Earthbound and
Zelda all work this way. You "hear" the words only because your eye is reading
them at the same rate your ear is getting blips.

So the synth style below never touches a TTS model. It is an oscillator and an
envelope, which is why it needs no Kokoro, no GPU, no ComfyUI, and renders a
seventy-line script in well under a second on any machine.

Animal Crossing is the other tradition, and it *is* derived from speech —
"Animalese" is recorded voice played back fast, which raises pitch and shortens
duration together. That is the `voice` style: Kokoro, then a plain resample. Not
a pitch shift — `narrate.py` goes to some trouble to move pitch *without*
changing duration, and here the duration change is the whole point.

    synth  ->  oscillator per character     no model, instant, Undertale-ish
    voice  ->  Kokoro resampled faster      needs Kokoro, Animal-Crossing-ish

Both write through the same output path `--narrate` uses, so a blip pack and a
spoken pack are drop-in interchangeable per line.
"""
import os
import sys

import narrate

SAMPLE_RATE = 24000          # matches narrate.py, so packs can be mixed freely

WAVES = ("square", "triangle", "sine", "saw", "noise")

# Punctuation drives the rhythm. Reading a line aloud, you pause at a comma and
# stop at a period; typing it into a text box does the same. Values are in
# "character slots", so they scale automatically with --blip-rate.
PAUSES = {".": 4.0, "!": 4.0, "?": 4.0, "…": 5.0,
          ",": 2.0, ";": 2.0, ":": 2.0, "—": 2.0, "-": 1.0}

VOWELS = set("aeiouyAEIOUY")


def _char_semitones(ch, jitter):
    """Pitch offset for one character, in semitones.

    Deterministic on purpose. A random offset per blip sounds like a machine
    malfunctioning; the same word producing the same little melody every time is
    what makes it read as a voice saying that word. Undertale gets this for free
    by using one fixed sample — we get more life out of it by letting the
    letters move, without giving up the repeatability.

    Vowels sit a couple of semitones above consonants, which is a crude nod to
    how vowels carry the pitched, sustained part of real speech.

    To change the character of the voice, change THIS function — it is the whole
    personality. Flat (`return 0.0`) is authentic Undertale. Widening the vowel
    lift makes it sing; a larger `jitter` makes it chatter.
    """
    base = 2.0 if ch in VOWELS else 0.0
    # Knuth multiplicative hash: a stable, well-spread value per codepoint, with
    # no RNG to seed and no dependence on position in the string.
    h = ((ord(ch) * 2654435761) % 1000) / 1000.0
    return base + (h - 0.5) * 2.0 * jitter


def _osc(freq, n, sr, wave, np, rng_seed=0):
    t = np.arange(n, dtype=np.float64) / sr
    ph = 2.0 * np.pi * freq * t
    if wave == "square":
        return np.sign(np.sin(ph))
    if wave == "triangle":
        return (2.0 / np.pi) * np.arcsin(np.sin(ph))
    if wave == "saw":
        return 2.0 * ((freq * t) % 1.0) - 1.0
    if wave == "noise":
        # Seeded per blip so a given character always sounds the same, for the
        # same reason _char_semitones is deterministic.
        return np.random.default_rng(rng_seed).uniform(-1.0, 1.0, n)
    return np.sin(ph)


def _blip(freq, dur_s, sr, wave, np, seed=0):
    """One character's worth of sound: fast attack, exponential decay."""
    n = max(4, int(sr * dur_s))
    w = _osc(freq, n, sr, wave, np, seed)
    t = np.arange(n, dtype=np.float64) / sr
    # ~5 time constants over the blip, so it has visibly decayed by the end but
    # is not cut off — a hard cut on a square wave is a click, not a blip.
    env = np.exp(-t * (5.0 / max(dur_s, 1e-4)))
    att = max(2, min(n // 4, int(sr * 0.002)))
    env[:att] *= np.linspace(0.0, 1.0, att)
    return w * env


def _synth_line(text, a, np):
    """Render one line as blips. Returns a float array at SAMPLE_RATE."""
    sr = SAMPLE_RATE
    slot = 1.0 / max(a.blip_rate, 0.1)          # seconds per character
    dur = slot * a.blip_duty                     # sounding part of the slot
    base_hz = 440.0 * (2.0 ** (a.blip_pitch / 12.0))

    pieces = []
    for i, ch in enumerate(text):
        if ch in PAUSES:
            pieces.append(np.zeros(int(sr * slot * PAUSES[ch])))
            continue
        if ch.isspace():
            pieces.append(np.zeros(int(sr * slot)))
            continue
        if not (ch.isalnum() or ch in "'\""):
            pieces.append(np.zeros(int(sr * slot)))
            continue
        hz = base_hz * (2.0 ** (_char_semitones(ch, a.blip_jitter) / 12.0))
        b = _blip(hz, dur, sr, a.blip_wave, np, seed=ord(ch))
        pad = int(sr * slot) - len(b)
        pieces.append(np.concatenate([b, np.zeros(pad)]) if pad > 0 else b[:int(sr * slot)])
    if not pieces:
        return np.zeros(1)
    return np.concatenate(pieces)


def _voice_line(text, a, np, pipeline):
    """Animalese: real speech, played back faster. Pitch and duration move together."""
    chunks = [np.asarray(au) for _, _, au in pipeline(text, voice=a.voice, speed=1.0)]
    if not chunks:
        return None
    audio = np.concatenate(chunks).astype(np.float64)
    r = max(1.01, a.blip_speed)
    # Plain resample. Deliberately NOT narrate._pitch_shift, which exists to
    # preserve duration — here the speed-up IS the effect.
    n = int(len(audio) / r)
    idx = np.linspace(0.0, len(audio) - 1, n)
    return np.interp(idx, np.arange(len(audio)), audio)


def run(a, slug, to_ogg=None, loudness_normalize=None):
    """Generate blippy narration. Same contract as narrate.run()."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        sys.exit(f"--blip needs numpy + soundfile: pip install soundfile numpy   ({e})")

    if a.blip_wave not in WAVES:
        sys.exit(f"unknown --blip-wave {a.blip_wave!r}. Choose from: {', '.join(WAVES)}")

    src = getattr(a, "blip_file", None) or a.narrate_file
    items = narrate.parse_lines(src) if src else [(a.name or slug(a.prompt), a.prompt)]
    if not items:
        sys.exit(f"--blip: nothing to read in {src}")

    dest = os.path.abspath(os.path.expanduser(a.output_to)) if a.output_to else os.getcwd()
    if not os.path.isdir(dest):
        if a.create_dirs or a.output_to:
            os.makedirs(dest, exist_ok=True)
        else:
            sys.exit(f"output dir does not exist: {dest} (add --create-dirs)")

    pipeline = None
    if a.blip_style == "voice":
        try:
            from kokoro import KPipeline
        except ImportError as e:
            sys.exit(f"--blip-style voice needs Kokoro: pip install kokoro   ({e})")
        if a.voice not in narrate.VOICES:
            sys.exit(f"unknown --voice {a.voice!r}. See --list-voices.")
        pipeline = KPipeline(lang_code="b" if a.voice[0] == "b" else "a",
                             device=narrate.RUN_DEVICE)
        print(f"\U0001f7e2 blip/voice  {a.voice}  |  {len(items)} line(s)  |  "
              f"{a.blip_speed:g}x  |  CPU")
    else:
        print(f"\U0001f7e2 blip/synth  {a.blip_wave}  |  {len(items)} line(s)  |  "
              f"{a.blip_rate:g} ch/s  |  pitch {a.blip_pitch:+g} st  |  no model")

    made = []
    for i, (key, text) in enumerate(items, 1):
        audio = (_voice_line(text, a, np, pipeline) if a.blip_style == "voice"
                 else _synth_line(text, a, np))
        if audio is None or not len(audio):
            print(f"   ⚠ [{i}/{len(items)}] {key}: nothing generated, skipped")
            continue

        peak = float(np.abs(audio).max())
        if peak > 1e-6:                      # same -1 dBFS target as every other engine
            audio = audio * ((10.0 ** (a.normalize_db / 20.0)) / peak)

        path = os.path.join(dest, f"{key}.wav")
        sf.write(path, audio, SAMPLE_RATE,
                 subtype=f"PCM_{a.bits}" if a.bits != 8 else "PCM_U8")
        if getattr(a, "lufs_target", None) is not None and loudness_normalize:
            loudness_normalize(path, a.lufs_target, a.true_peak)
        if getattr(a, "ogg", False) and to_ogg:
            path = to_ogg(path, a.ogg_quality, a.keep_wav)
        made.append(path)
        print(f"   ✅ [{i}/{len(items)}] {key:<28} {len(audio)/SAMPLE_RATE:5.2f}s  "
              f"{text[:52]}{'…' if len(text) > 52 else ''}")

    print(f"   all done  |  {len(made)} file(s) in {dest}")
