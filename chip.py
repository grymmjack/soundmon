#!/usr/bin/env python3
"""--chip: an actual PSG chiptune synthesizer. NES 2A03 voicing, no model.

WHY THIS EXISTS RATHER THAN A BETTER PROMPT

Asked for "chiptune", Stable Audio 3 produces modern chiptune-*influenced*
music: made in a DAW, with reverb, sampled drums and unlimited polyphony. That
is what its training data labels chiptune, and it is not wrong — that genre
exists. It is just not what a 2A03 sounds like.

Real chiptune is not a genre the model imitates, it is a **constraint**:

    2 pulse channels     one note each, duty 12.5 / 25 / 50 / 75 %
    1 triangle channel   one note, no volume control at all
    1 noise channel      15-bit LFSR, percussion
    no reverb, nothing sampled, four voices total, forever

A diffusion model has no way to honour that. So don't ask it to. This module
synthesizes the audio directly, which makes it authentic *by construction* — it
physically cannot emit reverb or a sampled drum, because there is no code path
that produces one. Same reasoning as blip.py: some retro audio is synthesis, not
recording, and the honest implementation is an oscillator.

TWO THINGS THAT FALL OUT FOR FREE

- **It loops perfectly.** Music is composed in whole bars, so the last sample
  runs into the first with no discontinuity. `--loop`'s crossfade exists to hide
  a composed ending; there is no composed ending here. Do not wrap chip output.
- **It costs nothing.** No model, no GPU, no ComfyUI. A 60-second track renders
  in well under a second, so `-n 20` and keeping the best one is free.

THE IDIOM THAT MATTERS MOST: ARPEGGIOS

With one note per channel you cannot play a chord. So chiptune fakes them by
cycling the chord tones every frame or two (~50 Hz) — fast enough that the ear
fuses it into a buzzy, shimmering chord. That fast arpeggio IS the sound of the
format, more than the square waves are, and `_arp_channel` below is the single
most important thing here for it reading as genuine.
"""
import math
import os
import sys

SAMPLE_RATE = 44100
STEPS = 16                        # sixteenth-note grid per 4/4 bar

# Scale degrees in semitones. Minor modes first — this is a dungeon crawler.
SCALES = {
    "minor":      [0, 2, 3, 5, 7, 8, 10],
    "harmonic":   [0, 2, 3, 5, 7, 8, 11],
    "dorian":     [0, 2, 3, 5, 7, 9, 10],
    "phrygian":   [0, 1, 3, 5, 7, 8, 10],
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "pentatonic": [0, 3, 5, 7, 10],
}

# Chord progressions as scale degrees (0-indexed). Chosen because they are what
# the era actually used, not because they are theoretically interesting.
PROGRESSIONS = {
    "minor":      [[0, 5, 2, 6], [0, 3, 4, 0], [0, 6, 5, 6], [0, 0, 5, 6]],
    "harmonic":   [[0, 3, 4, 0], [0, 5, 4, 0], [0, 6, 4, 0]],
    "dorian":     [[0, 3, 0, 6], [0, 6, 3, 0]],
    "phrygian":   [[0, 1, 0, 6], [0, 1, 5, 0]],
    "major":      [[0, 4, 5, 3], [0, 3, 4, 4], [0, 5, 3, 4]],
    "mixolydian": [[0, 6, 3, 0], [0, 3, 6, 0]],
    "pentatonic": [[0, 3, 4, 0], [0, 4, 0, 3]],
}

NOTE_BASE = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

# Melody rhythms as sixteenth-note durations summing to one bar.
RHYTHMS = [
    [4, 4, 4, 4], [8, 4, 4], [4, 4, 8], [2, 2, 4, 4, 4],
    [4, 2, 2, 8], [6, 2, 4, 4], [4, 4, 2, 2, 4], [2, 2, 2, 2, 8],
    [8, 8], [12, 4], [4, 8, 4],
]

# 16-step drum patterns. k=kick s=snare h=hat.
DRUMS = [
    {"k": "1000000010000000", "s": "0000100000001000", "h": "1010101010101010"},
    {"k": "1000001000100000", "s": "0000100000001000", "h": "1111111111111111"},
    {"k": "1000000010001000", "s": "0000100000001000", "h": "1010101010101110"},
    {"k": "1001000010010000", "s": "0000100000001000", "h": "0010001000100010"},
]

DUTIES = (0.125, 0.25, 0.5, 0.75)


def parse_key(text):
    """'D minor' / 'f# major' / 'A' -> (root_semitone, scale_name)."""
    t = (text or "C minor").strip().lower().replace("-", " ")
    parts = t.split()
    name = parts[0]
    root = NOTE_BASE.get(name[0], 0)
    if len(name) > 1:
        if name[1] in "#s":
            root += 1
        elif name[1] == "b":
            root -= 1
    scale = "minor"
    for p in parts[1:]:
        if p in SCALES:
            scale = p
            break
        if p.startswith("maj"):
            scale = "major"
            break
        if p.startswith("min"):
            scale = "minor"
            break
    return root % 12, scale


def _hz(midi):
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _pulse(freq, n, sr, duty, np):
    """Square wave with a duty cycle. The 2A03's only timbral control."""
    if freq <= 0:
        return np.zeros(n)
    t = np.arange(n, dtype=np.float64) / sr
    return np.where((freq * t) % 1.0 < duty, 1.0, -1.0)


def _tri(freq, n, sr, np):
    """NES triangle: 15 quantized steps, which is why its bass sounds gritty
    rather than smooth. Quantizing here is not an approximation of the hardware,
    it IS the hardware — a clean triangle sounds wrong in this context."""
    if freq <= 0:
        return np.zeros(n)
    t = np.arange(n, dtype=np.float64) / sr
    saw = (freq * t) % 1.0
    tri = 2.0 * np.abs(2.0 * saw - 1.0) - 1.0
    return np.round(tri * 7.5) / 7.5


def _noise(n, sr, period, np, seed=1):
    """15-bit LFSR, the real 2A03 noise generator.

    Taps at bits 0 and 1, feedback into bit 14 — this is the actual shift
    register, so the result has the specific metallic colour NES percussion has
    rather than the flat hiss of uniform random noise.
    """
    out = np.empty(n)
    reg = (seed & 0x7FFF) or 1
    step = max(1, int(sr / max(period, 1.0)))
    val = 1.0
    for i in range(n):
        if i % step == 0:
            fb = ((reg ^ (reg >> 1)) & 1)
            reg = (reg >> 1) | (fb << 14)
            val = 1.0 if (reg & 1) else -1.0
        out[i] = val
    return out


def _env(n, sr, attack_ms, decay, sustain, np):
    """Fast attack, exponential decay to a sustain floor. No release curve —
    the channel is simply reassigned on the next note, as on hardware."""
    e = np.empty(n)
    t = np.arange(n, dtype=np.float64) / sr
    e[:] = sustain + (1.0 - sustain) * np.exp(-t * decay)
    a = max(1, min(n, int(sr * attack_ms / 1000.0)))
    e[:a] *= np.linspace(0.0, 1.0, a)
    return e


def _vibrato(n, sr, depth, rate, np):
    if depth <= 0:
        return np.ones(n)
    t = np.arange(n, dtype=np.float64) / sr
    # Delay onset — vibrato from the very first sample sounds seasick.
    on = np.clip((t - 0.12) * 6.0, 0.0, 1.0)
    return 1.0 + depth * on * np.sin(2 * np.pi * rate * t)


class Track:
    """Accumulates one channel's audio; every voice mixes into a flat buffer."""

    def __init__(self, n, np):
        self.buf = np.zeros(n)
        self.np = np

    def add(self, start, audio, gain=1.0):
        end = min(len(self.buf), start + len(audio))
        if end > start:
            self.buf[start:end] += audio[:end - start] * gain


def compose(a, np, rng):
    """Build the note plan. Returns (events, bars, spb) — pure data, no audio.

    Kept separate from rendering so the musical decisions are inspectable and
    the synthesis stays dumb.
    """
    root, scale_name = parse_key(a.key)
    scale = SCALES.get(getattr(a, "chip_scale", None) or scale_name, SCALES["minor"])
    prog = PROGRESSIONS.get(scale_name, PROGRESSIONS["minor"])
    prog = prog[rng.integers(len(prog))]

    bpm = max(40, min(300, a.bpm))
    spb = 60.0 / bpm * 4.0                       # seconds per bar (4/4)
    bars = max(2, int(round(a.seconds / spb)))
    # Even bar count keeps the progression whole, which is what makes the loop
    # land musically rather than merely sample-accurately.
    if bars % len(prog):
        bars += len(prog) - (bars % len(prog))

    def deg(d, octave=0):
        """Scale degree -> midi, wrapping octaves."""
        o, i = divmod(d, len(scale))
        return 12 * (4 + octave + o) + root + scale[i]

    # One motif, reused with variation. Repetition is what makes a chiptune read
    # as a tune rather than as noodling — the format has no room for through-
    # composition and the era's music leans hard on the hook.
    motif_r = RHYTHMS[rng.integers(len(RHYTHMS))]
    motif_d = [int(rng.integers(0, 5)) for _ in motif_r]
    drums = DRUMS[rng.integers(len(DRUMS))]
    duty_lead = DUTIES[rng.integers(len(DUTIES))]

    ev = {"lead": [], "arp": [], "bass": [], "drum": []}
    for bar in range(bars):
        ch = prog[bar % len(prog)]
        chord = [deg(ch), deg(ch + 2), deg(ch + 4)]      # triad on that degree

        # --- lead: the motif, transposed onto the chord, varied every 4th bar
        pos = 0
        vary = (bar % 4 == 3)
        for j, d in enumerate(motif_r):
            step = motif_d[j]
            if vary and rng.random() < 0.5:
                step += int(rng.integers(-1, 3))
            n = deg(ch + step, 1)
            if j == 0:
                n = chord[0] + 12                        # land on the root
            ev["lead"].append((bar, pos, d, n, duty_lead))
            pos += d

        # --- arp: the signature. Chord tones cycled every `arp` sixteenth.
        rate = max(1, int(getattr(a, "chip_arp", 1)))
        k = 0
        for s in range(0, STEPS, rate):
            ev["arp"].append((bar, s, rate, chord[k % 3], 0.25))
            k += 1

        # --- bass: triangle, root with a fifth on the off-beat
        for s in (0, 4, 8, 12):
            n = chord[0] - 12 if s in (0, 8) else chord[2] - 12
            ev["bass"].append((bar, s, 4, n))

        # --- drums
        for s in range(STEPS):
            for kind in ("k", "s", "h"):
                if drums[kind][s] == "1":
                    ev["drum"].append((bar, s, kind))

    return ev, bars, spb, scale_name, prog


def render(a, ev, bars, spb, np):
    sr = SAMPLE_RATE
    total = int(round(bars * spb * sr))
    step_s = spb / STEPS
    tr = Track(total, np)

    def at(bar, step):
        return int(round((bar * spb + step * step_s) * sr))

    for bar, step, dur, note, duty in ev["lead"]:
        n = int(dur * step_s * sr)
        if n < 8:
            continue
        f = _hz(note)
        vib = _vibrato(n, sr, 0.006 if dur >= 6 else 0.0, 6.0, np)
        t = np.arange(n, dtype=np.float64) / sr
        ph = 2 * np.pi * f * np.cumsum(vib) / sr
        w = np.where((ph / (2 * np.pi)) % 1.0 < duty, 1.0, -1.0)
        tr.add(at(bar, step), w * _env(n, sr, 2, 6.0, 0.35, np), 0.26)

    for bar, step, dur, note, duty in ev["arp"]:
        n = int(dur * step_s * sr)
        if n < 4:
            continue
        w = _pulse(_hz(note), n, sr, duty, np)
        tr.add(at(bar, step), w * _env(n, sr, 1, 30.0, 0.0, np), 0.15)

    for bar, step, dur, note in ev["bass"]:
        n = int(dur * step_s * sr)
        if n < 8:
            continue
        w = _tri(_hz(note), n, sr, np)
        tr.add(at(bar, step), w * _env(n, sr, 3, 3.0, 0.75, np), 0.34)

    # Noise hits are short and reused, so synthesize each kind once.
    hits = {
        "k": (_noise(int(0.11 * sr), sr, 900.0, np, 7)
              * _env(int(0.11 * sr), sr, 1, 46.0, 0.0, np), 0.42),
        "s": (_noise(int(0.13 * sr), sr, 7000.0, np, 3)
              * _env(int(0.13 * sr), sr, 1, 30.0, 0.0, np), 0.24),
        "h": (_noise(int(0.035 * sr), sr, 17000.0, np, 11)
              * _env(int(0.035 * sr), sr, 1, 90.0, 0.0, np), 0.10),
    }
    for bar, step, kind in ev["drum"]:
        w, g = hits[kind]
        tr.add(at(bar, step), w, g)

    out = tr.buf
    # A real 2A03 sums its channels into one mono DAC. Keep it mono; fake stereo
    # would be the first thing that gives the game away.
    peak = float(np.abs(out).max())
    if peak > 1e-9:
        out = out * (10.0 ** (a.normalize_db / 20.0) / peak)
    return out


def run(a, slug, to_ogg=None, loudness_normalize=None):
    """Generate chiptune. Same contract as narrate.run()/blip.run()."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        sys.exit(f"--chip needs numpy + soundfile: pip install soundfile numpy   ({e})")

    dest = os.path.abspath(os.path.expanduser(a.output_to)) if a.output_to else os.getcwd()
    if not os.path.isdir(dest):
        if a.create_dirs or a.output_to:
            os.makedirs(dest, exist_ok=True)
        else:
            sys.exit(f"output dir does not exist: {dest} (add --create-dirs)")

    base = a.name or slug(a.prompt or "chiptune")
    n_out = max(1, a.number)
    made = []
    for i in range(n_out):
        # soundmon's convention is --seed -1 for "random", NOT None. Treating
        # None as the sentinel silently used -1 as a literal seed, which made
        # every unseeded run produce the same two tracks.
        if a.seed is not None and a.seed >= 0:
            seed = (a.seed + i) % (2 ** 32)
        else:
            seed = int.from_bytes(os.urandom(4), "big")
        rng = np.random.default_rng(seed)

        ev, bars, spb, scale_name, prog = compose(a, np, rng)
        audio = render(a, ev, bars, spb, np)

        # The seed goes in EVERY filename, as it does for every other engine.
        # That is the documented way to re-run a take you liked, and the pack
        # generator relies on the `<name>_*` shape to rename output to the
        # manifest key — writing a bare `<name>.wav` made it report "nothing
        # produced" for files that were sitting right there.
        name = f"{base}_s{seed}"
        path = os.path.join(dest, f"{name}.wav")
        sf.write(path, audio, SAMPLE_RATE,
                 subtype=f"PCM_{a.bits}" if a.bits != 8 else "PCM_U8")
        if getattr(a, "lufs_target", None) is not None and loudness_normalize:
            loudness_normalize(path, a.lufs_target, a.true_peak)
        if getattr(a, "ogg", False) and to_ogg:
            path = to_ogg(path, a.ogg_quality, a.keep_wav)
        made.append(path)
        shown = a.key if not getattr(a, "chip_scale", None) else f"{a.key}->{scale_name}"
        print(f"   ✅ [{i+1}/{n_out}] {os.path.basename(path):<30} "
              f"{len(audio)/SAMPLE_RATE:5.1f}s  {bars} bars  {shown}  "
              f"{a.bpm}bpm  seed={seed}")

    print(f"   all done  |  {len(made)} file(s) in {dest}")
    print("   ↻ loops seamlessly by construction — whole bars, no crossfade needed")
    return made
