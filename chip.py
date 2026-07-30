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

import theory

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

# --- moods -------------------------------------------------------------------
# A mood is a BUNDLE of composition parameters, not a label. Changing the scale
# alone does very little: "ominous" is a dark scale AND a slow tempo AND a thin
# duty cycle AND sparse drums AND a falling melodic contour, all at once. Pull
# one lever and you get minor-key cheerfulness, which is the usual failure of
# procedural music.
#
#   scales    preferred scales, in order of preference
#   progs     chord progressions as scale degrees
#   bpm       multiplier on the requested tempo
#   rhythms   which RHYTHMS entries the motif may use (dense vs sparse)
#   arp       arpeggio speed in 16ths — 1 is a buzzy chord, 4 is an audible figure
#   duty      index into DUTIES — 0 is thin/nasal, 2 is full/bold
#   drums     index into DRUMS
#   octave    lead register shift
#   contour   melodic bias: +1 tends to rise, -1 tends to fall
#   vib       vibrato depth on sustained notes
#   span      how far the motif roams from the chord (small = insistent)
MOODS = {
    "heroic":      dict(scales=("major", "mixolydian"), progs=([0, 4, 5, 3], [0, 3, 4, 4]),
                        bpm=1.00, rhythms=(0, 1, 2, 5), arp=1, duty=2, drums=1,
                        octave=1, contour=+1, vib=0.006, span=4, cadences=("authentic","half"),
                        contours=("ascending","arch","terraced"), seventh=0.25),
    "triumphant":  dict(scales=("major",), progs=([0, 3, 4, 0], [0, 5, 3, 4]),
                        bpm=1.05, rhythms=(1, 2, 5, 10), arp=1, duty=2, drums=1,
                        octave=1, contour=+1, vib=0.008, span=5, cadences=("authentic",),
                        contours=("ascending","arch"), seventh=0.2),
    "ominous":     dict(scales=("phrygian", "harmonic"), progs=([0, 1, 0, 6], [0, 6, 5, 6]),
                        bpm=0.80, rhythms=(8, 9, 3), arp=2, duty=0, drums=3,
                        octave=0, contour=-1, vib=0.004, span=3, cadences=("phrygian","plagal"),
                        contours=("descending","static","valley"), seventh=0.45),
    "eerie":       dict(scales=("harmonic", "phrygian"), progs=([0, 1, 5, 0], [0, 6, 4, 0]),
                        bpm=0.75, rhythms=(8, 9, 10), arp=3, duty=0, drums=3,
                        octave=1, contour=0, vib=0.012, span=6, cadences=("phrygian","deceptive"),
                        contours=("valley","static","arch"), seventh=0.55),
    "melancholy":  dict(scales=("minor", "dorian"), progs=([0, 5, 2, 6], [0, 3, 0, 6]),
                        bpm=0.78, rhythms=(8, 9, 2, 10), arp=4, duty=1, drums=3,
                        octave=0, contour=-1, vib=0.007, span=3, cadences=("plagal","deceptive"),
                        contours=("descending","arch"), seventh=0.4),
    "solemn":      dict(scales=("minor", "harmonic"), progs=([0, 3, 4, 0], [0, 5, 4, 0]),
                        bpm=0.72, rhythms=(8, 9), arp=4, duty=1, drums=3,
                        octave=0, contour=0, vib=0.005, span=2, cadences=("plagal","authentic"),
                        contours=("descending","static"), seventh=0.3),
    "mysterious":  dict(scales=("dorian", "minor"), progs=([0, 6, 3, 0], [0, 3, 0, 6]),
                        bpm=0.88, rhythms=(3, 4, 7, 10), arp=2, duty=1, drums=0,
                        octave=1, contour=0, vib=0.009, span=5, cadences=("deceptive","plagal"),
                        contours=("valley","terraced","arch"), seventh=0.45),
    "tense":       dict(scales=("phrygian", "minor"), progs=([0, 0, 5, 6], [0, 1, 0, 6]),
                        bpm=1.02, rhythms=(7, 3, 4), arp=1, duty=0, drums=1,
                        octave=0, contour=0, vib=0.003, span=2, cadences=("half","phrygian"),
                        contours=("static","terraced"), seventh=0.5),
    "frantic":     dict(scales=("phrygian", "harmonic", "minor"),
                        progs=([0, 6, 5, 6], [0, 1, 0, 6]),
                        bpm=1.22, rhythms=(3, 7, 4), arp=1, duty=0, drums=1,
                        octave=1, contour=+1, vib=0.004, span=4, cadences=("authentic","phrygian"),
                        contours=("ascending","valley"), seventh=0.35),
    "driving":     dict(scales=("minor", "dorian"), progs=([0, 6, 5, 6], [0, 3, 4, 0]),
                        bpm=1.12, rhythms=(0, 3, 5, 7), arp=1, duty=2, drums=1,
                        octave=0, contour=+1, vib=0.005, span=3, cadences=("authentic","plagal"),
                        contours=("ascending","terraced"), seventh=0.3),
    "playful":     dict(scales=("pentatonic", "major", "mixolydian"),
                        progs=([0, 3, 4, 0], [0, 4, 0, 3]),
                        bpm=1.08, rhythms=(3, 4, 6, 7), arp=2, duty=2, drums=0,
                        octave=1, contour=+1, vib=0.006, span=4, cadences=("authentic","plagal"),
                        contours=("arch","ascending","terraced"), seventh=0.15),
    "serene":      dict(scales=("major", "dorian", "pentatonic"),
                        progs=([0, 5, 3, 4], [0, 3, 0, 4]),
                        bpm=0.82, rhythms=(8, 9, 2), arp=4, duty=1, drums=0,
                        octave=1, contour=0, vib=0.010, span=3, cadences=("plagal",),
                        contours=("arch","static"), seventh=0.2),
    "grand":       dict(scales=("harmonic", "minor"), progs=([0, 3, 4, 0], [0, 5, 4, 0]),
                        bpm=0.90, rhythms=(1, 2, 9), arp=2, duty=2, drums=2,
                        octave=0, contour=+1, vib=0.008, span=4, cadences=("authentic","plagal"),
                        contours=("ascending","arch"), seventh=0.35),
    "wondrous":    dict(scales=("major", "dorian"), progs=([0, 3, 5, 4], [0, 4, 5, 3]),
                        bpm=0.95, rhythms=(2, 4, 6, 10), arp=2, duty=1, drums=0,
                        octave=2, contour=+1, vib=0.011, span=6, cadences=("plagal","deceptive"),
                        contours=("arch","ascending","valley"), seventh=0.3),
}
DEFAULT_MOOD = "mysterious"

# Words -> mood, matched against the asset's description. The manifest already
# says how each track should FEEL ("crushing finality and grand grieving
# menace"), so the mood does not need to be specified twice — read it from the
# text that is already there. Longer, more specific keys are checked first.
MOOD_WORDS = [
    ("triumphant", ("triumph", "victor", "fanfare", "champion", "glory", "won")),
    ("heroic", ("heroic", "hero", "brave", "noble", "valiant", "hopeful", "destiny")),
    ("frantic", ("frantic", "panic", "desperate", "chaos", "furious", "intense", "climax")),
    ("driving", ("driving", "combat", "battle", "pursuit", "urgent", "relentless", "march")),
    ("tense", ("tense", "tension", "uneasy", "nervous", "danger", "threat", "stalk", "dread")),
    ("ominous", ("ominous", "menace", "malevolent", "sinister", "foreboding", "evil", "doom")),
    ("eerie", ("eerie", "haunt", "ghost", "spectral", "unsettl", "otherworldly", "horror")),
    ("melancholy", ("melancholy", "mourn", "grief", "griev", "sorrow", "lament", "sad", "lost")),
    ("solemn", ("solemn", "funeral", "dirge", "memorial", "requiem", "reverent", "finality")),
    ("grand", ("grand", "vast", "epic", "monument", "cathedral", "immense", "awe")),
    ("wondrous", ("wonder", "wondrous", "shimmer", "glitter", "magic", "arcane", "treasure")),
    ("mysterious", ("mystery", "mysterious", "curious", "secret", "hidden", "riddle", "unknown")),
    ("playful", ("playful", "cheerful", "jaunty", "whimsic", "bright", "village", "merry")),
    ("serene", ("serene", "calm", "quiet", "gentle", "peace", "contemplat", "reflect", "still")),
]


def infer_mood(text):
    """Pick a mood from a description. Falls back to DEFAULT_MOOD."""
    hay = (text or "").lower()
    best, score = DEFAULT_MOOD, 0
    for mood, words in MOOD_WORDS:
        hits = sum(1 for w in words if w in hay)
        # First match wins ties, so the table order encodes priority: a track
        # described as both "combat" and "tense" should read as combat.
        if hits > score:
            best, score = mood, hits
    return best


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
    """Build the note plan. Pure data — no audio.

    Structure comes from theory.plan_track(), which is deterministic on the TRACK
    NAME rather than on the seed. That split is the whole diversity fix: 24 names
    in a pack produce 24 different plans — key, form, progressions, contours,
    tempo — while `-n 8` on one name gives eight takes of the SAME piece instead
    of eight different pieces.

    Everything remains bounded by the mood, so a crypt never comes out cheerful;
    it just stops sounding like the armoury.
    """
    mname = getattr(a, "mood", None)
    if not mname or mname == "auto":
        mname = infer_mood(getattr(a, "prompt", "") or "")
    mood = MOODS.get(mname, MOODS[DEFAULT_MOOD])

    home_root, key_scale = parse_key(a.key)
    track = a.name or "untitled"

    # Scale: explicit override wins, then a mode spelled out in --key, else the
    # mood's — chosen by track identity so two `solemn` tracks can differ.
    explicit = getattr(a, "chip_scale", None)
    if explicit:
        scale_name = explicit
    elif any(w in (a.key or "").lower() for w in
             ("dorian", "phrygian", "harmonic", "mixolydian", "pentatonic", "lydian")):
        scale_name = key_scale
    else:
        scale_name = theory.Ident(track, "scale:" + mname).pick(mood["scales"])
    scale = theory.SCALES.get(scale_name, theory.SCALES["minor"])

    # The key ROTATES through closely related keys across a pack (i/iv/v/bIII/
    # bVII/bVI), weighted toward home. That is what stops 24 tracks sharing one
    # key without scattering them into 12 unrelated ones.
    root = theory.key_for_track(track, home_root, mname)

    est_spb = 60.0 / max(40.0, min(300.0, a.bpm * mood["bpm"])) * 4.0
    plan = theory.plan_track(track, mood, mname, scale_name,
                             max(4, int(round(a.seconds / est_spb))))

    bpm = max(40.0, min(300.0, a.bpm * plan["tempo_mul"]))
    # Bar length now follows the METER, not a hardcoded 4 beats. This is what
    # stops every track sharing one meter — 3/4, 6/8, 7/8 and 12/8 all produce
    # genuinely different bar lengths and accent patterns.
    steps = plan["steps"]
    spb_steps = plan["spb_steps"]
    spb = 60.0 / bpm * plan["quarters"]
    form = plan["form"]
    # Re-size to the REAL bar length. plan_track had to guess with a 4/4 bar, so
    # a 2/4 track came out half as long as asked and a 12/8 one half again over.
    want_bars = max(len(form), int(round(a.seconds / spb)))
    bps = max(2, -(-want_bars // len(form)))          # ceil, so we never undershoot
    bars = bps * len(form)

    def deg(d, octave=0):
        o, i = divmod(d, len(scale))
        return 12 * (4 + octave + o) + root + scale[i]

    # One motif for the first section; every other section DEVELOPS it via a
    # transformation (inversion, sequence, retrograde, augmentation...). Wholly
    # new material per section fragments a track; verbatim repetition bores.
    first = form[0]
    motifs, kits, basses = {}, {}, {}
    for letter, sec in plan["sections"].items():
        m = theory.make_motif(theory.Ident(track, "motif:%s:%s" % (letter, mname)),
                              sec["rhythm"], sec["contour"], mood["span"], spb_steps)
        if letter != first and sec["transform"]:
            m = theory.transform(m, sec["transform"],
                                 theory.Ident(track, "tr:" + letter))
        motifs[letter] = m
        # Per-section kit and bass style, so the B section does not merely change
        # notes — it changes groove. One bass rhythm for the whole pack was the
        # loudest single cause of "the meter feels the same".
        kits[letter] = theory.kit_pattern(sec["kit"], steps, spb_steps)
        basses[letter] = theory.bass_onsets(sec["bass_style"], steps, spb_steps)

    duty_lead = DUTIES[plan["duty"] % len(DUTIES)]

    ev = {"lead": [], "arp": [], "bass": [], "drum": []}
    prev_bass = None
    bar = 0
    for letter in form:
        sec = plan["sections"][letter]
        prog = plan["progs"][letter]
        motif = motifs[letter]
        lead_oct = 1 + sec["octave"]
        arp_rate = max(1, int(getattr(a, "chip_arp", 0) or sec["arp"]))
        for b in range(bps):
            degree, seventh = prog[b % len(prog)]
            chord = [deg(degree), deg(degree + 2), deg(degree + 4)]
            if seventh:
                chord.append(deg(degree + 6))

            # --- lead: the motif over this chord, resolving at the section end.
            # Rests are real silences, not notes — a melody that never stops has
            # no phrase structure, which is most of why tracks felt identical.
            pos = 0
            last = len(motif) - 1
            for j, (d, step, is_rest) in enumerate(motif):
                if pos >= steps:
                    break
                if not is_rest:
                    n = deg(degree + step, lead_oct)
                    if j == 0:
                        n = deg(degree, lead_oct)          # anchor on the chord
                    if b == bps - 1 and j == last:
                        n = deg(0, lead_oct)               # cadence -> tonic
                    ev["lead"].append((bar, pos, min(d, steps - pos), n, duty_lead))
                pos += d

            # --- arp: the signature. Chord tones cycled every `arp_rate` step.
            k = 0
            for s in range(0, steps, arp_rate):
                ev["arp"].append((bar, s, arp_rate, chord[k % len(chord)], 0.25))
                k += 1

            # --- bass: the section's rhythm STYLE, voice-led for motion
            low = [c - 12 for c in chord]
            for si, (s, d) in enumerate(basses[letter]):
                if s >= steps:
                    break
                if si == 0:
                    bn = theory.voice_bass(prev_bass, low,
                                           ident=theory.Ident(track, "bass:%d" % bar))
                    prev_bass = bn
                else:
                    bn = low[si % len(low)] if si % 2 else prev_bass
                ev["bass"].append((bar, s, min(d, steps - s), bn))

            # --- drums: kit realized for THIS meter
            kit = kits[letter]
            for kind in ("k", "s", "h"):
                for s in kit[kind]:
                    ev["drum"].append((bar, s, kind))
            bar += 1

    plan["_steps"] = steps
    return ev, bars, spb, scale_name, plan["progs"][first], mname, mood, plan


def render(a, ev, bars, spb, np, mood=None, steps=STEPS):
    sr = SAMPLE_RATE
    total = int(round(bars * spb * sr))
    step_s = spb / steps
    tr = Track(total, np)

    def at(bar, step):
        return int(round((bar * spb + step * step_s) * sr))

    for bar, step, dur, note, duty in ev["lead"]:
        n = int(dur * step_s * sr)
        if n < 8:
            continue
        f = _hz(note)
        depth = (mood or {}).get("vib", 0.006)
        vib = _vibrato(n, sr, depth if dur >= 6 else 0.0, 6.0, np)
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

        mfile = getattr(a, "from_midi", None)
        if mfile:
            # MIDI: nothing is inferred. Pitches, durations, tempo and metre all
            # come from the file, so this is strictly better than --from-audio
            # when a MIDI exists. Mood still picks the voicing.
            import midi as midimod
            mname = getattr(a, "mood", None)
            if not mname or mname == "auto":
                mname = infer_mood(getattr(a, "prompt", "") or "")
            mood = MOODS.get(mname, MOODS[DEFAULT_MOOD])
            got = midimod.to_events(mfile, np, seconds=a.seconds,
                                    transpose=getattr(a, "transpose", 0),
                                    start_frac=getattr(a, "midi_start", 0.15))
            if not got:
                sys.exit(f"--from-midi: no playable notes in {mfile}")
            ev, bars, spb, info, scale_name = got
            steps = info["steps"]; meter_s = info["timesig"]
            print(f"   \u266a {info['title'][:40]}: {midimod.describe(info)}")
        else:
            src = getattr(a, "from_audio", None)
        if mfile:
            pass
        elif src:
            # Transcribed: the composition comes from the reference recording.
            # Mood is still resolved, because it decides how this is VOICED —
            # notes from SA3, timbre from the chip.
            import transcribe
            mname = getattr(a, "mood", None)
            if not mname or mname == "auto":
                mname = infer_mood(getattr(a, "prompt", "") or "")
            mood = MOODS.get(mname, MOODS[DEFAULT_MOOD])
            got = transcribe.to_events_hi(src, np, sf, seconds=a.seconds,
                                          beats_per_bar=4, div=4)
            if not got:
                sys.exit(f"--from-audio: could not analyze {src}")
            ev, bars, spb, ana, scale_name = got
            steps = STEPS; meter_s = '4/4'
            print(f"   ♪ {os.path.basename(src)}: "
                  f"{transcribe.NOTE_NAMES[ana['root']]} {ana['mode']}  "
                  f"{ana['bpm']:.0f}bpm  {ana['notes']} notes  "
                  f"{ana['drum_hits']} hits")
        else:
            ev, bars, spb, scale_name, prog, mname, mood, plan = compose(a, np, rng)
            steps = plan["_steps"]; meter_s = plan["meter"]
        audio = render(a, ev, bars, spb, np, mood, steps)

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
        if getattr(a, "write_midi", False):
            # Additional output, not a substitute: the chip render is the point.
            import midi as _midiw
            mp = os.path.splitext(path)[0] + ".mid"
            try:
                _midiw.write_smf(mp, ev, bars, spb, steps, mood=mname,
                                 timesig=meter_s, title=base)
                print(f"   \u266b also wrote {os.path.basename(mp)}")
            except Exception as e:
                print(f"   \u26a0 midi write failed: {e}")
        made.append(path)
        print(f"   ✅ [{i+1}/{n_out}] {os.path.basename(path):<30} "
              f"{len(audio)/SAMPLE_RATE:5.1f}s  {bars}bar {meter_s:<5}{mname:<11}"
              f"{scale_name:<11}{a.bpm*mood['bpm']:.0f}bpm  seed={seed}")

    print(f"   all done  |  {len(made)} file(s) in {dest}")
    print("   ↻ loops seamlessly by construction — whole bars, no crossfade needed")
    return made
