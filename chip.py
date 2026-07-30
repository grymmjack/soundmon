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

    # Identity is PACK + TRACK, not track alone.
    #
    # Deriving it from the track name only meant `level9` composed identically in
    # every pack: same mood (same manifest description), therefore same scale,
    # form, progression, meter and motif. Two packs then differed solely in key,
    # tempo and timbre — which is why chiptune and adlib sounded like one piece
    # played twice rather than two soundtracks.
    #
    # The pack comes from the output directory, so nothing needs a new flag and
    # every existing caller gets distinct music for free. Still fully
    # deterministic: the same pack+track always regenerates the same plan.
    track = a.name or "untitled"
    _pack = (os.path.basename(os.path.normpath(a.output_to))
             if getattr(a, "output_to", None) else "")
    if _pack:
        track = f"{_pack}/{track}"

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
        mid_ = theory.Ident(track, "motif:%s:%s" % (letter, mname))
        # Learned interval walk when a corpus exists; hand-written contour
        # archetypes otherwise.
        m = (theory.corpus_motif(mid_, sec["rhythm"], scale_name, len(scale),
                                 mood["span"])
             or theory.make_motif(mid_, sec["rhythm"], sec["contour"],
                                  mood["span"], spb_steps))
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
                    ev["lead"].append((bar, pos, min(d, steps - pos), n,
                                       duty_lead, theory.accent(pos, spb_steps, 104)))
                pos += d

            # --- arp: the signature. Chord tones cycled every `arp_rate` step.
            k = 0
            for s in range(0, steps, arp_rate):
                ev["arp"].append((bar, s, arp_rate, chord[k % len(chord)], 0.25,
                                  theory.accent(s, spb_steps, 74)))
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
                ev["bass"].append((bar, s, min(d, steps - s), bn,
                                   theory.accent(s, spb_steps, 92)))

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

    for _it in ev["lead"]:
        bar, step, dur, note, duty = _it[:5]
        vgain = (_it[5] / 104.0) if len(_it) > 5 else 1.0
        n = int(dur * step_s * sr)
        if n < 8:
            continue
        f = _hz(note)
        depth = (mood or {}).get("vib", 0.006)
        vib = _vibrato(n, sr, depth if dur >= 6 else 0.0, 6.0, np)
        t = np.arange(n, dtype=np.float64) / sr
        ph = 2 * np.pi * f * np.cumsum(vib) / sr
        w = np.where((ph / (2 * np.pi)) % 1.0 < duty, 1.0, -1.0)
        tr.add(at(bar, step), w * _env(n, sr, 2, 6.0, 0.35, np), 0.26 * vgain)

    for _it in ev["arp"]:
        bar, step, dur, note, duty = _it[:5]
        vgain = (_it[5] / 74.0) if len(_it) > 5 else 1.0
        n = int(dur * step_s * sr)
        if n < 4:
            continue
        w = _pulse(_hz(note), n, sr, duty, np)
        tr.add(at(bar, step), w * _env(n, sr, 1, 30.0, 0.0, np), 0.15 * vgain)

    for _it in ev["bass"]:
        bar, step, dur, note = _it[:4]
        vgain = (_it[4] / 92.0) if len(_it) > 4 else 1.0
        n = int(dur * step_s * sr)
        if n < 8:
            continue
        w = _tri(_hz(note), n, sr, np)
        tr.add(at(bar, step), w * _env(n, sr, 3, 3.0, 0.75, np), 0.34 * vgain)

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

    out = _dc_block(tr.buf, np, sr)
    # A real 2A03 sums its channels into one mono DAC. Keep it mono; fake stereo
    # would be the first thing that gives the game away.
    peak = float(np.abs(out).max())
    if peak > 1e-9:
        out = out * (10.0 ** (a.normalize_db / 20.0) / peak)
    return out


def run(a, slug, to_ogg=None, loudness_normalize=None, to_flac=None):
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
            # EXACT timing. MIDI carries its own; quantizing it only destroys
            # information that was already correct.
            timed = midimod.to_timed_events(
                mfile, seconds=a.seconds,
                transpose=getattr(a, "transpose", 0),
                start_frac=getattr(a, "midi_start", 0.15))
            if not timed:
                sys.exit(f"--from-midi: no playable notes in {mfile}")
            tnotes, tdrums, info = timed
            ev, bars, spb, scale_name = None, 0, info["spb"], "minor"
            steps = STEPS; meter_s = info["timesig"]
            print(f"   \u266a {info['title'][:38]}: {info['timesig']} "
                  f"{info['bpm']:.0f}bpm  {info['duration']:.1f}s  "
                  f"{info['notes']} notes  {info['drum_hits']} hits")
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
        if mfile:
            audio = render_timed(a, tnotes, tdrums, info["duration"], np, mood)
        else:
            # Procedural through the same renderer: inherits --chippy arpeggios,
            # slides and the register-based voicing. Grid positions, exact-time
            # representation.
            pnotes, pdrums = events_to_timed(ev, bars, spb, steps, mname)
            audio = render_timed(a, pnotes, pdrums, bars * spb, np, mood)

        # The seed goes in EVERY filename, as it does for every other engine.
        # That is the documented way to re-run a take you liked, and the pack
        # generator relies on the `<name>_*` shape to rename output to the
        # manifest key — writing a bare `<name>.wav` made it report "nothing
        # produced" for files that were sitting right there.
        name = f"{base}_s{seed}"
        # Hardware format lock, if asked. Applied here rather than in the
        # ComfyUI node because these engines never touch the graph.
        out_sr = SAMPLE_RATE
        out_bits = a.bits
        if getattr(a, "format", "none") not in (None, "none"):
            _mod = sys.modules.get("chip") or __import__("chip")
            audio, out_sr = _mod.format_lock_np(audio, SAMPLE_RATE, a.format, np)
            # Store at the format's OWN bit depth. Writing 8-bit audio into a
            # 16-bit container wastes half the file and is a lie about the data;
            # it also stops FLAC from compressing what is really an 8-bit signal.
            _spec = _mod.FORMATS.get(a.format) or {}
            _fb = int(_spec.get("bits") or 0)
            if _fb:
                out_bits = 8 if _fb <= 8 else (16 if _fb <= 16 else 24)
        path = os.path.join(dest, f"{name}.wav")
        sf.write(path, audio, out_sr,
                 subtype=f"PCM_{out_bits}" if out_bits != 8 else "PCM_U8")
        if getattr(a, "lufs_target", None) is not None and loudness_normalize:
            loudness_normalize(path, a.lufs_target, a.true_peak)
        # FLAC before OGG: it is lossless, so for format-locked audio it is the
        # only compression that preserves the crush. Asking for both means FLAC.
        if getattr(a, "flac", False) and to_flac:
            path = to_flac(path, a.keep_wav)
        elif getattr(a, "ogg", False) and to_ogg:
            path = to_ogg(path, a.ogg_quality, a.keep_wav)
        if getattr(a, "write_mod", False):
            import mod as _modw
            gev, gbars = (ev, bars) if ev is not None else timed_to_grid(
                tnotes, tdrums, spb, STEPS)
            mp = os.path.splitext(path)[0] + ".mod"
            try:
                _modw.write_mod(mp, gev, gbars, spb, STEPS, np,
                                title=base[:20], bpm=int(a.bpm))
                print(f"   \u266b also wrote {os.path.basename(mp)}")
            except Exception as e:
                print(f"   \u26a0 mod write failed: {e}")
        if getattr(a, "write_midi", False) and ev is not None:
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


def render_timed(a, notes, drums, duration, np, mood=None):
    """Render exact-time MIDI notes on the 2A03 voices. No grid.

    The OPL path moved to event timing but this one was left on a 16th grid, so
    91% of notes still had their onsets and lengths rounded. Same fix here: each
    note is placed at its real sample offset for its real duration.

    Voice assignment is by REGISTER, which is how a 4-channel arrangement works
    anyway: low notes take the triangle (the 2A03's bass voice), everything else
    takes a pulse, with a thinner duty for the upper register so the melody sits
    above the harmony instead of blending into it.
    """
    sr = SAMPLE_RATE
    total = int(round((duration + 0.4) * sr))       # tail for the last release
    buf = np.zeros(total)
    depth = (mood or {}).get("vib", 0.006)

    cfg = CHIPPY.get(getattr(a, "chippy", "off") or "off", CHIPPY["off"])
    depth = max(depth, cfg["vib"])

    # ONE arpeggio implementation, shared with the OPL path. This function used to
    # carry its own copy, which is exactly how chip and opl drifted apart before.
    notes = chippify(notes, getattr(a, "chippy", "off"))

    for n in notes:
        start = int(round(n["t"] * sr))
        ln = max(int(round(n["dur"] * sr)), int(0.02 * sr))
        if start >= total:
            continue
        ln = min(ln, total - start)
        pitch = n["pitch"]
        vel = (n.get("vel") or 100) / 127.0
        hz = 440.0 * (2.0 ** ((pitch - 69) / 12.0))

        if pitch < BASS_MAX_PITCH:                  # triangle bass, never arped
            buf[start:start + ln] += (_tri_sweep_note(hz, ln, sr, np)
                                      * _env(ln, sr, 3.0, 3.0, 0.75, np)
                                      * 0.34 * vel)
            continue

        duty = 0.125 if pitch >= 72 else (0.25 if pitch >= 60 else 0.5)
        if cfg["duty_mod"] > 1:
            duty = 0.125
        sf_from = n.get("slide_from")
        if sf_from is not None:
            hz0 = 440.0 * (2.0 ** ((sf_from - 69) / 12.0))
            w = _slide_pulse(hz0, hz, ln, sr, duty,
                             n.get("slide_ms", cfg["slide_ms"]), np, depth)
        else:
            vib = _vibrato(ln, sr, depth if n["dur"] > 0.35 else 0.0, 6.0, np)
            t = np.arange(ln, dtype=np.float64) / sr
            ph = 2 * np.pi * hz * np.cumsum(vib) / sr
            w = np.where((ph / (2 * np.pi)) % 1.0 < duty, 1.0, -1.0)
        buf[start:start + ln] += w * _env(ln, sr, 2.0, 4.0, 0.55, np) * 0.20 * vel

    for t, gm in (drums or []):
        start = int(round(t * sr))
        if start >= total:
            continue
        kind = GM_DRUM_KIND.get(gm, "h")
        period, dec, g, dur_s = {"k": (900.0, 46.0, 0.42, 0.11),
                                 "s": (7000.0, 30.0, 0.24, 0.13),
                                 "h": (17000.0, 90.0, 0.10, 0.035)}[kind]
        ln = min(int(dur_s * sr), total - start)
        if ln <= 4:
            continue
        w = _lfsr_note(ln, sr, period, np) * _env(ln, sr, 1.0, dec, 0.0, np)
        buf[start:start + ln] += w * g

    buf = _chip_verb(buf, sr, cfg["verb"], np)
    buf = _dc_block(buf, np, sr)          # before normalising: DC steals headroom
    peak = float(np.abs(buf).max())
    if peak > 1e-9:
        buf = buf * (10.0 ** (a.normalize_db / 20.0) / peak)
    return buf


# GM percussion note -> the 2A03's single noise channel. Three categories is all
# one noise generator can express, unlike a WOPL percussion bank.
GM_DRUM_KIND = {35: "k", 36: "k", 41: "k", 43: "k", 45: "k", 47: "k",
                38: "s", 40: "s", 37: "s", 39: "s", 48: "s", 50: "s"}


def _tri_sweep_note(hz, n, sr, np):
    t = np.arange(n, dtype=np.float64) / sr
    saw = (hz * t) % 1.0
    return np.round((2.0 * np.abs(2.0 * saw - 1.0) - 1.0) * 7.5) / 7.5


def _lfsr_note(n, sr, period, np, seed=1):
    out = np.empty(n)
    reg = (int(seed) & 0x7FFF) or 1
    step = max(1, int(sr / max(period, 1.0)))
    val = 1.0
    for i in range(n):
        if i % step == 0:
            fb = (reg ^ (reg >> 1)) & 1
            reg = (reg >> 1) | (fb << 14)
            val = 1.0 if (reg & 1) else -1.0
        out[i] = val
    return out


# =============================================================================
# CHIPPINESS
#
# A 2A03 has one note per channel. It cannot play a chord — so the era's composers
# faked one by cycling the chord tones every frame or two, fast enough that the ear
# fuses them into a buzzy shimmer. That fast arpeggio is more the sound of the
# format than the square waves are. They also used pitch SLIDES constantly, because
# a pitch register is cheap to ramp and it disguises how few voices there are.
#
# Playing a MIDI chord as three simultaneous notes is therefore both unfaithful and
# less characteristic: technically more accurate, audibly less chip. This dial
# trades accuracy for idiom, which matches the stated priority — music first, then
# timbre, then accuracy.
#
#   off    exact polyphony, as written
#   some   chords of 3+ arpeggiate at 32 Hz; slides on big leaps
#   lots   all chords arpeggiate at 50 Hz; most leaps slide; vibrato
#   max    frame-rate 60 Hz arpeggios, aggressive slides, duty modulation
# =============================================================================
CHIPPY = {
    "off":  dict(arp_hz=0.0,  min_notes=99, slide=0.00, slide_ms=0,
                 vib=0.000, duty_mod=0, verb=0.00),
    "some": dict(arp_hz=32.0, min_notes=3,  slide=0.35, slide_ms=45,
                 vib=0.004, duty_mod=0, verb=0.00),
    "lots": dict(arp_hz=50.0, min_notes=2,  slide=0.60, slide_ms=60,
                 vib=0.010, duty_mod=1, verb=0.10),
    "max":  dict(arp_hz=60.0, min_notes=2,  slide=0.90, slide_ms=80,
                 vib=0.018, duty_mod=2, verb=0.22),
}


def _dc_block(buf, np, sr, hz=12.0):
    """Remove DC and sub-audible drift with a one-pole high-pass.

    A pulse wave with duty != 50% has an inherent DC offset — a 12.5% duty square
    sits at a mean of -0.75 — so thin, characteristically chippy voices carry a
    lot of it. Measured on a real render: -0.088, about -21 dBFS of pure DC.

    It has to go for two reasons. It wastes headroom, because peak normalisation
    then scales a DC-shifted waveform and the audio itself ends up quieter than
    the peak suggests. And it is inaudible but not harmless: a DC step at the start
    or end of a file clicks, and it makes speakers work for nothing.

    Real hardware blocks DC at the output coupling capacitor, so removing it is
    what the chip's own analogue stage did — not a departure from authenticity.

    12 Hz corner: below anything musical, above the DC we are removing.
    """
    x = np.asarray(buf, dtype=np.float64)
    a = float(np.exp(-2.0 * np.pi * hz / sr))
    # y[n] = x[n] - x[n-1] + a*y[n-1], vectorised via lfilter when available.
    try:
        from scipy.signal import lfilter
        return lfilter([1.0, -1.0], [1.0, -a], x, axis=0)
    except Exception:
        return x - x.mean(axis=0, keepdims=True) if x.ndim > 1 else x - x.mean()


def _chip_verb(buf, sr, amount, np):
    """Subtle multi-tap echo.

    A 2A03 cannot do this — but chiptune as *heard* usually has it, because
    trackers had echo commands and emulator captures pick up room. So it belongs
    on the stylistic end of the dial rather than in the synthesis: at "max" the
    goal is what chip music sounds like on a record, not what the silicon emits.

    Three prime-ish taps rather than one, so it reads as space instead of as a
    slapback repeat.
    """
    if amount <= 0:
        return buf
    out = buf.copy()
    for ms, g in ((53.0, 1.00), (97.0, 0.62), (149.0, 0.38)):
        d = int(sr * ms / 1000.0)
        if d < len(buf):
            out[d:] += buf[:-d] * (amount * g)
    return out


def _cluster_chords(notes, tol=0.045):
    """Group near-simultaneous overlapping notes into chords.

    `tol` is how close two onsets must be to count as one chord. 45 ms is roughly
    the limit at which the ear hears a single event rather than two.
    """
    out = []
    cur = []
    for n in sorted(notes, key=lambda x: x["t"]):
        if cur and abs(n["t"] - cur[0]["t"]) <= tol:
            cur.append(n)
        else:
            if cur:
                out.append(cur)
            cur = [n]
    if cur:
        out.append(cur)
    return out


def _arp_voice(pitches, n, sr, arp_hz, duty, np, vib_depth=0.0):
    """One voice cycling through `pitches` at `arp_hz` — the chip chord."""
    per = max(1, int(sr / max(arp_hz, 1.0)))
    out = np.zeros(n)
    k = 0
    pos = 0
    while pos < n:
        ln = min(per, n - pos)
        hz = 440.0 * (2.0 ** ((pitches[k % len(pitches)] - 69) / 12.0))
        t = np.arange(ln, dtype=np.float64) / sr
        ph = 2 * np.pi * hz * t
        if vib_depth:
            ph = ph * (1.0 + vib_depth * np.sin(2 * np.pi * 6.0 * (pos / sr + t)))
        out[pos:pos + ln] = np.where((ph / (2 * np.pi)) % 1.0 < duty, 1.0, -1.0)
        pos += ln
        k += 1
    return out


def _slide_pulse(hz0, hz1, n, sr, duty, slide_ms, np, vib_depth=0.0):
    """A pulse wave that glides from hz0 to hz1 over slide_ms, then holds."""
    t = np.arange(n, dtype=np.float64) / sr
    ns = max(1, min(n, int(sr * slide_ms / 1000.0)))
    f = np.full(n, hz1)
    f[:ns] = np.linspace(hz0, hz1, ns)
    if vib_depth:
        f = f * (1.0 + vib_depth * np.sin(2 * np.pi * 6.0 * t))
    ph = 2 * np.pi * np.cumsum(f) / sr
    return np.where((ph / (2 * np.pi)) % 1.0 < duty, 1.0, -1.0)


# Never arpeggiate low notes. A fast arp in the bass register turns into mud —
# the technique works because the ear fuses RAPID HIGH tones into a chord, and
# below roughly A3 it just hears a stuttering bass. The era used arps up top and
# held or walked the bass underneath. A source that already arpeggiates low is
# untouched, because chippify only ever collapses SIMULTANEOUS notes; a written-out
# arpeggio is already sequential.
ARP_MIN_PITCH = 57          # A3
BASS_MAX_PITCH = 48         # below this goes to the triangle


def chippify(notes, level, sr_hint=44100):
    """Rewrite an exact-time note list into chip idiom. Engine-agnostic.

    Chord clusters become ONE fast-alternating line, which is how a 1-note-per-
    channel machine plays a chord and is more characteristic of the format than
    the waveform is. Returned notes are still plain exact-time dicts, so any
    renderer plays them — that is why OPL gets this for free even though an OPL3
    *can* hold a chord: AdLib-era DOS music arpeggiated constantly anyway.

    Leaps are marked with "slide_from" so a renderer that can ramp pitch does so;
    one that cannot simply ignores the key.
    """
    cfg = CHIPPY.get(level or "off", CHIPPY["off"])
    if cfg["arp_hz"] <= 0:
        return notes

    # Three registers: bass passes through, mid passes through, only the upper
    # register is eligible to be collapsed into an arpeggio.
    passthru = [n for n in notes if n["pitch"] < ARP_MIN_PITCH]
    top = [n for n in notes if n["pitch"] >= ARP_MIN_PITCH]
    step = 1.0 / cfg["arp_hz"]
    out = []

    handled = set()
    for group in _cluster_chords(top):
        pitches = sorted({g["pitch"] for g in group})
        if len(group) < cfg["min_notes"] or len(pitches) < 2:
            continue
        t0 = min(g["t"] for g in group)
        t1 = max(g["t"] + g["dur"] for g in group)
        vel = max((g.get("vel") or 100) for g in group)
        prog = group[0].get("prog", 0)
        k = 0
        t = t0
        while t < t1 - 1e-6:
            out.append({"id": ("arp", t0, k), "t": t,
                        "dur": min(step, t1 - t),
                        "pitch": pitches[k % len(pitches)],
                        "prog": prog, "vel": vel})
            t += step
            k += 1
        handled.update(id(g) for g in group)

    # Notes not swallowed by a chord keep their own timing, plus a slide marker.
    # Slides are computed over the melodic line only, so a bass note between two
    # melody notes cannot trigger a spurious glide.
    rest = [n for n in top if id(n) not in handled]
    rest.sort(key=lambda x: x["t"])
    prev = None
    for n in rest:
        m = dict(n)
        if prev is not None and cfg["slide"] > 0:
            leap = abs(m["pitch"] - prev["pitch"])
            gap = m["t"] - (prev["t"] + prev["dur"])
            if 3 <= leap <= 19 and gap < 0.12:
                m["slide_from"] = prev["pitch"]
                m["slide_ms"] = cfg["slide_ms"]
        out.append(m)
        prev = m
    out.extend(passthru)
    out.sort(key=lambda x: x["t"])
    return out


def events_to_timed(ev, bars, spb, steps, mood_name, drums_only=False):
    """Convert grid-composed events into the exact-time note list.

    This is what lets PROCEDURAL music share the renderer the MIDI path uses, and
    therefore inherit everything built for it: the DMXOPL 128-instrument bank,
    per-patch level calibration, 18-channel polyphony, the GM percussion kit and
    --chippy arpeggios. Previously the procedural path had none of those — it ran
    four fixed channels through a hand-authored 12-patch bank.

    This is NOT un-quantizing. The notes were composed on a grid and stay on it;
    their positions are merely expressed in seconds instead of step indices, which
    is the form the renderer consumes. Generated material may be gridded — only
    MIDI input must never be.

    GM programs come from the mood, so a solemn track voices as church organ and
    choir rather than defaulting to whatever patch happens to be first.
    """
    import midi as _m
    lead_gm, arp_gm, bass_gm = _m.MOOD_GM.get(mood_name, _m.MOOD_GM["mysterious"])
    step_s = spb / float(steps)

    def at(bar, st):
        return (bar * steps + st) * step_s

    out = []
    if not drums_only:
        for role, gm, key in (("lead", lead_gm, "lead"),
                              ("arp", arp_gm, "arp"),
                              ("bass", bass_gm, "bass")):
            for i, it in enumerate(ev.get(key, [])):
                bar, st, dur, pitch = it[0], it[1], it[2], it[3]
                vel = it[5] if (key != "bass" and len(it) > 5) else (
                      it[4] if (key == "bass" and len(it) > 4) else 96)
                out.append({"id": (key, i), "t": at(bar, st),
                            "dur": max(0.02, dur * step_s),
                            "pitch": int(pitch), "prog": gm,
                            "vel": int(vel) if isinstance(vel, int) else 96})
    dr = []
    kind_gm = {"k": 36, "s": 38, "h": 42}
    for bar, st, kind in ev.get("drum", []):
        dr.append((at(bar, st), kind_gm.get(kind, 42)))
    out.sort(key=lambda n: n["t"])
    return out, dr


# --- format lock for the SYNTHESIS engines -----------------------------------
# The RetroSFX node's _format_lock only runs in the ComfyUI graph, i.e. the
# diffusion path. --chip/--opl/--chipfx/--blip write their audio directly, so
# --format would have silently done nothing for exactly the engines most likely
# to want it. Same two decisions as the node version, for the same reasons:
#
#   NAIVE stride decimation, no anti-aliasing filter — the aliasing IS the sound.
#   PLAIN quantization, no dither — dither smooths the grit a ceiling exists to
#   create.
#
# Tracker musicians turn interpolation OFF for this reason; a filtered downsample
# gives you a dull quiet recording that happens to be 8 kHz.
FORMATS = {
    "none":    None,
    "amiga":   dict(rate=8363,  bits=8,  mono=True),
    "sb":      dict(rate=11025, bits=8,  mono=True),
    "sb22":    dict(rate=22050, bits=8,  mono=True),
    "gameboy": dict(rate=8192,  bits=4,  mono=True),
    "nes":     dict(rate=4096,  bits=7,  mono=True),
    "snes":    dict(rate=32000, bits=16, mono=False),
    "psx":     dict(rate=22050, bits=16, mono=False),
    "cd":      dict(rate=44100, bits=16, mono=False),
    "mod8":    dict(rate=11025, bits=8,  mono=True),
    "mod8s":   dict(rate=22050, bits=8,  mono=False),
    "mod6":    dict(rate=11025, bits=6,  mono=True),
    "mod6s":   dict(rate=22050, bits=6,  mono=False),
    "crush":   dict(rate=8000,  bits=6,  mono=True),
}


def format_lock_np(audio, sr, fmt, np):
    """Crush a mono/stereo numpy array to a hardware format. Returns (audio, sr)."""
    spec = FORMATS.get(fmt or "none")
    if not spec:
        return audio, sr
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if spec.get("mono") and a.shape[1] > 1:
        a = a.mean(axis=1, keepdims=True)
    target = int(spec.get("rate") or sr)
    if 0 < target < sr:
        stepf = sr / float(target)
        n = int(a.shape[0] / stepf)
        if n > 1:
            idx = np.minimum((np.arange(n) * stepf).astype(np.int64), a.shape[0] - 1)
            a = a[idx]
            sr = target
    bits = int(spec.get("bits") or 0)
    if 0 < bits < 24:
        peak = float(2 ** bits) / 2.0 - 1.0
        a = np.round(np.clip(a, -1.0, 1.0) * peak) / peak
    return (a[:, 0] if a.shape[1] == 1 else a), sr


def timed_to_grid(notes, drums, spb, steps, bars=None):
    """Exact-time notes -> a grid event dict, for tracker output.

    Quantizing here is not the mistake the MIDI rule warns about. That rule exists
    because a rendered WAVEFORM must honour the source's real timing. A tracker
    pattern IS a grid — rows are its only representation of time — so snapping to
    rows is the format's native form, not a loss imposed on it.

    Voices are assigned by register, matching every other renderer here: low notes
    to bass, the highest sounding note to lead, a middle voice to arp.
    """
    step_s = spb / float(steps)
    ev = {"lead": [], "arp": [], "bass": [], "drum": []}
    buckets = {}
    for n in notes:
        row = int(round(n["t"] / step_s))
        buckets.setdefault(row, []).append(n)
    maxbar = 0
    for row, group in sorted(buckets.items()):
        bar, st = divmod(row, steps)
        maxbar = max(maxbar, bar)
        group.sort(key=lambda x: x["pitch"])
        dur = max(1, int(round(max(g["dur"] for g in group) / step_s)))
        low, high = group[0], group[-1]
        if high["pitch"] >= 48:
            ev["lead"].append((bar, st, dur, high["pitch"], 0.5,
                               high.get("vel") or 96))
        if len(group) > 1:
            ev["bass"].append((bar, st, dur, low["pitch"], low.get("vel") or 96))
        else:
            ev["bass"].append((bar, st, dur, low["pitch"] - 12
                               if low["pitch"] >= 60 else low["pitch"],
                               low.get("vel") or 96))
        if len(group) > 2:
            mid = group[len(group) // 2]
            ev["arp"].append((bar, st, dur, mid["pitch"], 0.25,
                              mid.get("vel") or 80))
    kind_of = {36: "k", 35: "k", 38: "s", 40: "s", 42: "h", 44: "h", 46: "h"}
    for t, gm in (drums or []):
        row = int(round(t / step_s))
        bar, st = divmod(row, steps)
        maxbar = max(maxbar, bar)
        ev["drum"].append((bar, st, kind_of.get(gm, "h")))
    return ev, (bars or maxbar + 1)
