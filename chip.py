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
NOTE_NAMES = "C C# D D# E F F# G G# A A# B".split()
MAJOR_TYPE = ("major", "mixolydian", "lydian")      # everything else in SCALES is a minor mode

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
    """'D minor' / 'f# major' / 'A' / 'Bb dorian' -> (root_semitone, scale_name).

    Raises ValueError for text that names no note — "   " used to crash with an
    IndexError and "H major" quietly became C minor.
    """
    t = (text if text is not None else "C minor").strip().lower().replace("-", " ")
    parts = t.split()
    if not parts or parts[0][0] not in NOTE_BASE:
        raise ValueError(f"--key {text!r}: expected a note A-G (with # or b) and a scale, "
                         f"e.g. \"D minor\"")
    name = parts[0]
    root = NOTE_BASE[name[0]]
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


def key_names_scale(text):
    """True when --key spells out a scale or mode ("E major", "D dorian"), as
    opposed to a bare root ("E")."""
    for p in (text or "").lower().replace("-", " ").split()[1:]:
        if p in SCALES or p.startswith(("maj", "min")) or p == "lydian":
            return True
    return False


def _flag_given(*flags):
    """Was one of these flags typed on the command line? Same test as
    soundmon.given(): comparing a value with its default cannot tell a typed
    `--key "C minor"` from no --key at all."""
    return any(x == f or x.startswith(f + "=") for x in sys.argv[1:] for f in flags)


def track_name(a):
    """The track's name: --name, else the prompt's slug, else "untitled".

    Without the prompt here, every `--chip` run with no --name was the track
    "untitled" — so every prompt composed the identical piece.
    """
    import re
    nm = getattr(a, "name", None)
    if nm:
        return nm
    words = re.sub(r"[^a-z0-9]+", "_", (getattr(a, "prompt", "") or "").lower()).strip("_")[:40]
    return words or "untitled"


def track_identity(a):
    """PACK + TRACK: what decides the PIECE (key, form, harmony, metre, tempo).

    Identity is pack + track, not track alone: deriving it from the track name
    only meant `level9` composed identically in every pack. The pack comes from
    the output directory, so nothing needs a new flag and every existing caller
    gets distinct music for free.
    """
    track = track_name(a)
    _pack = (os.path.basename(os.path.normpath(a.output_to))
             if getattr(a, "output_to", None) else "")
    return f"{_pack}/{track}" if _pack else track


def take_identity(track, rng):
    """PIECE + SEED: what decides the TAKE (melody, groove, bass line, kit).

    The seed used to be passed in and ignored, so `-n 4` wrote four
    byte-identical files and --seed did nothing. A take keeps the piece's key,
    form, chord progression, metre and tempo, and gets its own tune and groove
    over them — enough to choose between, still recognisably one cue.
    """
    if rng is None:
        return track
    return f"{track}#take{int(rng.integers(0, 2 ** 62))}"


MAX_SECONDS = 600.0          # --chip/--opl: ten minutes is a long game loop


def check_seconds(a, what):
    """--seconds as a usable length, or a one-line exit. nan/inf used to crash
    and 1e9 used to hang composing a billion bars."""
    s = getattr(a, "seconds", None)
    if s is None:
        return
    try:
        s = float(s)
    except (TypeError, ValueError):
        sys.exit(f"{what}: --seconds must be a number, got {a.seconds!r}")
    if not math.isfinite(s) or s <= 0:
        sys.exit(f"{what}: --seconds must be more than 0, got {a.seconds}")
    if s > MAX_SECONDS:
        sys.exit(f"{what}: --seconds {s:g} is too long (at most {MAX_SECONDS:g} s; "
                 f"it loops, so render a shorter piece and repeat it)")


def apply_format(audio, sr, a, np):
    """Hardware format lock for the synthesis engines -> (audio, sr, bits).

    Stores at the format's OWN bit depth: writing 8-bit audio into a 16-bit
    container wastes half the file and is a lie about the data. When that
    overrides a typed --bits, say so instead of silently ignoring it.
    """
    bits = a.bits
    fmt = getattr(a, "format", "none")
    if fmt in (None, "none"):
        return audio, sr, bits
    if fmt not in FORMATS:
        sys.exit(f"unknown --format {fmt!r}. One of: {', '.join(FORMATS)}")
    audio, sr = format_lock_np(audio, sr, fmt, np)
    _fb = int((FORMATS.get(fmt) or {}).get("bits") or 0)
    if _fb:
        bits = 8 if _fb <= 8 else (16 if _fb <= 16 else 24)
        if bits != a.bits and _flag_given("--bits") and not getattr(a, "_bits_noted", False):
            print(f"   \u2139 --format {fmt} stores {bits}-bit audio; --bits {a.bits} ignored")
            a._bits_noted = True
    return audio, sr, bits


def wav_subtype(bits):
    return f"PCM_{bits}" if bits != 8 else "PCM_U8"


def note_unused_writers(a, engine):
    """--write-mod belongs to --chip and --write-rad to --opl. Accepting the
    other one and writing nothing is how a flag silently does nothing."""
    if engine == "chip" and getattr(a, "write_rad", False):
        print("   \u2139 --write-rad is for --opl (an OPL tracker); --chip writes a .MOD with --write-mod")
    if engine == "opl" and getattr(a, "write_mod", False):
        print("   \u2139 --write-mod is for --chip (a sample tracker); --opl writes a .RAD with --write-rad")


def fold_tail(buf, n):
    """Wrap everything past sample `n` back onto the head, return exactly n.

    A loop's last notes ring into its FIRST bar — that is where a listener hears
    them when it repeats. Adding the release onto the head therefore makes the
    file exactly whole bars long and the seam continuous by construction, where
    a 0.35-0.4 s tail made every loop land late.
    """
    out = buf[:n].copy()
    tail = buf[n:]
    while len(tail):
        k = min(len(tail), n)
        out[:k] += tail[:k]
        tail = tail[k:]
    return out


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

    # Identity is PACK + TRACK (+ the take). See track_identity(): the piece —
    # key, form, harmony, metre, tempo — comes from the pack and the track name
    # (or the prompt, when there is no --name), and the seed picks the take: its
    # tune and groove over that piece. Same pack + name/prompt + seed always
    # regenerates the same bytes.
    track = track_identity(a)
    take = take_identity(track, rng)

    # Scale: explicit override wins, then a scale spelled out in a TYPED --key
    # ("E major" used to come out minor or dorian: only mode names counted),
    # else the mood's — chosen by track identity so two `solemn` tracks differ.
    # The untyped default "C minor" must not count, or every mood would be minor.
    #
    # "E major" / "D minor" fix the QUALITY and the mood still picks the mode
    # within it: D minor + ominous stays phrygian or harmonic, E major + ominous
    # becomes major (the mood has no major mode), D minor + heroic becomes minor.
    # A mode name ("D dorian") is taken literally.
    explicit = getattr(a, "chip_scale", None)
    key_typed = _flag_given("--key") or (a.key or "C minor").strip().lower() != "c minor"
    words = (a.key or "").lower().replace("-", " ").split()[1:]
    if explicit:
        scale_name = explicit
    elif key_typed and any(w in theory.SCALES and w not in ("major", "minor") or w == "lydian"
                           for w in words):
        scale_name = key_scale if key_scale in theory.SCALES else "minor"
    elif key_typed and key_names_scale(a.key):
        same = [m for m in mood["scales"]
                if (m in MAJOR_TYPE) == (key_scale in MAJOR_TYPE)]
        scale_name = (theory.Ident(track, "scale:" + mname).pick(same) if same
                      else ("major" if key_scale in MAJOR_TYPE else "minor"))
    else:
        scale_name = theory.Ident(track, "scale:" + mname).pick(mood["scales"])
    scale = theory.SCALES.get(scale_name, theory.SCALES["minor"])

    # The key ROTATES through closely related keys across a pack (i/iv/v/bIII/
    # bVII/bVI), weighted toward home. That is what stops 24 tracks sharing one
    # key without scattering them into 12 unrelated ones. (--key is the pack's
    # HOME key; the scale it names is honoured, the root may rotate.)
    root = theory.key_for_track(track, home_root, mname)

    est_spb = 60.0 / max(40.0, min(300.0, a.bpm * mood["bpm"])) * 4.0
    plan = theory.plan_track(track, mood, mname, scale_name,
                             max(4, int(round(a.seconds / est_spb))), take=take)

    bpm = max(40.0, min(300.0, a.bpm * plan["tempo_mul"]))
    # Bar length now follows the METER, not a hardcoded 4 beats. This is what
    # stops every track sharing one meter — 3/4, 6/8, 7/8 and 12/8 all produce
    # genuinely different bar lengths and accent patterns.
    steps = plan["steps"]
    spb_steps = plan["spb_steps"]
    spb = 60.0 / bpm * plan["quarters"]
    form = list(plan["form"])
    # Length in WHOLE BARS, as close to --seconds as whole bars allow. The form
    # is played whole when there is room (bars per section >= 1, rounded, not
    # ceiled: ceil plus a 2-bar floor made 0.5 s come out as 19 s). A request
    # shorter than one pass of the form plays its first one or two sections —
    # a 1-bar loop is still a loop — so the floor is ONE bar.
    want_bars = max(1, int(round(a.seconds / spb)))
    if want_bars < len(form):
        # 1 or 2 bars (a 3-bar loop is a phrase nobody writes), nearest wins
        n_short = min((1, 2, len(form)), key=lambda b: abs(b * spb - a.seconds))
        form, bps = form[:n_short], 1
    else:
        bps = max(1, int(want_bars / float(len(form)) + 0.5))      # half rounds up
    bars = bps * len(form)
    plan["bpm"] = bpm
    plan["key_name"] = f"{NOTE_NAMES[root % 12]} {scale_name}"

    def deg(d, octave=0):
        o, i = divmod(d, len(scale))
        return 12 * (4 + octave + o) + root + scale[i]

    # One motif for the first section; every other section DEVELOPS it via a
    # transformation (inversion, sequence, retrograde, augmentation...). Wholly
    # new material per section fragments a track; verbatim repetition bores.
    first = form[0]
    motifs, kits, basses = {}, {}, {}
    for letter, sec in plan["sections"].items():
        if letter not in form:
            continue
        mid_ = theory.Ident(take, "motif:%s:%s" % (letter, mname))
        # Learned interval walk when a corpus exists; hand-written contour
        # archetypes otherwise.
        m = (theory.corpus_motif(mid_, sec["rhythm"], scale_name, len(scale),
                                 mood["span"])
             or theory.make_motif(mid_, sec["rhythm"], sec["contour"],
                                  mood["span"], spb_steps))
        if letter != first and sec["transform"]:
            m = theory.transform(m, sec["transform"],
                                 theory.Ident(take, "tr:" + letter))
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
                # Clamped to the bar: a 3- or 4-step arp in a 14-step 7/8 bar ran
                # into the next bar's first arp — two arp notes at once, which with
                # the lead is three pulse channels on a chip that has two.
                ev["arp"].append((bar, s, min(arp_rate, steps - s), chord[k % len(chord)], 0.25,
                                  theory.accent(s, spb_steps, 74)))
                k += 1

            # --- bass: the section's rhythm STYLE, voice-led for motion
            low = [c - 12 for c in chord]
            for si, (s, d) in enumerate(basses[letter]):
                if s >= steps:
                    break
                if si == 0:
                    bn = theory.voice_bass(prev_bass, low,
                                           ident=theory.Ident(take, "bass:%d" % bar))
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
    plan["form"] = tuple(form)
    # The duty in each lead/arp event is the MOOD's choice (composed), not a
    # placeholder the way a transcription's 0.5 is — let the renderer use it.
    ev["_duty"] = True
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


def resolve_mood(a):
    mname = getattr(a, "mood", None)
    if not mname or mname == "auto":
        mname = infer_mood(getattr(a, "prompt", "") or "")
    return mname


def load_midi_source(a, label):
    """--from-midi -> (notes, drums, info), or a one-line exit.

    A missing, empty, truncated or non-MIDI file used to end in a traceback.
    """
    import midi as midimod
    mfile = a.from_midi
    try:
        timed = midimod.to_timed_events(
            mfile, seconds=a.seconds,
            transpose=getattr(a, "transpose", 0),
            start_frac=getattr(a, "midi_start", 0.15))
    except (OSError, ValueError) as e:
        sys.exit(f"{label} --from-midi {mfile}: {e}")
    if not timed or not timed[0] and not timed[1]:
        sys.exit(f"{label} --from-midi {mfile}: no playable notes")
    if timed[2].get("folded"):
        print(f"   \u2139 {timed[2]['folded']} note(s) moved by octaves into the playable range "
              f"(MIDI {midimod.PITCH_LO}-{midimod.PITCH_HI}) after --transpose")
    return timed


def load_audio_source(a, np, sf, label):
    """--from-audio -> (ev, bars, spb, analysis, scale_name), or a one-line exit.

    Silence, noise, speech and other non-music used to be "transcribed" anyway
    (a silent reference rendered a loud arpeggio); transcribe now says when it
    finds no steady beat or key, and that is an error here, not a track.
    """
    import transcribe
    src = a.from_audio
    try:
        got = transcribe.to_events_hi(src, np, sf, seconds=a.seconds,
                                      beats_per_bar=4, div=4)
    except (OSError, ValueError) as e:
        sys.exit(f"{label} --from-audio {src}: {e}")
    if not got:
        sys.exit(f"{label} --from-audio {src}: no steady beat or key found — "
                 f"it needs music with a pulse, not silence, noise or speech")
    return got


def out_dir(a):
    dest = os.path.abspath(os.path.expanduser(a.output_to)) if a.output_to else os.getcwd()
    if not os.path.isdir(dest):
        if os.path.exists(dest):
            sys.exit(f"--output-to {dest}: that is a file, not a folder")
        if a.create_dirs or a.output_to:
            os.makedirs(dest, exist_ok=True)
        else:
            sys.exit(f"output dir does not exist: {dest} (add --create-dirs)")
    return dest


def base_name(a, slug, fallback):
    """The output file stem: --name, else the prompt's slug. Through the same
    safe-name rule as manifest keys, so `--name ../x` cannot leave --output-to."""
    import narrate
    raw = a.name or slug(a.prompt or fallback)
    safe = narrate.safe_key(raw) or fallback
    if a.name and safe != a.name:
        print(f"   \u2139 --name {a.name!r} is not a safe file name; using {safe!r}")
    return safe


def takes_for(a):
    """How many takes -n can actually make. A MIDI file or a recording plays the
    same way every time — there is nothing for a seed to change — so `-n 4`
    would write four identical files; make one and say so."""
    n = max(1, int(a.number))
    if n > 1 and (getattr(a, "from_midi", None) or getattr(a, "from_audio", None)):
        src = "--from-midi" if getattr(a, "from_midi", None) else "--from-audio"
        print(f"   \u2139 -n {n}: {src} plays the same notes every time, so one take is made")
        return 1
    return n


def length_note(a, real_s):
    """Whole bars cannot hit every --seconds. Say what was made instead."""
    if _flag_given("--seconds") and abs(real_s - float(a.seconds)) > max(0.5, 0.1 * float(a.seconds)):
        return f"  (asked {float(a.seconds):g}s; whole bars give {real_s:.1f}s)"
    return ""


def run(a, slug, to_ogg=None, loudness_normalize=None, to_flac=None):
    """Generate chiptune. Same contract as narrate.run()/blip.run()."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        sys.exit(f"--chip needs numpy + soundfile: pip install soundfile numpy   ({e})")

    check_seconds(a, "--chip")
    try:
        parse_key(a.key)
    except ValueError as e:
        sys.exit(f"--chip: {e}")
    note_unused_writers(a, "chip")
    dest = out_dir(a)
    base = base_name(a, slug, "chiptune")
    if getattr(a, "from_midi", None) and getattr(a, "write_midi", False):
        print("   \u2139 --write-midi writes composed music; with --from-midi you already have the MIDI")

    # Sources are read ONCE: a MIDI file or a recording does not change per take.
    mfile = getattr(a, "from_midi", None)
    src = getattr(a, "from_audio", None)
    mname = resolve_mood(a)
    mood = MOODS.get(mname, MOODS[DEFAULT_MOOD])
    if mfile:
        tnotes, tdrums, info = load_midi_source(a, "--chip")
    elif src:
        import transcribe
        got = load_audio_source(a, np, sf, "--chip")

    n_out = takes_for(a)
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

        loop = True
        key_name = ""
        if mfile:
            # MIDI: nothing is inferred. Pitches, durations, tempo and metre all
            # come from the file. Mood still picks the voicing. EXACT timing —
            # MIDI carries its own; quantizing it only destroys information.
            ev, bars, spb, scale_name = None, 0, info["spb"], "(from MIDI)"
            steps = STEPS; meter_s = info["timesig"]; bpm = info["bpm"]
            loop = False            # an excerpt of a piece, not whole bars
            if i == 0:
                print(f"   \u266a {info['title'][:38]}: {info['timesig']} "
                      f"{info['bpm']:.0f}bpm  {info['duration']:.1f}s  "
                      f"{info['notes']} notes  {info['drum_hits']} hits")
            audio = render_timed(a, tnotes, tdrums, info["duration"], np, mood)
        else:
            if src:
                # Transcribed: the composition comes from the reference recording.
                # Mood is still resolved, because it decides how this is VOICED.
                ev, bars, spb, ana, scale_name = got
                steps = STEPS; meter_s = '4/4'; bpm = ana["bpm"]
                key_name = f"{transcribe.NOTE_NAMES[ana['root']]} {ana['mode']}"
                if i == 0:
                    print(f"   \u266a {os.path.basename(src)}: {key_name}  "
                          f"{ana['bpm']:.0f}bpm  {ana['notes']} notes  "
                          f"{ana['drum_hits']} hits")
            else:
                ev, bars, spb, scale_name, prog, mname, mood, plan = compose(a, np, rng)
                steps = plan["_steps"]; meter_s = plan["meter"]; bpm = plan["bpm"]
                key_name = plan.get("key_name", "")
            # Procedural through the same renderer: inherits --chippy arpeggios,
            # slides and the part-based voicing. Grid positions, exact-time
            # representation; exactly bars*spb long, so it loops on the bar.
            pnotes, pdrums = events_to_timed(ev, bars, spb, steps, mname)
            audio = render_timed(a, pnotes, pdrums, bars * spb, np, mood, loop=True)

        # The seed goes in EVERY filename, as it does for every other engine.
        # That is the documented way to re-run a take you liked, and the pack
        # generator relies on the `<name>_*` shape to rename output to the
        # manifest key.
        name = f"{base}_s{seed}"
        audio, out_sr, out_bits = apply_format(audio, SAMPLE_RATE, a, np)
        path = os.path.join(dest, f"{name}.wav")
        sf.write(path, audio, out_sr, subtype=wav_subtype(out_bits))
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
            if ev is not None:
                # Procedural: the plan's own steps-per-bar, which is NOT always 16
                # \u2014 a 6/8 mood uses 12, and writing those rows as 16 per bar
                # shifted every bar progressively further out of place.
                gev, gbars, grows = ev, bars, steps
            else:
                grows = best_grid(tnotes, spb, bars=None)
                gev, gbars = timed_to_grid(tnotes, tdrums, spb, grows)
            mp = os.path.splitext(path)[0] + ".mod"
            try:
                _modw.write_mod(mp, gev, gbars, spb, grows, np,
                                title=base[:20],
                                chippy=getattr(a, "chippy", "off"),
                                rows_per_bar=grows)
                _t, _b = _modw.timing_for(spb, grows)
                print(f"   \u266b also wrote {os.path.basename(mp)}"
                      f"  ({grows} rows/bar, speed {_t}, {_b} bpm)")
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
        real_s = len(audio) / float(out_sr)
        print(f"   \u2705 [{i+1}/{n_out}] {os.path.basename(path):<30} "
              f"{real_s:5.1f}s  {(str(bars) + 'bar') if bars else '  -  '} {meter_s:<5}{mname:<11}"
              f"{(key_name or scale_name):<14}{bpm:.0f}bpm  seed={seed}"
              f"{length_note(a, real_s) if loop else ''}")

    print(f"   all done  |  {len(made)} file(s) in {dest}")
    if not mfile:
        print("   \u21bb loops seamlessly by construction — whole bars, no crossfade needed")
    return made


def render_timed(a, notes, drums, duration, np, mood=None, loop=False):
    """Render exact-time notes on the 2A03 voices. No grid.

    The OPL path moved to event timing but this one was left on a 16th grid, so
    91% of notes still had their onsets and lengths rounded. Same fix here: each
    note is placed at its real sample offset for its real duration.

    Voice assignment follows the PART when the notes carry one (composed music:
    the bass part is the triangle, lead and arp are the two pulses, at the duty
    the mood chose — so the 4-voice limit holds). A MIDI file's notes carry no
    part, so they go by REGISTER, which is how a 4-channel arrangement works
    anyway: low notes take the triangle, everything else a pulse, thinner in the
    upper register so the melody sits above the harmony.

    `loop`: the piece is whole bars that repeat. The file is then EXACTLY
    `duration` long and anything ringing past the end is folded onto the head
    (fold_tail) — where it is heard when the loop comes round. Without `loop`
    (a MIDI excerpt) a short tail is kept so the last release is not cut.
    """
    sr = SAMPLE_RATE
    n_out = int(round(duration * sr))
    tail = int(round(0.4 * sr))                     # room for the last release
    total = n_out + tail
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
        role = n.get("role")

        if role == "bass" or (role is None and pitch < BASS_MAX_PITCH):
            # triangle bass, never arped
            buf[start:start + ln] += (_tri_sweep_note(hz, ln, sr, np)
                                      * _env(ln, sr, 3.0, 3.0, 0.75, np)
                                      * 0.34 * vel)
            continue

        duty = n.get("duty") or (0.125 if pitch >= 72 else (0.25 if pitch >= 60 else 0.5))
        if cfg["duty_mod"] > 1:
            duty = 0.125
        sf_from = n.get("slide_from")
        if sf_from is not None:
            hz0 = 440.0 * (2.0 ** ((sf_from - 69) / 12.0))
            w = _slide_pulse(hz0, hz, ln, sr, duty,
                             n.get("slide_ms", cfg["slide_ms"]), np, depth)
        else:
            vib = _vibrato(ln, sr, depth if n["dur"] > 0.35 else 0.0, 6.0, np)
            ph = 2 * np.pi * hz * np.cumsum(vib) / sr
            w = np.where((ph / (2 * np.pi)) % 1.0 < duty, 1.0, -1.0)
        buf[start:start + ln] += w * _env(ln, sr, 2.0, 4.0, 0.55, np) * 0.20 * vel

    hit_cache = {}                       # the LFSR is a per-sample Python loop: make each hit once
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
        if (kind, ln) not in hit_cache:
            hit_cache[(kind, ln)] = _lfsr_note(ln, sr, period, np) * _env(ln, sr, 1.0, dec, 0.0, np)
        buf[start:start + ln] += hit_cache[(kind, ln)] * g

    if loop and n_out > 0:
        buf = fold_tail(buf, n_out)
        buf = _chip_verb(buf, sr, cfg["verb"], np, circular=True)
        # Primed with its own end: the filter state at sample 0 is the state it
        # has when the loop comes round, so the seam stays continuous.
        buf = _dc_block(buf, np, sr, prime=buf[-min(len(buf), sr):])
    else:
        buf = _chip_verb(buf, sr, cfg["verb"], np)
        buf = _dc_block(buf, np, sr)      # before normalising: DC steals headroom
        f = min(len(buf), int(0.01 * sr))  # an excerpt ends mid-note: land on zero
        if f > 1:
            buf[-f:] *= np.linspace(1.0, 0.0, f)
    peak = float(np.abs(buf).max()) if len(buf) else 0.0
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


def _dc_block(buf, np, sr, hz=12.0, prime=None):
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

    It used to call scipy's lfilter and, without scipy, subtract the mean —
    which does not block DC, it CREATES a step at both ends of the file (the
    exact click this exists to prevent). This is the same filter in numpy, so
    the result no longer depends on what is installed.

    `prime`: samples that precede `buf` — for a loop, its own end. The filter
    then starts in the state it will have when the loop comes round, so the seam
    stays continuous instead of restarting from zero.
    """
    x = np.asarray(buf, dtype=np.float64)
    if x.ndim > 1:
        return np.stack([_dc_block(x[:, c], np, sr, hz,
                                   None if prime is None else np.asarray(prime)[:, c])
                         for c in range(x.shape[1])], axis=1)
    a = float(np.exp(-2.0 * np.pi * hz / sr))
    x_prev, y_prev = 0.0, 0.0
    if prime is not None and len(prime):
        _, x_prev, y_prev = _hp_one_pole(np.asarray(prime, dtype=np.float64), a, 0.0, 0.0, np)
    y, _, _ = _hp_one_pole(x, a, x_prev, y_prev, np)
    return y


def _hp_one_pole(x, a, x_prev, y_prev, np, block=2048):
    """y[n] = x[n] - x[n-1] + a*y[n-1], vectorised in blocks.

    Within a block the recursion has a closed form,
        y[k] = a^k * (a*y_prev + sum_{j<=k} d[j] * a^-j),   d = diff(x)
    so each block is one cumsum. 2048 keeps a^-k under ~35 at 44.1 kHz/12 Hz:
    no overflow and no precision worth measuring (agrees with lfilter to 1e-12).
    """
    n = len(x)
    y = np.empty(n)
    if n == 0:
        return y, x_prev, y_prev
    d = np.diff(x, prepend=x_prev)
    k = np.arange(block, dtype=np.float64)
    up, down = a ** -k, a ** k
    for s in range(0, n, block):
        e = min(n, s + block)
        m = e - s
        y[s:e] = down[:m] * (y_prev * a + np.cumsum(d[s:e] * up[:m]))
        y_prev = y[e - 1]
    return y, float(x[-1]), float(y_prev)


def _chip_verb(buf, sr, amount, np, circular=False):
    """Subtle multi-tap echo.

    A 2A03 cannot do this — but chiptune as *heard* usually has it, because
    trackers had echo commands and emulator captures pick up room. So it belongs
    on the stylistic end of the dial rather than in the synthesis: at "max" the
    goal is what chip music sounds like on a record, not what the silicon emits.

    Three prime-ish taps rather than one, so it reads as space instead of as a
    slapback repeat.

    `circular`: the buffer is a loop, so the echo of its end lands on its head
    (where it is heard when the loop repeats) rather than falling off the end.
    """
    if amount <= 0:
        return buf
    out = buf.copy()
    for ms, g in ((53.0, 1.00), (97.0, 0.62), (149.0, 0.38)):
        d = int(sr * ms / 1000.0)
        if circular and 0 < d:
            out += np.roll(buf, d) * (amount * g)
        elif d < len(buf):
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
    low = lambda n: n["pitch"] < ARP_MIN_PITCH or n.get("role") == "bass"   # noqa: E731
    passthru = [n for n in notes if low(n)]
    top = [n for n in notes if not low(n)]
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
                n = {"id": (key, i), "t": at(bar, st),
                     "dur": max(0.02, dur * step_s),
                     "pitch": int(pitch), "prog": gm,
                     "vel": int(vel) if isinstance(vel, int) else 96,
                     # The PART, so a renderer voices by role, not by guessing
                     # from pitch: the composed bass reaches up to ~B3 and was
                     # landing on a pulse (three pulses at once on a 2A03).
                     "role": key}
                if ev.get("_duty") and key != "bass" and len(it) > 4 and isinstance(it[4], float):
                    n["duty"] = it[4]          # the mood's duty: timbre on a 2A03
                out.append(n)
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


def best_grid(notes, spb, cap_rows=8192, bars=None):
    """Pick rows-per-bar that actually fits the source's rhythm.

    HOW THIS WAS DIAGNOSED, because the obvious reading was wrong. Quantizing the
    Zelda title theme to 16ths left a mean onset error of 0.25 steps — and 0.25 is
    exactly what a UNIFORM distribution over half a step gives, i.e. the notes
    looked unquantizable. Doubling the grid did not help, and neither did any phase
    offset, which ruled out both "too coarse" and "off by a constant".

    Looking at the raw ticks explained it: at PPQ 120, onsets cluster at ticks 0,
    5, 10, 15, 20 and 25 of every 30-tick 16th. Five ticks is a 24th of a quarter
    note, so the piece is written on a 96-per-bar grid — sextuplets. A 16th grid
    cannot express that at ANY phase, which is why rolled figures came out as
    blocks.

    So: measure, don't assume. Candidates include both the power-of-two divisions
    and the triple ones, since only the latter can hold triplets.
    """
    if not notes:
        return 16
    cands = (16, 24, 32, 48, 64, 96)
    scored = []
    for rpb in cands:
        if bars and bars * rpb > cap_rows:
            continue
        step = float(spb) / rpb
        err = sum(abs(n["t"] / step - round(n["t"] / step)) for n in notes)
        # Error as a fraction of a 16th note, so the numbers are comparable
        # across candidates rather than shrinking automatically with the grid.
        scored.append((err / len(notes) * (16.0 / rpb), rpb))
    if not scored:
        return 16
    # Prefer the coarsest grid that is already accurate: finer rows cost pattern
    # count and make the file harder to edit by hand, which is half the point of
    # emitting a module at all.
    good = [(rpb, e) for e, rpb in scored if e <= 0.02]
    return good[0][0] if good else min(scored)[1]


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

        def rows(n):
            """That note's own length. Using the group's longest for all of them
            made a short melody note inherit a held bass note's duration, so every
            chord came out legato — the "everything is legato unless the source is"
            complaint, in the tracker path this time."""
            return max(1, int(round(n["dur"] / step_s)))

        low, high = group[0], group[-1]
        if high["pitch"] >= 48:
            ev["lead"].append((bar, st, rows(high), high["pitch"], 0.5,
                               high.get("vel") or 96))
        if len(group) > 1:
            ev["bass"].append((bar, st, rows(low), low["pitch"],
                               low.get("vel") or 96))
        else:
            ev["bass"].append((bar, st, rows(low), low["pitch"] - 12
                               if low["pitch"] >= 60 else low["pitch"],
                               low.get("vel") or 96))
        if len(group) > 2:
            mid = group[len(group) // 2]
            ev["arp"].append((bar, st, rows(mid), mid["pitch"], 0.25,
                              mid.get("vel") or 80))
    kind_of = {36: "k", 35: "k", 38: "s", 40: "s", 42: "h", 44: "h", 46: "h"}
    for t, gm in (drums or []):
        row = int(round(t / step_s))
        bar, st = divmod(row, steps)
        maxbar = max(maxbar, bar)
        ev["drum"].append((bar, st, kind_of.get(gm, "h")))
    return ev, (bars or maxbar + 1)
