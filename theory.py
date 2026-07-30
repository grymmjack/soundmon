#!/usr/bin/env python3
"""Music theory for the synthesis engines: harmony, melody, form, and a key plan.

THE PROBLEM THIS SOLVES

The first version of `chip.compose()` gave every track in a pack one key, one
tempo, one 4-bar progression drawn from a pool of two, and a single motif looped
for the whole piece. Measured across a 24-track pack, mean pairwise similarity
was 0.90 for chiptune and 0.93 for AdLib — audibly "the same notes, same meter,
same melody" every time.

The fix is NOT more randomness. Random music is not diverse, it is uniformly
mushy, and it would destroy the thing worth keeping: a pack that sounds like one
soundtrack. What was missing is *structure* — the axes real soundtracks vary along
while staying coherent:

    key         related keys, not one key and not twelve unrelated ones
    harmony     progressions generated from FUNCTION, not picked from a list
    form        AABA / ABAB / ABAC — sections that differ from each other
    melody      contour archetypes, cadences, non-chord tones, motif development
    tempo       a little per-track spread inside the mood's range

TWO SOURCES OF RANDOMNESS, DELIBERATELY SEPARATED

    identity(name)  -> stable per-track character. The SAME track name always
                       gets the same key, form and progression shape, across
                       regenerations. This is what makes a pack diverse: 24
                       names produce 24 different plans.
    seed            -> performance detail. `-n 8` gives you eight takes of the
                       same piece, not eight different pieces.

Conflating those is why `-n 8` used to feel like rolling dice rather than
auditioning takes.

WHAT STAYS STEERED BY MOOD

Everything here is *bounded* by the mood. `solemn` picks slow tempos, dark
scales, plagal cadences and sparse forms; `frantic` picks fast tempos, phrygian,
authentic cadences and dense forms. Diversity happens INSIDE those bounds, so a
crypt never sounds cheerful — it just stops sounding like the armoury.
"""
import hashlib

# --- scales & chords ---------------------------------------------------------
SCALES = {
    "minor":      [0, 2, 3, 5, 7, 8, 10],
    "harmonic":   [0, 2, 3, 5, 7, 8, 11],
    "dorian":     [0, 2, 3, 5, 7, 9, 10],
    "phrygian":   [0, 1, 3, 5, 7, 8, 10],
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "pentatonic": [0, 3, 5, 7, 10],
    "aeolian":    [0, 2, 3, 5, 7, 8, 10],
    "lydian":     [0, 2, 4, 6, 7, 9, 11],
}

# Harmonic FUNCTION by scale degree. Generating progressions from function
# instead of listing them is what turns 2 options into dozens that all still
# resolve properly — the era's music is overwhelmingly functional, so this is
# both more varied AND more idiomatic than a hand-written list.
#   T tonic (home)  S subdominant (departure)  D dominant (tension)
FUNCTION = {"T": (0, 5, 2), "S": (3, 1), "D": (4, 6)}

# Function sequences. Each ends somewhere a cadence can follow.
TEMPLATES = [
    ("T", "S", "D"), ("T", "D", "S"), ("T", "S", "T", "D"),
    ("T", "T", "S", "D"), ("T", "D", "T", "S"), ("T", "S", "S", "D"),
    ("T", "D", "D"), ("S", "T", "S", "D"), ("T", "T", "D"),
]

# Cadences, by character. A plagal (iv-i) cadence is soft and liturgical; an
# authentic (V-i) one is decisive; a phrygian (bII-i) one is the dark-fantasy
# sound. The mood chooses which are allowed, which matters more for "feel" than
# the chord list does.
CADENCES = {
    "authentic": (4, 0),
    "plagal":    (3, 0),
    "phrygian":  (1, 0),
    "deceptive": (4, 5),
    "half":      (0, 4),
}

# Melodic contour archetypes. A phrase shape, sampled across its length.
CONTOURS = {
    "arch":       lambda t: 1.0 - abs(2.0 * t - 1.0),
    "descending": lambda t: 1.0 - t,
    "ascending":  lambda t: t,
    "valley":     lambda t: abs(2.0 * t - 1.0),
    "static":     lambda t: 0.5,
    "terraced":   lambda t: min(1.0, (int(t * 3) / 2.0)),
}

# Song forms. Distinct sections are the single biggest cure for a track that
# "sounds the same all the way through".
FORMS = [("A", "A", "B", "A"), ("A", "B", "A", "B"), ("A", "A", "B", "B"),
         ("A", "B", "A", "C"), ("A", "A", "B", "C"), ("A", "B", "B", "A")]

# Keys related to the home key, as semitone offsets. Restricted on purpose: any
# of these shares most of its notes with the home key, so a pack that rotates
# through them still sounds like one soundtrack. Twelve arbitrary keys would not.
RELATED = (0, 0, 0, 5, 7, 3, 10, 8)     # i, i, i, iv, v, bIII, bVII, bVI


def identity(name, salt=""):
    """A stable integer for a track NAME. Same name -> same character, always."""
    h = hashlib.sha256(f"{name}\x00{salt}".encode()).digest()
    return int.from_bytes(h[:8], "big")


class Ident:
    """Deterministic chooser driven by a track name rather than a random seed."""

    def __init__(self, name, salt=""):
        self.n = identity(name, salt)
        self._i = 0

    def _next(self):
        self._i += 1
        return identity(str(self.n), f"{self._i}")

    def pick(self, seq):
        seq = list(seq)
        return seq[self._next() % len(seq)] if seq else None

    def rint(self, lo, hi):
        """Inclusive integer in [lo, hi]."""
        return lo + (self._next() % max(1, hi - lo + 1))

    def frac(self):
        return (self._next() % 10000) / 10000.0


def build_progression(ident, scale_len, template=None, cadence="authentic",
                      seventh_on_dominant=False):
    """Realize a function template into concrete scale degrees, plus a cadence.

    Returns a list of (degree, add_seventh) pairs. Degrees are scale-relative, so
    the same progression works in any mode — which is what lets one plan be
    reused across a pack in different keys.
    """
    tmpl = template or ident.pick(TEMPLATES)
    out = []
    for fn in tmpl:
        opts = [d for d in FUNCTION[fn] if d < scale_len]
        deg = ident.pick(opts) if opts else 0
        out.append((deg, fn == "D" and seventh_on_dominant))
    a, b = CADENCES.get(cadence, CADENCES["authentic"])
    if a < scale_len and b < scale_len:
        out.append((a, seventh_on_dominant))
        out.append((b, False))
    return out


def triad(scale, degree, seventh=False):
    """Stack thirds within the scale — so quality (major/minor/dim) comes out of
    the mode automatically instead of being hard-coded."""
    n = len(scale)
    idx = [degree, degree + 2, degree + 4] + ([degree + 6] if seventh else [])
    return [12 * (i // n) + scale[i % n] for i in idx]


def voice_bass(prev_note, chord, root_pref=0.6, ident=None):
    """Pick a bass note from the chord, preferring small motion from the last one.

    Real bass lines move by step where they can. Always playing the root gives
    the four-square sound the first version had; choosing an inversion when it
    is *closer* produces walking motion for free.
    """
    cands = [chord[0], chord[0] + 12, chord[2] if len(chord) > 2 else chord[0]]
    if len(chord) > 1:
        cands.append(chord[1])
    if prev_note is None:
        return chord[0]
    if ident is not None and ident.frac() < root_pref:
        return chord[0]
    return min(cands, key=lambda c: abs(c - prev_note))


def make_motif(ident, rhythm, contour_name, span, spb_steps=4):
    """A motif as (duration, scale-step, is_rest) triples following a contour.

    `rhythm` is the (duration, is_rest) list from gen_rhythm, so the melody
    inherits the meter's own grouping and its rests. Strong positions get chord
    tones; weak ones may take a passing or neighbour tone — that single rule is
    most of what separates a melody from a sequence of in-key notes.
    """
    shape = CONTOURS.get(contour_name, CONTOURS["arch"])
    k = len(rhythm)
    out = []
    pos = 0
    for j, item in enumerate(rhythm):
        dur, is_rest = item if isinstance(item, tuple) else (item, False)
        t = j / max(k - 1, 1)
        step = int(round(shape(t) * span))
        if pos % spb_steps and ident.frac() < 0.5:
            step += ident.pick((-1, 1))          # passing / neighbour tone
        out.append((dur, step, is_rest))
        pos += dur
    return out


def transform(motif, kind, ident):
    """Develop a motif rather than repeating or replacing it.

    This is why a B section can feel like the same *piece* while being different
    material — it is the A motif, altered. Wholly new material per section
    fragments the track; verbatim repetition bores.
    """
    if kind == "invert":
        top = max(s for _, s, _ in motif)
        return [(d, top - s, r) for d, s, r in motif]
    if kind == "sequence":
        shift = ident.pick((2, 3, -2, -3))
        return [(d, s + shift, r) for d, s, r in motif]
    if kind == "retrograde":
        return list(reversed(motif))
    if kind == "augment":
        return [(max(2, int(d * 1.5)), s, r) for d, s, r in motif]
    if kind == "diminish":
        return [(max(2, d // 2), s, r) for d, s, r in motif]
    if kind == "ornament":
        out = []
        for d, s, r in motif:
            if d >= 4 and not r and ident.frac() < 0.5:
                out.append((d // 2, s, r))
                out.append((d - d // 2, s + ident.pick((1, 2)), r))
            else:
                out.append((d, s, r))
        return out
    return list(motif)


TRANSFORMS = ("invert", "sequence", "retrograde", "augment", "ornament", "diminish")


def key_for_track(name, home_root, mood_name=""):
    """Offset the home key for this track, staying within closely related keys.

    The point is a pack that rotates through i / iv / v / bIII / bVII rather than
    sitting in one key for 24 tracks. Weighted toward the home key so the pack
    still has a tonal centre.
    """
    ident = Ident(name, f"key:{mood_name}")
    return (home_root + ident.pick(RELATED)) % 12


def plan_track(name, mood, mood_name, scale_name, bars_target):
    """Everything about a track that should be stable across regenerations.

    Returns a dict the caller turns into notes. Kept as data so it can be
    printed, diffed and reasoned about without rendering audio.
    """
    ident = Ident(name, f"plan:{mood_name}")
    scale = SCALES.get(scale_name, SCALES["minor"])

    form = ident.pick(FORMS)
    cadence = ident.pick(mood.get("cadences", ("authentic", "plagal")))
    seventh = ident.frac() < mood.get("seventh", 0.3)

    # One progression per distinct section letter, so A and B genuinely differ
    # harmonically rather than only melodically.
    progs = {}
    for letter in dict.fromkeys(form):
        pid = Ident(name, f"prog:{letter}:{mood_name}")
        # Learned transitions when we have them; hand-written functional
        # templates otherwise. Both end with a mood-appropriate cadence.
        learned = corpus_progression(pid, scale_name, 4, len(scale))
        if learned:
            a_c, b_c = CADENCES.get(cadence, CADENCES["authentic"])
            if a_c < len(scale) and b_c < len(scale):
                learned = learned + [(a_c, seventh), (b_c, False)]
            progs[letter] = learned
        else:
            progs[letter] = build_progression(
                pid, len(scale), cadence=cadence, seventh_on_dominant=seventh)

    # METER. The single biggest cause of "every track has the same meter" was
    # that there was only ever one. Mood-scoped, so 7/8 lands on dread rather
    # than on a fanfare.
    meter = (corpus_meter(ident, MOOD_METERS.get(mood_name, ("4/4",)))
             or ident.pick(MOOD_METERS.get(mood_name, ("4/4",))))
    steps, spb_steps, quarters = meter_grid(meter)

    contours = mood.get("contours", ("arch", "descending", "ascending"))
    density = mood.get("density", 0.4)
    rest_chance = mood.get("rest", 0.15)
    syncopate = mood.get("syncopate", 0.1)
    kits = MOOD_KITS.get(mood_name, ("backbeat",))

    sections = {}
    first = form[0]
    for letter in dict.fromkeys(form):
        sid = Ident(name, f"sec:{letter}:{mood_name}")
        sections[letter] = {
            "contour": sid.pick(contours),
            # Rhythm is GENERATED for this meter, not indexed out of a 4/4 table.
            "rhythm": (corpus_rhythm(sid, meter, steps)
                       or gen_rhythm(sid, steps, spb_steps, density,
                                     rest_chance, syncopate)),
            "bass_style": sid.pick(BASS_STYLES),
            "kit": sid.pick(kits),
            "transform": None if letter == first else sid.pick(TRANSFORMS),
            "octave": mood["octave"] + (0 if letter == first else sid.pick((0, 0, 1, -1))),
            "arp": mood["arp"] if letter == first else max(1, mood["arp"] + sid.pick((0, 1, 2))),
        }

    # Tempo spread inside the mood, so two `driving` tracks are not the same BPM.
    tempo_mul = mood["bpm"] * (0.94 + 0.12 * ident.frac())

    return {"form": form, "progs": progs, "sections": sections,
            "cadence": cadence, "seventh": seventh, "tempo_mul": tempo_mul,
            "scale": scale, "scale_name": scale_name,
            "meter": meter, "steps": steps, "spb_steps": spb_steps,
            "quarters": quarters, "duty": mood["duty"],
            "bars_per_section": max(2, bars_target // max(1, len(form)))}


# =============================================================================
# METER AND RHYTHM
#
# The original engine hardcoded 4/4 (STEPS = 16), ONE bass rhythm
# (`for s in (0, 4, 8, 12)`) and four drum patterns that all shared the SAME
# snare placement. Different notes on an identical rhythmic chassis reads as
# "the same music", which is exactly the complaint this fixes.
#
# No theory library helps here — music21 knows what 7/8 is, but it cannot know
# that this synthesizer only ever emitted 4/4. The fix is to stop hardcoding.
# =============================================================================

# (beats_per_bar, beat_unit). Everything is quantized to a 16th grid, so a /4
# meter gets 4 steps per beat and a /8 meter gets 2.
METERS = {
    "4/4":  (4, 4), "3/4": (3, 4), "5/4": (5, 4),
    "6/8":  (6, 8), "12/8": (12, 8), "7/8": (7, 8), "9/8": (9, 8),
    "2/4":  (2, 4),
}


def meter_grid(meter):
    """(steps_per_bar, steps_per_beat, quarter_notes_per_bar)."""
    beats, unit = METERS.get(meter, METERS["4/4"])
    spb_steps = 4 if unit == 4 else 2
    return beats * spb_steps, spb_steps, beats * (4.0 / unit)


# Which meters suit which mood. Compound meters (6/8, 12/8) lilt; 5/4 and 7/8
# feel unresolved, which is useful for dread and useless for a fanfare.
MOOD_METERS = {
    "heroic":     ("4/4", "4/4", "12/8"),
    "triumphant": ("4/4", "4/4", "3/4"),
    "ominous":    ("4/4", "3/4", "5/4", "7/8"),
    "eerie":      ("5/4", "7/8", "3/4", "6/8"),
    "melancholy": ("3/4", "6/8", "4/4"),
    "solemn":     ("3/4", "4/4", "12/8"),
    "mysterious": ("6/8", "5/4", "4/4", "7/8"),
    "tense":      ("7/8", "5/4", "4/4"),
    "frantic":    ("4/4", "7/8", "2/4"),
    "driving":    ("4/4", "4/4", "12/8"),
    "playful":    ("6/8", "2/4", "4/4"),
    "serene":     ("3/4", "6/8", "4/4"),
    "grand":      ("4/4", "3/4", "12/8"),
    "wondrous":   ("6/8", "12/8", "3/4"),
}

# Bass rhythm styles, as functions of the meter's grid. This is the single
# biggest contributor to "the meter feels the same" — a pedal note and a walking
# eighth line over the same chords are unrecognisable as the same arrangement.
BASS_STYLES = ("beats", "pedal", "eighths", "offbeat", "dotted", "halves",
               "firstlast", "gallop")


def bass_onsets(style, steps, spb_steps):
    """Step indices where the bass articulates, plus the duration of each."""
    if style == "pedal":                       # one long note, whole bar
        return [(0, steps)]
    if style == "beats":                       # on every beat
        return [(s, spb_steps) for s in range(0, steps, spb_steps)]
    if style == "eighths":
        h = max(1, spb_steps // 2)
        return [(s, h) for s in range(0, steps, h)]
    if style == "offbeat":                     # skip the downbeat entirely
        h = max(1, spb_steps // 2)
        return [(s, h) for s in range(h, steps, spb_steps)]
    if style == "dotted":                      # 3+3+2 feel, meter permitting
        out, s = [], 0
        for d in (3, 3, 2) * 4:
            if s >= steps:
                break
            out.append((s, min(d, steps - s)))
            s += d
        return out
    if style == "halves":
        two = spb_steps * 2
        return [(s, two) for s in range(0, steps, two)]
    if style == "firstlast":
        return [(0, spb_steps), (max(0, steps - spb_steps), spb_steps)]
    if style == "gallop":                      # long-short-short
        out, s = [], 0
        while s < steps:
            out.append((s, min(spb_steps, steps - s)))
            s += spb_steps
            for _ in range(2):
                if s >= steps:
                    break
                h = max(1, spb_steps // 2)
                out.append((s, min(h, steps - s)))
                s += h
        return out
    return [(s, spb_steps) for s in range(0, steps, spb_steps)]


# Drum kits described by FUNCTION rather than as fixed 16-char strings, so they
# work in 3/4, 7/8 and 12/8 instead of only 4/4. Values are beat positions
# (floats, in beats) — fractional entries land off the beat.
KITS = {
    "backbeat":  dict(k=(0, 2), s=(1, 3), h=0.5),
    "fourfloor": dict(k=(0, 1, 2, 3), s=(2,), h=0.5),
    "halftime":  dict(k=(0,), s=(2,), h=1.0),
    "march":     dict(k=(0, 1, 2, 3), s=(1.5, 3.5), h=0.0),
    "waltz":     dict(k=(0,), s=(1, 2), h=1.0),
    "gallop":    dict(k=(0, 1.5, 2.5), s=(2,), h=0.5),
    "sparse":    dict(k=(0,), s=(), h=0.0),
    "tribal":    dict(k=(0, 0.75, 1.5, 2.25), s=(3,), h=0.0),
    "heartbeat": dict(k=(0, 0.5), s=(), h=0.0),
    "none":      dict(k=(), s=(), h=0.0),
    "hatsonly":  dict(k=(), s=(), h=0.5),
    "syncop":    dict(k=(0, 1.75, 3), s=(1, 2.5), h=0.25),
}

MOOD_KITS = {
    "heroic":     ("march", "backbeat", "fourfloor"),
    "triumphant": ("march", "fourfloor", "backbeat"),
    "ominous":    ("halftime", "sparse", "heartbeat", "tribal"),
    "eerie":      ("sparse", "none", "heartbeat", "hatsonly"),
    "melancholy": ("waltz", "halftime", "sparse"),
    "solemn":     ("halftime", "waltz", "sparse", "none"),
    "mysterious": ("hatsonly", "sparse", "syncop", "tribal"),
    "tense":      ("syncop", "heartbeat", "halftime", "tribal"),
    "frantic":    ("fourfloor", "gallop", "syncop"),
    "driving":    ("fourfloor", "backbeat", "gallop"),
    "playful":    ("backbeat", "gallop", "waltz"),
    "serene":     ("none", "hatsonly", "sparse"),
    "grand":      ("march", "halftime", "tribal"),
    "wondrous":   ("hatsonly", "waltz", "sparse"),
}


def kit_pattern(kit_name, steps, spb_steps):
    """Realize a kit into {kind: set(step indices)} for this meter."""
    kit = KITS.get(kit_name, KITS["backbeat"])
    beats = steps / float(spb_steps)
    out = {"k": set(), "s": set(), "h": set()}
    for kind in ("k", "s"):
        for b in kit[kind]:
            if b < beats:
                out[kind].add(int(round(b * spb_steps)) % steps)
    if kit["h"] > 0:
        stride = max(1, int(round(kit["h"] * spb_steps)))
        out["h"] = set(range(0, steps, stride))
    return out


def gen_rhythm(ident, steps, spb_steps, density, rest_chance=0.0, syncopate=0.0):
    """Generate a motif rhythm summing to `steps`, as (duration, is_rest) pairs.

    Replaces the fixed 11-entry RHYTHMS table, which could only ever describe
    4/4 bars and had no rests — so every lead line played continuously, in the
    same meter, forever.

    `density` 0..1 biases toward shorter notes. Rests matter more than they
    sound like they should: a melody that never stops is a melody with no phrase
    structure, and it is a large part of why every track felt the same.
    """
    # Durations available, longest first, expressed in 16th steps.
    pool = [spb_steps * 2, spb_steps, spb_steps // 2 or 1, spb_steps + spb_steps // 2]
    pool = [d for d in dict.fromkeys(pool) if d >= 1]
    out, pos = [], 0
    while pos < steps:
        room = steps - pos
        cands = [d for d in pool if d <= room] or [room]
        # Denser moods weight short values; sparse ones weight long.
        if ident.frac() < density:
            d = min(cands)
        else:
            d = cands[ident._next() % len(cands)]
        if syncopate and pos % spb_steps == 0 and ident.frac() < syncopate:
            half = max(1, spb_steps // 2)
            if half <= room:
                out.append((half, True))       # push the phrase off the beat
                pos += half
                continue
        out.append((d, ident.frac() < rest_chance))
        pos += d
    return out


# =============================================================================
# LEARNED IDIOM (corpus.json, from tools/mine-corpus.py)
#
# Everything above is hand-written: my guesses about what game music does. Where
# a mined corpus has real measurements, they win. Two of my guesses were already
# shown wrong by 1,107 real game MIDIs:
#
#   METER      I spread moods across 7/8, 5/4, 12/8 and 2/4. The corpus is 90%
#              4/4 and 5.6% 3/4; everything else is under 1%. Exotic metres are a
#              spice, not a palette.
#   CADENCE    I defaulted minor keys to authentic (V-i) as classical practice
#              suggests. In the corpus, iv->i (plagal) outnumbers V->i by 1976 to
#              1189. Game music leans modal, not classical.
#
# The corpus is OPTIONAL. Without it everything falls back to the tables above,
# so soundmon works on a machine with no MIDI collection.
# =============================================================================
CORPUS = None


def load_corpus(path=None):
    """Load corpus.json if present. Cached; returns None when absent."""
    global CORPUS
    if CORPUS is not None:
        return CORPUS or None
    import json
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in ([path] if path else []) + [os.path.join(here, "corpus.json")]:
        if cand and os.path.exists(cand):
            try:
                with open(cand) as fh:
                    CORPUS = json.load(fh)
                return CORPUS
            except Exception:
                pass
    CORPUS = {}
    return None


def _weighted(ident, pairs):
    """Pick from [(value, weight), ...] using the track's identity."""
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[0][0] if pairs else None
    r = ident.frac() * total
    for v, w in pairs:
        r -= w
        if r <= 0:
            return v
    return pairs[-1][0]


def corpus_meter(ident, mood_meters, bias=0.65):
    """Sample a time signature from the corpus, nudged toward the mood's taste.

    `bias` is how much weight the mood's preferred metres get on top of their
    corpus frequency. At 0.65 a mood that likes 7/8 will pick it sometimes rather
    than 0.3% of the time, without pretending game music is mostly in 7/8.
    """
    c = load_corpus()
    if not c or not c.get("timesigs"):
        return None
    pref = set(mood_meters or ())
    pairs = []
    for ts, count in c["timesigs"]:
        if ts not in METERS:
            continue
        w = float(count)
        if ts in pref:
            w += bias * sum(n for _, n in c["timesigs"])
        pairs.append((ts, w))
    return _weighted(ident, pairs) if pairs else None


def corpus_progression(ident, mode, length, scale_len=7):
    """Walk the corpus's degree-transition matrix.

    A Markov walk over real transitions produces progressions that are idiomatic
    by measurement rather than by my assertion — and it naturally reproduces the
    corpus's strong preference for lingering on the tonic, which hand-written
    templates never did.
    """
    c = load_corpus()
    if not c:
        return None
    tab = (c.get("transitions") or {}).get(
        "minor" if mode != "major" else "major")
    if not tab:
        return None
    nxt = {}
    for key, count in tab.items():
        try:
            a, b = (int(x) for x in key.split(">"))
        except ValueError:
            continue
        if a < scale_len and b < scale_len:
            nxt.setdefault(a, []).append((b, count))
    if not nxt:
        return None
    deg = 0
    out = [(0, False)]
    for _ in range(max(1, length - 1)):
        opts = nxt.get(deg)
        if not opts:
            deg = 0
        else:
            # Suppress self-transitions a little: the corpus counts every bar a
            # chord is held, so 0>0 dominates and a raw walk would sit still.
            adj = [(b, w * (0.25 if b == deg else 1.0)) for b, w in opts]
            deg = _weighted(ident, adj)
        out.append((deg, False))
    return out


def corpus_rhythm(ident, timesig, steps):
    """Return a real observed rhythm for this metre, as (duration, is_rest)."""
    c = load_corpus()
    if not c:
        return None
    pool = (c.get("rhythms") or {}).get(timesig)
    if not pool:
        return None
    pairs = [(tuple(d), w) for d, w in pool if d and sum(d) <= steps * 2]
    if not pairs:
        return None
    durs = list(_weighted(ident, pairs))
    out, pos = [], 0
    for d in durs:
        d = max(1, min(int(d), steps - pos))
        if d <= 0:
            break
        out.append((d, False))
        pos += d
        if pos >= steps:
            break
    if pos < steps:
        out.append((steps - pos, ident.frac() < 0.35))   # tail: note or rest
    return out or None
