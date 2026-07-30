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


def make_motif(ident, rhythm, contour_name, span, chord_len=3):
    """A motif as (duration, scale-step) pairs following a contour archetype.

    Strong positions get chord tones; weak ones may take a passing or neighbour
    tone. That single rule is most of what separates a melody from a sequence of
    in-key notes.
    """
    shape = CONTOURS.get(contour_name, CONTOURS["arch"])
    k = len(rhythm)
    out = []
    pos = 0
    for j, dur in enumerate(rhythm):
        t = j / max(k - 1, 1)
        step = int(round(shape(t) * span))
        strong = (pos % 4 == 0)
        if not strong and ident.frac() < 0.5:
            step += ident.pick((-1, 1))          # passing / neighbour tone
        out.append((dur, step))
        pos += dur
    return out


def transform(motif, kind, ident):
    """Develop a motif rather than repeating or replacing it.

    This is why a B section can feel like the same *piece* while being different
    material — it is the A motif, altered. Wholly new material per section
    fragments the track; verbatim repetition bores.
    """
    if kind == "invert":
        top = max(s for _, s in motif)
        return [(d, top - s) for d, s in motif]
    if kind == "sequence":
        shift = ident.pick((2, 3, -2, -3))
        return [(d, s + shift) for d, s in motif]
    if kind == "retrograde":
        return list(reversed(motif))
    if kind == "augment":
        return [(max(2, int(d * 1.5)), s) for d, s in motif]
    if kind == "diminish":
        return [(max(2, d // 2), s) for d, s in motif]
    if kind == "ornament":
        out = []
        for d, s in motif:
            if d >= 4 and ident.frac() < 0.5:
                out.append((d // 2, s))
                out.append((d - d // 2, s + ident.pick((1, 2))))
            else:
                out.append((d, s))
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
        progs[letter] = build_progression(
            Ident(name, f"prog:{letter}:{mood_name}"), len(scale),
            cadence=cadence, seventh_on_dominant=seventh)

    contours = mood.get("contours", ("arch", "descending", "ascending"))
    sections = {}
    first = form[0]
    for letter in dict.fromkeys(form):
        sid = Ident(name, f"sec:{letter}:{mood_name}")
        sections[letter] = {
            "contour": sid.pick(contours),
            "rhythm_idx": sid.pick(mood["rhythms"]),
            "transform": None if letter == first else sid.pick(TRANSFORMS),
            "octave": mood["octave"] + (0 if letter == first else sid.pick((0, 0, 1, -1))),
            "arp": mood["arp"] if letter == first else max(1, mood["arp"] + sid.pick((0, 1, 2))),
        }

    # Tempo spread inside the mood, so two `driving` tracks are not the same BPM.
    tempo_mul = mood["bpm"] * (0.94 + 0.12 * ident.frac())

    return {"form": form, "progs": progs, "sections": sections,
            "cadence": cadence, "seventh": seventh, "tempo_mul": tempo_mul,
            "scale": scale, "scale_name": scale_name,
            "drums": ident.rint(0, 3), "duty": mood["duty"],
            "bars_per_section": max(2, bars_target // max(1, len(form)))}
