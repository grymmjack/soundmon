#!/usr/bin/env python3
"""--chipfx / --oplfx: synthesized retro sound effects, sfxr-style.

WHY, AGAIN

Same argument as chip.py, and it applies even harder to short effects. A 0.12 s
footstep is 5,300 samples; a diffusion model asked for one produces a fragment of
a recording of a footstep, at 44.1 kHz, with a room around it. A 2A03 footstep is
a 30 ms burst of LFSR noise with a fast decay, and there is nothing to record.

So these are BUILT, not generated. Every effect is a recipe — a pitch contour, a
waveform, an envelope — which is exactly how sfxr/bfxr and the era's actual sound
designers worked. That makes them:

    authentic     a noise burst is a noise burst, not an approximation of one
    instant       46 effects in under a second, no GPU, nothing to install
    tweakable     don't like `hit`? change six numbers, not a prompt
    reproducible  the seed is in the filename and the recipe is deterministic

TWO RENDERERS, ONE RECIPE LAYER

    --chipfx   PSG: pulse / triangle / 15-bit LFSR noise    (NES, Amiga, generic)
    --oplfx    FM through the Nuked OPL3 core                (AdLib, Sound Blaster)

The archetype and its parameters are shared; only the voicing differs — the same
split chip.py and opl.py use for music. An FM explosion is a high-feedback patch
with an extreme modulator ratio swept downward, because that is how DOS games
made explosions when all they had was two operators.

MAPPING NAMES TO ARCHETYPES

`classify()` matches the asset name and its description against keywords. This is
deliberately a lookup table and not a model: there are ~20 archetypes, the game
has 46 effects, and a table is inspectable, instant and never surprises you.
An unmatched name falls back to `blip`, which is short and inoffensive rather
than silent — a missing effect should be audible as "wrong", not as nothing.
"""
import os
import sys

SAMPLE_RATE = 44100

# name/description keyword -> archetype. First match wins, so order matters:
# more specific keys must precede the generic ones they contain.
KEYWORDS = [
    (("breakdoor", "shatter", "splinter", "break"), "crash"),
    (("strongdoor", "door", "chest", "creak", "hinge"), "creak"),
    (("boom", "explos", "blast"), "boom"),
    (("lightning", "thunder", "crack", "spark"), "crack"),
    (("fireball", "flame", "fire"), "fire"),
    (("frost", "ice", "shimmer", "sparkle"), "shimmer"),
    (("teleport", "warp", "portal"), "sweepup"),
    (("poison", "fizzle", "sizzle", "acid"), "fizzle"),
    (("hiss", "steam", "static"), "noise"),
    (("alarm", "trap", "warning", "siren"), "alarm"),
    (("levelup", "level up", "fanfare", "win", "victor", "triumph"), "fanfare"),
    (("treasure", "coin", "gold", "pickup", "loot"), "coin"),
    (("secret", "curio", "reveal", "mystery", "discover"), "chime"),
    (("saveok", "confirm", "accept", "ok"), "confirmup"),
    (("savebad", "error", "deny", "fumble", "fail", "lose", "invalid"), "errordown"),
    (("maxhit", "crit"), "bighit"),
    (("monster-death", "monster-pain", "growl", "roar", "monster"), "growl"),
    (("death", "die"), "death"),
    (("player-pain", "pain", "hurt", "grunt"), "hurt"),
    (("heartbeat", "heart", "pulse"), "heartbeat"),
    (("diceroll", "rattle", "roll"), "rattle"),
    (("dice_settle", "settle"), "rattle"),
    (("dice", "click", "land", "edge", "tick"), "click"),
    (("move", "step", "footstep", "walk"), "step"),
    (("bump", "thud", "dull"), "thud"),
    (("hit", "impact", "strike", "damage"), "hit"),
    (("miss", "whoosh", "swing", "swipe"), "whoosh"),
    (("key", "metal", "jingle", "unlock"), "metal"),
    (("idle", "ambien", "drone", "hum"), "drone"),
    (("search", "scan", "look"), "search"),
    (("select", "menu", "cursor", "blip", "beep"), "blip"),
    (("voice", "text", "talk", "speak"), "blip"),
]

ARCHETYPES = ("click", "step", "thud", "creak", "crash", "boom", "crack", "fire",
              "shimmer", "sweepup", "fizzle", "noise", "alarm", "fanfare", "coin",
              "chime", "confirmup", "errordown", "hit", "bighit", "whoosh",
              "growl", "death", "hurt", "heartbeat", "rattle", "metal", "drone",
              "search", "blip")


# Explicit archetype per known asset. Keyword inference is a FALLBACK, not the
# primary route, because substring matching on short words is a trap: "dice"
# contains "ice", so `diceroll` classified as a frost shimmer until this table
# existed. With a known, finite asset list, naming each one is both safer and
# better — it lets the choice be hand-tuned per effect instead of inferred.
NAME_MAP = {
    "move": "step", "bump": "thud", "door": "creak", "strongdoor": "creak",
    "breakdoor": "crash", "secret": "chime", "secretpass": "chime",
    "key": "metal", "idle": "drone", "treasure": "coin", "trap": "alarm",
    "hit": "hit", "miss": "whoosh", "crit": "bighit", "fumble": "errordown",
    "search": "search", "win": "fanfare", "lose": "errordown",
    "saveok": "confirmup", "savebad": "errordown", "chest": "creak",
    "boom": "boom", "hiss": "noise", "fizzle": "fizzle", "alarm": "alarm",
    "select": "blip", "levelup": "fanfare", "voice": "blip",
    "diceroll": "rattle", "diceland": "click", "dice_edge": "click",
    "dice_settle": "rattle", "dice-math-1": "blip", "dice-math-2": "blip",
    "monster-pain": "growl", "player-pain": "hurt", "death": "death",
    "monster-death": "growl", "maxhit": "bighit", "heartbeat": "heartbeat",
    "curio": "chime", "poison-proc": "fizzle", "frost-proc": "shimmer",
    "teleport": "sweepup", "fireball": "fire", "lightning-bolt": "crack",
}


def classify(name, desc=""):
    key = (name or "").strip().lower()
    if key in NAME_MAP:
        return NAME_MAP[key]
    hay = f"{name} {desc}".lower()
    for keys, arch in KEYWORDS:
        for k in keys:
            if k in hay:
                return arch
    return "blip"


# --- PSG primitives ----------------------------------------------------------
def _pulse_sweep(np, sr, n, f0, f1, duty=0.5, curve=1.0):
    """Pulse wave whose pitch glides f0 -> f1. The glide is the single most
    useful gesture in 8-bit sound design: up reads as gain/reward, down as
    damage/failure, and steepness reads as force."""
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    f = f0 + (f1 - f0) * (t ** curve)
    ph = np.cumsum(f) / sr
    return np.where(ph % 1.0 < duty, 1.0, -1.0)


def _tri_sweep(np, sr, n, f0, f1, curve=1.0):
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    f = f0 + (f1 - f0) * (t ** curve)
    ph = np.cumsum(f) / sr
    saw = ph % 1.0
    return np.round((2.0 * np.abs(2.0 * saw - 1.0) - 1.0) * 7.5) / 7.5


def _lfsr(np, n, sr, period, seed=1):
    """Real 15-bit LFSR noise — the 2A03 generator, not uniform random. Taps at
    bits 0 and 1 give it the specific metallic colour NES percussion has."""
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


def _noise_sweep(np, n, sr, p0, p1, seed=1):
    """LFSR noise whose clock rate glides — a pitched noise sweep, which is how
    the era did whooshes, fire and steam."""
    out = np.empty(n)
    reg = (int(seed) & 0x7FFF) or 1
    val, acc = 1.0, 0.0
    for i in range(n):
        frac = i / max(n - 1, 1)
        period = p0 + (p1 - p0) * frac
        acc += max(period, 1.0) / sr
        if acc >= 1.0:
            acc -= 1.0
            fb = (reg ^ (reg >> 1)) & 1
            reg = (reg >> 1) | (fb << 14)
            val = 1.0 if (reg & 1) else -1.0
        out[i] = val
    return out


def _ad(np, n, sr, a_ms=1.0, decay=8.0, sustain=0.0):
    t = np.arange(n, dtype=np.float64) / sr
    e = sustain + (1.0 - sustain) * np.exp(-t * decay)
    a = max(1, min(n, int(sr * a_ms / 1000.0)))
    e[:a] *= np.linspace(0.0, 1.0, a)
    # Always land on zero — a hard cut on a square wave is a click, and 46 of
    # these play constantly during a game.
    r = max(2, min(n // 4, int(sr * 0.004)))
    e[-r:] *= np.linspace(1.0, 0.0, r)
    return e


def _arp(np, sr, n, notes, duty=0.5, decay=12.0):
    """A sequence of pitched blips — chime, coin, fanfare all share this shape."""
    k = len(notes)
    seg = max(4, n // k)
    parts = []
    for i, hz in enumerate(notes):
        m = seg if i < k - 1 else max(seg, n - seg * (k - 1))
        parts.append(_pulse_sweep(np, sr, m, hz, hz, duty) * _ad(np, m, sr, 1.0, decay))
    return np.concatenate(parts)[:n]


def render_psg(arch, n, sr, rng, np):
    """One archetype -> audio. Numbers here are the sound design; tune freely."""
    R = lambda a, b: float(rng.uniform(a, b))
    sd = int(rng.integers(1, 32767))

    if arch == "click":
        return _lfsr(np, n, sr, R(6000, 9000), sd) * _ad(np, n, sr, 0.3, 220.0)
    if arch == "step":
        return _noise_sweep(np, n, sr, R(2200, 3200), R(700, 1100), sd) * \
               _ad(np, n, sr, 0.3, 120.0) * 0.8
    if arch == "thud":
        return _tri_sweep(np, sr, n, R(150, 190), R(55, 70), 0.6) * _ad(np, n, sr, 0.5, 34.0)
    if arch == "creak":
        # Two detuned slow sweeps: the beating between them is the "groan".
        a = _pulse_sweep(np, sr, n, R(120, 160), R(230, 300), 0.16, 1.4)
        b = _pulse_sweep(np, sr, n, R(126, 168), R(240, 312), 0.22, 1.4)
        gr = _lfsr(np, n, sr, 900.0, sd) * 0.22
        return (a * 0.5 + b * 0.4 + gr) * _ad(np, n, sr, 12.0, 2.2, 0.45)
    if arch == "crash":
        return (_noise_sweep(np, n, sr, R(9000, 13000), R(500, 900), sd) +
                _tri_sweep(np, sr, n, 220.0, 60.0, 0.5) * 0.5) * _ad(np, n, sr, 0.5, 11.0)
    if arch == "boom":
        return (_noise_sweep(np, n, sr, R(1600, 2400), R(180, 320), sd) * 1.0 +
                _tri_sweep(np, sr, n, R(90, 120), 34.0, 0.45) * 0.7) * _ad(np, n, sr, 1.0, 6.5)
    if arch == "crack":
        return (_lfsr(np, n, sr, R(14000, 19000), sd) * _ad(np, n, sr, 0.2, 90.0) +
                _noise_sweep(np, n, sr, 8000.0, 600.0, sd + 1) * _ad(np, n, sr, 0.2, 13.0) * 0.7)
    if arch == "fire":
        return _noise_sweep(np, n, sr, R(700, 1100), R(3200, 4200), sd) * \
               _ad(np, n, sr, 8.0, 5.0, 0.25)
    if arch == "shimmer":
        base = R(1500, 1900)
        return _arp(np, sr, n, [base, base * 1.5, base * 2.0, base * 2.5, base * 3.0],
                    0.125, 26.0) * 0.75
    if arch == "sweepup":
        return _pulse_sweep(np, sr, n, R(180, 260), R(2400, 3200), 0.25, 0.6) * \
               _ad(np, n, sr, 2.0, 4.5, 0.35)
    if arch == "fizzle":
        return _noise_sweep(np, n, sr, R(5000, 7000), R(900, 1400), sd) * \
               _ad(np, n, sr, 2.0, 7.0, 0.2)
    if arch == "noise":
        return _lfsr(np, n, sr, R(6500, 9000), sd) * _ad(np, n, sr, 15.0, 1.2, 0.7)
    if arch == "alarm":
        lo, hi = R(520, 620), R(820, 960)
        reps = max(2, int(n / (sr * 0.11)))
        return _arp(np, sr, n, [lo, hi] * (reps // 2 + 1), 0.5, 5.0)
    if arch == "fanfare":
        r = R(392, 466)
        return _arp(np, sr, n, [r, r * 1.26, r * 1.5, r * 2.0], 0.5, 7.0)
    if arch == "coin":
        r = R(920, 1080)
        # The two-note coin: a grace note, then the held one. Length ratio is the
        # whole trick — an even split sounds like a doorbell.
        g = max(4, int(n * 0.22))
        return np.concatenate([
            _pulse_sweep(np, sr, g, r, r, 0.5) * _ad(np, g, sr, 0.5, 18.0),
            _pulse_sweep(np, sr, n - g, r * 1.5, r * 1.5, 0.5) * _ad(np, n - g, sr, 0.5, 7.0)])
    if arch == "chime":
        r = R(680, 820)
        return _arp(np, sr, n, [r, r * 1.5, r * 2.0], 0.25, 9.0)
    if arch == "confirmup":
        r = R(600, 720)
        return _arp(np, sr, n, [r, r * 1.335], 0.5, 14.0)
    if arch == "errordown":
        r = R(320, 400)
        return _arp(np, sr, n, [r, r * 0.75], 0.5, 9.0) * 0.9
    if arch == "hit":
        return (_lfsr(np, n, sr, R(3200, 4600), sd) * 0.85 +
                _tri_sweep(np, sr, n, R(300, 380), R(80, 110), 0.5) * 0.6) * \
               _ad(np, n, sr, 0.3, 40.0)
    if arch == "bighit":
        return (_noise_sweep(np, n, sr, R(5000, 7000), R(600, 1000), sd) +
                _tri_sweep(np, sr, n, R(260, 330), 60.0, 0.45) * 0.8) * \
               _ad(np, n, sr, 0.4, 15.0)
    if arch == "whoosh":
        return _noise_sweep(np, n, sr, R(1200, 1700), R(4200, 5600), sd) * \
               _ad(np, n, sr, 6.0, 6.0, 0.15) * 0.7
    if arch == "growl":
        t = np.arange(n, dtype=np.float64) / sr
        wob = 1.0 + 0.22 * np.sin(2 * np.pi * R(11, 17) * t)
        base = _pulse_sweep(np, sr, n, R(150, 200), R(70, 95), 0.35, 1.2)
        return (base * wob + _lfsr(np, n, sr, 1400.0, sd) * 0.3) * _ad(np, n, sr, 6.0, 3.2, 0.4)
    if arch == "death":
        return (_pulse_sweep(np, sr, n, R(420, 520), R(50, 70), 0.35, 1.6) * 0.9 +
                _lfsr(np, n, sr, 1100.0, sd) * 0.2) * _ad(np, n, sr, 3.0, 2.4, 0.35)
    if arch == "hurt":
        return _pulse_sweep(np, sr, n, R(340, 420), R(150, 200), 0.35, 1.3) * \
               _ad(np, n, sr, 1.0, 16.0)
    if arch == "heartbeat":
        h = n // 2
        one = lambda m, f: _tri_sweep(np, sr, m, f, f * 0.55, 0.6) * _ad(np, m, sr, 1.0, 26.0)
        return np.concatenate([one(h, 96.0), one(n - h, 78.0)])
    if arch == "rattle":
        # Discrete clatters, thinning out — dice coming to rest.
        parts, pos = [], 0
        while pos < n:
            m = min(int(sr * R(0.012, 0.03)), n - pos)
            if m <= 2:
                break
            parts.append(_lfsr(np, m, sr, R(3500, 7000), int(rng.integers(1, 32767))) *
                         _ad(np, m, sr, 0.2, 150.0))
            gap = min(int(sr * R(0.015, 0.05)), n - pos - m)
            if gap > 0:
                parts.append(np.zeros(gap))
            pos += m + max(gap, 0)
        out = np.concatenate(parts)[:n] if parts else np.zeros(n)
        if len(out) < n:
            out = np.concatenate([out, np.zeros(n - len(out))])
        return out * np.linspace(1.0, 0.25, n)
    if arch == "metal":
        r = R(1500, 1900)
        return (_arp(np, sr, n, [r, r * 1.19], 0.125, 20.0) +
                _lfsr(np, n, sr, 11000.0, sd) * _ad(np, n, sr, 0.3, 60.0) * 0.35)
    if arch == "drone":
        t = np.arange(n, dtype=np.float64) / sr
        v = 1.0 + 0.02 * np.sin(2 * np.pi * 4.5 * t)
        return _tri_sweep(np, sr, n, R(85, 105), R(85, 105), 1.0) * v * \
               _ad(np, n, sr, 40.0, 0.6, 0.85) * 0.8
    if arch == "search":
        r = R(700, 860)
        return _arp(np, sr, n, [r, r * 1.12, r], 0.25, 22.0) * 0.8
    # blip — the fallback. Short, clean, obviously a UI sound.
    r = R(760, 940)
    return _pulse_sweep(np, sr, n, r, r, 0.5) * _ad(np, n, sr, 0.5, 22.0)


# --- OPL renderer ------------------------------------------------------------
# FM effects are pitch sweeps over a patch, which is what AdLib-only games did.
# Noise is faked with an extreme modulator ratio plus high feedback — an
# inharmonic wash. That is not a workaround, it is the period-correct technique:
# the OPL has no noise channel outside fixed rhythm mode.
OPL_FX = {
    #            mult ksl tl  ar dr sl rr wv    fb  sweep(f0mul,f1mul,curve)
    "click":     ((8, 0, 8, 15, 15, 0, 15, 0), 7, (1.0, 0.9, 1.0)),
    "step":      ((12, 0, 10, 15, 14, 0, 14, 0), 7, (1.0, 0.5, 1.0)),
    "thud":      ((1, 0, 6, 15, 10, 0, 10, 0), 3, (1.0, 0.35, 0.6)),
    "creak":     ((7, 0, 12, 6, 4, 8, 5, 0), 6, (1.0, 1.9, 1.4)),
    "crash":     ((14, 0, 6, 15, 8, 0, 8, 0), 7, (1.0, 0.15, 0.5)),
    "boom":      ((13, 0, 4, 15, 6, 0, 6, 0), 7, (1.0, 0.2, 0.45)),
    "crack":     ((15, 0, 6, 15, 13, 0, 13, 0), 7, (1.0, 0.25, 0.5)),
    "fire":      ((11, 0, 10, 12, 5, 4, 5, 0), 7, (1.0, 2.6, 1.0)),
    "shimmer":   ((4, 0, 10, 15, 11, 0, 11, 1), 5, (1.0, 2.2, 0.7)),
    "sweepup":   ((1, 0, 8, 14, 6, 2, 6, 0), 4, (1.0, 6.0, 0.6)),
    "fizzle":    ((10, 0, 12, 13, 7, 3, 7, 0), 6, (1.0, 0.4, 1.0)),
    "noise":     ((12, 0, 12, 10, 2, 10, 3, 0), 7, (1.0, 1.05, 1.0)),
    "alarm":     ((1, 0, 6, 15, 4, 6, 5, 0), 2, (1.0, 1.0, 1.0)),
    "fanfare":   ((1, 0, 6, 14, 5, 4, 6, 0), 4, (1.0, 1.0, 1.0)),
    "coin":      ((3, 0, 7, 15, 9, 0, 9, 0), 5, (1.0, 1.0, 1.0)),
    "chime":     ((3, 0, 8, 15, 8, 0, 8, 0), 5, (1.0, 1.0, 1.0)),
    "confirmup": ((1, 0, 8, 15, 10, 0, 10, 0), 3, (1.0, 1.0, 1.0)),
    "errordown": ((1, 0, 8, 15, 8, 0, 8, 0), 3, (1.0, 0.75, 1.0)),
    "hit":       ((13, 0, 7, 15, 12, 0, 12, 0), 7, (1.0, 0.3, 0.5)),
    "bighit":    ((14, 0, 5, 15, 9, 0, 9, 0), 7, (1.0, 0.22, 0.45)),
    "whoosh":    ((10, 0, 11, 11, 6, 3, 6, 0), 7, (1.0, 3.4, 1.0)),
    "growl":     ((2, 0, 8, 12, 4, 6, 5, 0), 6, (1.0, 0.55, 1.2)),
    "death":     ((2, 0, 7, 13, 4, 5, 5, 0), 5, (1.0, 0.14, 1.6)),
    "hurt":      ((2, 0, 8, 15, 11, 0, 11, 0), 4, (1.0, 0.45, 1.3)),
    "heartbeat": ((1, 0, 6, 15, 11, 0, 11, 0), 2, (1.0, 0.55, 0.6)),
    "rattle":    ((11, 0, 9, 15, 14, 0, 14, 0), 7, (1.0, 0.85, 1.0)),
    "metal":     ((5, 0, 8, 15, 10, 0, 10, 1), 6, (1.0, 1.15, 1.0)),
    "drone":     ((1, 0, 10, 8, 0, 12, 4, 0), 1, (1.0, 1.0, 1.0)),
    "search":    ((3, 0, 9, 15, 12, 0, 12, 0), 4, (1.0, 1.1, 1.0)),
    "blip":      ((1, 0, 8, 15, 11, 0, 11, 0), 3, (1.0, 1.0, 1.0)),
}

# Archetypes whose PSG version is a note sequence: on OPL they retrigger.
_SEQ = {"alarm": 2, "fanfare": 4, "coin": 2, "chime": 3, "confirmup": 2,
        "errordown": 2, "shimmer": 5, "metal": 2, "search": 3, "rattle": 6}


def render_opl(arch, n, sr, rng, np, oplmod):
    """Render an archetype through the real OPL3 core."""
    spec = OPL_FX.get(arch, OPL_FX["blip"])
    (mult, ksl, tl, ar, dr, sl, rr, wv), fb, (m0, m1, curve) = spec
    base = float(rng.uniform(300.0, 460.0))
    if arch in ("thud", "boom", "heartbeat", "drone", "growl", "death"):
        base = float(rng.uniform(110.0, 170.0))
    elif arch in ("click", "crack", "shimmer", "metal"):
        base = float(rng.uniform(900.0, 1400.0))

    chip = oplmod.OPL3()
    ins = oplmod.Instrument(arch,
                            (mult, ksl, tl, ar, dr, sl, rr, wv, 0, 0, 0, 0),
                            (1, 0, 0, ar, dr, sl, rr, wv, 0, 0, 0, 0), fb=fb, con=0)
    chip.program(0, ins)

    reps = _SEQ.get(arch, 1)
    ratios = {"fanfare": (1.0, 1.26, 1.5, 2.0), "coin": (1.0, 1.5),
              "chime": (1.0, 1.5, 2.0), "confirmup": (1.0, 1.335),
              "errordown": (1.0, 0.75), "alarm": (1.0, 1.6),
              "shimmer": (1.0, 1.5, 2.0, 2.5, 3.0), "metal": (1.0, 1.19),
              "search": (1.0, 1.12, 1.0), "rattle": (1.0, 0.95, 1.05, 0.9, 1.0, 0.97)}
    rs = ratios.get(arch, (1.0,))

    # Sweep the F-number in small slices; that is the only way to bend pitch on
    # an OPL, and it is exactly what the era's sound code did.
    SLICES = 24
    out = []
    per = max(1, n // reps)
    for r_i in range(reps):
        f_start = base * rs[r_i % len(rs)]
        chip.key_off(0)
        chip.key_on(0, f_start * m0)
        seg = per if r_i < reps - 1 else max(per, n - per * (reps - 1))
        for s in range(SLICES):
            frac = (s / max(SLICES - 1, 1)) ** curve
            chip.write(0xA0, 0)          # touch nothing; keeps timing uniform
            fn, bl = oplmod.fnum_block(f_start * (m0 + (m1 - m0) * frac))
            chip.write(0xA0, fn & 0xFF)
            chip.write(0xB0, 0x20 | ((bl & 7) << 2) | ((fn >> 8) & 3))
            m = seg // SLICES + (1 if s < seg % SLICES else 0)
            if m > 0:
                out.append(chip.render(m, np))
        chip.key_off(0)
    a = np.concatenate(out) if out else np.zeros(n)
    if len(a) < n:
        a = np.concatenate([a, np.zeros(n - len(a))])
    return a[:n] * _ad(np, n, SAMPLE_RATE, 1.0, 3.0, 0.6)


def run(a, slug, to_ogg=None, loudness_normalize=None):
    """Generate a synthesized sound effect. Same contract as chip.run()."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        sys.exit(f"--chipfx needs numpy + soundfile: pip install soundfile numpy  ({e})")

    use_opl = bool(getattr(a, "oplfx", False))
    oplmod = None
    if use_opl:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import opl as oplmod                                    # noqa: N813

    dest = os.path.abspath(os.path.expanduser(a.output_to)) if a.output_to else os.getcwd()
    if not os.path.isdir(dest):
        if a.create_dirs or a.output_to:
            os.makedirs(dest, exist_ok=True)
        else:
            sys.exit(f"output dir does not exist: {dest} (add --create-dirs)")

    base = a.name or slug(a.prompt or "sfx")
    arch = a.fx_archetype or classify(base, a.prompt or "")
    if arch not in ARCHETYPES:
        sys.exit(f"unknown --fx-archetype {arch!r}. One of: {', '.join(ARCHETYPES)}")

    # Effects are short; --seconds defaults to something musical, so prefer an
    # explicit --max-seconds (which the pack generator passes as the target).
    dur = a.max_seconds if getattr(a, "max_seconds", 0) else a.seconds
    dur = max(0.03, min(float(dur), 10.0))
    n = int(SAMPLE_RATE * dur)

    n_out = max(1, a.number)
    made = []
    for i in range(n_out):
        if a.seed is not None and a.seed >= 0:
            seed = (a.seed + i) % (2 ** 32)
        else:
            seed = int.from_bytes(os.urandom(4), "big")
        rng = np.random.default_rng(seed)

        audio = (render_opl(arch, n, SAMPLE_RATE, rng, np, oplmod) if use_opl
                 else render_psg(arch, n, SAMPLE_RATE, rng, np))
        audio = np.asarray(audio, dtype=np.float64)[:n]
        peak = float(np.abs(audio).max())
        if peak > 1e-9:
            audio = audio * (10.0 ** (a.normalize_db / 20.0) / peak)

        name = f"{base}_s{seed}"
        path = os.path.join(dest, f"{name}.wav")
        sf.write(path, audio, SAMPLE_RATE,
                 subtype=f"PCM_{a.bits}" if a.bits != 8 else "PCM_U8")
        if getattr(a, "lufs_target", None) is not None and loudness_normalize:
            loudness_normalize(path, a.lufs_target, a.true_peak)
        if getattr(a, "ogg", False) and to_ogg:
            path = to_ogg(path, a.ogg_quality, a.keep_wav)
        made.append(path)
        print(f"   ✅ [{i+1}/{n_out}] {os.path.basename(path):<30} "
              f"{len(audio)/SAMPLE_RATE:5.2f}s  {'FM' if use_opl else 'PSG'}:{arch:<10} "
              f"seed={seed}")

    print(f"   all done  |  {len(made)} file(s) in {dest}")
    return made
