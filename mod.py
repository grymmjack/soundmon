#!/usr/bin/env python3
"""--write-mod: emit a real ProTracker .MOD file. Samples and patterns, not audio.

WHY THIS INSTEAD OF A MODEL

There are no generative models for tracker formats — searched, and the only hits
are audio diffusion experiments that *sound* chiptune rather than emitting module
files. There are structural reasons: a module is a score AND its instruments (two
unrelated generative problems), its effect columns are effectively a small program,
and nobody has even packaged a corpus.

But a model was never needed. A MOD file *is* samples plus pattern data, and
soundmon already produces both — it was just mixing them down to audio instead of
writing them out separately. Same insight as --write-midi: emit the SOURCE, not
only the render.

The payoff is that the output is EDITABLE. Open it in Schism Tracker or OpenMPT,
hear the crispness natively, rewrite a pattern by hand. No rendered file can offer
that.

FORMAT (ProTracker, 31-sample "M.K.")

    0      20 bytes   song title
    20     31 x 30    sample headers: 22-byte name, length in WORDS (BE),
                      finetune, volume 0-64, repeat point + length in words
    950    1 byte     song length in patterns (1-128)
    951    1 byte     127, for historical reasons
    952    128 bytes  pattern order table
    1080   4 bytes    "M.K."
    1084   ...        patterns: 64 rows x 4 channels x 4 bytes
    ...    ...        sample data, signed 8-bit, concatenated in order

Each 4-byte cell packs sample number across two nibbles either side of a 12-bit
Amiga period — a layout that only makes sense once you know the sample number grew
from 4 bits to 5 after the format shipped.
"""
import os
import struct

# Amiga period table, finetune 0. ProTracker itself only offered these three
# octaves, because that is all the Amiga's DMA could tune a sample across.
_PT3 = [
    856, 808, 762, 720, 678, 640, 604, 570, 538, 508, 480, 453,   # MIDI 36-47
    428, 404, 381, 360, 339, 320, 302, 285, 269, 254, 240, 226,   # MIDI 48-59
    214, 202, 190, 180, 170, 160, 151, 143, 135, 127, 120, 113,   # MIDI 60-71
]
# ...but the period field is 12 bits, and every modern replayer (libxmp, OpenMPT,
# Schism) honours the full range. Doubling a period drops an octave, halving it
# raises one, so extending the table is just arithmetic.
#
# THIS MATTERS: the three-octave table forced _period() to FOLD out-of-range
# notes by octave, and a folded note is not a slightly-wrong note — it is a
# different one. 12 of 328 notes in the Zelda title theme sit above MIDI 71, and
# they are the melody's peaks, so the phrase that should rise instead drops an
# octave at its high point.
# The extremes are the ones every replayer agrees on: 1712 is the standard
# extended low note, and 113 // 2 == 56 is exactly the lowest period ProTracker's
# own Amiga limit allows. Going one octave further up would need period 53, which
# replayers in ProTracker-compatible mode clamp — so five octaves is the honest
# range, not an arbitrary stop.
PERIODS = ([p * 2 for p in _PT3[:12]]                 # MIDI 24-35
           + _PT3                                     # MIDI 36-71
           + [p // 2 for p in _PT3[24:]])             # MIDI 72-83
MIDI_BASE = 24
ROWS = 64                    # rows per pattern, fixed by the format
CHANNELS = 4                 # ProTracker: exactly four

# Channel assignment. Four voices is the format's limit and also the 2A03's, so
# the mapping is the same one chip.py already uses.
CH_LEAD, CH_ARP, CH_BASS, CH_DRUM = 0, 1, 2, 3


def _period(pitch):
    """MIDI note -> Amiga period, folded by octaves into the table's range."""
    p = int(pitch)
    while p < MIDI_BASE:
        p += 12
    while p > MIDI_BASE + len(PERIODS) - 1:
        p -= 12
    return PERIODS[p - MIDI_BASE]


# A looping tone sample must be tuned to the period table, and the traditional
# 64-byte single cycle is NOT.
#
#   period 428 plays at 3546894.6 / 428 = 8287.1 Hz
#   64 bytes of it loops at 8287.1 / 64 = 129.49 Hz
#   but period 428 is the table's C-3, and C3 is 130.81 Hz
#
# That is 17.7 cents flat on every tone in the module — a fixed detune against the
# source it was transcribed from, and it is period-independent, so it does not
# average out across the range.
#
# The fix I reached for first was a better RATIO: three cycles in 190 bytes lands
# within 0.5 cents. Measuring the rendered output killed it. The cycle length is
# 63.33 samples, so the three cycles are NOT byte-identical, which makes the
# sample's true period the whole 190 bytes — a component at a THIRD of the
# fundamental. Autocorrelation on the libxmp render locked onto 43.6 Hz for a
# 130.8 Hz note, exactly 1/3, which is that subharmonic being the strongest
# periodicity in the signal.
#
# So: one cycle, and correct the tuning with FINETUNE, which is the field the
# format provides for precisely this. Searching cycle length against the 16
# finetune values (steps of 1/8 semitone) gives 62 bytes at finetune -3, off by
# 0.18 cents — better than the ratio trick AND with a single unambiguous period.
CYCLE_LEN = 62
FINETUNE = -3


def _cycle(kind, np, length=CYCLE_LEN, duty=0.5):
    """One cycle of a waveform in `length` bytes, as signed 8-bit.

    A single cycle with the loop covering the whole sample is how trackers sustain
    a tone: the replayer just repeats it, exactly as a pulse channel would.
    """
    t = np.arange(length) / float(length)
    if kind == "pulse":
        w = np.where(t < duty, 1.0, -1.0)
    elif kind == "triangle":
        w = 2.0 * np.abs(2.0 * t - 1.0) - 1.0
        w = np.round(w * 7.5) / 7.5                # 15 steps, like the 2A03's
    else:
        w = np.sin(2 * np.pi * t)
    return np.clip(np.round(w * 127.0), -128, 127).astype(np.int8)


def _noise(np, n, period, seed=1, decay=6.0):
    """A one-shot LFSR noise burst for percussion. No loop — drums are one-shots."""
    out = np.empty(n)
    reg = (int(seed) & 0x7FFF) or 1
    step = max(1, int(n / max(period, 1.0)))
    val = 1.0
    for i in range(n):
        if i % step == 0:
            fb = (reg ^ (reg >> 1)) & 1
            reg = (reg >> 1) | (fb << 14)
            val = 1.0 if (reg & 1) else -1.0
        out[i] = val
    env = np.exp(-np.arange(n) / float(n) * decay)
    return np.clip(np.round(out * env * 127.0), -128, 127).astype(np.int8)


def build_samples(np):
    """The instrument set.

    Returns [(name, int8 data, loop_start, loop_len, vol, finetune)]. Only the
    looping tones carry finetune — the drums are one-shot noise with no pitch to
    correct, and detuning them would just change their character.
    """
    L, F = CYCLE_LEN, FINETUNE
    return [
        ("lead pulse 25%",  _cycle("pulse", np, L, 0.25),      0, L,  64, F),
        ("arp pulse 50%",   _cycle("pulse", np, L, 0.50),      0, L,  48, F),
        ("bass triangle",   _cycle("triangle", np, L),         0, L,  64, F),
        ("kick",            _noise(np, 1024, 40, 7, 9.0),      0, 0,  64, 0),
        ("snare",           _noise(np, 1024, 220, 3, 7.0),     0, 0,  52, 0),
        ("hat",             _noise(np, 512, 400, 11, 16.0),    0, 0,  36, 0),
    ]


def _cell(sample, period, effect=0, param=0):
    """Pack one 4-byte pattern cell.

    The sample number is split across two nibbles either side of the 12-bit
    period — an artifact of the field growing from 4 bits to 5 after the format
    shipped, and the single easiest thing to get wrong here.
    """
    s = int(sample) & 0x1F
    p = int(period) & 0xFFF
    b0 = ((s >> 4) << 4) | ((p >> 8) & 0x0F)
    b1 = p & 0xFF
    b2 = ((s & 0x0F) << 4) | (int(effect) & 0x0F)
    b3 = int(param) & 0xFF
    return bytes((b0, b1, b2, b3))


EMPTY = _cell(0, 0)


def timing_for(spb, rows_per_bar):
    """Pick (ticks_per_row, bpm) so one row lasts exactly spb / rows_per_bar.

    A MOD's row duration is `ticks_per_row * 2.5 / bpm` seconds. Effect F sets
    ticks-per-row when its parameter is below 32 and BPM when it is 32 or above,
    so BOTH are writable — and both must be, because a single BPM cannot express
    a fine row grid: 96 rows per bar at 75 bpm needs BPM 450 at the default 6
    ticks, well past the 255 the byte holds.

    Prefer 6 ticks per row (ProTracker's default, and enough ticks for arpeggio
    and vibrato to have somewhere to move) and drop only as far as needed to keep
    BPM in range.
    """
    row_s = float(spb) / max(1, int(rows_per_bar))
    for ticks in (6, 5, 4, 3, 2, 1):
        bpm = ticks * 2.5 / row_s
        if 32 <= bpm <= 255:
            return ticks, int(round(bpm))
    # Outside anything expressible: clamp and accept the tempo error.
    ticks = 6
    return ticks, max(32, min(255, int(round(ticks * 2.5 / row_s))))


def write_mod(path, ev, bars, spb, steps, np, title="soundmon", bpm=None,
              chippy="off", rows_per_bar=None):
    """Write composed events as a ProTracker MOD. Returns the path.

    `ev` is chip.compose()'s grid-based event dict — which is exactly the shape a
    tracker wants, since patterns ARE a grid. This is the one output where
    soundmon's quantized composition is not a compromise but the native form.

    Tempo is DERIVED from `spb` (seconds per bar), which is the composition's own
    tempo. It used to be passed in from --bpm, which for --from-midi is the
    unrelated CLI default: a 75 bpm source was written as 120 and played 1.6x too
    fast. The tempo of the notes and the tempo in the header cannot be two
    different numbers.
    """
    samples = build_samples(np)
    # A bar is allowed to straddle a pattern boundary. Patterns are a storage
    # unit, not a musical one — the order list plays them back to back — so
    # clamping rows-per-bar to 64 was capping the time resolution for no reason.
    rows_per_bar = max(1, int(rows_per_bar or min(steps, ROWS)))
    ticks, mod_bpm = timing_for(spb, rows_per_bar)
    n_patterns = min(128, max(1, -(-(bars * rows_per_bar) // ROWS)))

    # grid[pattern][row][channel] -> cell
    grid = [[[EMPTY] * CHANNELS for _ in range(ROWS)] for _ in range(n_patterns)]

    def place(bar, step, ch, sample, pitch, effect=0, param=0):
        if step >= rows_per_bar:
            return
        abs_row = bar * rows_per_bar + step
        pat, row = divmod(abs_row, ROWS)
        if 0 <= pat < n_patterns:
            grid[pat][row][ch] = _cell(sample, _period(pitch), effect, param)

    def place_off(bar, step, ch):
        """MOD has NO note-off event. A note ends when you set its channel volume
        to zero (effect C, parameter 0) or retrigger it. The tone samples here loop
        forever by design — that is how a tracker sustains — so without this every
        note rings until the next one on that channel, which is what "no noteoffs,
        something is missing" sounds like."""
        if step >= rows_per_bar:
            return
        abs_row = bar * rows_per_bar + step
        pat, row = divmod(abs_row, ROWS)
        if 0 <= pat < n_patterns and grid[pat][row][ch] == EMPTY:
            grid[pat][row][ch] = _cell(0, 0, 0xC, 0)

    # Every pitch sounding at each grid position, so an arpeggio can use the real
    # chord instead of assuming a major triad.
    chords = {}
    for vname in ("lead", "arp", "bass"):
        for it in ev.get(vname, []):
            chords.setdefault((it[0], it[1]), set()).add(it[3])

    def place_voice(items, ch, sample):
        """Lay one voice down a channel, notes and releases together.

        This has to see the WHOLE voice at once, not one note at a time. A MOD
        channel is monophonic, so a note's release row is bounded by the next
        note's onset — and the source's note lengths often overlap, so placing a
        release blindly at onset+duration dropped a C00 into the middle of the
        FOLLOWING note and cut it off. Per-note placement cannot know that; it has
        no view of what comes next.
        """
        seq = sorted(items, key=lambda it: it[0] * rows_per_bar + it[1])
        prev_pitch = None
        legato = False               # did the previous note run INTO this one?
        for i, it in enumerate(seq):
            bar, step, dur, pitch = it[0], it[1], it[2], it[3]
            start = bar * rows_per_bar + step
            nxt = (seq[i + 1][0] * rows_per_bar + seq[i + 1][1]
                   if i + 1 < len(seq) else None)
            # Effects in the era's idiom, matching what --chippy does in
            # synthesis. Deterministic on the row so a regenerated file is
            # byte-identical.
            frac = ((bar * 131 + step * 17 + ch * 7) % 100) / 100.0
            e, p = chip_effects(frac, pitch, prev_pitch, int(dur),
                                is_chord_top=(ch == CH_ARP), chippy=chippy,
                                legato=legato, ticks_per_row=ticks,
                                chord=chords.get((bar, step), ()))
            prev_pitch = pitch
            place(bar, step, ch, sample, pitch, e, p)
            # Release one row past the last sounding row, clamped to the next
            # onset. Landing exactly ON the next onset needs no release at all:
            # retriggering a sample ends the previous note by itself — and that
            # unbroken hand-off is also the only place a glide can work, so the
            # next iteration is told about it.
            end = start + max(1, int(dur))
            legato = nxt is not None and end >= nxt
            if legato:
                continue
            eb, es = divmod(end, rows_per_bar)
            place_off(eb, es, ch)

    place_voice(ev.get("lead", []), CH_LEAD, 1)
    place_voice(ev.get("arp", []), CH_ARP, 2)
    place_voice(ev.get("bass", []), CH_BASS, 3)
    for bar, step, kind in ev.get("drum", []):
        smp = {"k": 4, "s": 5, "h": 6}.get(kind)
        if smp:
            # Drums are one-shots at a fixed pitch; C-2 is the tuned reference.
            place(bar, step, CH_DRUM, smp, 48)

    # Tempo, on row 0. Effect F is overloaded: parameter < 32 sets ticks per row,
    # >= 32 sets BPM. Both are needed, so they go on two different channels of the
    # same row — one cell holds one effect. A cell can carry a note AND an effect,
    # so merge rather than overwrite: clobbering row 0 would drop the downbeat.
    row0 = list(grid[0][0])
    # Two DISTINCT channels. dict.fromkeys dedupes while keeping order: the free
    # channels are already in the fallback list, so a plain concatenation aimed
    # both commands at the same cell and the BPM silently ate the speed.
    pref = ([c for c in range(CHANNELS) if row0[c] == EMPTY]
            + [CH_DRUM, CH_BASS, CH_ARP, CH_LEAD])
    slots = list(dict.fromkeys(pref))[:2]
    for c, param in zip(slots, (ticks & 0x1F, mod_bpm)):
        b0, b1, b2, _b3 = row0[c]
        row0[c] = bytes((b0, b1, (b2 & 0xF0) | 0xF, param))
    grid[0][0] = row0

    # Trim the tail. Patterns are 64 rows whether or not the music fills them, so
    # a piece that ends mid-pattern leaves silence before the order list wraps —
    # dead air in the middle of what should be a seamless game loop. D00 breaks
    # out of the pattern at the end of its row; on the last pattern that wraps to
    # the song start, which is exactly the loop point.
    last = None
    for pi in range(n_patterns - 1, -1, -1):
        for r in range(ROWS - 1, -1, -1):
            if any(c != EMPTY for c in grid[pi][r]):
                last = (pi, r)
                break
        if last:
            break
    if last and last != (n_patterns - 1, ROWS - 1):
        pi, r = last
        row = list(grid[pi][r])
        # Prefer a cell with no effect of its own, so a break never displaces an
        # arpeggio or a release.
        cand = [c for c in range(CHANNELS) if (row[c][2] & 0x0F) == 0 and row[c][3] == 0]
        c = cand[0] if cand else CH_DRUM
        b0, b1, b2, _b3 = row[c]
        row[c] = bytes((b0, b1, (b2 & 0xF0) | 0xD, 0))
        grid[pi][r] = row
        n_patterns = pi + 1
        grid = grid[:n_patterns]

    out = bytearray()
    out += title.encode("ascii", "replace")[:20].ljust(20, b"\0")
    for i in range(31):
        if i < len(samples):
            name, data, lstart, llen, vol, ft = samples[i]
            words = len(data) // 2
            out += name.encode("ascii", "replace")[:22].ljust(22, b"\0")
            out += struct.pack(">H", words)
            # Finetune is a 4-bit SIGNED value in 1/8-semitone steps, stored as
            # two's complement in the low nibble. Writing 0 here left every tone
            # 17.7 cents flat; the field is not decoration.
            out += bytes((int(ft) & 0x0F, int(vol) & 0x7F))
            # A repeat length of 0 or 1 word means "no loop" to a replayer.
            out += struct.pack(">HH", lstart // 2, (llen // 2) if llen >= 2 else 0)
        else:
            out += b"\0" * 22 + struct.pack(">H", 0) + bytes((0, 0)) + struct.pack(">HH", 0, 0)
    out += bytes((n_patterns, 127))
    order = bytes(range(n_patterns)) + b"\0" * (128 - n_patterns)
    out += order
    out += b"M.K."

    for pat in grid:
        for row in pat:
            for cell in row:
                out += cell
    for name, data, *_ in samples:
        out += data.tobytes()

    with open(path, "wb") as fh:
        fh.write(bytes(out))
    return path


# =============================================================================
# THE FULL PROTRACKER EFFECT SET
#
# All sixteen commands. The effect column is where a MOD stops being a note list
# and becomes a small program — arpeggio, slides and vibrato are how four channels
# were made to sound like more, and they are the same techniques --chippy applies
# in the synthesis path. Emitting them means the tracker file carries the idiom
# itself rather than a rendering of it.
#
#   0xy  arpeggio: cycle note, note+x, note+y semitones every tick
#   1xx  portamento up      2xx  portamento down
#   3xx  tone portamento (slide toward the new note)
#   4xy  vibrato: x = speed, y = depth
#   5xy  tone portamento + volume slide      6xy  vibrato + volume slide
#   7xy  tremolo
#   8xx  set panning (not in ProTracker; widely honoured)
#   9xx  sample offset, in 256-byte units
#   Axy  volume slide: x up, y down
#   Bxx  position jump      Cxx  set volume (00 = the note-off this format lacks)
#   Dxx  pattern break
#   Exy  extended: E1 fine slide up, E2 fine down, E9 retrigger, EA/EB fine
#        volume slide, EC note cut, ED note delay, EE pattern delay
#   Fxx  set speed (<32 = ticks per row) or BPM (>=32)
# =============================================================================
FX = {
    "arpeggio": 0x0, "porta_up": 0x1, "porta_down": 0x2, "tone_porta": 0x3,
    "vibrato": 0x4, "porta_vol": 0x5, "vib_vol": 0x6, "tremolo": 0x7,
    "pan": 0x8, "offset": 0x9, "vol_slide": 0xA, "jump": 0xB,
    "volume": 0xC, "break": 0xD, "extended": 0xE, "speed": 0xF,
}
# Extended (Exy) sub-commands, in the high nibble of the parameter.
EXT = {
    "filter": 0x0, "fine_up": 0x1, "fine_down": 0x2, "glissando": 0x3,
    "vib_wave": 0x4, "finetune": 0x5, "loop_start": 0x6, "loop_end": 0x7,
    "trem_wave": 0x8, "retrigger": 0x9, "fine_vol_up": 0xA,
    "fine_vol_down": 0xB, "note_cut": 0xC, "note_delay": 0xD,
    "pattern_delay": 0xE, "invert_loop": 0xF,
}


def fx(name, x=0, y=None):
    """Build (effect, param) for an effect by name.

    Two-nibble effects take x and y; single-byte ones take x as the whole value.
    `fx("vibrato", 4, 3)` -> vibrato speed 4 depth 3; `fx("volume", 0)` -> C00.
    """
    e = FX.get(name)
    if e is None:
        raise ValueError(f"unknown MOD effect {name!r}")
    param = ((x & 0x0F) << 4) | (y & 0x0F) if y is not None else (x & 0xFF)
    return e, param


def ext(name, value=0):
    """Build (effect, param) for an Exy extended command."""
    sub = EXT.get(name)
    if sub is None:
        raise ValueError(f"unknown extended command {name!r}")
    return FX["extended"], ((sub & 0x0F) << 4) | (value & 0x0F)


def porta_rate(pitch, prev_pitch, ticks_per_row, rows=1):
    """Tone-portamento parameter to glide from prev_pitch to pitch in `rows` rows.

    THE UNITS ARE THE WHOLE PROBLEM. 3xx slides the channel's current period
    toward the target by xx PERIOD UNITS PER TICK — it is not a semitone rate and
    not a duration. Passing something semitone-shaped (I used 2 + interval) makes
    an octave leap, which is 214 period units, slide at 14 units per tick: at 3
    ticks per row that needs five rows to arrive, and the next note lands long
    before then. The pitch never reaches its target, so every slide just smears
    off somewhere wrong.

    Period distance also depends on register: the same interval is far more period
    units low than high, because period is inversely proportional to frequency. So
    the rate has to be computed from the actual endpoints, every time.
    """
    d = abs(_period(pitch) - _period(prev_pitch))
    ticks = max(1, int(ticks_per_row) * max(1, int(rows)))
    return max(1, min(0xFF, -(-d // ticks)))          # ceil, so it does arrive


def chip_effects(ident_frac, pitch, prev_pitch, dur, is_chord_top=False,
                 chippy="off", legato=False, ticks_per_row=6, chord=()):
    """Choose an effect for a note, in the era's idiom.

    Mirrors what --chippy does in the synthesis path, so a MOD carries the same
    character:

        a slurred step gets tone portamento — how the era connected intervals
        a sustained note gets vibrato       — cheap, and it hides a static wave
        a chord top gets arpeggio           — a chord on one channel

    Returns (effect, param) or (0, 0) for none.
    """
    if chippy == "off":
        return 0, 0
    strength = {"some": 0.35, "lots": 0.6, "max": 0.9}.get(chippy, 0.0)

    if is_chord_top and ident_frac < strength:
        # 0xy cycles note, note+x, note+y every tick. The intervals come from the
        # ACTUAL chord when we know it — a hard-coded 4,7 major triad over the
        # minor harmony this piece is in is not an ornament, it is a wrong note.
        iv = [i for i in sorted({(int(p) - int(pitch)) % 12 for p in chord})
              if 1 <= i <= 15]
        x, y = (iv + [4, 7])[:2] if iv else (4, 7)
        return fx("arpeggio", x, y)

    if prev_pitch is not None and legato:
        # ONLY when the previous note ran straight into this one. Tone portamento
        # does not retrigger the sample, which is what makes it a glide — but it
        # also means the channel volume is whatever it already was. After a C00
        # release that is zero, so a slide there is perfectly silent.
        step = abs(int(pitch) - int(prev_pitch))
        if 1 <= step <= 7 and ident_frac < strength:
            return fx("tone_porta", porta_rate(pitch, prev_pitch,
                                               ticks_per_row))

    if dur >= 4 and ident_frac < strength * 0.7:
        return fx("vibrato", 4, 3)
    return 0, 0
