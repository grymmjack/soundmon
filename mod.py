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

# Amiga period table, finetune 0, three octaves. Index 0 is MIDI 36 (C2), so
# index 12 is period 428 — the reference pitch a MOD sample is tuned to.
PERIODS = [
    856, 808, 762, 720, 678, 640, 604, 570, 538, 508, 480, 453,   # MIDI 36-47
    428, 404, 381, 360, 339, 320, 302, 285, 269, 254, 240, 226,   # MIDI 48-59
    214, 202, 190, 180, 170, 160, 151, 143, 135, 127, 120, 113,   # MIDI 60-71
]
MIDI_BASE = 36
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


def _cycle(kind, np, length=64, duty=0.5):
    """One looping cycle of a waveform, as signed 8-bit.

    A single cycle with the loop covering the whole sample is how trackers
    sustain a tone: the replayer just repeats it, exactly as a pulse channel
    would. 64 bytes is the traditional size — long enough to be smooth, short
    enough that the loop is sample-accurate.
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
    """The instrument set. Returns [(name, int8 data, loop_start, loop_len, vol)]."""
    return [
        ("lead pulse 25%",  _cycle("pulse", np, 64, 0.25),     0, 64, 64),
        ("arp pulse 50%",   _cycle("pulse", np, 64, 0.50),     0, 64, 48),
        ("bass triangle",   _cycle("triangle", np, 64),        0, 64, 64),
        ("kick",            _noise(np, 1024, 40, 7, 9.0),      0, 0,  64),
        ("snare",           _noise(np, 1024, 220, 3, 7.0),     0, 0,  52),
        ("hat",             _noise(np, 512, 400, 11, 16.0),    0, 0,  36),
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


def write_mod(path, ev, bars, spb, steps, np, title="soundmon", bpm=125):
    """Write composed events as a ProTracker MOD. Returns the path.

    `ev` is chip.compose()'s grid-based event dict — which is exactly the shape a
    tracker wants, since patterns ARE a grid. This is the one output where
    soundmon's quantized composition is not a compromise but the native form.
    """
    samples = build_samples(np)
    rows_per_bar = max(1, min(steps, ROWS))
    bars_per_pattern = max(1, ROWS // rows_per_bar)
    n_patterns = max(1, -(-bars // bars_per_pattern))
    n_patterns = min(n_patterns, 128)

    # grid[pattern][row][channel] -> cell
    grid = [[[EMPTY] * CHANNELS for _ in range(ROWS)] for _ in range(n_patterns)]

    def place(bar, step, ch, sample, pitch, effect=0, param=0):
        if step >= rows_per_bar:
            return
        abs_row = bar * rows_per_bar + step
        pat, row = divmod(abs_row, ROWS)
        if 0 <= pat < n_patterns:
            grid[pat][row][ch] = _cell(sample, _period(pitch), effect, param)

    for it in ev.get("lead", []):
        place(it[0], it[1], CH_LEAD, 1, it[3])
    for it in ev.get("arp", []):
        place(it[0], it[1], CH_ARP, 2, it[3])
    for it in ev.get("bass", []):
        place(it[0], it[1], CH_BASS, 3, it[3])
    for bar, step, kind in ev.get("drum", []):
        smp = {"k": 4, "s": 5, "h": 6}.get(kind)
        if smp:
            # Drums are one-shots at a fixed pitch; C-2 is the tuned reference.
            place(bar, step, CH_DRUM, smp, 48)

    # Tempo: effect F with a parameter >= 32 sets BPM directly. Put it on row 0.
    b = int(max(32, min(255, round(bpm))))
    first = grid[0][0]
    grid[0][0] = [first[0], first[1], first[2], _cell(0, 0, 0xF, b)]

    out = bytearray()
    out += title.encode("ascii", "replace")[:20].ljust(20, b"\0")
    for i in range(31):
        if i < len(samples):
            name, data, lstart, llen, vol = samples[i]
            words = len(data) // 2
            out += name.encode("ascii", "replace")[:22].ljust(22, b"\0")
            out += struct.pack(">H", words)
            out += bytes((0, int(vol) & 0x7F))
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
