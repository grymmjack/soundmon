#!/usr/bin/env python3
"""--write-rad: emit a Reality AdLib Tracker (.RAD) file. OPL instruments + patterns.

WHY THIS IS THE BEST POSSIBLE OUTPUT FOR THE OPL SIDE

QB64 plays .RAD natively, so this drops straight into the game with no decoder and
no audio file. And unlike a render, it is EDITABLE: open it in Reality AdLib
Tracker, retune an operator, rewrite a pattern.

It is also the format that matches what we already have. A RAD instrument is a set
of OPL operator registers — which is exactly what the DMXOPL patches are — and a
RAD pattern is a grid of notes, which is exactly what chip.compose() produces. No
conversion loss in either direction.

FORMAT (RAD 2.1, per the Reality AdLib Tracker technical specification)

    0x00  16 bytes   "RAD by REALiTY!!"
    0x10  1 byte     version, BCD: 0x21
    0x11  1 byte     bit6 slow timer | bit5 BPM present | bits4-0 initial speed
    +     2 bytes    BPM, little-endian, only if bit5 set
    +     ...        description, 0x00-terminated (0x01 = newline,
                     0x02-0x1F = that many spaces)
    +     ...        instruments: {id, name-len, name, data} until id == 0x00
    +     1 byte     order list length, then that many entries
    +     ...        patterns: {id, size LE16, packed lines} until id == 0xFF
    +     ...        riffs, same shape, until id == 0xFF

The operator bytes are OPL registers with two fields INVERTED — RAD stores volume
where the chip stores attenuation, for total level and sustain level both. Getting
that backwards produces a file that loads and plays silently, so it is the first
thing to check if a RAD sounds wrong.
"""
import os
import struct

MAGIC = b"RAD by REALiTY!!"
VERSION = 0x21
CHANNELS = 9                 # OPL2 melodic channels
CH_LEAD, CH_ARP, CH_ARP2, CH_BASS = 0, 1, 2, 3


def _note_octave(pitch):
    """MIDI note -> (RAD note 1-12, octave 0-7).

    RAD numbers notes 1..12 as C#..C, so C sits at the TOP of a block rather than
    the bottom. Shifting by one semitone before dividing puts C# at index 0, which
    makes the octave arithmetic fall out correctly instead of being off by one for
    every C.
    """
    m = int(pitch) - 1
    note = (m % 12) + 1
    octv = (m // 12) - 1
    while octv < 0:
        octv += 1
    return note, min(7, octv)


def _op_bytes(op):
    """One WOPL operator -> five RAD operator bytes.

    WOPL op tuple is (AVEKM, KSL|TL, AR|DR, SL|RR, waveform) — raw OPL registers.
    RAD keeps the same bit layout but stores VOLUME where OPL stores ATTENUATION,
    so total level and sustain level are both inverted.
    """
    avekm, ksltl, ardr, slrr, ws = (int(x) & 0xFF for x in op)
    ksl = ksltl & 0xC0
    tl = ksltl & 0x3F
    b1 = ksl | ((63 - tl) & 0x3F)               # attenuation -> volume
    sl = (slrr >> 4) & 0x0F
    rr = slrr & 0x0F
    b3 = (((15 - sl) & 0x0F) << 4) | rr         # sustain attenuation -> level
    return bytes((avekm, b1, ardr, b3, ws & 0x07))


def instrument_from_wopl(ins):
    """A DMXOPL patch -> a RAD FM instrument body (24 bytes)."""
    con = ins["fbc1"] & 0x01
    fb = (ins["fbc1"] >> 1) & 0x07
    # Algorithm 0 is 2-op FM (modulator into carrier), 1 is 2-op AM (both audible).
    alg = 1 if con else 0
    out = bytearray()
    out.append(alg & 0x07)                      # no riff, no panning bits set
    out.append(fb & 0x0F)                       # feedback for ops 1-2
    out.append(0x00)                            # detune / riff speed
    out.append(0x3F)                            # instrument volume, max
    # RAD op1 is the MODULATOR and op2 the CARRIER; WOPL stores carrier first.
    out += _op_bytes(ins["ops"][1])
    out += _op_bytes(ins["ops"][0])
    out += b"\0" * 10                           # ops 3-4 unused by algorithm 0/1
    return bytes(out)


def _fallback_instrument(kind="lead"):
    """A hand-authored instrument, for when no WOPL bank is available."""
    presets = {
        # (AVEKM, KSL|TL, AR|DR, SL|RR, WS) modulator then carrier
        "lead": ((0x01, 0x8F, 0xF2, 0x53, 0), (0x01, 0x00, 0xF4, 0x44, 0)),
        "bass": ((0x01, 0x93, 0xE7, 0x28, 0), (0x01, 0x00, 0xE6, 0x38, 0)),
        "perc": ((0x0E, 0x80, 0xF8, 0xF8, 0), (0x01, 0x00, 0xF8, 0xF8, 0)),
    }
    mod, car = presets.get(kind, presets["lead"])
    out = bytearray((0x00, 0x06, 0x00, 0x3F))
    out += _op_bytes(mod)
    out += _op_bytes(car)
    out += b"\0" * 10
    return bytes(out)


def _pack_line(line_no, is_last, chans):
    """Pack one pattern line. `chans` is [(ch, note, octave, inst, fx, param)]."""
    out = bytearray()
    out.append((0x80 if is_last else 0) | (line_no & 0x7F))
    if not chans:
        # An empty line still needs a channel byte with the last-channel flag, or
        # the reader keeps consuming.
        out.append(0x80 | 0x0F)
        return bytes(out)
    for i, (ch, note, octv, inst, fx, param) in enumerate(chans):
        last = (i == len(chans) - 1)
        hdr = (0x80 if last else 0) | (ch & 0x0F)
        if note is not None:
            hdr |= 0x40
        if inst is not None:
            hdr |= 0x20
        if fx is not None:
            hdr |= 0x10
        out.append(hdr)
        if note is not None:
            out.append(((octv & 0x07) << 4) | (note & 0x0F))
        if inst is not None:
            out.append(inst & 0x7F)
        if fx is not None:
            out.append(fx & 0x1F)
            out.append(param & 0x7F)
    return bytes(out)


def write_rad(path, ev, bars, spb, steps, np, title="soundmon", bpm=125,
              wopl=None, progs=None, rows=64):
    """Write composed events as a RAD file. Returns the path.

    `ev` is chip.compose()'s grid event dict; `progs` optionally maps voice name
    to a GM program so instruments come from the DMXOPL bank.
    """
    rows = max(1, min(int(rows), 64))
    rows_per_bar = max(1, min(steps, rows))
    bars_per_pattern = max(1, rows // rows_per_bar)
    n_patterns = max(1, min(100, -(-bars // bars_per_pattern)))

    # --- instruments --------------------------------------------------------
    # One per voice, taken from the bank when we have it. RAD instrument numbers
    # start at 1; 0 means "end of list" in the file.
    voices = [("lead", 1), ("arp", 2), ("bass", 3),
              ("kick", 4), ("snare", 5), ("hat", 6)]
    progs = progs or {}
    bodies = {}
    names = {}
    for vname, num in voices:
        ins = None
        if wopl and vname in progs:
            p = progs[vname]
            if 0 <= p < len(wopl.get("melodic", [])):
                ins = wopl["melodic"][p]
        if ins is not None:
            bodies[num] = instrument_from_wopl(ins)
            names[num] = (ins["name"] or vname)[:22]
        else:
            kind = "perc" if vname in ("kick", "snare", "hat") else (
                   "bass" if vname == "bass" else "lead")
            bodies[num] = _fallback_instrument(kind)
            names[num] = vname

    # --- patterns ----------------------------------------------------------
    # grid[pattern][row] -> {channel: (note, octave, inst)}
    grid = [[{} for _ in range(rows)] for _ in range(n_patterns)]

    def place(bar, step, ch, inst, pitch):
        if step >= rows_per_bar:
            return
        abs_row = bar * rows_per_bar + step
        pat, row = divmod(abs_row, rows)
        if 0 <= pat < n_patterns:
            n, o = _note_octave(pitch)
            grid[pat][row][ch] = (n, o, inst)

    for it in ev.get("lead", []):
        place(it[0], it[1], CH_LEAD, 1, it[3])
    for it in ev.get("arp", []):
        place(it[0], it[1], CH_ARP, 2, it[3])
    for it in ev.get("bass", []):
        place(it[0], it[1], CH_BASS, 3, it[3])
    for bar, step, kind in ev.get("drum", []):
        inst = {"k": 4, "s": 5, "h": 6}.get(kind)
        if inst:
            # Percussion on its own channels, at a fixed pitch — the instrument
            # envelope is the sound, not the note.
            ch = {"k": 6, "s": 7, "h": 8}[kind]
            place(bar, step, ch, inst, 48)

    out = bytearray()
    out += MAGIC
    out.append(VERSION)
    speed = 6
    out.append(0x20 | (speed & 0x1F))            # bit5: a BPM value follows
    out += struct.pack("<H", int(max(1, min(65535, round(bpm)))))
    desc = title.encode("ascii", "replace")[:60]
    out += desc + b"\0"

    for _vname, num in voices:
        out.append(num)
        nm = names[num].encode("ascii", "replace")[:22]
        out.append(len(nm))
        out += nm
        out += bodies[num]
    out.append(0x00)                             # end of instruments

    out.append(n_patterns & 0x7F)
    out += bytes(range(n_patterns))

    for pi in range(n_patterns):
        body = bytearray()
        used = [r for r in range(rows) if grid[pi][r]]
        if not used:
            used = [rows - 1]
        for idx, r in enumerate(used):
            chans = []
            for ch in sorted(grid[pi][r]):
                n, o, inst = grid[pi][r][ch]
                chans.append((ch, n, o, inst, None, None))
            body += _pack_line(r, idx == len(used) - 1, chans)
        out.append(pi & 0x7F)
        out += struct.pack("<H", len(body))
        out += body
    out.append(0xFF)                             # end of patterns
    out.append(0xFF)                             # end of riffs

    with open(path, "wb") as fh:
        fh.write(bytes(out))
    return path
