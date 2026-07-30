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
LINES = 64                   # lines per pattern, fixed: "each pattern is made up
                             # of 64 lines"
MAX_PATTERNS = 100           # "RAD allows up to 100 separate patterns"
KEYOFF = 15                  # "1..12 which indicate one of the 12 notes from
                             # C#(1) to C(12), or 15 for a key-off"


def timing_for(spb, lines_per_bar):
    """Pick (speed, bpm) so one pattern line lasts exactly spb / lines_per_bar.

    The spec says "RAD uses a 50Hz timer which is delayed by the speed value
    between each line", and that BPM defaults to 125. Those two facts pin the
    relation: 50 Hz IS the tick rate at 125 BPM, so the rate is BPM / 2.5 and

        line seconds = speed * 2.5 / BPM

    — the same relation ProTracker uses, which is unsurprising given the shared
    lineage. So this mirrors mod.timing_for rather than inventing a second scheme.
    """
    line_s = float(spb) / max(1, int(lines_per_bar))
    for speed in (6, 5, 4, 3, 2, 1):
        bpm = speed * 2.5 / line_s
        if 32 <= bpm <= 255:
            return speed, int(round(bpm))
    speed = 6
    return speed, max(32, min(255, int(round(speed * 2.5 / line_s))))


def _note_octave(pitch):
    """MIDI note -> (RAD note 1-12, octave 0-7).

    RAD numbers notes "1..12 which indicate one of the 12 notes from C#(1) to
    C(12)", so C is numbered LAST even though it is lowest in the octave. That
    numbering quirk is about the note field only — the octave field is still the
    plain octave number, because a player has to combine them as an OPL block plus
    an F-number from a twelve-entry table indexed by note % 12, and note 12 wraps
    to index 0, i.e. C.

    I originally shifted the pitch down a semitone before dividing, reasoning that
    C sat at the "top" of a block. That is wrong, and wrong for exactly one note:
    every C landed an octave low while its neighbours were right, so C4 sounded
    below the D4 next to it. Chroma 0 keeps its own octave; only its NUMBER is 12.
    """
    p = int(pitch)
    chroma = p % 12
    note = 12 if chroma == 0 else chroma
    octv = (p // 12) - 1
    return note, max(0, min(7, octv))


def _op_bytes(op):
    """One WOPL operator -> five RAD operator bytes.

    WOPL op tuple is (AVEKM, KSL|TL, AR|DR, SL|RR, waveform) — raw OPL registers.
    RAD keeps the same bit layout but stores VOLUME where OPL stores ATTENUATION,
    so total level and sustain level are both inverted.
    """
    # PASS THE OPL REGISTERS THROUGH UNCHANGED.
    #
    # The spec annotates these fields "(inverted)", which describes their SEMANTICS
    # — higher means quieter, i.e. inverted relative to volume — not that RAD stores
    # them flipped from the chip. RAD is an OPL tracker; it holds raw register
    # values.
    #
    # I read it the other way first and wrote `63 - TL`, which turned every
    # loudest-possible patch (TL=0) into 63 = maximum attenuation. The file loaded
    # and played essentially silently, which is exactly the failure this format
    # invites: nothing errors, you just hear nothing.
    avekm, ksltl, ardr, slrr, ws = (int(x) & 0xFF for x in op)
    return bytes((avekm, ksltl, ardr, slrr, ws & 0x07))


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
    out.append(0x3F)                            # instrument volume: 6-bit, max
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


def write_rad(path, ev, bars, spb, steps, np, title="soundmon", bpm=None,
              wopl=None, progs=None, rows=None, lines_per_bar=None):
    """Write composed events as a RAD file. Returns the path.

    `ev` is chip.compose()'s grid event dict; `progs` optionally maps voice name
    to a GM program so instruments come from the DMXOPL bank.

    Tempo is DERIVED from `spb`, exactly as in mod.py and for the same reason: it
    used to come from --bpm, which for --from-midi is an unrelated CLI default, so
    a 75 bpm source was written as 120 and played 1.6x too fast.
    """
    # A bar may straddle a pattern boundary — patterns are storage, the order list
    # plays them back to back — so the line grid is not capped at 64.
    lines_per_bar = max(1, int(lines_per_bar or rows or steps))
    speed, rad_bpm = timing_for(spb, lines_per_bar)
    n_patterns = max(1, min(MAX_PATTERNS,
                            -(-(bars * lines_per_bar) // LINES)))
    rows = LINES

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
    # grid[pattern][line] -> {channel: (note, octave, inst)}
    grid = [[{} for _ in range(rows)] for _ in range(n_patterns)]

    def place(bar, step, ch, inst, pitch):
        abs_row = bar * lines_per_bar + step
        pat, row = divmod(abs_row, rows)
        if 0 <= pat < n_patterns:
            n, o = _note_octave(pitch)
            grid[pat][row][ch] = (n, o, inst)

    def place_off(bar, step, ch):
        """Emit a key-off. RAD DOES have one — the spec lists note value 15
        alongside the twelve pitches — and without it an OPL voice is never
        released, so it sustains through the rest of the tune. Patches with a long
        release turn the whole channel into a drone."""
        abs_row = bar * lines_per_bar + step
        pat, row = divmod(abs_row, rows)
        if 0 <= pat < n_patterns and ch not in grid[pat][row]:
            grid[pat][row][ch] = (KEYOFF, 0, None)

    def place_voice(items, ch, inst):
        """Notes and releases down one channel. An OPL channel is monophonic, so a
        release is bounded by the next onset — same constraint as a MOD channel,
        and the same bug if ignored: a release placed at onset+duration lands
        inside the FOLLOWING note and cuts it off."""
        seq = sorted(items, key=lambda it: it[0] * lines_per_bar + it[1])
        for i, it in enumerate(seq):
            bar, step, dur, pitch = it[0], it[1], it[2], it[3]
            start = bar * lines_per_bar + step
            nxt = (seq[i + 1][0] * lines_per_bar + seq[i + 1][1]
                   if i + 1 < len(seq) else None)
            place(bar, step, ch, inst, pitch)
            end = start + max(1, int(dur))
            if nxt is not None:
                if end >= nxt:
                    continue          # runs into the next note; it retriggers
                end = min(end, nxt)
            place_off(*divmod(end, lines_per_bar), ch)

    place_voice(ev.get("lead", []), CH_LEAD, 1)
    place_voice(ev.get("arp", []), CH_ARP, 2)
    place_voice(ev.get("bass", []), CH_BASS, 3)
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
    out.append(0x20 | (speed & 0x1F))            # bit5: a BPM value follows
    out += struct.pack("<H", int(max(1, min(65535, rad_bpm))))
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
