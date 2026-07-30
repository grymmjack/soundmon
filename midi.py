#!/usr/bin/env python3
"""Play a Standard MIDI File on a chip. The best possible input for --chip/--opl.

WHY THIS BEATS AUDIO TRANSCRIBING

`--from-audio` has to *infer* everything from a spectrum: pitch (often landing on
a harmonic instead of the fundamental), tempo (autocorrelation, biased), note
boundaries, which sound is a kick. Every stage adds error.

A MIDI file already contains the answers, exactly:

    pitch            integer note numbers, no octave errors possible
    timing           note-on/note-off ticks, so durations are real
    tempo            explicit, in a meta event, including tempo CHANGES
    time signature   explicit
    instrumentation  channels already separate melody, bass and percussion
    velocity         per-note dynamics

And playing game MIDI through an OPL3 is not a hack — it is *literally* what
AdLib-era DOS games did. This is the historically correct signal path.

VOICE ASSIGNMENT

A chip has four voices; a MIDI file has sixteen channels and arbitrary polyphony,
so something must be discarded. Assignment is by REGISTER rather than by channel
number, because channel numbering is a convention that arrangers ignore
constantly:

    highest sounding note   -> lead      (the melody, where the tune lives)
    lowest sounding note    -> bass      (triangle / FM bass)
    a middle voice          -> arp       (harmony filler)
    GM channel 10           -> drums     (mapped to kick / snare / hat)

Taking the extremes is deliberate: melody and bass are what make a piece
recognisable, and inner voices are the least missed when you only have four.

NO DEPENDENCIES. SMF is a simple chunked binary format; parsing it is ~120 lines
and avoids adding a package for something this small.
"""
import os
import struct
import sys

# General MIDI percussion -> our three drum voices. GM channel 10 is fixed by the
# standard, which is why drums need no guessing at all.
GM_DRUM = {
    35: "k", 36: "k", 41: "k", 43: "k", 45: "k", 47: "k",     # kicks & low toms
    38: "s", 40: "s", 37: "s", 39: "s", 48: "s", 50: "s",     # snares & high toms
    42: "h", 44: "h", 46: "h", 49: "h", 51: "h", 52: "h",     # hats & cymbals
    53: "h", 55: "h", 57: "h", 59: "h",
}


def _vlq(data, i):
    """Variable-length quantity: 7 bits per byte, high bit = continue."""
    v = 0
    while i < len(data):
        b = data[i]
        i += 1
        v = (v << 7) | (b & 0x7F)
        if not (b & 0x80):
            break
    return v, i


def parse(path):
    """Parse an SMF into (notes, drums, ticks_per_beat, tempo_map, timesig).

    notes  : list of (start_tick, end_tick, pitch, velocity, channel)
    drums  : list of (tick, kind)
    tempo_map : list of (tick, microseconds_per_quarter)
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] != b"MThd":
        raise ValueError("not a Standard MIDI File")
    hlen = struct.unpack(">I", data[4:8])[0]
    fmt, ntrks, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        # SMPTE timing. Rare in game MIDI; treat as 480 tpb rather than fail.
        tpb = 480
    else:
        tpb = division or 480

    pos = 8 + hlen
    notes, drums, tempo_map, timesig = [], [], [], (4, 4)

    for _ in range(ntrks):
        if pos + 8 > len(data) or data[pos:pos + 4] != b"MTrk":
            break
        tlen = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        tstart = pos + 8
        tend = min(len(data), tstart + tlen)
        pos = tend

        i = tstart
        tick = 0
        status = 0
        sounding = {}                       # (chan, pitch) -> (start_tick, vel)
        program = [0] * 16                  # current GM program per channel
        while i < tend:
            delta, i = _vlq(data, i)
            tick += delta
            if i >= tend:
                break
            b = data[i]
            if b & 0x80:
                status = b
                i += 1
            # else: running status — reuse the previous status byte

            if status == 0xFF:              # meta
                mtype = data[i]; i += 1
                mlen, i = _vlq(data, i)
                payload = data[i:i + mlen]
                i += mlen
                if mtype == 0x51 and mlen == 3:
                    tempo_map.append((tick, (payload[0] << 16) |
                                      (payload[1] << 8) | payload[2]))
                elif mtype == 0x58 and mlen >= 2:
                    timesig = (payload[0] or 4, 1 << payload[1])
                elif mtype == 0x2F:
                    break
                continue
            if status in (0xF0, 0xF7):      # sysex
                slen, i = _vlq(data, i)
                i += slen
                continue

            hi = status & 0xF0
            chan = status & 0x0F
            if hi in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                d1 = data[i] if i < tend else 0
                d2 = data[i + 1] if i + 1 < tend else 0
                i += 2
                if hi == 0x90 and d2 > 0:                    # note on
                    if chan == 9:
                        k = GM_DRUM.get(d1)
                        if k:
                            drums.append((tick, k))
                    else:
                        sounding[(chan, d1)] = (tick, d2)
                elif hi in (0x80, 0x90):                     # note off
                    st = sounding.pop((chan, d1), None)
                    if st and chan != 9:
                        notes.append((st[0], tick, d1, st[1], chan, program[chan]))
            elif hi == 0xC0:                             # program change
                # This is what makes the GM bank usable: without recording it,
                # every instrument in the file plays on one patch.
                program[chan] = data[i] if i < tend else 0
                i += 1
            elif hi == 0xD0:
                i += 1
            else:
                i += 1
        # Anything still held at end-of-track ends there.
        for (chan, pitch), (st, vel) in sounding.items():
            if chan != 9:
                notes.append((st, tick, pitch, vel, chan, program[chan]))

    if not tempo_map:
        tempo_map = [(0, 500000)]                  # 120 bpm default
    tempo_map.sort()
    notes.sort()
    drums.sort()
    return notes, drums, tpb, tempo_map, timesig


def _tick_seconds(tempo_map, tpb):
    """Return a function mapping tick -> seconds, honouring tempo changes."""
    pts = [(0, 0.0, tempo_map[0][1])]
    for tk, us in tempo_map[1:]:
        ptk, psec, pus = pts[-1]
        pts.append((tk, psec + (tk - ptk) * pus / 1e6 / tpb, us))

    def f(tick):
        lo = 0
        for j, (tk, _, _) in enumerate(pts):
            if tk <= tick:
                lo = j
            else:
                break
        tk, sec, us = pts[lo]
        return sec + (tick - tk) * us / 1e6 / tpb
    return f


def to_events(path, np, seconds=None, steps_per_bar=None, transpose=0,
              start_frac=0.0):
    """Convert a MIDI file into the (ev, bars, spb, info, scale_name) shape.

    Emits exactly what chip.compose() emits, so both synthesis back-ends play a
    MIDI file with no changes — the renderers still do not care where notes come
    from.
    """
    notes, drums, tpb, tempo_map, timesig = parse(path)
    if not notes and not drums:
        return None
    beats_per_bar, unit = timesig
    div = 4 if unit == 4 else 2
    steps = steps_per_bar or int(beats_per_bar * div)
    us = tempo_map[0][1]
    bpm = 60_000_000.0 / us
    # Bar length in seconds, from the file's own tempo and metre.
    spb = 60.0 / bpm * beats_per_bar * (4.0 / unit)
    ticks_per_bar = tpb * beats_per_bar * (4.0 / unit)
    step_ticks = ticks_per_bar / steps

    end_tick = max([n[1] for n in notes] + [d[0] for d in drums] + [1])
    total_bars = max(1, int(end_tick / ticks_per_bar) + 1)

    # Optionally take a slice from the middle: game MIDIs often open with a few
    # bars of nothing, and an intro is rarely the memorable part.
    first_bar = int(total_bars * max(0.0, min(0.9, start_frac)))
    want_bars = total_bars - first_bar
    if seconds:
        want_bars = max(1, min(want_bars, int(round(seconds / spb))))

    def bar_step(tick):
        b = int(tick / ticks_per_bar) - first_bar
        s = int(round((tick % ticks_per_bar) / step_ticks)) % steps
        return b, s

    # Bucket melodic notes by step so registers can be compared at each moment.
    buckets = {}
    for st, en, pitch, vel, chan, prog in notes:
        b, s = bar_step(st)
        if b < 0 or b >= want_bars:
            continue
        dur = max(1, int(round((en - st) / step_ticks)))
        buckets.setdefault((b, s), []).append((pitch + transpose, vel,
                                               min(dur, steps - s), prog))

    ev = {"lead": [], "arp": [], "bass": [], "drum": []}
    # GM program per voice per step, so a renderer with a real bank can select the
    # instrument the composer actually asked for.
    progs = {"lead": {}, "arp": {}, "bass": {}}
    for (b, s), group in sorted(buckets.items()):
        group.sort()                                    # by pitch, ascending
        lowest = group[0]
        highest = group[-1]
        # Highest = melody, lowest = bass. With one note only, it is the melody:
        # a single line is a tune, not a bass part.
        ev["lead"].append((b, s, highest[2], highest[0], 0.5))
        progs["lead"][(b, s)] = highest[3]
        if len(group) > 1:
            ev["bass"].append((b, s, lowest[2], lowest[0] - 12
                               if lowest[0] > 60 else lowest[0]))
            progs["bass"][(b, s)] = lowest[3]
        if len(group) > 2:
            mid = group[len(group) // 2]
            ev["arp"].append((b, s, max(1, mid[2]), mid[0], 0.25))
            progs["arp"][(b, s)] = mid[3]
    for tick, kind in drums:
        b, s = bar_step(tick)
        if 0 <= b < want_bars:
            ev["drum"].append((b, s, kind))

    # FULL polyphony, alongside the 4-voice reduction above. A chip renderer with
    # a voice allocator can play the arrangement as written; collapsing to
    # highest/lowest/middle turned every chord into three notes and deleted the
    # inner parts, which is a large part of why the result sounded thin.
    poly = []
    for (b, st), group in sorted(buckets.items()):
        for pitch, vel, dur, prog in group:
            poly.append((b, st, dur, pitch, prog))

    info = {"bpm": bpm, "timesig": f"{beats_per_bar}/{unit}", "steps": steps,
            "poly": poly, "poly_notes": len(poly),
            "max_poly": max([len(g) for g in buckets.values()] or [0]),
            "bars": want_bars, "total_bars": total_bars,
            "notes": len(ev["lead"]), "drum_hits": len(ev["drum"]),
            "tempo_changes": len(tempo_map),
            "title": os.path.basename(path), "progs": progs,
            "gm_used": sorted({p for d in progs.values() for p in d.values()})}
    return ev, want_bars, spb, info, "minor"


def describe(info):
    return (f"{info['timesig']}  {info['bpm']:.0f}bpm  "
            f"{info['bars']}/{info['total_bars']}bars  "
            f"{info['notes']} notes  {info['drum_hits']} hits")
