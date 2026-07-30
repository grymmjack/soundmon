#!/usr/bin/env python3
"""Render a .RAD to WAV by parsing it and driving Nuked OPL3. VERIFICATION ONLY.

WHY THIS EXISTS

The RAD path has no listening loop. pixel-viewer plays mod/xm/s3m/it but not rad,
and the only real players are Reality AdLib Tracker and QB64 — so a broken RAD gets
discovered in the game, late. That already happened once: inverting total level
turned every loudest patch into a silent one, and the file loaded and "played" fine.
Nothing about the bytes looked wrong.

WHAT THIS DOES AND DOES NOT PROVE

It plays the file according to the SPEC AS I READ IT: note + octave combined as an
OPL block and an F-number, instrument bytes written straight to the operator
registers, line duration speed * 2.5 / BPM. So it confirms the file is internally
consistent and audibly correct under that reading, and it catches whole classes of
defect — silence, wrong octaves, wrong tempo, notes that never release.

It does NOT prove byte-for-byte agreement with Reality AdLib Tracker's own
replayer. Where the spec is ambiguous, this shares my interpretation rather than
checking it, so it cannot detect that the interpretation is wrong. The RAD in QB64
remains the final word.

    tools/render-rad.py song.rad             # -> song-opl.wav
    tools/render-rad.py song.rad --seconds 30
"""
import argparse
import os
import struct
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

# The classic AdLib F-number table, C through B, to be used with block = octave.
# Note field n maps to n % 12, so RAD's note 12 wraps to index 0 = C.
FNUM = [0x157, 0x16B, 0x181, 0x198, 0x1B0, 0x1CA,
        0x1E5, 0x202, 0x220, 0x241, 0x263, 0x287]
KEYOFF = 15
LINES = 64


def parse(path):
    """RAD 2.1 -> {speed, bpm, instruments, order, patterns}."""
    d = open(path, "rb").read()
    if d[:16] != b"RAD by REALiTY!!":
        sys.exit(f"{path}: not a RAD file")
    o = 16
    ver = d[o]; o += 1
    flags = d[o]; o += 1
    speed = flags & 0x1F
    bpm = 125
    if flags & 0x20:
        bpm = struct.unpack("<H", d[o:o + 2])[0]; o += 2
    slow = bool(flags & 0x40)
    end = d.index(0, o)
    desc = d[o:end]; o = end + 1

    instruments = {}
    while d[o]:
        num = d[o]; o += 1
        ln = d[o]; o += 1
        name = d[o:o + ln].decode("ascii", "replace"); o += ln
        instruments[num] = (name, d[o:o + 24]); o += 24
    o += 1
    nord = d[o]; o += 1
    order = list(d[o:o + nord]); o += nord

    patterns = {}
    while o < len(d) and d[o] != 0xFF:
        pi = d[o]; o += 1
        sz = struct.unpack("<H", d[o:o + 2])[0]; o += 2
        body = d[o:o + sz]; o += sz
        lines = {}
        i = 0
        while i < len(body):
            lb = body[i]; i += 1
            ln_no = lb & 0x7F
            chans = []
            while i < len(body):
                h = body[i]; i += 1
                ch = h & 0x0F
                note = octv = inst = fx = param = None
                if h & 0x40:
                    nb = body[i]; i += 1
                    note = nb & 0x0F
                    octv = (nb >> 4) & 0x07
                if h & 0x20:
                    inst = body[i] & 0x7F; i += 1
                if h & 0x10:
                    fx = body[i] & 0x1F; param = body[i + 1] & 0x7F; i += 2
                chans.append((ch, note, octv, inst, fx, param))
                if h & 0x80:
                    break
            lines[ln_no] = chans
            if lb & 0x80:
                break
        patterns[pi] = lines
    return {"version": ver, "speed": speed, "bpm": bpm, "slow": slow,
            "desc": desc.decode("ascii", "replace"), "instruments": instruments,
            "order": order, "patterns": patterns}


def program(chip, ch, body):
    """Write a RAD 24-byte instrument body to one OPL channel's registers.

    Byte layout: alg, feedback, detune, volume, then op1 (modulator) and op2
    (carrier) as five raw register bytes each. RAD holds RAW OPL values — the
    fields the spec calls "inverted" are attenuations, which are inverted in
    MEANING, not in storage. Subtracting them from 63 is the bug that made every
    loud patch silent.
    """
    import opl
    alg = body[0] & 0x07
    fb = body[1] & 0x0F
    mod = body[4:9]
    car = body[9:14]
    op1, op2 = opl._op_pair(ch)
    for op, regs in ((op1, mod), (op2, car)):
        chip.write(0x20 + op, regs[0])      # AM/VIB/EG/KSR/MULT
        chip.write(0x40 + op, regs[1])      # KSL / total level
        chip.write(0x60 + op, regs[2])      # attack / decay
        chip.write(0x80 + op, regs[3])      # sustain / release
        chip.write(0xE0 + op, regs[4] & 0x07)
    # 0x30 turns BOTH speakers on. A channel with neither bit set is silent, which
    # is an extremely confusing way to hear nothing -- and the second time this
    # exact trap would have produced a silent RAD.
    chip.write(opl._ch_reg(ch, 0xC0),
               0x30 | ((fb & 0x07) << 1) | (alg & 0x01))


def render(path, seconds=None, rate=49716):
    import numpy as np
    import opl
    r = parse(path)
    chip = opl.OPL3()
    chip.write(0x105, 0x01)                 # OPL3 mode: 18 channels
    chip.write(0x104, 0x00)                 # no 4-op fusion
    chip.write(0x01, 0x20)                  # waveform select enable

    line_s = r["speed"] * 2.5 / max(1, r["bpm"])
    cur = {}                                # channel -> instrument number
    out = []
    total = 0.0
    limit = seconds if seconds else 1e9
    counts = {"notes": 0, "keyoffs": 0}

    for pi in r["order"]:
        lines = r["patterns"].get(pi, {})
        last = max(lines) if lines else LINES - 1
        for ln in range(last + 1):
            for ch, note, octv, inst, _fx, _param in lines.get(ln, []):
                if ch >= 9:
                    continue
                if inst is not None and inst in r["instruments"]:
                    program(chip, ch, r["instruments"][inst][1])
                    cur[ch] = inst
                if note is None:
                    continue
                if note == KEYOFF:
                    chip.key_off(ch)
                    counts["keyoffs"] += 1
                elif 1 <= note <= 12:
                    if ch not in cur and 1 in r["instruments"]:
                        program(chip, ch, r["instruments"][1][1])
                        cur[ch] = 1
                    chip.key_off(ch)
                    f = FNUM[note % 12]
                    chip.write(opl._ch_reg(ch, 0xA0), f & 0xFF)
                    chip.write(opl._ch_reg(ch, 0xB0),
                               0x20 | ((octv & 0x07) << 2) | ((f >> 8) & 0x03))
                    counts["notes"] += 1
            out.append(chip.render(int(line_s * rate), np))
            total += line_s
            if total >= limit:
                break
        if total >= limit:
            break
    # OPL3.render() mono-sums, so this is 1-D. Kept that way deliberately: the OPL
    # side of soundmon is mono everywhere else too.
    au = np.concatenate(out) if out else np.zeros(1)
    return au, rate, r, counts, line_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rad")
    ap.add_argument("-o", "--out")
    ap.add_argument("--seconds", type=float, default=None)
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf
    au, rate, r, counts, line_s = render(a.rad, a.seconds)
    out = a.out or (os.path.splitext(a.rad)[0] + "-opl.wav")
    sf.write(out, au, rate, subtype="PCM_16")
    m = au
    print(f"  {os.path.basename(a.rad)} -> {os.path.basename(out)}")
    print(f"  RAD v{r['version']:X}  speed {r['speed']}  {r['bpm']} bpm  "
          f"line {line_s*1000:.1f} ms  {len(r['patterns'])} patterns  "
          f"{len(r['instruments'])} instruments")
    print(f"  {counts['notes']} notes, {counts['keyoffs']} key-offs   "
          f"{len(au)/rate:.1f}s")
    print(f"  peak {20*np.log10(max(abs(m).max(),1e-9)):+.1f} dBFS   "
          f"rms {20*np.log10(max(np.sqrt((m**2).mean()),1e-9)):+.1f} dBFS   "
          f"silent {float((np.abs(au) < 1e-4).mean())*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
