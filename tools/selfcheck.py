#!/usr/bin/env python3
"""Assert the things that broke silently. Run after touching an engine.

WHY THIS EXISTS

Every defect in this repo's recent history was invisible to inspection and only
surfaced by ear:

    a -16 LUFS ceiling that should not apply to music, and then a fix for it that
    sat AFTER the engine hand-offs and so never ran at all
    8-bit format-locked audio silently promoted back to 16-bit by ffmpeg
    -21 dBFS of DC from thin duty cycles
    note durations clamped at bar lines
    note_off matched by pitch, so doubled notes killed each other
    two renderers drifting apart because a fix landed in only one

Documenting a lesson does not install it — I wrote the hand-off gotcha in AGENTS.md
and then walked into it the same day. An assertion re-checks it every time, at the
moment it matters, for free.

These are cheap invariants, not a substitute for listening. Audio quality still
needs ears; this only catches the class of bug where a value is silently wrong.

    tools/selfcheck.py            # all checks
    tools/selfcheck.py -v         # show each result
"""
import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
PY = os.path.expanduser("~/ComfyUI/.venv/bin/python")
SM = os.path.join(HERE, "soundmon.py")

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return ok


def _render(tmp, *args):
    """Run soundmon and return the single file it produced."""
    before = set(os.listdir(tmp))
    r = subprocess.run([PY, SM, "test tone", "--output-to", tmp, "--no-open",
                        "--seconds", "4", *args],
                       capture_output=True, text=True)
    made = [f for f in os.listdir(tmp) if f not in before]
    if not made:
        return None, r.stderr[-300:]
    return os.path.join(tmp, sorted(made)[0]), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import chip
    import theory

    # --- pure-logic invariants ----------------------------------------------
    # Arpeggios must never reach the bass. A fast arp below ~A3 is mud, not a
    # chord, because the ear stops fusing the tones.
    notes = [{"id": i, "t": 0.0, "dur": 1.0, "pitch": p, "prog": 0, "vel": 100}
             for i, p in enumerate((36, 40, 43, 62, 65, 69))]
    out = chip.chippify(notes, "max")
    arped = [n["pitch"] for n in out if isinstance(n.get("id"), tuple)]
    check("chippify never arpeggiates below ARP_MIN_PITCH",
          all(p >= chip.ARP_MIN_PITCH for p in arped),
          f"lowest arped = {min(arped) if arped else 'none'} "
          f"(floor {chip.ARP_MIN_PITCH})")

    # Pack identity must change the plan, or every pack composes one piece.
    mood = chip.MOODS["solemn"]
    p1 = theory.plan_track("packA/level9", mood, "solemn", "minor", 12)
    p2 = theory.plan_track("packB/level9", mood, "solemn", "minor", 12)
    check("pack identity changes the composition",
          (p1["progs"], p1["form"]) != (p2["progs"], p2["form"]),
          "same track name in two packs must not produce one plan")

    # DC blocking must actually remove DC.
    x = np.ones(4096) * 0.5
    check("_dc_block removes a constant offset",
          abs(float(chip._dc_block(x, np, 44100)[2048:].mean())) < 0.02,
          "a 12.5% duty pulse sits at a mean of -0.75")

    # --- tracker output -----------------------------------------------------
    # These four were ALL shipped broken and all were invisible to inspection:
    # correct-looking bytes, wrong interpretation. Only a replayer or arithmetic
    # catches them, so check the arithmetic here every time.
    import math
    import mod as modw
    import rad as radw

    # Tempo must come from the composition. It used to come from --bpm, so a
    # 75 bpm MIDI was written as 120 and played 1.6x too fast.
    worst, detail = 0.0, ""
    for spb in (3.2, 2.0, 1.714, 4.5):
        for rows in (16, 24, 32, 48, 96):
            for name, fn in (("mod", modw.timing_for), ("rad", radw.timing_for)):
                t, b = fn(spb, rows)
                err = abs(t * 2.5 / b - spb / rows) / (spb / rows)
                if err > worst:
                    worst, detail = err, f"{name} spb={spb} rows={rows} -> {t}/{b}"
    check("tracker row duration matches the source tempo", worst < 0.01,
          f"worst {worst*100:.2f}% off ({detail})")

    # The MOD period table must span real game-music range. Three octaves forced
    # octave FOLDING, which turned melody peaks into different notes.
    lo, hi = 41, 82
    folded = [p for p in range(lo, hi + 1)
              if not (modw.MIDI_BASE <= p < modw.MIDI_BASE + len(modw.PERIODS))]
    check("MOD periods cover MIDI 41-82 without folding", not folded,
          f"{len(folded)} of {hi-lo+1} notes outside the table")

    # A looping single cycle is only in tune if the length and finetune agree.
    # A 64-byte cycle at period 428 is 17.7 cents flat, on every note.
    f = 3546894.6 / 428 / modw.CYCLE_LEN * 2 ** (modw.FINETUNE / 96.0)
    cents = 1200 * math.log2(f / (440 * 2 ** ((48 - 69) / 12)))
    check("MOD sample tuning is within 3 cents", abs(cents) < 3.0,
          f"{cents:+.2f} cents (len {modw.CYCLE_LEN}, finetune {modw.FINETUNE})")

    # RAD numbers C as 12, which is a NOTE-field quirk only. Treating it as an
    # octave boundary put every C an octave below its neighbours.
    worst, detail = 0.0, ""
    for p in range(36, 84):
        note, octv = radw._note_octave(p)
        fnum = [0x157, 0x16B, 0x181, 0x198, 0x1B0, 0x1CA,
                0x1E5, 0x202, 0x220, 0x241, 0x263, 0x287][note % 12]
        got = fnum * 49716.0 / (1 << (20 - octv))
        c = abs(1200 * math.log2(got / (440 * 2 ** ((p - 69) / 12))))
        if c > worst:
            worst, detail = c, f"MIDI {p} -> note {note} oct {octv} = {got:.1f} Hz"
    check("RAD note/octave reconstructs the right pitch", worst < 25.0,
          f"worst {worst:.1f} cents ({detail})")

    # A release must never land inside the NEXT note. MOD and RAD channels are
    # both monophonic, so a release at onset+duration silences its successor.
    ev = {"lead": [(0, 0, 8, 60, 0.5, 100), (0, 4, 8, 64, 0.5, 100),
                   (0, 12, 2, 67, 0.5, 100)],
          "arp": [], "bass": [], "drum": []}
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, "t.mod")
        modw.write_mod(mp, ev, 1, 2.0, 16, np, rows_per_bar=16)
        d = open(mp, "rb").read()
        onsets, offs = set(), set()
        for r in range(64):
            b0, b1, b2, b3 = d[1084 + r * 16:1084 + r * 16 + 4]
            if ((b0 & 0x0F) << 8) | b1:
                onsets.add(r)
            elif (b2 & 0x0F) == 0xC and b3 == 0:
                offs.add(r)
        check("MOD release never lands on a following onset",
              not (offs & onsets) and offs,
              f"onsets {sorted(onsets)} releases {sorted(offs)}")

    # Vibrato depth is in PERIOD units, so one fixed depth is a swing that widens
    # without limit going up. A hard-coded 3 measured +/-162 cents at MIDI 80.
    worst, detail = 0.0, ""
    for p in range(36, 84):
        d = modw.vib_depth(p)
        if not d:
            continue
        per = modw._period(p)
        c = 1200 * math.log2(per / max(1.0, per - d * 2.0))
        if c > worst:
            worst, detail = c, f"MIDI {p} depth {d}"
    check("MOD vibrato swing stays under 45 cents", worst < 45.0,
          f"worst {worst:.0f} cents ({detail})")

    # A slide must ARRIVE. 3xx only acts on rows it is written to, so a rate sized
    # for N rows but written on one stops 1/N of the way: 83 of 83 slides landed
    # short, up to 500 cents from the written note.
    notes = [(0, i * 6, 6, 60 + (i % 5) * 2, 0.5, 100) for i in range(24)]
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, "p.mod")
        modw.write_mod(mp, {"lead": notes, "arp": [], "bass": [], "drum": []},
                       3, 3.2, 96, np, chippy="max", rows_per_bar=96)
        d = open(mp, "rb").read()
        npat = d[950]
        cells = [d[1084 + pat * 1024 + r * 16:1084 + pat * 1024 + r * 16 + 4]
                 for pat in range(npat) for r in range(64)]
        ticks, _bpm = modw.timing_for(3.2, 96)
        cur = None
        i = 0
        tot = bad = 0
        while i < len(cells):
            b0, b1, b2, b3 = cells[i]
            per = ((b0 & 0x0F) << 8) | b1
            if (b2 & 0x0F) == 3 and cur and per and per != cur:
                tot += 1
                tgt, rate, pp, up, j = per, b3, cur, per > cur, i
                while j < len(cells) and (cells[j][2] & 0x0F) == 3:
                    rate = cells[j][3] or rate
                    # ticks - 1: tick 0 of a row sets the target and does not
                    # slide. Simulating `ticks` here is what made this check agree
                    # with a writer that was stopping every slide short.
                    for _ in range(max(1, ticks - 1)):
                        pp = pp + rate if up else pp - rate
                        if (up and pp >= tgt) or (not up and pp <= tgt):
                            pp = tgt
                            break
                    j += 1
                    if pp == tgt:
                        break
                bad += pp != tgt
                cur = tgt
                i = j
                continue
            if per:
                cur = per
            i += 1
        check("MOD tone portamento always reaches its target",
              tot > 0 and bad == 0, f"{bad} of {tot} slides land short")

    # --- end-to-end invariants ----------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        # THE ONE THAT KEEPS BREAKING: the loudness ceiling must not apply to
        # music, and the override must be set ABOVE the engine hand-offs, which
        # return early (AGENTS.md gotcha 10).
        p, err = _render(tmp, "--chip", "--bpm", "120")
        if check("--chip renders", p, err):
            au, sr = sf.read(p, always_2d=True)
            m = au[:, 0]
            peak = 20 * np.log10(max(abs(m).max(), 1e-9))
            check("--chip is NOT held down by the LUFS ceiling", peak > -2.0,
                  f"peak {peak:.2f} dBFS (a -16 LUFS ceiling lands near -3)")
            check("--chip output carries no DC", abs(float(m.mean())) < 0.02,
                  f"dc {float(m.mean()):+.4f}")

        # A format lock must actually change rate AND bit depth, and survive
        # loudness normalisation, which used to promote everything to 16-bit.
        p, err = _render(tmp, "--chip", "--format", "mod6")
        if check("--format renders", p, err):
            i = sf.info(p)
            check("--format sets the sample rate", i.samplerate == 11025,
                  f"{i.samplerate} Hz (expected 11025)")
            check("--format sets the stored bit depth", i.subtype == "PCM_U8",
                  f"{i.subtype} (expected PCM_U8; ffmpeg used to force PCM_16)")
            au, _ = sf.read(p, always_2d=True)
            lv = len(np.unique(np.round(au[:, 0], 6)))
            check("--format quantizes amplitude", lv <= 70,
                  f"{lv} distinct levels (6-bit allows 64)")

        # encode() MUST be idempotent. On the diffusion path --flac attaches a
        # ComfyUI SaveAudio node, so the server already returns a .flac —
        # re-encoding resolved the output to the same path, ffmpeg refused, and the
        # cleanup deleted the file. Every asset downloaded fine and was then
        # destroyed, reported as "nothing produced".
        import soundmon as _smm
        import types as _t
        for fmt, flags in (("flac", dict(flac=True, ogg=False)),
                           ("ogg", dict(flac=False, ogg=True))):
            p = os.path.join(tmp, f"idem.{fmt}")
            sf.write(p, np.zeros(2205, dtype="float32"), 44100,
                     format=fmt.upper())
            n0 = os.path.getsize(p)
            aa = _t.SimpleNamespace(keep_wav=False, ogg_quality=5, **flags)
            out = _smm.encode(p, aa)
            check(f"encode() leaves an existing .{fmt} alone",
                  out == p and os.path.exists(p) and os.path.getsize(p) == n0,
                  f"-> {os.path.basename(out)}, exists={os.path.exists(p)}")
            os.path.exists(p) and os.remove(p)

        # EVERY engine must honour --flac. It was wired into three of seven, and
        # blip.py/narrate.py accepted a to_flac callable and never called it — so
        # the flag was accepted, plumbed, and silently dropped. Only the engines
        # that need no model or GPU are checked here; that is enough to catch a
        # new engine copying the old if-ogg-only shape.
        for eng, extra in (("--chip", []), ("--opl", []),
                           ("--chipfx", []), ("--blip", [])):
            before = set(os.listdir(tmp))
            r = subprocess.run([PY, SM, "beep", "--output-to", tmp, "--no-open",
                                "--seconds", "2", eng, "--flac", *extra],
                               capture_output=True, text=True)
            made = [f for f in os.listdir(tmp) if f not in before]
            got = [f for f in made if f.endswith(".flac")]
            check(f"{eng} honours --flac", bool(got),
                  f"produced {made or r.stderr[-160:]}")
            for f in made:
                os.remove(os.path.join(tmp, f))

        # FLAC must be lossless. Test the CODEC, not two renders: since the
        # pack-identity fix, --output-to feeds the composition, so rendering into
        # two directories deliberately produces two different pieces. My first
        # attempt did exactly that and "failed" on a 7436-frame length difference
        # that was the feature working. Convert one file instead.
        p, err = _render(tmp, "--chip", "--format", "mod6", "--seed", "42")
        if check("--flac source renders", p, err):
            sys.path.insert(0, HERE)
            import soundmon as _sm
            import shutil as _sh
            src = os.path.join(tmp, "flacsrc.wav")
            _sh.copy(p, src)
            out = _sm.to_flac(src, keep=True)
            ok = out.endswith(".flac") and os.path.exists(out)
            if check("to_flac produces a .flac", ok, out):
                x1, _ = sf.read(src, always_2d=True)
                x2, _ = sf.read(out, always_2d=True)
                same = (len(x1) == len(x2)
                        and np.allclose(x1[:, 0], x2[:, 0], atol=1e-6))
                check("FLAC is bit-identical to its WAV", same,
                      f"{len(x1)} vs {len(x2)} frames, "
                      f"max diff {float(np.abs(x1[:min(len(x1),len(x2)),0] - x2[:min(len(x1),len(x2)),0]).max()):.2e}")

    width = max(len(n) for n, _, _ in RESULTS)
    bad = 0
    for name, ok, detail in RESULTS:
        bad += not ok
        if a.verbose or not ok:
            mark = "ok  " if ok else "FAIL"
            print(f"  {mark} {name:<{width}}  {detail}")
    print(f"\n  {len(RESULTS) - bad}/{len(RESULTS)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
