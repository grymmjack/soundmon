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
