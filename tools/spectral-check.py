#!/usr/bin/env python3
"""Report the spectral cutoff of generated audio.

The gap this fills: soundmon verifies duration, peak, LUFS, true-peak and
readability, and a spectrally hollow file passes every one of them. 1,122 files
shipped sounding "spectrally processed, gaps of frequencies just gone" while
every automated check was green — the problem was only caught by ear.

A cutoff is the cheapest proxy for that. Measure where the average spectrum
falls 40 dB below the 500-2000 Hz reference band. Measured on real output:

    source          as WAV     as OGG
    ACE-Step turbo  16.0 kHz   15.9 kHz   <- its audio VAE; not recoverable
    Stable Audio 3  22.1 kHz   17.3 kHz   <- WAV is Nyquist, i.e. no ceiling

COMPARE LIKE WITH LIKE. Vorbis imposes its own rolloff, so an ogg of perfect
source still reads ~17 kHz — that is the codec, not a defect. Thresholds:

    --min 20   for WAV      (anything less lost content upstream)
    --min 17   for OGG      (below that the source was already hollow)

The band table printed on a failure is the better discriminator anyway: at
12-16 kHz the same two files read -23 dB (ACE) versus -11 dB (SA3), a 12 dB
gap that survives the codec because it sits below Vorbis's rolloff.

    ./spectral-check.py --min 17 assets/music/soundmon-souls/*.ogg
    ./spectral-check.py --min 20 raw-render.wav
"""
import argparse
import os
import sys

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    # In a soundmon install the deps already live in ComfyUI's venv (soundfile
    # arrives with Kokoro for --narrate), so point there rather than telling
    # someone to install a second copy.
    venv = os.path.expanduser("~/ComfyUI/.venv/bin/python")
    hint = f"\n  try:  {venv} {' '.join(sys.argv)}" if os.path.exists(venv) else ""
    sys.exit(f"needs numpy + soundfile:  pip install soundfile numpy{hint}")

REF_LO, REF_HI = 500.0, 2000.0     # the band everything is measured against
DROP_DB = 40.0                     # how far below reference counts as "gone"


def cutoff(path):
    """Return (cutoff_khz, samplerate, duration, band_table)."""
    audio, sr = sf.read(path)
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    win = 16384
    if len(mono) < win:            # very short one-shots: use what there is
        win = 1 << max(8, (len(mono) - 1).bit_length() - 1)
    if win < 256:
        return None, sr, len(mono) / sr, []

    acc = np.zeros(win // 2 + 1)
    frames = 0
    for i in range(0, len(mono) - win + 1, win // 2):
        acc += np.abs(np.fft.rfft(mono[i:i + win] * np.hanning(win)))
        frames += 1
    if not frames:
        return None, sr, len(mono) / sr, []
    acc /= frames
    freq = np.fft.rfftfreq(win, 1 / sr)

    ref_sel = (freq >= REF_LO) & (freq < REF_HI)
    ref = acc[ref_sel].mean() if ref_sel.any() else acc.max()
    db = 20 * np.log10(acc / (ref + 1e-12) + 1e-12)

    below = np.where(db < -DROP_DB)[0]
    # Ignore an early dip: the cutoff is where it drops AND stays down.
    cut = None
    for idx in below:
        if (db[idx:] < -DROP_DB).mean() > 0.9:
            cut = freq[idx]
            break
    if cut is None:
        cut = sr / 2

    bands = []
    for lo, hi in ((4000, 8000), (8000, 12000), (12000, 16000),
                   (16000, 18000), (18000, 22000)):
        sel = (freq >= lo) & (freq < hi)
        bands.append((lo, hi, db[sel].mean() if sel.any() else float("-inf")))
    return cut / 1000.0, sr, len(mono) / sr, bands


def main():
    ap = argparse.ArgumentParser(description="report spectral cutoff of audio files")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--min", type=float, default=None, metavar="KHZ",
                    help="fail (exit 1) if any file cuts off below this")
    ap.add_argument("--quiet", action="store_true", help="only show failures")
    a = ap.parse_args()

    worst, failures = None, 0
    for p in a.files:
        try:
            cut, sr, dur, bands = cutoff(p)
        except Exception as e:
            print(f"  {os.path.basename(p):<34} unreadable: {e}")
            failures += 1
            continue
        if cut is None:
            continue
        nyq = sr / 2000.0
        bad = a.min is not None and cut < a.min
        failures += bad
        worst = cut if worst is None else min(worst, cut)
        if a.quiet and not bad:
            continue
        flag = " ← LOW" if bad else ""
        print(f"  {os.path.basename(p):<34} {dur:6.2f}s {sr/1000:5.1f}k  "
              f"cutoff {cut:5.1f} kHz (nyq {nyq:.1f}){flag}")
        if bad:
            print("      " + "  ".join(f"{lo//1000}-{hi//1000}k {v:.0f}dB"
                                       for lo, hi, v in bands))

    if worst is not None:
        print(f"\n  lowest cutoff seen: {worst:.1f} kHz over {len(a.files)} file(s)")
    if a.min is not None:
        print(f"  below {a.min:g} kHz: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
