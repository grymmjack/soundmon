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
REF_FALLBACK_DB = 70.0             # reference band this far under the loudest octave (only leakage: a
                                   # pure tone below 500 Hz) - use that octave. Bass-heavy SFX sit at -40..-60


def read(path):
    """(mono samples, rate): soundfile, else ffmpeg (m4a/AAC, opus... whatever it decodes)."""
    try:
        audio, sr = sf.read(path)
        return (audio.mean(axis=1) if audio.ndim > 1 else audio), sr
    except Exception as e:
        import shutil
        import subprocess
        if not shutil.which("ffmpeg"):
            raise
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
                            "-of", "csv=p=0", path], capture_output=True, text=True)
        try:
            sr = int(r.stdout.strip().split(",")[0])
        except ValueError:
            raise e
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-f", "f32le", "-"],
                           capture_output=True)
        if r.returncode:
            lines = r.stderr.decode(errors="replace").strip().splitlines()
            raise RuntimeError(lines[-1] if lines else "ffmpeg couldn't decode it")
        return np.frombuffer(r.stdout, dtype=np.float32).astype(np.float64), sr


def cutoff(path):
    """Return (cutoff_khz, samplerate, duration, band_table, non-finite samples)."""
    mono, sr = read(path)
    bad = int((~np.isfinite(mono)).sum())
    if bad:                        # NaN/inf: broken audio, whatever its spectrum says
        return None, sr, len(mono) / sr, [], bad
    win = 16384
    if len(mono) < win:            # very short one-shots: use what there is
        win = 1 << max(8, (len(mono) - 1).bit_length() - 1)
    if win < 256:
        return None, sr, len(mono) / sr, [], 0

    acc = np.zeros(win // 2 + 1)
    frames = 0
    for i in range(0, len(mono) - win + 1, win // 2):
        acc += np.abs(np.fft.rfft(mono[i:i + win] * np.hanning(win)))
        frames += 1
    if not frames:
        return None, sr, len(mono) / sr, [], 0
    acc /= frames
    freq = np.fft.rfftfreq(win, 1 / sr)

    # The reference: the 500-2000 Hz band, unless it holds almost nothing (a 300 Hz tone, a bass
    # line): then the file's own loudest octave. Measured against an empty band, a pure 440 Hz tone
    # read 22 kHz and a 300 Hz one 2.3 kHz; now both read just above their pitch.
    ref_sel = (freq >= REF_LO) & (freq < REF_HI)
    octs = [acc[(freq >= lo) & (freq < 2 * lo)].mean() for lo in 2.0 ** np.arange(5, 15) if (freq >= lo).any()
            and ((freq >= lo) & (freq < 2 * lo)).any()]
    top = max(octs) if octs else acc.max()
    ref = acc[ref_sel].mean() if ref_sel.any() else top
    if ref < top * 10 ** (-REF_FALLBACK_DB / 20):
        ref = top
    if top <= 0:                   # digital silence: no spectrum to have a cutoff
        return None, sr, len(mono) / sr, [], 0
    db = 20 * np.log10(acc / (ref + 1e-12) + 1e-12)

    below = np.where(db < -DROP_DB)[0]
    below = below[below > int(np.argmax(acc))]     # a cutoff is above the loudest content, not under it
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
    return cut / 1000.0, sr, len(mono) / sr, bands, 0


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
            cut, sr, dur, bands, nonfinite = cutoff(p)
        except Exception as e:
            print(f"  {os.path.basename(p):<34} unreadable: {e}")
            failures += 1
            continue
        if nonfinite:
            print(f"  {os.path.basename(p):<34} {nonfinite} NaN/inf sample(s) ← BROKEN")
            failures += 1
            continue
        if cut is None:            # too short or silent: say so (it can't pass a --min check)
            print(f"  {os.path.basename(p):<34} {dur:6.3f}s  too short or silent to measure"
                  + (" ← can't check" if a.min is not None else ""))
            failures += a.min is not None
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
