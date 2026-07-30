#!/usr/bin/env python3
"""Render a tracker module to WAV with libxmp, so a .mod can be VERIFIED.

WHY THIS IS NOT OPTIONAL

Every .mod bug so far was structurally invisible. The file parsed, the note cells
were present and correct, and it still played 1.6x too fast because the tempo in
the header was the CLI's --bpm rather than the source's. Decoding the pattern data
back to (row, pitch) cannot catch that class of defect: the bytes are right, the
INTERPRETATION is wrong. Only a replayer settles it.

libxmp is the replayer that matters here, because it is the one pixel-viewer uses —
so what this renders is what grymmjack hears, not an approximation of it. It is
vendored in ~/git/pixel-viewer, and its build leaves a static archive behind, which
is enough to wrap:

    gcc -shared -o libxmp.so -Wl,--whole-archive libxmp.a \
        -Wl,--no-whole-archive -lm

Same approach as the Nuked OPL3 wrapper in opl.py: don't reimplement a replayer,
drive the real one.

    tools/render-mod.py song.mod                 # -> song.mod.wav
    tools/render-mod.py song.mod -o out.wav --seconds 30
    tools/render-mod.py song.mod --compare source.mid
"""
import argparse
import ctypes
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
VENDOR = os.path.expanduser("~/git/pixel-viewer")

XMP_PLAYER_INTERP = 2
XMP_INTERP_NEAREST = 0          # no oversampling: the crispness grymmjack keeps


class ModuleInfo(ctypes.Structure):
    """Only the head of xmp_module_info matters here; the tail is padding."""
    _fields_ = [("md5", ctypes.c_ubyte * 16),
                ("vol_base", ctypes.c_int),
                ("mod", ctypes.c_void_p),
                ("comment", ctypes.c_char_p),
                ("num_sequences", ctypes.c_int),
                ("seq_data", ctypes.c_void_p)]


class FrameInfo(ctypes.Structure):
    _fields_ = [("pos", ctypes.c_int), ("pattern", ctypes.c_int),
                ("row", ctypes.c_int), ("num_rows", ctypes.c_int),
                ("frame", ctypes.c_int), ("speed", ctypes.c_int),
                ("bpm", ctypes.c_int), ("time", ctypes.c_int),
                ("total_time", ctypes.c_int), ("frame_time", ctypes.c_int),
                ("buffer", ctypes.c_void_p), ("buffer_size", ctypes.c_int),
                ("total_size", ctypes.c_int), ("volume", ctypes.c_int),
                ("loop_count", ctypes.c_int), ("virt_channels", ctypes.c_int),
                ("virt_used", ctypes.c_int), ("sequence", ctypes.c_int),
                ("channel_info", ctypes.c_ubyte * (64 * 24))]


def find_lib(cache=None):
    """Locate or build libxmp.so from pixel-viewer's vendored copy."""
    for env in ("SOUNDMON_LIBXMP",):
        p = os.environ.get(env)
        if p and os.path.exists(p):
            return p
    cache = cache or os.path.join(tempfile.gettempdir(), "soundmon-libxmp.so")
    if os.path.exists(cache):
        return cache
    archives = []
    for root, _d, files in os.walk(os.path.join(VENDOR, "target")):
        if "libxmp.a" in files:
            archives.append(os.path.join(root, "libxmp.a"))
    if not archives:
        sys.exit("no libxmp.a found — build pixel-viewer first "
                 "(cd ~/git/pixel-viewer && cargo build --release)")
    archives.sort(key=os.path.getmtime, reverse=True)
    subprocess.run(["gcc", "-shared", "-o", cache, "-Wl,--whole-archive",
                    archives[0], "-Wl,--no-whole-archive", "-lm"], check=True)
    return cache


def render(path, rate=44100, seconds=None, loops=1, interp_nearest=True):
    """Module -> (float32 stereo array, rate, info dict)."""
    import numpy as np
    lib = ctypes.CDLL(find_lib())
    lib.xmp_create_context.restype = ctypes.c_void_p
    ctx = ctypes.c_void_p(lib.xmp_create_context())
    if not ctx:
        sys.exit("xmp_create_context failed")
    if lib.xmp_load_module(ctx, path.encode()) != 0:
        sys.exit(f"libxmp could not load {path}")

    mi = ModuleInfo()
    lib.xmp_get_module_info(ctx, ctypes.byref(mi))
    fi = FrameInfo()

    if lib.xmp_start_player(ctx, rate, 0) != 0:
        sys.exit("xmp_start_player failed")
    if interp_nearest:
        # grymmjack turns bilinear/oversampling OFF to keep modules crisp, so
        # verify what he listens to, not a smoothed version of it.
        lib.xmp_set_player(ctx, XMP_PLAYER_INTERP, XMP_INTERP_NEAREST)

    n = 4096
    buf = (ctypes.c_short * (n * 2))()
    chunks = []
    frames_max = int(seconds * rate) if seconds else None
    total = 0
    while lib.xmp_play_buffer(ctx, buf, ctypes.sizeof(buf), loops) == 0:
        chunks.append(np.frombuffer(bytes(buf), dtype="<i2").copy())
        total += n
        if frames_max and total >= frames_max:
            break
    lib.xmp_get_frame_info(ctx, ctypes.byref(fi))
    info = {"total_time": fi.total_time / 1000.0, "bpm": fi.bpm,
            "speed": fi.speed, "num_rows": fi.num_rows}
    lib.xmp_end_player(ctx)
    lib.xmp_release_module(ctx)
    lib.xmp_free_context(ctx)

    if not chunks:
        sys.exit("libxmp produced no audio")
    au = np.concatenate(chunks).astype(np.float32) / 32768.0
    au = au.reshape(-1, 2)
    if frames_max:
        au = au[:frames_max]
    return au, rate, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("-o", "--out")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--rate", type=int, default=44100)
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--smooth", action="store_true",
                    help="enable interpolation (default is nearest, i.e. crisp)")
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf
    au, rate, info = render(a.module, a.rate, a.seconds, a.loops,
                            interp_nearest=not a.smooth)
    out = a.out or (os.path.splitext(a.module)[0] + "-xmp.wav")
    sf.write(out, au, rate, subtype="PCM_16")
    m = au[:, 0]
    silent = float((np.abs(au).max(axis=1) < 1e-4).mean())
    print(f"  {os.path.basename(a.module)} -> {os.path.basename(out)}")
    print(f"  {len(au)/rate:.1f}s   libxmp reports {info['total_time']:.1f}s, "
          f"{info['bpm']} bpm, speed {info['speed']}")
    print(f"  peak {20*np.log10(max(abs(m).max(),1e-9)):+.1f} dBFS   "
          f"rms {20*np.log10(max(np.sqrt((m**2).mean()),1e-9)):+.1f} dBFS   "
          f"silent {silent*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
