"""--master: make a set of sounds sit together — neon-caster's mastering chain, for any game.

Each sound is fine alone; in a fight they stack, and the ones heavy in 2.5–10 kHz for long
stretches are what grates. This measures every sound and gives each its own amounts of one chain:

    high-pass 30 Hz (rumble)
    -> low shelf 180 Hz up      body, only where there is low end to lift
    -> high shelf 3 kHz down    by how far its tilt is above -3 dB/octave (darker than pink,
                                as a comfortable mix is)
    -> cut at 3.2 kHz           when the 2.5–5 kHz presence band piles up
    -> de-esser                 when its sibilant bursts are sharp
    -> low-pass 12 kHz          sound effects only (air)
    -> gain back towards its own loudest 400 ms — at most +3 dB and never past -1 dBTP.
       No limiter, so nothing is distorted, and nothing is normalized to full scale.

The make-up gain is capped ON PURPOSE (neon-caster's design): at most +3 dB, and none at all for
a sound with over 40 % of its energy above 10 kHz (white noise, a whistle) — for those, the level
the EQ and low-pass took away IS the treatment. So very bright sounds come out quieter than they
went in (white noise: about -9 dB at its loudest 400 ms); every line of --master says by how much.

Voices ("speech") are left alone unless they are bright, and keep their air. On NEON-CASTER this
took the sound effects' energy above 2.5 kHz from 34 % to 21 % and their harsh seconds from 64 to
40, while voices barely changed. Originally neon-caster/sfx/mastering/{analyze,master}.py.

True peak is measured with 4x+ oversampling (soxr), the written file is measured again, and a
file that lands over -1 dBTP (rounding, a lossy encode) is written again that much quieter.

Needs numpy and ffmpeg. Imports nothing heavy at module level (soundmon-gui imports this).
"""
import os
import re
import subprocess
import sys
import tempfile

SR = 44100
CENTRES = [31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
           2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000]
REGIONS = {"sub": (20, 80), "low": (80, 250), "lowmid": (250, 1000), "mid": (1000, 2500),
           "presence": (2500, 5000), "sibilance": (5000, 10000), "air": (10000, 20000)}
TARGET_TILT = -3.0          # dB per octave in 1/3-octave band power
AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3", ".opus", ".m4a", ".aif", ".aiff")
KINDS = ("sfx", "speech")
# never mastered from a folder: our own output, backups, recorder takes, GUI previews/measurements
# (_mastered/, _analysis/ — any folder starting with "_" or "."), and hidden files
SKIP_DIRS = {"mastered", ".soundmon-backup", ".pixelmon-backup", "takes", "node_modules", ".git"}
CEILING = -1.0              # dBTP: the only guard (no limiter)
MARGIN = 0.1                # aim this far under it; the written file is measured again
SILENT = 1e-6               # peak below -120 dBFS: nothing to measure


def _need_ffmpeg():
    from shutil import which
    if not which("ffmpeg"):
        sys.exit("--master needs ffmpeg on PATH")


def _reason(stderr):
    """ffmpeg's own explanation, short: its last meaningful lines."""
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="replace")
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    lines = [re.sub(r"^\[[^\]]*\]\s*", "", ln) for ln in lines]
    lines = [ln for ln in lines if not ln.startswith(("Conversion failed", "Error opening output"))] or lines
    last = lines[-1] if lines else "unknown error"
    if last.startswith("Error opening input") and ": " in last:
        last = last.split(": ", 1)[1]
    return last[-240:]


def _ffmpeg(args, what="ffmpeg"):
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y"] + args, capture_output=True)
    if r.returncode:
        raise RuntimeError(f"{what}: {_reason(r.stderr)}")
    return r


def _tmp(near, suffix):
    """A temp file beside `near` (same filesystem, so os.replace is atomic), hidden, unique."""
    d = os.path.dirname(os.path.abspath(near))
    os.makedirs(d, exist_ok=True)
    fd, p = tempfile.mkstemp(prefix="." + os.path.basename(near) + ".", suffix=suffix, dir=d)
    os.close(fd)
    return p


def _chunks(path, np, seconds=10.0):
    """Mono float64 at 44.1 kHz, `seconds` at a time — a 1-hour file never sits in memory whole."""
    with tempfile.TemporaryFile() as err:
        p = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
                              "-f", "f32le", "-"], stdout=subprocess.PIPE, stderr=err)
        want, rest = int(SR * seconds) * 4, b""
        try:
            while True:
                raw = p.stdout.read(want)
                if not raw:
                    break
                raw = rest + raw
                cut = len(raw) - len(raw) % 4
                rest = raw[cut:]
                if cut:
                    yield np.frombuffer(raw[:cut], dtype=np.float32).astype(np.float64)
        finally:
            p.stdout.close()
            code = p.wait()
        if code:
            err.seek(0)
            raise RuntimeError(f"can't read {os.path.basename(path)}: {_reason(err.read())}")


def load(path, np):
    """Mono float64 at 44.1 kHz, whatever the file was."""
    parts = list(_chunks(path, np))
    return np.concatenate(parts) if parts else np.zeros(0)


def _db_num(s):
    return -150.0 if s in ("-inf", "inf") or s is None else max(-150.0, float(s))


def loudness(path):
    """(its loudest 400 ms in LUFS, its true peak in dBTP) — ffmpeg's EBU R128 scanner (0.1 dB)."""
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "verbose", "-i", path, "-af",
                        "apad=whole_dur=0.5,ebur128=framelog=verbose:peak=true", "-f", "null", "-"],
                       capture_output=True, text=True, errors="replace").stderr
    m = [float(x) for x in re.findall(r"M:\s*(-?[\d.]+)", r)]
    tp = re.findall(r"Peak:\s*(-?[\d.]+|-inf) dBFS", r)
    return (max(m) if m else -99.0), (_db_num(tp[-1]) if tp else 0.0)


def true_peak(path):
    """A file's true peak in dBTP to 0.01 dB: oversampled 4x (at least to 176.4 kHz) with soxr, as a
    DAC would rebuild it. ffmpeg's ebur128 prints 0.1 dB and does not oversample a 96 kHz file."""
    try:
        sr = int(_info(path)[0])
    except ValueError:
        sr = SR
    up = max(sr * 4, 176400)
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "info", "-i", path, "-af",
                        f"aresample={up}:resampler=soxr:precision=28,aformat=sample_fmts=dbl,"     # (unpinned, it clips at 0 dBFS)
                        "astats=measure_overall=Peak_level:measure_perchannel=none", "-f", "null", "-"],
                       capture_output=True, text=True, errors="replace")
    pk = re.findall(r"Peak level dB:\s*(-?[\d.]+|-inf)", r.stderr)
    if r.returncode or not pk:
        return loudness(path)[1]
    return round(_db_num(pk[-1]), 2)


def _frames(path):
    """Sample frames in a WAV ffmpeg wrote (its data chunk), or None."""
    import struct
    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
            if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                return None
            align = None
            while True:
                h = fh.read(8)
                if len(h) < 8:
                    return None
                cid, size = h[:4], struct.unpack("<I", h[4:])[0]
                if cid == b"fmt ":
                    body = fh.read(size)
                    align = struct.unpack("<H", body[12:14])[0]
                    if size & 1:
                        fh.read(1)
                elif cid == b"data":
                    return size // align if align else None
                else:
                    fh.seek(size + (size & 1), 1)
    except (OSError, struct.error):
        return None


def analyse(path, sonogram=False):
    """What a sound is made of: 1/3-octave spectrum (silence ignored), tilt in dB/octave, share of
    energy per region, sibilant bursts, harsh seconds (loud and mostly above 2.5 kHz), loudness.

    Read in 10 s chunks, keeping only ~40 numbers per 11.6 ms frame (the band sums everything
    below is made of), so a 1-hour file fits. A silent file gets "silent": True and neutral numbers;
    NaN/inf samples are read as 0 and counted in "nonfinite"."""
    import numpy as np
    n, hop = 2048, 512
    win = np.hanning(n)
    fr = np.fft.rfftfreq(n, 1 / SR)
    cols = [(fr >= c / 2 ** (1 / 6)) & (fr < c * 2 ** (1 / 6)) for c in CENTRES]          # 0..27 bands
    cols += [(fr >= lo) & (fr < hi) for lo, hi in REGIONS.values()]                       # 28..34 regions
    cols += [fr >= 20, (fr >= 5000) & (fr < 9000), fr >= 2500, np.ones_like(fr, bool)]   # 35 tot, 36 sib, 37 hf, 38 all
    P = np.array(cols, dtype=np.float64).T
    NB, NR = len(CENTRES), len(REGIONS)
    I_TOT, I_SIB, I_HF, I_E = NB + NR, NB + NR + 1, NB + NR + 2, NB + NR + 3
    rows, buf, total, peak, bad = [], np.zeros(0), 0, 0.0, 0

    def frames_of(b):
        k = (len(b) - n) // hop + 1 if len(b) >= n else 0
        if k > 0:
            f = np.lib.stride_tricks.sliding_window_view(b, n)[::hop][:k] * win
            rows.append((np.abs(np.fft.rfft(f, axis=1)) ** 2) @ P)
        return k

    for x in _chunks(path, np):
        nf = ~np.isfinite(x)
        if nf.any():
            bad += int(nf.sum())
            x = np.where(nf, 0.0, x)
        if len(x):
            peak = max(peak, float(np.abs(x).max()))
        total += len(x)
        buf = np.concatenate([buf, x])
        buf = buf[frames_of(buf) * hop:]
    if total < n:                                         # shorter than one frame: padded, as before
        frames_of(np.pad(buf, (0, n - len(buf))))
    B = np.concatenate(rows) if rows else np.zeros((1, P.shape[1]))
    e = B[:, I_E]
    seconds = round(total / SR, 3 if total < SR else 2)
    silent = peak <= SILENT or not e.max() > 0
    if silent:
        out = {"bands": [-120.0] * NB, "tilt": 0.0, "regions": {k: 0.0 for k in REGIONS}, "sib": 0.0,
               "harsh": 0.0, "harsh_s": 0.0, "seconds": seconds, "peak": round(float(20 * np.log10(peak + 1e-9)), 1),
               "lufs": -99.0, "true_peak": -150.0, "silent": True}
    else:
        keep = e > e.max() * 10 ** (-40 / 10)             # frames within 40 dB of the loudest
        mean = B[keep].mean(axis=0)
        bands = 10 * np.log10(mean[:NB] + 1e-20)
        tot = mean[I_TOT] + 1e-20
        region = {k: float(mean[NB + i] / tot) for i, k in enumerate(REGIONS)}
        sel = slice(3, 27)                                # tilt: 63 Hz – 12.5 kHz, weighted by what is there
        w = 10 ** ((bands[sel] - bands[sel].max()) / 20) + 0.05
        tilt = float(np.polyfit(np.log2(np.array(CENTRES[sel]) / 1000), bands[sel], 1, w=w)[0])
        sibr = B[keep, I_SIB] / (e[keep] + 1e-20)
        loud = e > e.max() * 10 ** (-20 / 10)
        lufs, tp = loudness(path)
        out = {"bands": [round(float(b - bands.max()), 1) for b in bands], "tilt": round(tilt, 2),
               "regions": {k: round(v, 4) for k, v in region.items()},
               "sib": round(float(np.percentile(sibr, 95)) if len(sibr) else 0.0, 3),
               "harsh": round(region["presence"] + region["sibilance"] + region["air"], 4),
               "harsh_s": round(float(((B[:, I_HF] / (e + 1e-20) > 0.5) & loud).sum() * hop / SR), 2),
               "seconds": seconds, "peak": round(float(20 * np.log10(peak + 1e-9)), 1),
               "lufs": round(lufs, 1), "true_peak": round(tp, 1)}
    if bad:
        out["nonfinite"] = bad
    if sonogram:                                         # 40 ms columns x the bands 63 Hz – 16 kHz, 0..255
        step = max(1, int(round(0.04 * SR / hop)))
        g = B[:, 3:NB].T
        pad = (-g.shape[1]) % step
        if pad:
            g = np.pad(g, ((0, 0), (0, pad)), mode="edge")
        g = g.reshape(len(g), -1, step).mean(axis=2)[:, :250]
        if silent:
            gd = np.zeros_like(g)
        else:
            gd = np.clip((10 * np.log10(g + 1e-20) - 10 * np.log10(g.max() + 1e-20) + 60) / 60, 0, 1)
        out["sonogram"] = {"rows": int(gd.shape[0]), "cols": int(gd.shape[1]),
                           "data": (gd * 255).astype(np.uint8).T.flatten().tolist()}
    return out


def chain(a, kind="sfx"):
    """(ffmpeg filters, what was done) for one sound's analysis."""
    speech = kind == "speech"
    f = ["highpass=f=30"]
    if a.get("silent"):                                   # nothing to shape: written as it is
        return ["anull"], {"lowshelf": 0, "highshelf": 0, "peak3k": 0, "deess": 0, "lowpass": 0}
    ex = a["tilt"] - TARGET_TILT
    b = dict(zip(CENTRES, a["bands"]))
    has_low = max(b[100], b[125], b[160], b[200]) > -30   # something down there to lift
    hs = ls = pk = de = 0
    if speech and ex < 1.5:
        ex = 0                                            # voices: left alone unless bright
    if ex > 0.5:
        hs = -min(12.0, round(ex * 1.6, 1))
        if has_low:
            ls = min(5.0, round(ex * 0.7, 1))
    elif ex < -3.0 and not speech:                        # a dull one: a touch of air back
        hs = min(3.0, round(-ex * 0.5, 1))
    if a["regions"]["presence"] > 0.30:                   # 2.5–5 kHz piling up
        pk = -min(6.0, round((a["regions"]["presence"] - 0.25) * 20, 1))
    if ls:
        f.append("lowshelf=f=180:g=%g:t=s:w=0.7" % ls)
    if hs:
        f.append("highshelf=f=3000:g=%g:t=s:w=0.7" % hs)
    if pk:
        f.append("equalizer=f=3200:t=o:w=1:g=%g" % pk)
    if (a["sib"] > 0.35 and not speech) or a["sib"] > 0.5:
        de = 0.6 if a["sib"] > 0.6 else 0.4
        f.append("deesser=i=%g:m=0.5:f=0.5:s=o" % de)
    if not speech:
        f.append("lowpass=f=12000:p=2")
    return f, {"lowshelf": ls, "highshelf": hs, "peak3k": pk, "deess": de, "lowpass": 0 if speech else 12000}


def _info(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels",
                        "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip().split(",")
    return (r[0], r[1]) if len(r) >= 2 and r[0] else ("44100", "2")


def _sample_codec(path, ext):
    """ffmpeg output options that keep src's sample format when writing ext (u8 stays u8, 24-bit stays
    24-bit, a 16-bit FLAC stays 16-bit), and whether src is format-locked (8-bit or under 16 kHz)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                        "stream=codec_name,sample_fmt,sample_rate,bits_per_sample,bits_per_raw_sample",
                        "-of", "default=nw=1", path], capture_output=True, text=True).stdout
    s = dict(ln.split("=", 1) for ln in r.splitlines() if "=" in ln)
    codec, fmt = s.get("codec_name", ""), s.get("sample_fmt", "")
    bits = next((int(s[k]) for k in ("bits_per_raw_sample", "bits_per_sample") if s.get(k, "").isdigit()
                 and int(s[k]) > 0), 0)
    try:
        rate = int(s.get("sample_rate", SR))
    except ValueError:
        rate = SR
    locked = codec in ("pcm_u8", "pcm_s8") or bits and bits <= 8 or rate < 16000
    if ext == ".wav":
        pcm = {"pcm_u8": "pcm_u8", "pcm_s8": "pcm_u8", "pcm_s24le": "pcm_s24le", "pcm_s24be": "pcm_s24le",
               "pcm_s32le": "pcm_s32le", "pcm_f32le": "pcm_f32le", "pcm_f32be": "pcm_f32le",
               "pcm_f64le": "pcm_f64le"}.get(codec)
        if not pcm and codec in ("flac", "alac", "pcm_s24le") and bits == 24:
            pcm = "pcm_s24le"
        return ["-c:a", pcm or "pcm_s16le"], locked
    if ext == ".flac":
        return ["-c:a", "flac", "-sample_fmt", "s32" if bits > 16 else "s16"]\
            + (["-bits_per_raw_sample", "24"] if bits > 16 else []), locked
    return [], locked


def _write_gain(src, dst, g, sr, ch, codec, ext=".wav"):
    """src (float WAV) at g dB into dst, then MEASURED: if the written file — a lossy encode can
    overshoot between samples — lands over the ceiling, written again that much quieter. -> (g, dBTP)."""
    for _ in range(4):
        part = _tmp(dst, ext)
        try:
            _ffmpeg(["-i", src, "-af", "volume=%.3fdB" % g, "-ar", sr, "-ac", ch] + codec + [part])
            tp = true_peak(part)
            if tp <= CEILING - 0.02:
                os.replace(part, dst)
                return g, tp
            g -= tp - (CEILING - MARGIN)
        finally:
            if os.path.exists(part):
                os.remove(part)
    raise RuntimeError(f"couldn't get it under {CEILING:g} dBTP (last {tp:+.2f})")


def master(src, dst, kind="sfx", a=None):
    """Master src into dst (a 16-bit WAV at src's rate and channels). Returns what was done."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    a = a or analyse(src)
    f, what = chain(a, kind)
    sr, ch = _info(src)
    tmp = _tmp(dst, ".wav")
    try:
        _ffmpeg(["-i", src, "-af", ",".join(f), "-ar", sr, "-ac", ch, "-c:a", "pcm_f32le", tmp])
        m0, (m1, _), tp1 = loudness(src)[0], loudness(tmp), true_peak(tmp)
        # back towards how loud it was (at most +3 dB), never past its headroom
        g = min(m0 - m1, CEILING - MARGIN - tp1, 3.0)
        why = "loudness" if g == m0 - m1 else "headroom" if g == CEILING - MARGIN - tp1 else "+3 dB cap"
        if a["regions"]["air"] > 0.4 and g > 0:
            g, why = 0.0, "bright"                        # a whistle up top: the low-pass took the harshness
        if a.get("silent"):
            g, why = 0.0, "silent"
        g, tp = _write_gain(tmp, dst, g, sr, ch, ["-c:a", "pcm_s16le"])
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    what["gain"] = round(g, 1)
    what["kind"] = kind
    what["true_peak"] = tp
    what["limit"] = why                                   # what decided the gain
    return what


def match_tone(src, dst, ref, a=None, r=None):
    """EQ src toward ref's spectrum and bring it to ref's loudness — pixelmon's colour match, for sound.

    Both 1/3-octave curves are taken relative to their own 125 Hz–8 kHz average (so it is the SHAPE
    that is matched, not the level), the difference is smoothed over neighbouring bands, clamped to
    ±9 dB and eased at the extremes, and only applied where src has something to boost. Then the
    gain goes to ref's loudest 400 ms, never past -1 dBTP. dst may be src (written via a temp file).

    The result is sample-exact: as long as src, aligned with it (firequalizer's linear-phase delay is
    trimmed off), in src's sample format (u8 stays u8, 24-bit stays 24-bit) — so a --loop stays
    seamless. A format-locked src (8-bit, or under 16 kHz: --format amiga/sb/nes/gameboy…) gets the
    level only, and only ever quieter: re-EQ'ing and boosting it would smooth and magnify the crush.
    """
    a = a or analyse(src)
    r = r or analyse(ref)
    if r.get("silent"):
        raise ValueError(f"the reference {os.path.basename(ref)} is silent")
    ext = os.path.splitext(dst)[1].lower() or ".wav"
    codec, locked = _sample_codec(src, ext)
    mid = [i for i, c in enumerate(CENTRES) if 125 <= c <= 8000]
    ma = sum(a["bands"][i] for i in mid) / len(mid)
    mr = sum(r["bands"][i] for i in mid) / len(mid)
    d = [(r["bands"][i] - mr) - (a["bands"][i] - ma) for i in range(len(CENTRES))]
    sm = [(d[max(0, i - 1)] + 2 * d[i] + d[min(len(d) - 1, i + 1)]) / 4 for i in range(len(d))]
    eq = []
    for i, c in enumerate(CENTRES):
        gdb = max(-9.0, min(9.0, sm[i]))
        if c < 60 or c > 14000:
            gdb *= 0.5
        if a["bands"][i] < -60 and gdb > 0:               # nothing there to lift: boosting only adds hiss
            gdb = 0.0
        eq.append(0.0 if locked or a.get("silent") else round(gdb, 1))
    entries = ";".join(f"entry({c:g},{g:g})" for c, g in zip(CENTRES, eq))
    sr, ch = _info(src)
    plain, filt, tmp = _tmp(dst, ".wav"), _tmp(dst, ".wav"), _tmp(dst, ".wav")
    try:
        _ffmpeg(["-i", src, "-ar", sr, "-ac", ch, "-c:a", "pcm_f32le", plain])
        n = _frames(plain)
        if any(eq):
            # firequalizer is linear-phase: it delays everything by its `delay` (10 ms) and adds as much
            # again at the end. Trim both off, so every sample stays where it was.
            _ffmpeg(["-i", plain, "-af", f"firequalizer=gain_entry='{entries}'", "-c:a", "pcm_f32le", filt])
            m, lag = _frames(filt), 0
            if n is not None and m is not None and m >= n:
                lag = (m - n) // 2
            _ffmpeg(["-i", filt, "-af", f"atrim=start_sample={lag}:end_sample={lag + (n or 0)},asetpts=N/SR/TB"
                     if n else "anull", "-c:a", "pcm_f32le", tmp])
        else:
            os.replace(plain, tmp)
        if n is not None and _frames(tmp) != n:
            raise RuntimeError(f"length changed ({n} -> {_frames(tmp)} samples)")
        m1, tp1 = loudness(tmp)[0], true_peak(tmp)
        g = min(r["lufs"] - m1, CEILING - MARGIN - tp1, 0.0 if locked else 12.0)
        if a.get("silent"):
            g = 0.0
        out = _tmp(dst, ext)
        try:
            osr = sr if ext != ".opus" or sr in ("8000", "12000", "16000", "24000", "48000") else "48000"
            g, tp = _write_gain(tmp, out, g, osr, ch, codec, ext)          # (Opus has its own rates)
            os.replace(out, dst)
        finally:
            if os.path.exists(out):
                os.remove(out)
    finally:
        for f in (plain, filt, tmp):
            if os.path.exists(f):
                os.remove(f)
    return {"eq": eq, "gain": round(g, 1), "kind": "like", "true_peak": tp, "level_only": bool(locked)}


SPEECH = re.compile(r"(speech|narration|voices?|vo|dialog(ue)?)")
SFXDIR = re.compile(r"(sfx|fx|sounds?|effects?|foley)")


def guess_kind(path):
    """speech for files in voice folders (speech/, narration/, voice…), else sfx. The NEAREST folder
    that says either decides: voice/proj/sfx/impact.wav is sfx, game/speech/line.wav is speech."""
    parts = [p.lower() for p in path.replace("\\", "/").split("/") if p not in ("", ".")][:-1]
    for p in reversed(parts):
        if SPEECH.fullmatch(p):
            return "speech"
        if SFXDIR.fullmatch(p):
            return "sfx"
    return "sfx"


def _walk(root):
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in SKIP_DIRS and not d.startswith(("_", ".")))
        for f in sorted(fns):
            if f.lower().endswith(AUDIO_EXT) and not f.startswith("."):
                yield os.path.join(dp, f)


def _plan(files, base, out):
    """Each source's output path. Same name, different extension (hit.wav + hit.flac) would land on
    one hit.wav: then the .wav keeps the plain name and the others keep their extension in it
    (hit.flac.wav). Any clash left (case-insensitive filesystems) gets -2, -3…"""
    rels = [os.path.relpath(f, base) for f in files]
    groups = {}
    for i, rel in enumerate(rels):
        groups.setdefault(os.path.splitext(rel)[0].lower(), []).append(i)
    names = [None] * len(files)
    for idx in groups.values():
        plain = next((i for i in idx if os.path.splitext(rels[i])[1] == ".wav"), None)
        if plain is None:
            plain = next((i for i in idx if os.path.splitext(rels[i])[1].lower() == ".wav"), None)
        for i in idx:
            stem, ext = os.path.splitext(rels[i])
            names[i] = stem + ".wav" if (i == plain or len(idx) == 1) else rels[i] + ".wav"
    taken, dsts = set(), []
    for nm in names:
        stem, k, cand = os.path.splitext(nm)[0], 1, nm
        while cand.lower() in taken:
            k += 1
            cand = f"{stem}-{k}.wav"
        taken.add(cand.lower())
        dsts.append(os.path.join(out, cand))
    return list(zip(files, rels, dsts))


def _same(path, ids):
    try:
        st = os.stat(path)
    except OSError:
        return False
    return (st.st_dev, st.st_ino) in ids


def _encode_guarded(dst, a, to_ogg, to_flac):
    """--flac / --ogg after mastering. FLAC is lossless; an OGG is measured, and if the encode put it
    over the ceiling the WAV goes in again that much quieter (as soundmon-gui's export does)."""
    if getattr(a, "flac", False) and to_flac:
        return to_flac(dst, keep=a.keep_wav)
    if not (getattr(a, "ogg", False) and to_ogg):
        return dst
    o = dst
    for _ in range(3):
        o = to_ogg(dst, a.ogg_quality, keep=True)
        if o == dst:
            return dst                                    # encode failed: the WAV stays
        tp = true_peak(o)
        if tp <= CEILING - 0.02:
            break
        tmp = _tmp(dst, ".wav")
        try:
            _ffmpeg(["-i", dst, "-af", "volume=%.3fdB" % -(tp - (CEILING - MARGIN)), "-c:a", "pcm_s16le", tmp])
            os.replace(tmp, dst)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    if not a.keep_wav and os.path.exists(dst):
        os.remove(dst)
    return o


def run(a, slug, to_ogg=None, loudness_normalize=None, to_flac=None):
    """soundmon --master FILE|DIR [--master-kind sfx|speech] [--output-to DIR] [--flac|--ogg]"""
    _need_ffmpeg()
    try:
        import numpy  # noqa: F401
    except ImportError:
        sys.exit("--master needs numpy (run soundmon with ComfyUI's venv)")
    src = os.path.expanduser(a.master)
    is_dir = os.path.isdir(src)
    if is_dir:
        files, base = list(_walk(src)), src
        root_name = os.path.basename(os.path.abspath(src))
    elif os.path.isfile(src):
        files, base = [src], os.path.dirname(os.path.abspath(src))
    else:
        sys.exit(f"--master: no such file or folder: {a.master}")
    if not files:
        sys.exit(f"--master: no audio files in {a.master} ({', '.join(e[1:] for e in AUDIO_EXT)})")
    like = os.path.expanduser(a.master_like) if getattr(a, "master_like", None) else None
    ref_a = None
    if like:
        if not os.path.isfile(like):
            sys.exit(f"--master-like: no such file: {a.master_like}")
        try:
            ref_a = analyse(like)
        except Exception as e:
            sys.exit(f"--master-like: {e}")
        if ref_a.get("silent"):
            sys.exit(f"--master-like: {a.master_like} is silent — nothing to match")
    default_out = os.path.join(base, "mastered")
    out = os.path.expanduser(a.output_to) if a.output_to else default_out
    rb, ro = os.path.realpath(base), os.path.realpath(out)
    if ro == rb:
        sys.exit("--master never overwrites its sources: pick a different --output-to")
    if is_dir and ro.startswith(rb.rstrip(os.sep) + os.sep) and ro != os.path.realpath(default_out):
        sys.exit(f"--master: --output-to {a.output_to} is inside {a.master}, the folder being mastered — "
                 f"it could overwrite sources and would be mastered again next time. Pick a folder outside it, "
                 f"or leave --output-to off for {os.path.join(a.master, 'mastered')}")
    ids = set()
    for f in files:
        try:
            st = os.stat(f)
            ids.add((st.st_dev, st.st_ino))
        except OSError:
            pass
    plan = _plan(files, base, out)
    print(f"🎛  mastering {len(files)} sound(s) -> {out}" + (f"  (like {os.path.basename(like)})" if like else ""))
    done, failed = 0, []
    for i, (f, rel, dst) in enumerate(plan, 1):
        shown = rel if os.path.basename(dst) == os.path.splitext(os.path.basename(rel))[0] + ".wav" \
            else f"{rel} -> {os.path.relpath(dst, out)}"
        if a.master_kind != "auto":
            kind = a.master_kind
        else:
            kind = guess_kind(os.path.join(root_name, rel) if is_dir else
                              (f if os.path.dirname(a.master) else os.path.join(os.path.basename(base), rel)))
        finals = [dst] + [os.path.splitext(dst)[0] + e for e, on in ((".flac", getattr(a, "flac", False)),
                                                                     (".ogg", getattr(a, "ogg", False))) if on]
        if any(_same(p, ids) for p in finals):
            print(f"   ⚠ {rel}: would overwrite a source file ({next(p for p in finals if _same(p, ids))}) — skipped")
            failed.append(rel)
            continue
        try:
            before = analyse(f)
            if before.get("nonfinite"):
                raise ValueError(f"{before['nonfinite']} NaN/inf samples — the file is damaged; fix or re-render it")
            if like:
                kind = "like"
                what = match_tone(f, dst, like, before, ref_a)
            else:
                what = master(f, dst, kind, before)
            after = analyse(dst)
            final = _encode_guarded(dst, a, to_ogg, to_flac)
            tp = true_peak(final) if final != dst else what["true_peak"]
        except Exception as e:
            print(f"   ⚠ {rel}: {e}")
            failed.append(rel)
            continue
        done += 1
        if before.get("silent"):
            print(f"   ✅ [{i}/{len(files)}] {shown:<32} silent — written as it is")
            continue
        drop = after["lufs"] - before["lufs"]
        note = {"bright": "  (bright: no make-up gain, by design)", "+3 dB cap": "  (make-up capped at +3 dB, by design)",
                "headroom": "  (kept under -1 dBTP)"}.get(what.get("limit"), "") if drop < -1 else ""
        print(f"   ✅ [{i}/{len(files)}] {shown:<32} {kind:<6} above 2.5k {before['harsh']:.0%}→{after['harsh']:.0%}  "
              f"harsh {before['harsh_s']:.2f}→{after['harsh_s']:.2f}s  loudest {before['lufs']:.1f}→{after['lufs']:.1f} LUFS  "
              f"gain {what['gain']:+.1f} dB  peak {tp:.1f} dBTP{note}")
    print(f"   all done  |  {done} file(s) in {out}" + (f"  |  {len(failed)} failed" if failed else ""))
    if failed:
        sys.exit(f"--master: {len(failed)} file(s) failed")
