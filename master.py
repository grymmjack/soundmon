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

Voices ("speech") are left alone unless they are bright, and keep their air. On NEON-CASTER this
took the sound effects' energy above 2.5 kHz from 34 % to 21 % and their harsh seconds from 64 to
40, while voices barely changed. Originally neon-caster/sfx/mastering/{analyze,master}.py.

Needs numpy and ffmpeg. Imports nothing heavy at module level (soundmon-gui imports this).
"""
import os
import re
import subprocess
import sys

SR = 44100
CENTRES = [31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
           2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000]
REGIONS = {"sub": (20, 80), "low": (80, 250), "lowmid": (250, 1000), "mid": (1000, 2500),
           "presence": (2500, 5000), "sibilance": (5000, 10000), "air": (10000, 20000)}
TARGET_TILT = -3.0          # dB per octave in 1/3-octave band power
AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3", ".opus")
KINDS = ("sfx", "speech")


def _need_ffmpeg():
    from shutil import which
    if not which("ffmpeg"):
        sys.exit("--master needs ffmpeg on PATH")


def load(path, np):
    """Mono float64 at 44.1 kHz, whatever the file was."""
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def loudness(path):
    """(its loudest 400 ms in LUFS, its true peak in dBTP) — ffmpeg's EBU R128 scanner."""
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "verbose", "-i", path, "-af",
                        "apad=whole_dur=0.5,ebur128=framelog=verbose:peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    m = [float(x) for x in re.findall(r"M:\s*(-?[\d.]+)", r)]
    tp = re.findall(r"Peak:\s*(-?[\d.]+) dBFS", r)
    return (max(m) if m else -99.0), (float(tp[-1]) if tp else 0.0)


def analyse(path, sonogram=False):
    """What a sound is made of: 1/3-octave spectrum (silence ignored), tilt in dB/octave, share of
    energy per region, sibilant bursts, harsh seconds (loud and mostly above 2.5 kHz), loudness."""
    import numpy as np
    x = load(path, np)
    n, hop = 2048, 512
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    fr = np.fft.rfftfreq(n, 1 / SR)
    e = spec.sum(axis=1)
    keep = e > e.max() * 10 ** (-40 / 10)                 # frames within 40 dB of the loudest
    mean = spec[keep].mean(axis=0) + 1e-20
    bands = np.array([10 * np.log10(mean[(fr >= c / 2 ** (1 / 6)) & (fr < c * 2 ** (1 / 6))].sum() + 1e-20) for c in CENTRES])
    tot = mean[fr >= 20].sum()
    region = {k: float(mean[(fr >= lo) & (fr < hi)].sum() / tot) for k, (lo, hi) in REGIONS.items()}
    sel = slice(3, 27)                                   # tilt: 63 Hz – 12.5 kHz, weighted by what is there
    w = 10 ** ((bands[sel] - bands[sel].max()) / 20) + 0.05
    tilt = float(np.polyfit(np.log2(np.array(CENTRES[sel]) / 1000), bands[sel], 1, w=w)[0])
    sibr = spec[:, (fr >= 5000) & (fr < 9000)].sum(axis=1)[keep] / (e[keep] + 1e-20)
    hfe = spec[:, fr >= 2500].sum(axis=1)
    loud = e > e.max() * 10 ** (-20 / 10)
    lufs, tp = loudness(path)
    out = {"bands": [round(float(b - bands.max()), 1) for b in bands], "tilt": round(tilt, 2),
           "regions": {k: round(v, 4) for k, v in region.items()},
           "sib": round(float(np.percentile(sibr, 95)) if len(sibr) else 0.0, 3),
           "harsh": round(region["presence"] + region["sibilance"] + region["air"], 4),
           "harsh_s": round(float(((hfe / (e + 1e-20) > 0.5) & loud).sum() * hop / SR), 2),
           "seconds": round(len(x) / SR, 2), "peak": round(float(20 * np.log10(np.abs(x).max() + 1e-9)), 1),
           "lufs": round(lufs, 1), "true_peak": round(tp, 1)}
    if sonogram:                                         # 40 ms columns x the bands 63 Hz – 16 kHz, 0..255
        step = max(1, int(round(0.04 * SR / hop)))
        g = np.array([spec[:, (fr >= c / 2 ** (1 / 6)) & (fr < c * 2 ** (1 / 6))].sum(axis=1) for c in CENTRES[3:]])
        pad = (-g.shape[1]) % step
        if pad:
            g = np.pad(g, ((0, 0), (0, pad)), mode="edge")
        g = g.reshape(len(g), -1, step).mean(axis=2)[:, :250]
        gd = np.clip((10 * np.log10(g + 1e-20) - 10 * np.log10(g.max() + 1e-20) + 60) / 60, 0, 1)
        out["sonogram"] = {"rows": int(gd.shape[0]), "cols": int(gd.shape[1]),
                           "data": (gd * 255).astype(np.uint8).T.flatten().tolist()}
    return out


def chain(a, kind="sfx"):
    """(ffmpeg filters, what was done) for one sound's analysis."""
    speech = kind == "speech"
    f = ["highpass=f=30"]
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


def master(src, dst, kind="sfx", a=None):
    """Master src into dst (a 16-bit WAV at src's rate and channels). Returns what was done."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    a = a or analyse(src)
    f, what = chain(a, kind)
    sr, ch = _info(src)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = dst + ".tmp.wav"
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-af", ",".join(f),
                        "-ar", sr, "-ac", ch, "-c:a", "pcm_f32le", tmp], check=True, capture_output=True)
        (m0, _), (m1, tp1) = loudness(src), loudness(tmp)
        g = min(m0 - m1, -1.0 - tp1, 3.0)                 # back towards how loud it was, never past its headroom
        if a["regions"]["air"] > 0.4:
            g = min(g, 0.0)                               # a whistle up top: the low-pass took the harshness
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", tmp, "-af", "volume=%gdB" % g,
                        "-ar", sr, "-ac", ch, "-c:a", "pcm_s16le", dst], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg: " + (e.stderr or b"").decode(errors="replace").strip()[-300:])
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    what["gain"] = round(g, 1)
    what["kind"] = kind
    return what


def match_tone(src, dst, ref, a=None, r=None):
    """EQ src toward ref's spectrum and bring it to ref's loudness — pixelmon's colour match, for sound.

    Both 1/3-octave curves are taken relative to their own 125 Hz–8 kHz average (so it is the SHAPE
    that is matched, not the level), the difference is smoothed over neighbouring bands, clamped to
    ±9 dB and eased at the extremes, and only applied where src has something to boost. Then the
    gain goes to ref's loudest 400 ms, never past -1 dBTP. dst may be src (written via a temp file).
    """
    a = a or analyse(src)
    r = r or analyse(ref)
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
        eq.append(round(gdb, 1))
    entries = ";".join(f"entry({c:g},{g:g})" for c, g in zip(CENTRES, eq))
    sr, ch = _info(src)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    ext = os.path.splitext(dst)[1].lower() or ".wav"
    tmp = dst + ".tmp.wav"
    out = dst + ".out" + ext
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-af", f"firequalizer=gain_entry='{entries}'",
                        "-ar", sr, "-ac", ch, "-c:a", "pcm_f32le", tmp], check=True, capture_output=True)
        m1, tp1 = loudness(tmp)
        g = min(r["lufs"] - m1, -1.0 - tp1, 12.0)
        codec = ["-c:a", "pcm_s16le"] if ext == ".wav" else []
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", tmp, "-af", "volume=%gdB" % g,
                        "-ar", sr, "-ac", ch] + codec + [out], check=True, capture_output=True)
        os.replace(out, dst)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("ffmpeg: " + (e.stderr or b"").decode(errors="replace").strip()[-300:])
    finally:
        for f in (tmp, out):
            if os.path.exists(f):
                os.remove(f)
    return {"eq": eq, "gain": round(g, 1), "kind": "like"}


def guess_kind(path):
    """speech for voice folders and files (speech/, narration/, voice…), else sfx."""
    p = path.replace("\\", "/").lower()
    return "speech" if re.search(r"(^|/)(speech|narration|voices?|vo|dialog(ue)?)(/|$)", p) else "sfx"


def run(a, slug, to_ogg=None, loudness_normalize=None, to_flac=None):
    """soundmon --master FILE|DIR [--master-kind sfx|speech] [--output-to DIR] [--flac|--ogg]"""
    _need_ffmpeg()
    try:
        import numpy  # noqa: F401
    except ImportError:
        sys.exit("--master needs numpy (run soundmon with ComfyUI's venv)")
    src = os.path.expanduser(a.master)
    if os.path.isdir(src):
        files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(src) for f in fs
                       if f.lower().endswith(AUDIO_EXT) and "mastered" not in dp.split(os.sep))
        base = src
    elif os.path.isfile(src):
        files, base = [src], os.path.dirname(os.path.abspath(src))
    else:
        sys.exit(f"--master: no such file or folder: {a.master}")
    if not files:
        sys.exit(f"--master: no audio files in {a.master}")
    like = os.path.expanduser(a.master_like) if getattr(a, "master_like", None) else None
    if like and not os.path.isfile(like):
        sys.exit(f"--master-like: no such file: {a.master_like}")
    ref_a = analyse(like) if like else None
    out = os.path.expanduser(a.output_to) if a.output_to else os.path.join(base, "mastered")
    if os.path.realpath(out) == os.path.realpath(base):
        sys.exit("--master never overwrites its sources: pick a different --output-to")
    print(f"🎛  mastering {len(files)} sound(s) -> {out}" + (f"  (like {os.path.basename(like)})" if like else ""))
    done = 0
    for i, f in enumerate(files, 1):
        rel = os.path.relpath(f, base)
        kind = a.master_kind if a.master_kind != "auto" else guess_kind(rel if os.path.isdir(src) else f)
        dst = os.path.join(out, os.path.splitext(rel)[0] + ".wav")
        try:
            before = analyse(f)
            if like:
                kind = "like"
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                what = match_tone(f, dst, like, before, ref_a)
            else:
                what = master(f, dst, kind, before)
            after = analyse(dst)
        except Exception as e:
            print(f"   ⚠ {rel}: {e}")
            continue
        if a.flac and to_flac:
            dst = to_flac(dst, keep=a.keep_wav)
        elif a.ogg and to_ogg:
            dst = to_ogg(dst, a.ogg_quality, keep=a.keep_wav)
        done += 1
        print(f"   ✅ [{i}/{len(files)}] {rel:<32} {kind:<6} above 2.5k {before['harsh']:.0%}→{after['harsh']:.0%}  "
              f"harsh {before['harsh_s']:.2f}→{after['harsh_s']:.2f}s  gain {what['gain']:+.1f} dB  "
              f"peak {after['true_peak']:.1f} dBTP")
    print(f"   all done  |  {done} file(s) in {out}")
    if done < len(files):
        sys.exit(f"--master: {len(files) - done} file(s) failed")
