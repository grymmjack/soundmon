#!/usr/bin/env python3
"""Transcribe a reference recording into note events, to be played on a chip.

THE IDEA

`--chip` and `--opl` synthesize authentically but compose procedurally, and
procedural composition is where the craftsmanship runs out. Stable Audio 3 is the
opposite: it cannot honour a channel count, but it *can* write music — real
phrasing, real harmonic motion, real rhythmic interest.

So use each engine for what it is good at. Take the musical CONTENT from a
recording and play it through the chip:

    reference.ogg  ->  tempo, key, chords, melody, bass, drums  ->  2A03 / OPL3

This is a cover version, not a remix. The arrangement is SA3's; the instrument is
a Yamaha YMF262. And diversity comes free — 24 different reference tracks give 24
genuinely different chip arrangements, because a human-recognisable composition is
doing the varying rather than a hash function.

WHAT THIS IS NOT

Not a full audio-to-MIDI transcriber. It extracts a monophonic melody, a bass
line, a chord per bar and a drum pattern — which is exactly a 4-channel chip
arrangement and no more. Trying to recover every voice would be both harder and
useless: there are only four channels to play it on.

METHOD (numpy + scipy only, no librosa)

    onset envelope   spectral flux over an STFT
    tempo            the tatum in the onset autocorrelation, sub-frame accurate,
                     beat = tatum or tatum/2 by accent, 50-200 BPM
    beat phase       cross-correlation of the envelope against a pulse train
    chroma           STFT bins folded to 12 pitch classes, per beat
    key              chroma (+ bass line) against Krumhansl-Kessler profiles;
                     the first bass note settles relative major vs minor
    not music?       no tatum, or no key  ->  None, not an invented answer
    chords           per-bar chroma matched to triad templates within the key
    melody / bass    strongest pitch per beat in a high / low band
    drums            onsets binned by spectral centroid into kick/snare/hat
"""
import os
import sys

# Krumhansl-Kessler key profiles: how strongly each pitch class belongs to a
# major or minor key. Correlating a chroma vector against all 24 rotations is the
# standard key-finding method and is far more robust than "loudest note wins".
KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

NOTE_NAMES = "C C# D D# E F F# G G# A A# B".split()


def _load(path, np, sf, target_sr=22050):
    """Mono float audio at <= target_sr. Raises ValueError with a one-line reason.

    libsndfile reads WAV/FLAC/OGG/AIFF; anything else it cannot open (.m4a,
    .mp3 on older builds, .opus) is decoded through ffmpeg when it is installed.
    A missing path, a folder, or a file that is not audio at all is a clear
    error instead of a LibsndfileError traceback.
    """
    p = os.path.expanduser(str(path))
    if not os.path.exists(p):
        raise ValueError("no such file")
    if os.path.isdir(p):
        raise ValueError("is a folder, not an audio file")
    try:
        audio, sr = sf.read(p, always_2d=True)
    except Exception as e:                          # LibsndfileError, RuntimeError
        audio, sr = _ffmpeg_decode(p, np, sf, target_sr, e)
    if audio.size == 0:
        raise ValueError("contains no audio")
    m = np.nan_to_num(np.asarray(audio, dtype=np.float64).mean(axis=1),
                      nan=0.0, posinf=0.0, neginf=0.0)
    if sr > target_sr:                      # cheap decimation; we only need pitch
        k = int(round(sr / target_sr))
        m = m[::k]
        sr = sr // k
    return m.astype(np.float64), sr


def _ffmpeg_decode(p, np, sf, sr, why):
    import io
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        raise ValueError(f"can't read it as audio ({why}); installing ffmpeg adds more formats")
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", p, "-vn", "-ac", "1",
                        "-ar", str(sr), "-f", "wav", "pipe:1"], capture_output=True)
    if r.returncode != 0 or len(r.stdout) < 64:
        msg = (r.stderr.decode("utf-8", "replace").strip().splitlines() or ["unknown format"])[-1]
        raise ValueError(f"not an audio file soundmon can read ({msg[:120]})")
    try:
        return sf.read(io.BytesIO(r.stdout), always_2d=True)
    except Exception as e:
        raise ValueError(f"not an audio file soundmon can read ({e})") from None


def _stft(x, np, n_fft=2048, hop=512):
    win = np.hanning(n_fft)
    frames = 1 + max(0, (len(x) - n_fft) // hop)
    out = np.empty((frames, n_fft // 2 + 1))
    for i in range(frames):
        seg = x[i * hop:i * hop + n_fft]
        out[i] = np.abs(np.fft.rfft(seg * win))
    return out


def _onset_env(S, np):
    """Spectral flux: how much the spectrum GREW frame to frame. Growth marks an
    attack; decay does not, which is why the positive part is kept."""
    d = np.diff(S, axis=0)
    return np.concatenate([[0.0], np.maximum(d, 0.0).sum(axis=1)])


# Tempo, by TATUM. Measured on synthetic loops at 14 known tempos (60-180) and on
# chip renders, the old estimate was coarse and pulled toward 100: the lag was a
# whole STFT frame (114.9-120.2 bpm shared one lag, and the prior then picked the
# one nearest 100), and folding into 70-150 halved everything fast (174 -> 88).
# 4 of 14 were right.
#
# What is stable across all of them is the METRICAL LADDER in the onset
# autocorrelation: bar ~ half-bar > beat > 8th, then a collapse at the 16th. So:
#
#   1. find the FASTEST strong periodicity whose double also recurs — the tatum
#      (the finest level the music really articulates; usually the 8th)
#   2. measure it precisely from a long lag (8x), with parabolic sub-frame
#      interpolation — resolution ~0.1 bpm instead of ~5
#   3. decide beat = tatum or tatum/2 from ACCENTS: if every other tatum is weak
#      (hats between kicks and snares) the tatum is a subdivision of the beat
#   4. keep the beat in 50-200
#
# 20 of 20 references land within 1 bpm, including 60, 160, 174 and 180. The
# accent threshold is a judgement — music with genuinely even 8ths at 60 bpm
# will read as 120 — and an octave is the one error tempo-from-audio cannot
# rule out without a listener.
TATUM_MIN_AC = 0.35         # normalised autocorrelation a tatum must reach
SUBDIVISION_ALT = 0.5       # offbeat/onbeat accent ratio below which tatum = 8th


def _onset_env_fine(x, sr, np, n_fft=1024, hop=256):
    """Log-compressed spectral flux at ~86 frames/s (twice the old resolution)."""
    if len(x) < n_fft * 2:
        return np.zeros(0), sr / float(hop)
    S = _stft(x, np, n_fft, hop)
    S = np.log1p(100.0 * S / (S.max() + 1e-12))
    d = np.maximum(np.diff(S, axis=0), 0.0).sum(axis=1)
    return np.concatenate([[0.0], d]), sr / float(hop)


def _acf(e, np):
    """Unbiased, normalised autocorrelation (via FFT)."""
    e = e - e.mean()
    n = len(e)
    f = np.fft.rfft(e, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / np.arange(n, 0, -1)
    return ac / (ac[0] + 1e-12) if ac[0] > 0 else ac * 0.0


def _parabolic(ac, j):
    if 0 < j < len(ac) - 1:
        d = ac[j - 1] - 2 * ac[j] + ac[j + 1]
        if d < 0:
            return j + 0.5 * (ac[j - 1] - ac[j + 1]) / d
    return float(j)


def _accent_alt(e, period, np):
    """Mean onset strength at odd tatum positions / at even ones, with the phase
    chosen so the even ones are the strong ones."""
    best, bo = -1.0, 0.0
    for off in np.arange(0.0, 2 * period, 0.5):
        idx = np.round(np.arange(off, len(e) - 1, 2 * period)).astype(int)
        v = float(e[idx].sum())
        if v > best:
            best, bo = v, off
    pk = lambda ii: np.array([e[max(0, i - 1):i + 2].max() for i in ii])   # noqa: E731
    on = pk(np.round(np.arange(bo, len(e) - 1, 2 * period)).astype(int))
    off = pk(np.round(np.arange(bo + period, len(e) - 1, 2 * period)).astype(int))
    if not len(on) or not len(off):
        return 1.0
    return float(off.mean() / (on.mean() + 1e-12))


def estimate_tempo(x, sr, np):
    """-> (bpm, clarity), or (None, 0.0) when nothing recurs (noise, a sweep,
    random hits, silence). `clarity` is the tatum's normalised autocorrelation."""
    e, fps = _onset_env_fine(x, sr, np)
    if len(e) < 64 or e.max() <= 0:
        return None, 0.0
    ac = _acf(e, np)
    lo, hi = fps * 60.0 / 600.0, fps * 60.0 / 80.0          # tatum 80..600 /min
    pk = []
    for i in range(max(2, int(lo)), min(len(ac) - 1, int(hi) + 1)):
        if ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] >= TATUM_MIN_AC:
            pk.append((_parabolic(ac, i), float(ac[i])))
    tat = None
    for L, v in sorted(pk):
        j = int(round(2 * L))
        if j + 3 < len(ac) and ac[max(0, j - 2):j + 3].max() >= TATUM_MIN_AC:
            tat, clarity = L, v
            break
    if tat is None:
        return None, 0.0
    for k in (8, 4, 2):                                      # precise period
        Lk = k * tat
        i0, w = int(round(Lk)), max(2, int(0.03 * Lk))
        if i0 - w > 0 and i0 + w + 1 < len(ac) * 0.5:
            j = int(np.argmax(ac[i0 - w:i0 + w + 1])) + i0 - w
            tat = _parabolic(ac, j) / k
            break
    if not (np.isfinite(tat) and tat > 0):
        return None, 0.0
    bpm = fps * 60.0 / tat
    if _accent_alt(e, tat, np) < SUBDIVISION_ALT:
        bpm /= 2.0
    while bpm > 200.0:
        bpm /= 2.0
    while bpm < 50.0:
        bpm *= 2.0
    return float(bpm), float(clarity)


def _beat_phase(env, sr, hop, bpm, np):
    """Where does beat 1 fall? Correlate against a pulse train at this tempo."""
    period = sr / hop * 60.0 / bpm
    n = len(env)
    best, best_v = 0, -1e30
    for off in range(int(round(period))):
        idx = np.arange(off, n, period).astype(int)
        idx = idx[idx < n]
        if len(idx) < 2:
            continue
        v = float(env[idx].sum())
        if v > best_v:
            best, best_v = off, v
    return best, period


def _chroma(S, sr, n_fft, np):
    """Fold spectrum bins into 12 pitch classes, weighted by magnitude."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    keep = (freqs > 55.0) & (freqs < 4000.0)
    mapping = np.array([int(round(12 * np.log2(f / 55.0))) % 12 if k else -1
                        for f, k in zip(freqs, keep)])
    out = np.zeros((S.shape[0], 12))
    for c in range(12):
        cols = np.where(mapping == c)[0]
        if len(cols):
            out[:, c] = S[:, cols].sum(axis=1)
    # A-based folding: shift so index 0 is C, matching NOTE_NAMES.
    return np.roll(out, -3, axis=1)


def _find_key(chroma, np, bass_pcs=None, first_pc=None):
    """Krumhansl-Kessler key, with the TONIC weighed properly.

    Chroma alone cannot separate a minor key from its relative major — they share
    every note, and a loop that cycles i-VI-III-VII has the same pitch content as
    vi-IV-I-V. Measured: 6 of 14 minor loops came back as their relative major.
    The bass says which note is HOME, so a bass-note histogram is scored against
    the same profiles, and the first bass note (where pieces and loops nearly
    always start on the tonic) breaks what is left of a tie.

    Returns (root, mode, conf); conf is the plain chroma correlation, which is
    what says whether this is tonal music at all.
    """
    v = chroma.sum(axis=0)
    if not np.isfinite(v).all() or v.sum() <= 0 or np.ptp(v) <= 0:
        return 0, "minor", -2.0
    v = v / v.sum()
    b = None
    if bass_pcs:
        b = np.bincount(np.asarray(bass_pcs, dtype=int) % 12, minlength=12).astype(np.float64)
        if np.ptp(b) <= 0:
            b = None
    maj = np.array(KK_MAJOR); mnr = np.array(KK_MINOR)
    scores = {}
    for r in range(12):
        rot = np.roll(v, -r)
        for name, prof in (("major", maj), ("minor", mnr)):
            c = float(np.corrcoef(rot, prof)[0, 1])
            score = c
            if b is not None:
                score += 0.5 * float(np.corrcoef(np.roll(b, -r), prof)[0, 1])
            scores[(r, name)] = (score, c)
    (r, name), (score, c) = max(scores.items(), key=lambda kv: kv[1][0])
    # Relative major/minor share every note, so pitch content alone often picks
    # the wrong one of the pair (vi-IV-I-V and i-VI-III-VII are the same notes).
    # Where the music STARTS settles it: if the first bass note is the other
    # one's tonic, and the two are not far apart on content, it is the other.
    rel = ((r + 9) % 12, "minor") if name == "major" else ((r + 3) % 12, "major")
    if first_pc is not None and first_pc % 12 == rel[0] and first_pc % 12 != r \
            and score - scores[rel][0] < REL_SWITCH_GAP:
        r, name = rel
    # Confidence that this is TONAL at all: the best plain-chroma fit of any key.
    # Which key wins is decided above; how tonal the music is does not depend on it.
    return r, name, max(c_ for _s, c_ in scores.values())


def _bass_pcs(S, sr, n_fft, np, beats):
    """Bass pitch class per beat -> (list, first one or None).

    The LOWEST strong peak in 40-200 Hz, with parabolic interpolation: at 22 kHz
    a 2048-point bin is 10.8 Hz, two semitones at E2, so the raw bin lands on
    the wrong note. HPS is no help here — on a sine bass it locks onto the
    chord's harmonics instead (E minor read as G).
    """
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    sel = np.where((freqs >= 40.0) & (freqs <= 200.0))[0]
    out = []
    for i, b in enumerate(beats):
        b2 = beats[i + 1] if i + 1 < len(beats) else S.shape[0]
        frame = S[b:max(b + 1, b2)].sum(axis=0)
        band = frame[sel]
        if not len(band) or band.max() <= 1e-9:
            continue
        j = int(sel[int(np.argmax(band))])
        if 0 < j < len(frame) - 1:
            a_, b_, c_ = np.log(frame[j - 1:j + 2] + 1e-12)
            d = a_ - 2 * b_ + c_
            j = j + (0.5 * (a_ - c_) / d if d < 0 else 0.0)
        f = j * sr / float(n_fft)
        if f > 0:
            out.append(int(round(69 + 12 * np.log2(f / 440.0))) % 12)
    return out, (out[0] if out else None)


# Not music: below this a reference has no tonal centre worth transcribing.
# Measured: tonal loops 0.73-0.95, modal chip music (phrygian, harmonic minor)
# down to 0.54; white noise 0.28, speech-like buzz 0.39. Most non-music never
# gets this far — noise, sweeps, random hits and silence have no tatum at all.
KEY_MIN_CONF = 0.45
REL_SWITCH_GAP = 0.5         # how far apart (score) the relative key may be and still win on the first bass note
SILENCE = 1e-4               # peak below -80 dBFS: nothing there


def _dominant_pitch(S, sr, n_fft, np, lo, hi):
    """Strongest spectral peak in a band, as a MIDI note (or None)."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    sel = (freqs >= lo) & (freqs <= hi)
    if not sel.any():
        return None
    band = S[sel]
    if band.max() <= 1e-9:
        return None
    f = float(freqs[sel][int(np.argmax(band))])
    if f <= 0:
        return None
    return int(round(69 + 12 * np.log2(f / 440.0)))


def _read_music(x, sr, np, n_fft, hop):
    """The shared front end: tempo, beat grid, chroma and key — or None when the
    reference is not music with a pulse and a key.

    Returning None for non-music is the point. Silence used to come back as
    "120 bpm C minor" (key-confidence sentinel -2.0), white noise as 148 bpm,
    a sweep as 148, speech-like buzz as 115 E minor — and --from-audio then
    "transcribed" them: a silent reference rendered a loud arpeggio. Callers
    already handle None ("no steady tempo or key").
    """
    if len(x) < n_fft * 4 or float(np.abs(x).max()) < SILENCE:
        return None
    bpm, clarity = estimate_tempo(x, sr, np)
    if bpm is None:
        return None
    # Pad one window of silence so the STFT reaches the LAST sample. Without it
    # the final ~90 ms had no frame, the grid came up a few steps short of the
    # last bar, and a 4-bar reference transcribed as 3 bars.
    # And one in FRONT: spectral flux needs a frame before the first onset, or
    # beat 1 at sample 0 is invisible and the grid starts a beat late.
    S = _stft(np.concatenate([np.zeros(n_fft), x, np.zeros(n_fft)]), np, n_fft, hop)
    env = _onset_env(S, np)
    off, period = _beat_phase(env, sr, hop, bpm, np)
    chroma = _chroma(S, sr, n_fft, np)
    beats = np.arange(off, S.shape[0], period).astype(int)
    beats = beats[beats < S.shape[0]]
    if len(beats) < 4:
        return None
    bass, first = _bass_pcs(S, sr, n_fft, np, beats)
    root, mode, conf = _find_key(chroma, np, bass, first)
    if conf < KEY_MIN_CONF:
        return None
    return S, env, bpm, clarity, off, period, chroma, beats, root, mode, conf


def analyze(path, np, sf, bars_max=32):
    """Extract a musical summary of a recording. Returns a plain dict."""
    n_fft, hop = 2048, 512
    x, sr = _load(path, np, sf)
    got = _read_music(x, sr, np, n_fft, hop)
    if not got:
        return None
    S, env, bpm, clarity, off, period, chroma, beats, root, mode, conf = got

    per_beat = []
    for i, b in enumerate(beats):
        b2 = beats[i + 1] if i + 1 < len(beats) else S.shape[0]
        seg = S[b:max(b + 1, b2)]
        frame = seg.sum(axis=0)
        per_beat.append({
            "chroma": np.roll(chroma[b:max(b + 1, b2)].sum(axis=0), 0),
            "melody": _dominant_pitch(frame, sr, n_fft, np, 220.0, 1800.0),
            "bass": _dominant_pitch(frame, sr, n_fft, np, 55.0, 220.0),
            "energy": float(env[b:max(b + 1, b2)].sum()),
            "centroid": float((np.fft.rfftfreq(n_fft, 1.0 / sr) * frame).sum()
                              / (frame.sum() + 1e-12)),
        })
    return {"bpm": bpm, "root": root, "mode": mode, "key_conf": conf,
            "clarity": clarity, "beats": per_beat, "duration": len(x) / sr,
            "onset_env": env, "fps": sr / hop, "beat_period": period}


# --- chord estimation --------------------------------------------------------
def _chord_for(chroma_vec, root, scale, np):
    """Best-fitting scale degree for this chroma, by triad template match."""
    best, best_v = 0, -1e30
    v = chroma_vec / (chroma_vec.sum() + 1e-12)
    for deg in range(len(scale)):
        tmpl = np.zeros(12)
        for k in (0, 2, 4):
            tmpl[(root + scale[(deg + k) % len(scale)] +
                  12 * ((deg + k) // len(scale))) % 12] = 1.0
        tmpl /= tmpl.sum()
        s = float(np.dot(v, tmpl))
        if s > best_v:
            best, best_v = deg, s
    return best


def to_events(path, np, sf, steps_per_bar=16, beats_per_bar=4, scale=None,
              seconds=None, quantize=2):
    """Transcribe `path` into the (ev, bars, spb) triple the renderers consume.

    The output format is identical to chip.compose()'s, so both synthesis
    back-ends play a transcription with no changes at all — the renderers never
    knew where their notes came from.
    """
    import theory
    a = analyze(path, np, sf)
    if not a:
        return None
    scale_name = "minor" if a["mode"] == "minor" else "major"
    scale = scale or theory.SCALES[scale_name]
    root = a["root"]

    beats = a["beats"]
    nbars = max(1, len(beats) // beats_per_bar)
    if seconds:
        want = int(round(seconds / (60.0 / a["bpm"] * beats_per_bar)))
        nbars = max(1, min(nbars, want)) if want else nbars
    spb = 60.0 / a["bpm"] * beats_per_bar
    spbeat = steps_per_bar // beats_per_bar

    # Drum classification thresholds from this track's own centroid spread, so a
    # dark track is not silently declared to be all kick.
    cents = np.array([b["centroid"] for b in beats])
    lo_c, hi_c = np.percentile(cents, 33), np.percentile(cents, 66)
    energies = np.array([b["energy"] for b in beats])
    e_thr = np.percentile(energies, 40)

    ev = {"lead": [], "arp": [], "bass": [], "drum": []}
    for bar in range(nbars):
        seg = beats[bar * beats_per_bar:(bar + 1) * beats_per_bar]
        if not seg:
            break
        chroma_bar = np.sum([b["chroma"] for b in seg], axis=0)
        degree = _chord_for(chroma_bar, root, scale, np)
        chord = [12 * 4 + root + scale[(degree + k) % len(scale)] +
                 12 * ((degree + k) // len(scale)) for k in (0, 2, 4)]

        for bi, b in enumerate(seg):
            step = bi * spbeat
            # --- melody: quantize to the scale so it plays cleanly on a chip
            if b["melody"]:
                n = b["melody"]
                pcs = [(root + s) % 12 for s in scale]
                while (n % 12) not in pcs and quantize:
                    n += 1 if (n % 12) < 6 else -1
                ev["lead"].append((bar, step, spbeat, int(n), 0.5))
            # --- bass
            if b["bass"]:
                ev["bass"].append((bar, step, spbeat, int(b["bass"])))
            else:
                ev["bass"].append((bar, step, spbeat, chord[0] - 12))
            # --- arp fills the chord underneath
            for k in range(spbeat // 2 or 1):
                ev["arp"].append((bar, step + k * 2, 2,
                                  chord[(bi + k) % len(chord)], 0.25))
            # --- drums from onset strength + brightness
            if b["energy"] >= e_thr:
                kind = "k" if b["centroid"] <= lo_c else ("h" if b["centroid"] >= hi_c else "s")
                ev["drum"].append((bar, step, kind))
    return ev, nbars, spb, a, scale_name


def describe(a):
    return (f"{NOTE_NAMES[a['root']]} {a['mode']}  {a['bpm']:.1f}bpm  "
            f"{a['duration']:.1f}s  key-confidence {a['key_conf']:.2f}")


# =============================================================================
# HIGH-RESOLUTION TRANSCRIPTION
#
# The first pass sampled ONE note per beat. At 100 bpm that is a 600 ms grid, so
# every melodic and rhythmic detail finer than a quarter note was discarded and
# the result was a slideshow of the original. This does the job properly:
#
#   16th-note grid          4x the resolution, so melodies actually move
#   HPS pitch detection     finds the FUNDAMENTAL, not the loudest harmonic
#   median smoothing        kills isolated octave jumps
#   note segmentation       consecutive equal pitches become one held note
#   per-beat chords         harmony can change inside a bar
#   banded percussion       kick/snare/hat from their own frequency bands
#   dynamics                per-note velocity from onset strength
# =============================================================================

def _hps(frame, sr, n_fft, np, lo=55.0, hi=2000.0, harmonics=5):
    """Harmonic Product Spectrum: find f0 by multiplying downsampled copies.

    A single spectral peak is often the 2nd or 3rd harmonic, which is why the
    first version reported melodies an octave or a fifth high. Multiplying
    decimated copies reinforces the true fundamental, where all the harmonics
    line up, and suppresses everything else.
    """
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    n = len(frame)
    prod = frame.astype(np.float64).copy()
    for h in range(2, harmonics + 1):
        dec = frame[::h]
        prod[:len(dec)] *= dec
    sel = (freqs >= lo) & (freqs <= hi)
    if not sel.any():
        return None
    idx = np.where(sel)[0]
    j = idx[int(np.argmax(prod[idx]))]
    if prod[j] <= 0:
        return None
    f = float(freqs[j])
    return f if f > 0 else None


def _median_smooth(vals, k=5):
    """Remove isolated pitch outliers — the classic octave-jump artifact."""
    out = list(vals)
    half = k // 2
    for i in range(len(vals)):
        win = [v for v in vals[max(0, i - half):i + half + 1] if v is not None]
        if not win:
            out[i] = None
            continue
        win.sort()
        out[i] = win[len(win) // 2]
    return out


def _band_energy(S, sr, n_fft, np, lo, hi):
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    sel = (freqs >= lo) & (freqs < hi)
    return S[:, sel].sum(axis=1) if sel.any() else np.zeros(S.shape[0])


def _segment(pitches, strengths, min_len=1):
    """Group runs of equal pitch into (start, length, pitch, strength) notes.

    Without this, a note held for a bar is re-triggered on every grid step, which
    on a chip sounds like a stutter rather than a sustained note — and is a large
    part of why the first transcriptions sounded mechanical.
    """
    out = []
    i = 0
    n = len(pitches)
    while i < n:
        p = pitches[i]
        if p is None:
            i += 1
            continue
        j = i
        while j + 1 < n and pitches[j + 1] == p:
            j += 1
        ln = j - i + 1
        if ln >= min_len:
            out.append((i, ln, int(p), float(max(strengths[i:j + 1]))))
        i = j + 1
    return out


def to_events_hi(path, np, sf, seconds=None, beats_per_bar=4, div=4,
                 scale=None, quantize=True):
    """High-resolution transcription. Same (ev, bars, spb, analysis, scale) shape.

    `div` is grid steps per beat (4 = sixteenths).
    """
    import theory
    n_fft, hop = 2048, 512
    x, sr = _load(path, np, sf)
    got = _read_music(x, sr, np, n_fft, hop)
    if not got:
        return None
    S, env, bpm, clarity, off, period, chroma, _beats, root, mode, conf = got
    scale_name = "minor" if mode == "minor" else "major"
    scale = scale or theory.SCALES[scale_name]
    pcs = [(root + t) % 12 for t in scale]

    # Grid: `div` steps per beat, so a 16th grid at 4 beats/bar = 16 steps/bar.
    step_frames = period / div
    n_steps = int(round((S.shape[0] - off) / step_frames))
    if n_steps < 0.75 * div * beats_per_bar:
        return None
    steps_per_bar = beats_per_bar * div

    # Percussion bands, measured independently rather than inferred from one
    # centroid — a kick and a hat overlap in time constantly.
    e_low = _band_energy(S, sr, n_fft, np, 30.0, 150.0)
    e_mid = _band_energy(S, sr, n_fft, np, 200.0, 900.0)
    e_high = _band_energy(S, sr, n_fft, np, 4000.0, 11000.0)
    d_low = np.concatenate([[0.0], np.maximum(np.diff(e_low), 0)])
    d_mid = np.concatenate([[0.0], np.maximum(np.diff(e_mid), 0)])
    d_high = np.concatenate([[0.0], np.maximum(np.diff(e_high), 0)])
    # 70 fired on ~62% of all steps — a hit on nearly every 16th reads as noise,
    # not a groove. 88/92 keeps only genuine accents.
    def thr(v, pct):
        return float(np.percentile(v[v > 0], pct)) if (v > 0).any() else 1e30
    t_low, t_mid, t_high = thr(d_low, 88), thr(d_mid, 90), thr(d_high, 92)

    mel_raw, bass_raw, strength, drums_at = [], [], [], []
    for i in range(n_steps):
        f0 = int(round(off + i * step_frames))
        f1 = max(f0 + 1, int(round(off + (i + 1) * step_frames)))
        f1 = min(f1, S.shape[0])
        if f0 >= S.shape[0]:
            break
        frame = S[f0:f1].sum(axis=0)
        fm = _hps(frame, sr, n_fft, np, 180.0, 2000.0)
        fb = _hps(frame, sr, n_fft, np, 45.0, 260.0)
        mel_raw.append(int(round(69 + 12 * np.log2(fm / 440.0))) if fm else None)
        bass_raw.append(int(round(69 + 12 * np.log2(fb / 440.0))) if fb else None)
        strength.append(float(env[f0:f1].sum()))
        hits = []
        if d_low[f0:f1].max(initial=0) >= t_low:
            hits.append("k")
        if d_mid[f0:f1].max(initial=0) >= t_mid:
            hits.append("s")
        if d_high[f0:f1].max(initial=0) >= t_high:
            hits.append("h")
        drums_at.append(hits)

    # k=3 on the melody: enough to kill isolated octave jumps, not enough to
    # flatten actual movement. Bass can take more smoothing — it moves less.
    mel = _median_smooth(mel_raw, 3)
    bass = _median_smooth(bass_raw, 7)
    if quantize:
        def snap(n):
            if n is None:
                return None
            for d in (0, 1, -1, 2, -2):
                if (n + d) % 12 in pcs:
                    return n + d
            return n
        mel = [snap(n) for n in mel]
        bass = [snap(n) for n in bass]

    # Whole bars, counting a final bar that is at least 3/4 there: the grid can
    # end a step or two short of the file's last sample, and flooring that
    # dropped the whole last bar.
    nbars = max(1, int(len(mel) / float(steps_per_bar) + 0.25))
    if seconds:
        spb_sec = 60.0 / bpm * beats_per_bar
        nbars = max(1, min(nbars, max(1, int(round(seconds / spb_sec)))))
    keep = nbars * steps_per_bar
    mel, bass, strength = mel[:keep], bass[:keep], strength[:keep]
    drums_at = drums_at[:keep]

    # Below the source's median onset strength, do not lay down an arp at all —
    # that is where the reference is quiet, and silence is part of the
    # arrangement.
    _pos = [v for v in strength if v > 0]
    arp_gate = float(np.median(_pos)) * 0.8 if _pos else float("inf")   # no onsets: no arp

    ev = {"lead": [], "arp": [], "bass": [], "drum": []}
    for st, ln, p, sg in _segment(mel, strength):
        ev["lead"].append((st // steps_per_bar, st % steps_per_bar,
                           min(ln, steps_per_bar - st % steps_per_bar), p, 0.5))
    for st, ln, p, sg in _segment(bass, strength, min_len=2):
        ev["bass"].append((st // steps_per_bar, st % steps_per_bar,
                           min(ln, steps_per_bar - st % steps_per_bar), p))
    # Chords per BEAT, from that beat's chroma — harmony now moves inside a bar.
    for bar in range(nbars):
        for beat in range(beats_per_bar):
            s0 = bar * steps_per_bar + beat * div
            f0 = int(round(off + s0 * step_frames))
            f1 = min(S.shape[0], int(round(off + (s0 + div) * step_frames)))
            if f0 >= f1:
                continue
            cv = np.roll(chroma[f0:f1].sum(axis=0), 0)
            degree = _chord_for(cv, root, scale, np)
            chord = [12 * 4 + root + scale[(degree + k) % len(scale)] +
                     12 * ((degree + k) // len(scale)) for k in (0, 2, 4)]
            # Gate the arp on the source's own activity. Filling every 16th of
            # every bar regardless of the reference made all tracks share one
            # dense texture — measured, it made the pack LESS diverse (0.798 ->
            # 0.819) even though the melody got better. The arp should follow the
            # music, not pave over it.
            for k in range(0, div, 2):                 # 8ths, not 16ths
                si = s0 + k
                if si >= len(strength):
                    break
                if strength[si] < arp_gate:
                    continue
                ev["arp"].append((bar, beat * div + k, 2,
                                  chord[(beat + k) % len(chord)], 0.25))
    for i, hits in enumerate(drums_at):
        for kind in hits:
            ev["drum"].append((i // steps_per_bar, i % steps_per_bar, kind))

    ana = {"bpm": bpm, "root": root, "mode": mode, "key_conf": conf, "clarity": clarity,
           "duration": len(x) / sr, "steps_per_bar": steps_per_bar,
           "notes": len(ev["lead"]), "drum_hits": len(ev["drum"])}
    return ev, nbars, 60.0 / bpm * beats_per_bar, ana, scale_name
