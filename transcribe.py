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
    tempo            autocorrelation of the onset envelope, 60-200 BPM
    beat phase       cross-correlation of the envelope against a pulse train
    chroma           STFT bins folded to 12 pitch classes, per beat
    key              chroma correlated against Krumhansl-Kessler profiles
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
    audio, sr = sf.read(path, always_2d=True)
    m = audio.mean(axis=1)
    if sr > target_sr:                      # cheap decimation; we only need pitch
        k = int(round(sr / target_sr))
        m = m[::k]
        sr = sr // k
    return m.astype(np.float64), sr


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


def _fold_tempo(bpm, lo=70.0, hi=150.0):
    """Pull an octave-error tempo into a musically plausible range.

    Autocorrelation happily locks onto twice or four times the real tempo, which
    is why a slow dirge measured 191 bpm. Halving/doubling until it lands in the
    range costs nothing and fixes the common case; genuinely fast music is still
    reachable because the range is generous.
    """
    while bpm > hi:
        bpm /= 2.0
    while bpm < lo:
        bpm *= 2.0
    return bpm


def _tempo(env, sr, hop, np, lo=60.0, hi=200.0):
    """Autocorrelate the onset envelope and pick the strongest plausible period."""
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[len(e) - 1:]
    fps = sr / hop
    best, best_v = 120.0, -1e30
    for bpm in np.arange(lo, hi + 0.5, 0.5):
        lag = int(round(fps * 60.0 / bpm))
        if lag < 2 or lag >= len(ac):
            continue
        # Sum a few harmonics of the lag so 2x/0.5x errors are less likely.
        v = ac[lag] + 0.5 * ac[min(len(ac) - 1, lag * 2)]
        if v > best_v:
            best, best_v = float(bpm), v
    return best


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


def _find_key(chroma, np):
    v = chroma.sum(axis=0)
    v = v / (v.sum() + 1e-12)
    maj = np.array(KK_MAJOR); mnr = np.array(KK_MINOR)
    best = (0, "minor", -2.0)
    for r in range(12):
        rot = np.roll(v, -r)
        for name, prof in (("major", maj), ("minor", mnr)):
            p = prof / prof.sum()
            c = float(np.corrcoef(rot, p)[0, 1])
            if c > best[2]:
                best = (r, name, c)
    return best[0], best[1], best[2]


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


def analyze(path, np, sf, bars_max=32):
    """Extract a musical summary of a recording. Returns a plain dict."""
    n_fft, hop = 2048, 512
    x, sr = _load(path, np, sf)
    if len(x) < n_fft * 4:
        return None
    S = _stft(x, np, n_fft, hop)
    env = _onset_env(S, np)
    bpm = _fold_tempo(_tempo(env, sr, hop, np))
    off, period = _beat_phase(env, sr, hop, bpm, np)
    chroma = _chroma(S, sr, n_fft, np)
    root, mode, conf = _find_key(chroma, np)

    # Beat frame indices
    beats = np.arange(off, S.shape[0], period).astype(int)
    beats = beats[beats < S.shape[0]]
    if len(beats) < 4:
        return None

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
            "beats": per_beat, "duration": len(x) / sr,
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
