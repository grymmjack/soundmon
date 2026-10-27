"""--index / --similar: find sounds on your disk that sound like this one (Sononym-style).

Every audio file gets a fingerprint made only from signal processing (no model, no GPU):

    timbre      32-band log-mel spectrum: its average shape and how much each band moves
    brightness  spectral centroid, flatness (tonal vs noisy) and roll-off, average and spread
    envelope    attack time, onsets per second, and its loudness over 8 slices of its length
    length      log duration

An index of folders you choose lives in ~/soundmon-gallery/similar/index.json. It is
incremental: only new or changed files are read again. A search z-scores every feature
across the index (so no one feature dominates by scale), measures the distance per group,
weights timbre highest, and returns the nearest files with a 0-100 "how alike" score and
which aspects match. Needs numpy and ffmpeg; imports nothing heavy at module level.
"""
import json
import math
import os
import subprocess
import sys
import time

SR = 22050
MAX_SECONDS = 30.0                      # long files: the first 30 s say enough
AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3", ".opus", ".aif", ".aiff", ".m4a")
SKIP_DIRS = {"mastered", ".soundmon-backup", ".pixelmon-backup", "takes", "node_modules", ".git"}
INDEX = os.path.expanduser("~/soundmon-gallery/similar/index.json")
METHOD = "dsp3"                         # bump when features() changes: the index rebuilds itself
# feature groups: (name, slice in the vector, weight)
GROUPS = [("timbre", slice(0, 64), 0.5), ("brightness", slice(64, 70), 0.2),
          ("envelope", slice(70, 80), 0.2), ("length", slice(80, 81), 0.1)]
DIMS = 81


def _load(path, np):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-t", str(MAX_SECONDS), "-i", path, "-ac", "1",
                          "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def _duration(path):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                           capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


_MEL = {}


def _mel_bank(np, n_fft, bands=32, lo=40.0, hi=10000.0):
    key = (n_fft, bands)
    if key not in _MEL:
        mel = lambda f: 2595 * np.log10(1 + f / 700.0)      # noqa: E731
        imel = lambda m: 700 * (10 ** (m / 2595.0) - 1)     # noqa: E731
        pts = imel(np.linspace(mel(lo), mel(hi), bands + 2))
        fr = np.fft.rfftfreq(n_fft, 1 / SR)
        fb = np.zeros((bands, len(fr)))
        for i in range(bands):
            a, b, c = pts[i], pts[i + 1], pts[i + 2]
            fb[i] = np.clip(np.minimum((fr - a) / (b - a + 1e-9), (c - fr) / (c - b + 1e-9)), 0, None)
        _MEL[key] = fb
    return _MEL[key]


def features(path):
    """The 81-number fingerprint of one file's audible part (see GROUPS), or None for silence."""
    import numpy as np
    x = _load(path, np)
    if len(x) < 256 or np.abs(x).max() < 1e-4:
        return None
    # Only the audible part: game files are often padded with silence (a 40 ms footstep in a 0.8 s
    # file), and the padding would otherwise decide its length and envelope.
    loud = np.where(np.abs(x) > np.abs(x).max() * 10 ** (-50 / 20))[0]
    x = x[loud[0]:loud[-1] + 1]
    n, hop = 1024, 256
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    fr = np.fft.rfftfreq(n, 1 / SR)
    e = spec.sum(axis=1)
    keep = e > e.max() * 1e-4                                  # frames within 40 dB of the loudest
    S = spec[keep]
    mel = np.log10(S @ _mel_bank(np, n).T + 1e-10) * 10        # dB per mel band
    mel -= mel.max()
    timbre = np.concatenate([mel.mean(axis=0), mel.std(axis=0)])
    tot = S.sum(axis=1) + 1e-20
    centroid = (S * fr).sum(axis=1) / tot / (SR / 2)
    flat = np.exp(np.log(S + 1e-20).mean(axis=1)) / (S.mean(axis=1) + 1e-20)
    cum = np.cumsum(S, axis=1) / tot[:, None]
    roll = fr[np.argmax(cum >= 0.85, axis=1)] / (SR / 2)
    bright = np.array([centroid.mean(), centroid.std(), flat.mean(), flat.std(), roll.mean(), roll.std()])
    rms = np.sqrt(e / n)
    peak_i = int(np.argmax(rms))
    attack = math.log10(1 + peak_i * hop / SR * 1000)             # ms to the loudest frame, log
    flux = np.maximum(0, np.diff(np.log10(spec + 1e-10), axis=0)).sum(axis=1)
    if len(flux) >= 3:                                         # a blip shorter than 3 frames has 1 onset
        thr = flux.mean() + flux.std()
        onsets = max(1, int(((flux[1:-1] > thr) & (flux[1:-1] > flux[:-2]) & (flux[1:-1] >= flux[2:])).sum()))
    else:
        onsets = 1
    secs = len(x) / SR
    shape = np.interp(np.linspace(0, len(rms) - 1, 8), np.arange(len(rms)), rms) / (rms.max() + 1e-12)
    env = np.concatenate([[attack, onsets / max(secs, 0.1)], shape])
    full = secs if len(x) < MAX_SECONDS * SR - 1 else (_duration(path) or secs)
    length = np.array([math.log10(max(full, 0.01))])
    v = np.nan_to_num(np.concatenate([timbre, bright, env, length]), nan=0.0, posinf=0.0, neginf=0.0)
    return [round(float(t), 4) for t in v]


def _walk(folders):
    for root in folders:
        root = os.path.expanduser(root)
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(("_", "."))]
            for f in fns:
                if f.lower().endswith(AUDIO_EXT) and not f.startswith("."):
                    yield os.path.join(dp, f)


class Index:
    def __init__(self, path=INDEX):
        self.path = path
        try:
            with open(path, encoding="utf-8") as fh:
                self.data = json.load(fh)
        except (OSError, ValueError):
            self.data = {}
        if self.data.get("method") != METHOD:            # a different fingerprint: start over
            self.data = {"method": METHOD, "folders": [], "files": {}}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh)
        os.replace(tmp, self.path)

    def update(self, folders, progress=None):
        """Index these folders (added to the list). Returns {"added", "kept", "removed", "failed"}."""
        folders = [os.path.realpath(os.path.expanduser(f)) for f in folders]
        for f in folders:
            if not os.path.isdir(f):
                raise ValueError(f"not a folder: {f}")
        self.data["folders"] = sorted(set(self.data["folders"]) | set(folders))
        files, stats = self.data["files"], {"added": 0, "kept": 0, "removed": 0, "failed": 0}
        todo = list(_walk(folders))
        seen = set()
        for i, p in enumerate(todo, 1):
            seen.add(p)
            try:
                st = os.stat(p)
            except OSError:
                continue
            old = files.get(p)
            if old and old["mtime"] == st.st_mtime and old["size"] == st.st_size:
                stats["kept"] += 1
            else:
                try:
                    v = features(p)
                except Exception:
                    v = None
                if v is None:
                    stats["failed"] += 1
                    files.pop(p, None)
                else:
                    files[p] = {"mtime": st.st_mtime, "size": st.st_size, "vec": v, "seconds": round(10 ** v[-1], 2)}
                    stats["added"] += 1
            if progress:
                progress(i, len(todo), p)
            if i % 200 == 0:
                self.save()
        for p in [p for p in files if any(p.startswith(f + os.sep) for f in folders) and p not in seen]:
            del files[p]
            stats["removed"] += 1
        self.data["updated"] = time.time()
        self.save()
        return stats

    def search(self, query, k=20, within=None, vec=None):
        """The k files most like `query` (a path, or a fingerprint via vec)."""
        import numpy as np
        files = self.data["files"]
        paths = [p for p in files if (not within or p.startswith(os.path.realpath(os.path.expanduser(within)) + os.sep))]
        qreal = os.path.realpath(query) if query else None
        paths = [p for p in paths if p != qreal]
        if not paths:
            return []
        q = np.array(vec if vec is not None else features(query) or [0.0] * DIMS)
        M = np.array([files[p]["vec"] for p in paths])
        allv = np.array([f["vec"] for f in files.values()] + [q.tolist()])
        mu, sd = allv.mean(axis=0), allv.std(axis=0) + 1e-6
        Z, zq = (M - mu) / sd, (q - mu) / sd
        per = {}
        dist = np.zeros(len(paths))
        for name, sl, w in GROUPS:
            d = np.sqrt(((Z[:, sl] - zq[sl]) ** 2).mean(axis=1))
            per[name] = d
            dist += w * d
        scale = float(np.median(dist)) or 1.0
        out, seen = [], set()
        for i in np.argsort(dist):
            if len(out) >= k:
                break
            same = (files[paths[i]]["size"], tuple(files[paths[i]]["vec"]))   # one copy of identical files
            if same in seen:
                continue
            seen.add(same)
            aspects = [name for name, _, _ in GROUPS if per[name][i] <= float(np.percentile(per[name], 15))]
            out.append({"path": paths[i], "score": int(round(100 * math.exp(-float(dist[i]) / scale * 0.7))),
                        "seconds": files[paths[i]].get("seconds"), "aspects": aspects})
        return out


def run(a, slug=None, to_ogg=None, loudness_normalize=None, to_flac=None):
    """soundmon --index DIR[,DIR]   /   soundmon --similar FILE [--in DIR] [-n 20]"""
    from shutil import which
    if not which("ffmpeg"):
        sys.exit("--index / --similar need ffmpeg on PATH")
    try:
        import numpy  # noqa: F401
    except ImportError:
        sys.exit("--index / --similar need numpy (run soundmon with ComfyUI's venv)")
    idx = Index()
    if a.index:
        folders = [f.strip() for f in a.index.split(",") if f.strip()]
        print(f"🔎 indexing {', '.join(folders)} -> {idx.path}")
        last = [0.0]

        def prog(i, n, p):
            if time.time() - last[0] > 2 or i == n:
                last[0] = time.time()
                print(f"   {i}/{n}  {os.path.basename(p)}", flush=True)
        try:
            st = idx.update(folders, prog)
        except ValueError as e:
            sys.exit(str(e))
        print(f"   all done  |  {len(idx.data['files'])} sound(s) indexed  "
              f"(new or changed {st['added']}, unchanged {st['kept']}, gone {st['removed']}, unreadable {st['failed']})")
    if a.similar:
        q = os.path.expanduser(a.similar)
        if not os.path.isfile(q):
            sys.exit(f"--similar: no such file: {a.similar}")
        if not idx.data["files"]:
            sys.exit("--similar: the index is empty — build it first with  soundmon --index DIR[,DIR]")
        k = a.number if a.number > 1 else 20
        res = idx.search(q, k, within=getattr(a, "within", None))
        print(f"🔎 sounds like {os.path.basename(q)}  ({len(idx.data['files'])} indexed)")
        for i, r in enumerate(res, 1):
            print(f"   {i:>3}. {r['score']:>3}%  {r['seconds'] or 0:6.2f}s  {r['path']}"
                  + (f"   [{', '.join(r['aspects'])}]" if r["aspects"] else ""))
