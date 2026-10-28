"""--index / --similar: find sounds on your disk that sound like this one (Sononym-style).

Every audio file gets a fingerprint made only from signal processing (no model, no GPU):

    timbre      32-band log-mel spectrum: its average shape and how much each band moves
    brightness  spectral centroid, flatness (tonal vs noisy) and roll-off, average and spread
    envelope    attack time, onsets per second, and its loudness over 8 slices of its length
    length      log duration

The fingerprint does not depend on volume: the sound is scaled to its own peak first, and mel
bands more than 60 dB under its loudest band all read -60 (a band at -90 dB is noise, not timbre).

An index of folders you choose lives in ~/soundmon-gallery/similar/index.json. It is
incremental: only new or changed files are read again (and a file that couldn't be read is not
tried again until it changes). Two --index runs at once are safe: saving takes a lock and merges
with what is on disk. A search z-scores every feature across the index (so no one feature
dominates by scale), measures the distance per group, weights timbre highest, and returns the
nearest files with a 0-100 "how alike" score and which aspects match. The score's scale comes
from the whole index (the typical distance between two of its sounds), not from the candidates,
so a near-duplicate reads high however few files --in leaves. Needs numpy and ffmpeg; imports
nothing heavy at module level.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import time

SR = 22050
MAX_SECONDS = 30.0                      # long files: the first 30 s say enough
MIN_SAMPLES = 64                        # ~3 ms: anything shorter has no spectrum to speak of
AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3", ".opus", ".aif", ".aiff", ".m4a")
SKIP_DIRS = {"mastered", ".soundmon-backup", ".pixelmon-backup", "takes", "node_modules", ".git"}
INDEX = os.path.expanduser("~/soundmon-gallery/similar/index.json")
METHOD = "dsp4"                         # bump when features() changes: the index rebuilds itself
# feature groups: (name, slice in the vector, weight)
GROUPS = [("timbre", slice(0, 64), 0.5), ("brightness", slice(64, 70), 0.2),
          ("envelope", slice(70, 80), 0.2), ("length", slice(80, 81), 0.1)]
DIMS = 81
TYPICAL = 1.3                           # the distance between two unrelated sounds, for a tiny index


def _reason(stderr):
    lines = [ln.strip() for ln in (stderr or b"").decode(errors="replace").splitlines() if ln.strip()]
    last = lines[-1] if lines else "unknown error"
    if last.startswith("[") and "] " in last:
        last = last.split("] ", 1)[1]
    return last.split(": ", 1)[1] if last.startswith("Error opening input") and ": " in last else last


def _load(path, np):
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-t", str(MAX_SECONDS), "-i", path, "-ac", "1",
                        "-ar", str(SR), "-f", "f32le", "-"], capture_output=True)
    if r.returncode:
        raise RuntimeError(_reason(r.stderr))
    x = np.frombuffer(r.stdout, dtype=np.float32).astype(np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


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
    """The 81-number fingerprint of one file's audible part (see GROUPS), or None when it is silent or
    too short (under ~3 ms) to have one. Raises RuntimeError (ffmpeg's reason) when it can't be read."""
    import numpy as np
    x = _load(path, np)
    if len(x) < MIN_SAMPLES or np.abs(x).max() < 1e-4:
        return None
    x = x / np.abs(x).max()                                    # volume is not what a sound IS
    # Only the audible part: game files are often padded with silence (a 40 ms footstep in a 0.8 s
    # file), and the padding would otherwise decide its length and envelope.
    loud = np.where(np.abs(x) > 10 ** (-50 / 20))[0]
    x = x[loud[0]:loud[-1] + 1]
    secs = len(x) / SR
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
    mel = np.maximum(mel, -60.0)                               # under -60 dB is noise floor, not timbre
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
    # onsets: spectral flux over bins within 60 dB of the loudest (below that, a 16-bit file's own
    # quantization noise flickers, and a quieter copy of the same sound would count more onsets)
    flux = np.maximum(0, np.diff(np.log10(spec + spec.max() * 1e-6), axis=0)).sum(axis=1)
    if len(flux) >= 3:                                         # a blip shorter than 3 frames has 1 onset
        thr = max(flux.mean() + flux.std(), 1.0)               # (a steady tone's flux is ~0.005: not onsets)
        onsets = max(1, int(((flux[1:-1] > thr) & (flux[1:-1] > flux[:-2]) & (flux[1:-1] >= flux[2:])).sum()))
    else:
        onsets = 1
    shape = np.interp(np.linspace(0, len(rms) - 1, 8), np.arange(len(rms)), rms) / (rms.max() + 1e-12)
    env = np.concatenate([[attack, onsets / max(secs, 0.1)], shape])
    full = secs if secs < MAX_SECONDS - 0.01 else (_duration(path) or secs)
    length = np.array([math.log10(max(full, 0.001))])
    v = np.nan_to_num(np.concatenate([timbre, bright, env, length]), nan=0.0, posinf=0.0, neginf=0.0)
    return [round(float(t), 4) for t in v]


def _prefix(folder):
    """folder as a path prefix: "/a/b/" (and "/" for the root, not "//")."""
    return folder if folder.endswith(os.sep) else folder + os.sep


def _walk(folders, problems=None):
    for root in folders:
        root = os.path.expanduser(root)

        def bad(err):
            if problems is not None:
                problems.append(getattr(err, "filename", None) or str(err))
        for dp, dns, fns in os.walk(root, onerror=bad):
            dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(("_", "."))]
            for f in fns:
                if f.lower().endswith(AUDIO_EXT) and not f.startswith("."):
                    yield os.path.join(dp, f)


def _valid_vec(v):
    return isinstance(v, list) and len(v) == DIMS and all(isinstance(t, (int, float)) and math.isfinite(t) for t in v)


class _Lock:
    """An exclusive lock on PATH.lock while saving (fcntl; msvcrt on Windows; else a no-op)."""

    def __init__(self, path):
        self.path, self.fh = path + ".lock", None

    def __enter__(self):
        self.fh = open(self.path, "a+")
        try:
            import fcntl
            fcntl.flock(self.fh, fcntl.LOCK_EX)
        except ImportError:
            try:
                import msvcrt
                for _ in range(600):
                    try:
                        msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        time.sleep(0.1)
            except ImportError:
                pass
        return self

    def __exit__(self, *exc):
        self.fh.close()                                    # closing releases it


class Index:
    def __init__(self, path=INDEX):
        self.path = path
        self.note = None                                   # why it was started over, when it was
        self._dirty, self._gone = set(), set()
        self.data = self._read()

    def _read(self):
        try:
            with open(self.path, encoding="utf-8") as fh:
                d = json.load(fh)
        except FileNotFoundError:
            return {"method": METHOD, "folders": [], "files": {}, "failed": {}}
        except (OSError, ValueError) as e:
            self.note = f"the index couldn't be read ({e.__class__.__name__}) and was started over"
            d = None
        if not isinstance(d, dict) or not isinstance(d.get("files"), dict):
            self.note = "the index file was malformed and was started over"
            return {"method": METHOD, "folders": [], "files": {}, "failed": {}}
        folders = [f for f in d.get("folders", []) if isinstance(f, str)] if isinstance(d.get("folders"), list) else []
        if d.get("method") != METHOD:                      # a different fingerprint: start over
            self.note = (f"the index was made with an older fingerprint ({d.get('method')}) and was reset — "
                         f"re-run  soundmon --index {','.join(folders) or 'DIR'}")
            return {"method": METHOD, "folders": folders, "files": {}, "failed": {}}
        files = {p: f for p, f in d["files"].items() if isinstance(f, dict) and _valid_vec(f.get("vec"))}
        if len(files) < len(d["files"]):
            self.note = f"{len(d['files']) - len(files)} malformed entr(ies) dropped from the index — re-run --index"
        failed = d.get("failed") if isinstance(d.get("failed"), dict) else {}
        return dict(d, folders=folders, files=files, failed={p: f for p, f in failed.items() if isinstance(f, dict)})

    def _put(self, sect, p, v):
        self.data[sect][p] = v
        self._dirty.add((sect, p))
        self._gone.discard((sect, p))

    def _drop(self, sect, p):
        if self.data[sect].pop(p, None) is not None:
            self._gone.add((sect, p))
            self._dirty.discard((sect, p))
            return True
        return False

    def save(self):
        """Write the index, merged with whatever another run saved meanwhile (under a lock)."""
        d = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(d, exist_ok=True)
        with _Lock(self.path):
            note, disk = self.note, self._read()
            self.note = note
            if disk.get("method") == METHOD:
                for sect in ("files", "failed"):
                    merged = dict(disk.get(sect, {}))
                    for p, v in self.data[sect].items():           # ours win where we changed them;
                        if (sect, p) in self._dirty or p not in merged:  # theirs win where we didn't
                            merged[p] = v
                    for s, p in self._gone:
                        if s == sect:
                            merged.pop(p, None)
                    self.data[sect] = merged
                self.data["folders"] = sorted(set(disk.get("folders", [])) | set(self.data["folders"]))
            fd, tmp = tempfile.mkstemp(prefix=".index.", suffix=".tmp", dir=d)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(self.data, fh)
                os.replace(tmp, self.path)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        self._dirty.clear()
        self._gone.clear()

    def update(self, folders, progress=None):
        """Index these folders (added to the list). Returns {"added", "kept", "removed", "failed",
        "short", "skipped", "problems"}: failed = unreadable, short = silent or too short to
        fingerprint, skipped = dangling links and folders that couldn't be listed."""
        folders = [os.path.realpath(os.path.expanduser(f)) for f in folders]
        for f in folders:
            if not os.path.isdir(f):
                raise ValueError(f"not a folder: {f}")
        self.data["folders"] = sorted(set(self.data["folders"]) | set(folders))
        self.data.setdefault("failed", {})
        stats = {"added": 0, "kept": 0, "removed": 0, "failed": 0, "short": 0, "skipped": 0}
        problems = []
        todo = list(_walk(folders, problems))
        seen = set()
        for i, p in enumerate(todo, 1):
            seen.add(p)
            try:
                st = os.stat(p)
            except OSError:
                problems.append(p)
                continue
            old, bad = self.data["files"].get(p), self.data["failed"].get(p)     # (save() may replace the dicts)
            if old and old.get("mtime") == st.st_mtime and old.get("size") == st.st_size:
                stats["kept"] += 1
            elif bad and bad.get("mtime") == st.st_mtime and bad.get("size") == st.st_size:
                stats["short" if bad.get("why") == "short" else "failed"] += 1    # not tried again until it changes
            else:
                why = "short"
                try:
                    v = features(p)
                except Exception:
                    v, why = None, "unreadable"
                if v is None:
                    stats["short" if why == "short" else "failed"] += 1
                    self._drop("files", p)
                    self._put("failed", p, {"mtime": st.st_mtime, "size": st.st_size, "why": why})
                else:
                    self._put("files", p, {"mtime": st.st_mtime, "size": st.st_size, "vec": v,
                                           "seconds": round(10 ** v[-1], 3)})
                    self._drop("failed", p)
                    stats["added"] += 1
            if progress:
                progress(i, len(todo), p)
            if i % 200 == 0:
                self.save()
        for sect in ("files", "failed"):
            for p in [p for p in self.data[sect] if any(p.startswith(_prefix(f)) for f in folders) and p not in seen]:
                self._drop(sect, p)
                stats["removed"] += sect == "files"
        stats["skipped"] = len(problems)
        self.data["updated"] = time.time()
        self.save()
        stats["problems"] = problems[:20]
        return stats

    def search(self, query, k=20, within=None, vec=None):
        """The k files most like `query` (a path, or a fingerprint via vec). Raises ValueError when
        the query can't be read or is silent; files deleted since indexing are left out."""
        import numpy as np
        files = self.data["files"]
        pre = _prefix(os.path.realpath(os.path.expanduser(within))) if within else None
        paths = [p for p in files if not pre or p.startswith(pre)]
        if not paths:
            return []
        if vec is None:
            try:
                vec = features(query)
            except Exception as e:
                raise ValueError(f"can't read {os.path.basename(query)}: {e}")
            if vec is None:
                raise ValueError(f"{os.path.basename(query)} is silent or too short to compare")
        q = np.array(vec, dtype=float)
        allp = list(files)
        A = np.array([files[p]["vec"] for p in allp])
        mu, sd = A.mean(axis=0), A.std(axis=0) + 1e-6
        Z = (A - mu) / sd
        zq = (q - mu) / sd

        def dist(X, y):
            per = {name: np.sqrt(((X[:, sl] - y[sl]) ** 2).mean(axis=1)) for name, sl, _ in GROUPS}
            return sum(w * per[name] for name, _, w in GROUPS), per
        # the score's scale: the typical distance between two sounds of the WHOLE index (a fixed
        # sample of it), so it doesn't move with --in or with how alike the candidates happen to be
        if len(allp) >= 8:
            pick = np.linspace(0, len(allp) - 1, min(len(allp), 200)).astype(int)
            ZS = Z[pick]
            pair = [dist(ZS[i + 1:], ZS[i])[0] for i in range(len(ZS) - 1)]
            scale = float(np.median(np.concatenate(pair))) or TYPICAL
        else:
            scale = TYPICAL
        where = {p: i for i, p in enumerate(allp)}
        idx = np.array([where[p] for p in paths])
        d, per = dist(Z[idx], zq)
        qreal = os.path.realpath(query) if query else None
        out, seen = [], set()
        for i in np.argsort(d, kind="stable"):
            if len(out) >= k:
                break
            p = paths[i]
            real = os.path.realpath(p)
            if real == qreal or real in seen or not os.path.isfile(p):   # itself (or a link to it), gone
                continue
            same = (files[p]["size"], tuple(files[p]["vec"]))            # one copy of identical files
            if same in seen:
                continue
            seen.update((same, real))
            aspects = [name for name, _, _ in GROUPS if per[name][i] <= float(np.percentile(per[name], 15))]
            out.append({"path": p, "score": int(round(100 * math.exp(-float(d[i]) / scale * 0.7))),
                        "seconds": files[p].get("seconds"), "aspects": aspects})
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
    if idx.note:
        print(f"   ⚠ {idx.note}")
    if getattr(a, "index", None) is not None:
        folders = [f.strip() for f in a.index.split(",") if f.strip()]
        if not folders:
            sys.exit("--index: give a folder (or several, comma-separated)")
        print(f"🔎 indexing {', '.join(folders)} -> {idx.path}")
        last = [0.0]

        def prog(i, n, p):
            if time.time() - last[0] > 2 or i == n:
                last[0] = time.time()
                print(f"   {i}/{n}  {os.path.basename(p)}", flush=True)
        try:
            st = idx.update(folders, prog)
        except ValueError as e:
            sys.exit(f"--index: {e}")
        except OSError as e:
            sys.exit(f"--index: can't write the index {idx.path}: {e.strerror or e}")
        print(f"   all done  |  {len(idx.data['files'])} sound(s) indexed  "
              f"(new or changed {st['added']}, unchanged {st['kept']}, gone {st['removed']}, "
              f"unreadable {st['failed']}, silent or too short {st['short']}"
              + (f", skipped {st['skipped']}: dangling links or folders that couldn't be read" if st["skipped"] else "")
              + ")")
        for p in st["problems"][:5]:
            print(f"      skipped: {p}")
    if getattr(a, "similar", None) is not None:
        q = os.path.expanduser(a.similar)
        if not a.similar.strip() or not os.path.isfile(q):
            sys.exit(f"--similar: no such file: {a.similar!r}")
        if not idx.data["files"]:
            sys.exit("--similar: the index is empty — build it first with  soundmon --index DIR[,DIR]"
                     if not idx.note else f"--similar: {idx.note}")
        # soundmon sets similar_k: the -n typed, or 20 when there was none (-n defaults to 1 for the
        # generators). Without it (an older caller), a bare default of 1 means 20.
        k = getattr(a, "similar_k", None)
        if k is None:
            n = getattr(a, "number", None)
            k = n if isinstance(n, int) and n != 1 else 20
        if k < 1:
            sys.exit(f"--similar: -n must be 1 or more, got {k}")
        within = getattr(a, "within", None)
        if within:
            w = os.path.expanduser(within)
            if not os.path.isdir(w):
                sys.exit(f"--similar: --in {within}: no such folder")
            rw = _prefix(os.path.realpath(w))
            if not any(p.startswith(rw) for p in idx.data["files"]):
                sys.exit(f"--similar: nothing indexed inside {within} — index it first:  soundmon --index {within}")
        try:
            res = idx.search(q, k, within=within)
        except ValueError as e:
            sys.exit(f"--similar: {e}")
        print(f"🔎 sounds like {os.path.basename(q)}  ({len(idx.data['files'])} indexed)")
        if not res:
            print("   no other sounds to compare with" + (f" in {within}" if within else ""))
        for i, r in enumerate(res, 1):
            print(f"   {i:>3}. {r['score']:>3}%  {r['seconds'] or 0:6.2f}s  {r['path']}"
                  + (f"   [{', '.join(r['aspects'])}]" if r["aspects"] else ""))
