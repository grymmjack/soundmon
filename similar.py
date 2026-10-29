"""--index / --similar / --find: your sound library, searchable (Sononym-style).

Every audio file gets a fingerprint made only from signal processing (no model, no GPU):

    timbre      32-band log-mel spectrum: its average shape and how much each band moves
    brightness  spectral centroid, flatness (tonal vs noisy) and roll-off, average and spread
    envelope    attack time, onsets per second, and its loudness over 8 slices of its length
    length      log duration

The fingerprint does not depend on volume: the sound is scaled to its own peak first, and mel
bands more than 60 dB under its loudest band all read -60 (a band at -90 dB is noise, not timbre).

Every file is also tagged from its own words (tags.py: kick, snare, pad, riser...; loop or one-shot;
tempo and key), and --listen (listen.py, optional: torch + transformers) adds what an audio model
hears - a tag for the files whose words say nothing, its "character" for --similar, and search by
description.

The index is one SQLite file, ~/soundmon-gallery/similar/index.db ($SOUNDMON_INDEX overrides). It is
incremental (only new or changed files are read; a file that couldn't be read isn't tried again until
it changes), indexes with several processes at once, saves as it goes (stop it any time and the next
run carries on), and two runs at once are safe. An index.json from an older soundmon is taken in
once. A search z-scores every feature across the index (so no one feature dominates by scale),
measures the distance per aspect, weights them (timbre highest, or as you set them), and returns the
nearest files with a 0-100 "how alike" score and which aspects match. The score's scale comes from
the whole index (the typical distance between two of its sounds), so a near-duplicate reads high
however few files the filters leave. Needs numpy and ffmpeg; imports nothing heavy at module level.
"""
import json
import math
import os
import sqlite3
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tags  # noqa: E402

SR = 22050
MAX_SECONDS = 30.0                      # long files: the first 30 s say enough
MIN_SAMPLES = 64                        # ~3 ms: anything shorter has no spectrum to speak of
AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3", ".opus", ".aif", ".aiff", ".m4a")
SKIP_DIRS = {"mastered", ".soundmon-backup", ".pixelmon-backup", "takes", "node_modules", ".git", "__MACOSX"}
HOME_DIR = os.path.expanduser("~/soundmon-gallery/similar")
INDEX = os.environ.get("SOUNDMON_INDEX") or os.path.join(HOME_DIR, "index.db")
OLD_JSON = os.path.join(HOME_DIR, "index.json")
METHOD = "dsp4"                         # bump when features() changes: fingerprints are made again
# feature groups: (name, slice in the vector, weight)
GROUPS = [("timbre", slice(0, 64), 0.5), ("brightness", slice(64, 70), 0.2),
          ("envelope", slice(70, 80), 0.2), ("length", slice(80, 81), 0.1)]
ASPECTS = [g[0] for g in GROUPS] + ["character"]     # character: what the audio model hears (--listen)
CHARACTER = 0.7                         # its share of the distance when both sounds have been listened to
DIMS = 81
TYPICAL = 1.3                           # the distance between two unrelated sounds, for a tiny index
OK, SHORT, UNREADABLE = 0, 1, 2         # files.status
_ONE_THREAD = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "NUMEXPR_NUM_THREADS")
WORDS_SURE = 0.3                        # a words tag at least this sure beats the ear's
MAX_TEMP = float(os.environ.get("SOUNDMON_MAX_TEMP") or 92)   # °C: indexing and listening rest above it (0: never)
COOL_BY = 5                             # ... until the CPU is this much under it again
EAR_SURE = 0.4                          # the ear tags a sound only when at least this sure (else: untagged);
                                        # listen.learn() measures a better bar for each library (meta ear_sure)
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS folders (path TEXT PRIMARY KEY, added REAL);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    root TEXT,                          -- the indexed folder it was found in (its words start below it)
    mtime REAL, size INTEGER,
    status INTEGER NOT NULL DEFAULT 0,  -- 0 ok, 1 silent or too short, 2 unreadable
    seconds REAL,
    vec BLOB,                           -- the fingerprint: DIMS float32
    method TEXT,
    emb TEXT,                           -- what the file says about itself (tags.embedded), JSON
    tver INTEGER,                       -- tags.VERSION the words were read with
    wtag TEXT, wsure REAL,              -- the words' tag and how sure
    kind TEXT, kind_by TEXT,            -- loop / oneshot, and how that was decided
    bpm REAL, key TEXT,
    clap BLOB,                          -- what the audio model hears: float16 embedding (--listen)
    stag TEXT, ssure REAL, skind TEXT,  -- the ear's tag, how sure, and loop / oneshot
    utag TEXT, ukind TEXT,              -- yours: set in the GUI, never overwritten
    indexed REAL
);
CREATE INDEX IF NOT EXISTS files_root ON files(root);
"""


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


def _path_map():
    pairs = []
    for part in (os.environ.get("SOUNDMON_PATH_MAP") or "").split(";"):
        a, eq, b = part.partition("=")
        if eq and a.strip() and b.strip():
            pairs.append((_prefix(a.strip()), _prefix(b.strip())))
    return pairs


# $SOUNDMON_PATH_MAP "INDEXED=HERE[;INDEXED=HERE...]": work on an index from another machine that sees the same
# files under another path (a share mounted at /mnt/windows/e on one, the disk itself at /mnt/e on the other).
# The index keeps its own paths; only reading the files goes through here().
PATH_MAP = _path_map()


def here(p):
    """an indexed path as this machine sees it"""
    for a, b in PATH_MAP:
        if p.startswith(a) or p + os.sep == a:
            return b + p[len(a):]
    return p


def indexed(p):
    """a path on this machine as the index keeps it"""
    for a, b in PATH_MAP:
        if p.startswith(b) or p + os.sep == b:
            return a + p[len(b):]
    return p


def _walk(folders, problems=None):
    """(path, root) for every audio file under the folders (each file once, from the first folder)."""
    seen = set()
    for root in folders:
        def bad(err):
            if problems is not None:
                problems.append(indexed(getattr(err, "filename", None) or str(err)))
        for dp, dns, fns in os.walk(here(root), onerror=bad):
            dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(("_", "."))]
            for f in fns:
                if f.lower().endswith(AUDIO_EXT) and not f.startswith("."):
                    p = indexed(os.path.join(dp, f))
                    if p not in seen:
                        seen.add(p)
                        yield p, root


def _valid_vec(v):
    return isinstance(v, list) and len(v) == DIMS and all(isinstance(t, (int, float)) and math.isfinite(t) for t in v)


def _blob(v):
    import array
    return array.array("f", v).tobytes()


def _unblob(b):
    import array
    a = array.array("f")
    a.frombytes(b)
    return list(a)


def _work(job):
    """One file, in a worker process: (path, root, need) -> a row. need "all" reads the fingerprint and the
    file's own words, "words" only its words (the fingerprint is already in the index)."""
    path, root, need = job
    lp = here(path)
    try:
        st = os.stat(lp)
    except OSError:
        return {"path": path, "gone": True}
    row = {"path": path, "root": root, "mtime": st.st_mtime, "size": st.st_size, "need": need}
    if need == "all":
        try:
            v = features(lp)
            row["status"] = OK if v else SHORT
        except Exception:
            v, row["status"] = None, UNREADABLE
        row["vec"], row["seconds"] = v, (round(10 ** v[-1], 3) if v else None)
        if row["status"] == UNREADABLE:
            row["emb"] = None
            return row
    try:
        row["emb"] = tags.embedded(lp)
    except Exception:
        row["emb"] = {}                                    # read once: an empty answer isn't asked again
    return row


def effective_tag(utag, wtag, wsure, stag, ssure, ear_sure=EAR_SURE):
    """(tag, by, sure): yours, else the words' when they're sure enough, else the ear's (when at least ear_sure:
    listen.learn() sets it from how often it was right), else the words'."""
    if utag:
        return (None if utag == "-" else utag), "you", 1.0
    if wtag and (wsure or 0) >= WORDS_SURE:
        return wtag, "words", wsure
    if stag and (ssure or 0) >= ear_sure:
        return stag, "sound", ssure
    if wtag:
        return wtag, "words", wsure or 0.0
    return None, None, 0.0


def effective_kind(ukind, kind, kind_by, skind):
    if ukind:
        return ukind, "you"
    if kind and kind_by in ("flag", "words", "sampler", "short"):   # the file's own say, or too short for a bar
        return kind, "words"
    if skind:
        return skind, "sound"
    if kind:
        return kind, "words"
    return None, None


class Cooler:
    """Holds heavy work off while the CPU runs hot: ok() is False from MAX_TEMP until it's COOL_BY under.
    A laptop's CPU and GPU often share their cooling, so a busy GPU heats the CPU too. Reads the CPU's own
    sensors (Linux hwmon: k10temp, coretemp, zenpower, cpu_thermal) at most once a second; where there are
    none it never holds anything off."""
    CPU = ("k10temp", "coretemp", "zenpower", "cpu_thermal", "soc_thermal")

    def __init__(self, limit=None):
        self.limit = MAX_TEMP if limit is None else limit
        self.hot, self.temp, self.at, self.waited = False, None, 0.0, 0.0
        self.files = []
        if self.limit > 0:
            import glob
            for h in glob.glob("/sys/class/hwmon/hwmon*"):
                try:
                    with open(os.path.join(h, "name")) as f:
                        if f.read().strip() in self.CPU:
                            self.files += glob.glob(os.path.join(h, "temp*_input"))
                except OSError:
                    pass

    def ok(self):
        if not self.files:
            return True
        now = time.time()
        if now - self.at >= 1:
            if self.hot:
                self.waited += now - self.at
            self.at, temps = now, []
            for fn in self.files:
                try:
                    with open(fn) as f:
                        temps.append(int(f.read()) / 1000)
                except (OSError, ValueError):
                    pass
            self.temp = max(temps) if temps else None
            if self.temp is not None and self.temp >= self.limit:
                self.hot = True
            elif self.temp is None or self.temp <= self.limit - COOL_BY:
                self.hot = False
        return not self.hot

    def note(self):
        """what to show while it waits"""
        return f"the CPU is at {self.temp:.0f} °C: waiting till it's under {self.limit - COOL_BY:.0f} °C"


class Index:
    def __init__(self, path=None):
        self.path = path or INDEX
        self.note = None                                   # something the user should know (an old index taken in)
        self._local = threading.local()
        self._cache, self._clap, self._scales = None, None, {}
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.db().executescript(SCHEMA)
        if os.path.abspath(self.path) == os.path.abspath(os.path.join(HOME_DIR, "index.db")):
            self._take_in_json()

    # ---------------------------------------------------------------- the database
    def db(self):
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=120)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.row_factory = sqlite3.Row
            self._local.c = c
        return c

    def _commit(self, db):
        db.execute("INSERT INTO meta VALUES ('gen', '1') ON CONFLICT(k) DO UPDATE SET v = CAST(v AS INTEGER) + 1")
        db.execute("INSERT INTO meta VALUES ('updated', ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
                   (str(time.time()),))
        db.commit()

    def meta(self, k, default=None):
        r = self.db().execute("SELECT v FROM meta WHERE k = ?", (k,)).fetchone()
        return r[0] if r else default

    def gen(self):
        return int(self.meta("gen", "0"))

    def folders(self):
        return [r[0] for r in self.db().execute("SELECT path FROM folders ORDER BY path")]

    def count(self):
        return self.db().execute("SELECT COUNT(*) FROM files WHERE status = 0").fetchone()[0]

    def has(self, path):
        return self.db().execute("SELECT 1 FROM files WHERE path = ? AND status = 0", (path,)).fetchone() is not None

    def info(self):
        db = self.db()
        n = db.execute("SELECT COUNT(*), SUM(length(clap) > 0), SUM(status != 0) FROM files").fetchone()
        ok = self.count()
        return {"count": ok, "listened": n[1] or 0, "skipped": n[2] or 0, "folders": self.folders(),
                "updated": float(self.meta("updated", "0")) or None, "path": self.path}

    def _take_in_json(self):
        """An index.json from an older soundmon: its fingerprints come in once (the file is left alone)."""
        if not os.path.isfile(OLD_JSON) or self.meta("json_taken") or self.count():
            return
        try:
            with open(OLD_JSON, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            return
        db = self.db()
        if isinstance(d, dict) and d.get("method") == METHOD and isinstance(d.get("files"), dict):
            folders = [f for f in d.get("folders", []) if isinstance(f, str)]
            for f in folders:
                db.execute("INSERT OR IGNORE INTO folders VALUES (?, ?)", (f, time.time()))
            n = 0
            for p, f in d["files"].items():
                if not (isinstance(f, dict) and _valid_vec(f.get("vec"))):
                    continue
                root = max((r for r in folders if p.startswith(_prefix(r))), key=len, default=os.path.dirname(p))
                secs = f.get("seconds") or round(10 ** f["vec"][-1], 3)
                w = tags.describe(p, root, None, secs)
                db.execute("INSERT OR IGNORE INTO files (path, root, mtime, size, status, seconds, vec, method, tver,"
                           " wtag, wsure, kind, kind_by, bpm, key, indexed) VALUES (?,?,?,?,0,?,?,?,NULL,?,?,?,?,?,?,?)",
                           (p, root, f.get("mtime"), f.get("size"), secs, _blob(f["vec"]), METHOD, w["tag"], w["sure"],
                            w["kind"], w["kind_by"], w["bpm"], w["key"], time.time()))
                n += 1
            self.note = f"took in {n} sound(s) from the old index.json (it's left where it was)"
        db.execute("INSERT INTO meta VALUES ('json_taken', '1') ON CONFLICT(k) DO NOTHING")
        self._commit(db)

    # ---------------------------------------------------------------- indexing
    def _rows_under(self, folder):
        pre = _prefix(folder)
        # noemb: the file's own words not read yet, or read by an older tags.embedded()
        return self.db().execute("SELECT path, mtime, size, status, method, tver, (emb IS NULL OR "
                                 "COALESCE(json_extract(emb, '$.v'), 1) < ?) AS noemb FROM files "
                                 "WHERE path >= ? AND path < ?", (tags.EMB_VERSION, pre, pre + "\U0010ffff"))

    def _write(self, db, row):
        if row.get("gone"):
            return
        emb = row.get("emb")
        if row["need"] == "all":
            w = tags.describe(row["path"], row["root"], emb, row["seconds"])
            db.execute(
                "INSERT INTO files (path, root, mtime, size, status, seconds, vec, method, emb, tver, wtag, wsure,"
                " kind, kind_by, bpm, key, indexed) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(path) DO UPDATE SET root=excluded.root, status=excluded.status,"
                " seconds=excluded.seconds, vec=excluded.vec, method=excluded.method, emb=excluded.emb,"
                " tver=excluded.tver, wtag=excluded.wtag, wsure=excluded.wsure, kind=excluded.kind,"
                " kind_by=excluded.kind_by, bpm=excluded.bpm, key=excluded.key, indexed=excluded.indexed,"
                # the ear heard the old file: when the file changed, it listens again
                " clap = CASE WHEN files.mtime = excluded.mtime AND files.size = excluded.size THEN files.clap END,"
                " stag = CASE WHEN files.mtime = excluded.mtime AND files.size = excluded.size THEN files.stag END,"
                " ssure = CASE WHEN files.mtime = excluded.mtime AND files.size = excluded.size THEN files.ssure END,"
                " skind = CASE WHEN files.mtime = excluded.mtime AND files.size = excluded.size THEN files.skind END,"
                " mtime=excluded.mtime, size=excluded.size",
                (row["path"], row["root"], row["mtime"], row["size"], row["status"], row["seconds"],
                 _blob(row["vec"]) if row["vec"] else None, METHOD, json.dumps(emb) if emb is not None else None,
                 tags.VERSION,
                 w["tag"], w["sure"], w["kind"], w["kind_by"], w["bpm"], w["key"], time.time()))
        else:
            r = db.execute("SELECT seconds FROM files WHERE path = ?", (row["path"],)).fetchone()
            w = tags.describe(row["path"], row["root"], emb, r[0] if r else None)
            db.execute("UPDATE files SET emb=?, tver=?, wtag=?, wsure=?, kind=?, kind_by=?, bpm=?, key=? WHERE path=?",
                       (json.dumps(emb if emb is not None else {"v": tags.EMB_VERSION}), tags.VERSION, w["tag"], w["sure"], w["kind"], w["kind_by"],
                        w["bpm"], w["key"], row["path"]))

    def retag(self, progress=None):
        """Read every file's words again with this tags.VERSION, from what the index keeps (no file reads)."""
        db = self.db()
        rows = db.execute("SELECT id, path, root, seconds, emb FROM files WHERE tver IS NOT ? AND emb IS NOT NULL",
                          (tags.VERSION,)).fetchall()
        last = time.time()
        for i, r in enumerate(rows, 1):
            emb = json.loads(r["emb"]) if r["emb"] else None
            w = tags.describe(r["path"], r["root"], emb, r["seconds"])
            db.execute("UPDATE files SET tver=?, wtag=?, wsure=?, kind=?, kind_by=?, bpm=?, key=? WHERE id=?",
                       (tags.VERSION, w["tag"], w["sure"], w["kind"], w["kind_by"], w["bpm"], w["key"], r["id"]))
            if progress and (time.time() - last > 1 or i == len(rows)):
                last = time.time()
                progress("retagging", i, len(rows), r["path"])
        if rows:
            self._commit(db)
        return len(rows)

    def update(self, folders, progress=None, workers=None, stop=None):
        """Index these folders (added to the list). progress(phase, done, total, path); stop() -> True ends it
        early (what's done is kept). Returns {"added", "kept", "removed", "failed", "short", "skipped",
        "problems", "stopped"}: failed = unreadable, short = silent or too short to fingerprint, skipped =
        dangling links and folders that couldn't be listed."""
        folders = [indexed(os.path.realpath(here(os.path.expanduser(f)))) for f in folders]
        for f in folders:
            if not os.path.isdir(here(f)):
                raise ValueError(f"not a folder: {f}")
        db = self.db()
        for f in folders:
            db.execute("INSERT OR IGNORE INTO folders VALUES (?, ?)", (f, time.time()))
        db.commit()
        stats = {"added": 0, "kept": 0, "removed": 0, "failed": 0, "short": 0, "skipped": 0, "stopped": False}
        problems = []
        known = {}
        for f in folders:
            for r in self._rows_under(f):
                known[r["path"]] = (r["mtime"], r["size"], r["status"], r["method"], r["tver"], r["noemb"])
        jobs, seen, last = [], set(), time.time()
        for p, root in _walk(folders, problems):
            seen.add(p)
            if progress and time.time() - last > 1:
                last = time.time()
                progress("listing", len(seen), 0, p)
            if stop and stop():
                break
            k = known.get(p)
            if k is None:
                jobs.append((p, root, "all"))
                continue
            try:
                st = os.stat(here(p))
            except OSError:
                problems.append(p)
                continue
            if k[0] != st.st_mtime or k[1] != st.st_size or (k[2] == OK and k[3] != METHOD):
                jobs.append((p, root, "all"))
            elif k[2] == OK and k[5]:
                jobs.append((p, root, "words"))
                stats["kept"] += 1
            else:
                stats[{OK: "kept", SHORT: "short", UNREADABLE: "failed"}[k[2]]] += 1
        stopped = bool(stop and stop())
        n = len(jobs)
        if progress:
            progress("reading", 0, n, "")
        if jobs and not stopped:
            # half the cores at most by default: indexing a big library takes a while, and the machine is
            # still yours meanwhile
            workers = max(1, int(workers or min(8, max(1, (os.cpu_count() or 2) // 2))))
            stopped = self._read_all(db, jobs, workers, stats, problems, progress, stop)
        if not stopped:
            bad = [_prefix(p) for p in problems if os.path.isdir(here(p))] if problems else []
            for f in folders:                                   # files gone from disk leave the index
                for r in list(self._rows_under(f)):
                    p = r["path"]
                    if p not in seen and not any(p.startswith(b) for b in bad):
                        db.execute("DELETE FROM files WHERE path = ?", (p,))
                        stats["removed"] += r["status"] == OK
        self._commit(db)
        if not stopped:
            self.retag(progress)
        stats["stopped"] = stopped
        stats["skipped"] = len(problems)
        stats["problems"] = problems[:20]
        return stats

    def _read_all(self, db, jobs, workers, stats, problems, progress, stop):
        """Read the jobs with `workers` worker processes (this file run as --worker: they import only what
        reading a sound needs, and do their maths on one thread at low priority). True if stopped."""
        import queue
        env = dict(os.environ, **{v: "1" for v in _ONE_THREAD})
        it, lock, q = iter(jobs), threading.Lock(), queue.Queue()
        halt, cool, cooler = threading.Event(), threading.Event(), Cooler()
        cool.set()

        def start():
            return subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker"], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env, bufsize=1)

        def drive():
            pr = start()
            try:
                while not halt.is_set():
                    if not cool.is_set():                  # too hot: wait (the file in hand is finished)
                        cool.wait(0.5)
                        continue
                    with lock:
                        j = next(it, None)
                    if j is None:
                        break
                    try:
                        pr.stdin.write(json.dumps(j) + "\n")
                        pr.stdin.flush()
                        line = pr.stdout.readline()
                    except OSError:
                        line = ""
                    if line:
                        q.put(json.loads(line))
                    else:                                  # it died on this file: note it, carry on with a new one
                        q.put({"path": j[0], "gone": True, "died": True})
                        pr.kill()
                        pr = start()
            finally:
                try:
                    pr.stdin.close()
                    pr.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pr.kill()
        threads = [threading.Thread(target=drive, daemon=True) for _ in range(min(workers, len(jobs)))]
        for t in threads:
            t.start()
        n, done, last, last_commit, stopped = len(jobs), 0, 0.0, time.time(), False
        while any(t.is_alive() for t in threads) or not q.empty():
            try:
                row = q.get(timeout=0.5)
            except queue.Empty:
                row = None
            if row is not None:
                self._write(db, row)
                done += 1
                if row.get("gone"):
                    problems.append(row["path"])
                elif row["need"] == "all":
                    stats[{OK: "added", SHORT: "short", UNREADABLE: "failed"}[row["status"]]] += 1
                if progress and (time.time() - last > 0.5 or done == n):
                    last = time.time()
                    progress("reading", done, n, row["path"])
            if cooler.ok():
                cool.set()
            else:
                cool.clear()
                if progress and time.time() - last > 2:
                    last = time.time()
                    progress("cooling", done, n, cooler.note())
            if time.time() - last_commit > 3:
                self._commit(db)
                last_commit = time.time()
            if not stopped and stop and stop():
                stopped = True
                halt.set()                                 # each worker finishes the file it's on
                cool.set()
        stats["cooled"] = round(cooler.waited)
        return stopped

    def forget(self, folder):
        """Take a folder (and every file under it) out of the index."""
        folder = os.path.realpath(os.path.expanduser(folder))
        db = self.db()
        pre = _prefix(folder)
        db.execute("DELETE FROM files WHERE path >= ? AND path < ?", (pre, pre + "\U0010ffff"))
        db.execute("DELETE FROM folders WHERE path = ?", (folder,))
        self._commit(db)

    def set_tag(self, path, tag=None, kind=None):
        """Your tag (or kind) for a file: kept over the words' and the ear's, and the ear learns from it.
        tag "" clears yours; "-" says none of the tags fits."""
        if tag is not None and tag not in ("", "-") and tag not in tags.IDS:
            raise ValueError(f"no such tag: {tag}")
        if kind is not None and kind not in ("", "loop", "oneshot"):
            raise ValueError(f"kind is loop or oneshot, not {kind}")
        db = self.db()
        if not self.has(path):
            raise ValueError("that file isn't in the index")
        if tag is not None:
            db.execute("UPDATE files SET utag = ? WHERE path = ?", (tag or None, path))
        if kind is not None:
            db.execute("UPDATE files SET ukind = ? WHERE path = ?", (kind or None, path))
        self._commit(db)

    # ---------------------------------------------------------------- reading it back
    def table(self):
        """Every fingerprinted file as arrays (cached until the index changes)."""
        import numpy as np
        g = self.gen()
        if self._cache and self._cache["gen"] == g:
            return self._cache
        rows = self.db().execute("SELECT id, path, root, seconds, vec, wtag, wsure, stag, ssure, utag, kind, kind_by,"
                                 " skind, ukind, bpm, key, length(clap) > 0 FROM files WHERE status = 0"
                                 " AND vec IS NOT NULL ORDER BY path").fetchall()
        n = len(rows)
        code = {t: i for i, t in enumerate(tags.IDS)}
        c = {"gen": g, "n": n, "ids": np.array([r[0] for r in rows], dtype=np.int64), "paths": [r[1] for r in rows],
             "roots": [r[2] for r in rows], "seconds": np.array([r[3] or 0 for r in rows], dtype=float),
             "bpm": np.array([r[14] or 0 for r in rows], dtype=float), "key": [r[15] for r in rows],
             "heard": np.array([bool(r[16]) for r in rows], dtype=bool)}
        X = np.frombuffer(b"".join(r[4] for r in rows), dtype=np.float32).reshape(n, DIMS) if n else \
            np.zeros((0, DIMS), np.float32)
        mu, sd = (X.mean(axis=0), X.std(axis=0) + 1e-6) if n else (np.zeros(DIMS), np.ones(DIMS))
        c.update(X=X, mu=mu, sd=sd, Z=((X - mu) / sd).astype(np.float32))
        tag, by, sure, kind, kby = [], [], [], [], []
        bar = float(self.meta("ear_sure", EAR_SURE))
        for r in rows:
            t, b, s = effective_tag(r[9], r[5], r[6], r[7], r[8], bar)
            k, kb = effective_kind(r[13], r[10], r[11], r[12])
            tag.append(t), by.append(b), sure.append(s or 0), kind.append(k), kby.append(kb)
        c.update(tag=tag, by=by, sure=np.array(sure, dtype=float), kind=kind, kind_by=kby,
                 code=np.array([code.get(t, -1) for t in tag], dtype=np.int16),
                 lower=[p.lower() for p in c["paths"]], where={p: i for i, p in enumerate(c["paths"])})
        self._cache, self._scales = c, {}
        return c

    def clap_matrix(self):
        """(rows of table() that have been listened to, their unit embeddings), or (None, None)."""
        import numpy as np
        c = self.table()
        if self._clap and self._clap[0] == c["gen"]:
            return self._clap[1], self._clap[2]
        rows = self.db().execute("SELECT path, clap FROM files WHERE status = 0 AND length(clap) > 0").fetchall()
        if not rows:
            self._clap = (c["gen"], None, None)
            return None, None
        idx = np.array([c["where"].get(r[0], -1) for r in rows])
        M = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float16).reshape(len(rows), -1).astype(np.float32)
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
        keep = idx >= 0
        self._clap = (c["gen"], idx[keep], M[keep])
        return idx[keep], M[keep]

    def mask(self, filters=None):
        """A boolean mask over table() for the filters: tags [ids] (or "none" for untagged), families
        [drums, tonal, fx], by [words, sound, you], kind loop/oneshot/unknown, min_s/max_s, bpm_min/bpm_max,
        key ("Am", "F#"...), text (all its words in the path), within (a folder)."""
        import numpy as np
        c, f = self.table(), filters or {}
        m = np.ones(c["n"], dtype=bool)
        want = [t for t in (f.get("tags") or []) if t]
        if want:
            codes = [tags.IDS.index(t) for t in want if t in tags.IDS]
            m &= np.isin(c["code"], codes) | (("none" in want) & (c["code"] < 0))
        fam = [x for x in (f.get("families") or []) if x]
        if fam:
            codes = [i for i, t in enumerate(tags.IDS) if tags.FAMILY[t] in fam]
            m &= np.isin(c["code"], codes)
        if f.get("by"):
            by = set(f["by"]) if isinstance(f["by"], list) else {f["by"]}
            m &= np.array([b in by for b in c["by"]], dtype=bool)
        k = f.get("kind")
        if k:
            m &= np.array([(x or "unknown") == k for x in c["kind"]], dtype=bool)
        for key, op in (("min_s", np.greater_equal), ("max_s", np.less_equal)):
            if f.get(key) not in (None, ""):
                m &= op(c["seconds"], float(f[key]))
        if f.get("bpm_min") not in (None, "") or f.get("bpm_max") not in (None, ""):
            b = c["bpm"]
            m &= (b > 0) & (b >= float(f.get("bpm_min") or 0)) & (b <= float(f.get("bpm_max") or 999))
        if f.get("key"):
            want_k = str(f["key"]).strip().lower()
            m &= np.array([(x or "").lower() == want_k for x in c["key"]], dtype=bool)
        if f.get("text"):
            words = [w for w in str(f["text"]).lower().split() if w]
            if words:
                m &= np.array([all(w in p for w in words) for p in c["lower"]], dtype=bool)
        if f.get("within"):
            pre = _prefix(os.path.realpath(os.path.expanduser(f["within"])))
            m &= np.array([p.startswith(pre) for p in c["paths"]], dtype=bool)
        return m

    def _row(self, c, i, score=None, aspects=None):
        return {"path": c["paths"][i], "score": score, "seconds": round(float(c["seconds"][i]), 3),
                "aspects": aspects or [], "tag": c["tag"][i], "tag_by": c["by"][i],
                "tag_label": tags.LABEL.get(c["tag"][i]), "sure": round(float(c["sure"][i]), 2),
                "kind": c["kind"][i], "bpm": float(c["bpm"][i]) or None, "key": c["key"][i],
                "heard": bool(c["heard"][i])}

    def _weights(self, weights):
        w = {name: wt for name, _, wt in GROUPS}
        w["character"] = None                                 # None: CHARACTER when there's an ear to ask
        for k, v in (weights or {}).items():
            if k in w and v is not None:
                w[k] = max(0.0, float(v))
        return w

    def _scale(self, c, w, Mc):
        """The typical distance between two of the index's sounds under these weights (a fixed sample)."""
        import numpy as np
        key = (c["gen"], tuple(sorted((k, v) for k, v in w.items() if k != "character")))
        if key in self._scales:
            return self._scales[key]
        if c["n"] < 8:
            return TYPICAL
        pick = np.linspace(0, c["n"] - 1, min(c["n"], 200)).astype(int)
        ZS = c["Z"][pick]
        tot = sum(w[g] for g, _, _ in GROUPS) or 1.0
        pair = []
        for i in range(len(ZS) - 1):
            d = sum(w[g] * np.sqrt(((ZS[i + 1:, sl] - ZS[i, sl]) ** 2).mean(axis=1)) for g, sl, _ in GROUPS) / tot
            pair.append(d)
        s = float(np.median(np.concatenate(pair))) or TYPICAL
        self._scales[key] = s
        return s

    def search(self, query, k=20, within=None, vec=None, filters=None, weights=None, listener=None):
        """The k files most like `query` (a path; or a fingerprint via vec), among those the filters leave.
        weights: {timbre, brightness, envelope, length, character} (any left out keep their default;
        character, what the audio model hears, counts when both sounds have been listened to). Raises
        ValueError when the query can't be read or is silent; files deleted since indexing are left out."""
        import numpy as np
        c = self.table()
        if within:
            filters = dict(filters or {}, within=within)
        m = self.mask(filters)
        if not c["n"] or not m.any():
            return []
        qi = c["where"].get(query) if query else None
        if vec is None and qi is not None:
            vec = c["X"][qi]
        if vec is None:
            try:
                vec = features(query)
            except Exception as e:
                raise ValueError(f"can't read {os.path.basename(query)}: {e}")
            if vec is None:
                raise ValueError(f"{os.path.basename(query)} is silent or too short to compare")
        zq = ((np.array(vec, dtype=float) - c["mu"]) / c["sd"]).astype(np.float32)
        w = self._weights(weights)
        idx = np.nonzero(m)[0]
        Z = c["Z"][idx]
        per = {g: np.sqrt(((Z[:, sl] - zq[sl]) ** 2).mean(axis=1)) for g, sl, _ in GROUPS}
        tot = sum(w[g] for g, _, _ in GROUPS)
        scale = self._scale(c, w, None)
        d = (sum(w[g] * per[g] for g, _, _ in GROUPS) / tot / scale) if tot else np.zeros(len(idx))
        # what the ear hears: when the query and the candidates have been listened to
        a = w["character"]
        rows, M = self.clap_matrix() if a != 0 else (None, None)
        if M is not None:
            cq = None
            if qi is not None and c["heard"][qi]:
                cq = M[np.nonzero(rows == qi)[0][0]]
            elif listener is not None:
                try:
                    cq = listener(query)
                except Exception:
                    cq = None
            if cq is not None:
                a = CHARACTER if a is None else a / (a + tot) if tot else 1.0
                pos = np.full(c["n"], -1)
                pos[rows] = np.arange(len(rows))
                pi = pos[idx]
                dc = np.full(len(idx), np.nan)
                have = pi >= 0
                dc[have] = 1.0 - M[pi[have]] @ cq
                med = float(np.nanmedian(1.0 - M[np.linspace(0, len(M) - 1, min(len(M), 400)).astype(int)] @ cq)) \
                    if len(M) else 1.0
                dc = dc / (med or 1.0)
                per["character"] = np.where(have, dc, np.nan)
                d = np.where(have, (1 - a) * d + a * dc, d + 0.25)   # unheard files: the fingerprint, a step behind
        qreal = os.path.realpath(query) if query else None
        out, seen = [], set()
        order = np.argsort(d, kind="stable")
        thr = {g: float(np.nanpercentile(v, 15)) for g, v in per.items() if np.isfinite(v).any()}
        for j in order:
            if len(out) >= k:
                break
            i = int(idx[j])
            p = c["paths"][i]
            real = os.path.realpath(p)
            if real == qreal or real in seen or not os.path.isfile(p):   # itself (or a link to it), gone
                continue
            same = (int(c["X"][i].tobytes().__hash__()), float(c["seconds"][i]))
            if same in seen:                                            # one copy of identical files
                continue
            seen.update((same, real))
            aspects = [g for g in thr if np.isfinite(per[g][j]) and per[g][j] <= thr[g]]
            out.append(self._row(c, i, int(round(100 * math.exp(-float(d[j]) * 0.7))), aspects))
        return out

    def browse(self, filters=None, k=50, offset=0, describe=None, listener=None):
        """The files the filters leave, by name - or, with describe ("a dark metallic hit") and an ear,
        those that sound most like the words. {"total", "results"}."""
        import numpy as np
        c = self.table()
        m = self.mask(filters)
        if describe:
            rows, M = self.clap_matrix()
            if M is None or listener is None:
                raise ValueError("searching by description needs the index listened to first (soundmon --listen)")
            t = listener(describe)
            keep = m[rows]
            sims = M[keep] @ t
            order = np.argsort(-sims)
            pick = rows[keep][order]
            total = int(keep.sum())
            res = []
            for j, i in enumerate(pick[offset:offset + k]):
                res.append(self._row(c, int(i), int(round(100 * max(0.0, float(sims[order[offset + j]]))))))
            return {"total": total, "results": res}
        idx = np.nonzero(m)[0]
        return {"total": int(len(idx)), "results": [self._row(c, int(i)) for i in idx[offset:offset + k]]}

    def summary(self, filters=None):
        """How many of the files the filters leave carry each tag, by whom, and each kind."""
        import collections
        c = self.table()
        m = self.mask(filters)
        t, by, kind = collections.Counter(), collections.Counter(), collections.Counter()
        for i in m.nonzero()[0]:
            t[c["tag"][i] or "none"] += 1
            by[c["by"][i] or "none"] += 1
            kind[c["kind"][i] or "unknown"] += 1
        return {"total": int(m.sum()), "tags": dict(t), "by": dict(by), "kinds": dict(kind)}


def _filters(a):
    f = {}
    if getattr(a, "find_tag", None):
        f["tags"] = [t.strip() for t in a.find_tag.split(",") if t.strip()]
        bad = [t for t in f["tags"] if t not in tags.IDS and t != "none"]
        if bad:
            sys.exit(f"--tag: no such tag {', '.join(bad)} — the tags are: {', '.join(tags.IDS)} (or none)")
    if getattr(a, "find_kind", None):
        if a.find_kind not in ("loop", "oneshot", "unknown"):
            sys.exit("--kind is loop, oneshot or unknown")
        f["kind"] = a.find_kind
    if getattr(a, "within", None):
        w = os.path.expanduser(a.within)
        if not os.path.isdir(w):
            sys.exit(f"--in {a.within}: no such folder")
        f["within"] = w
    return f


def _line(i, r):
    tag = r["tag_label"] or "-"
    by = {"words": "", "sound": "~", "you": "*"}.get(r["tag_by"] or "", "")
    extra = " ".join(x for x in (r["kind"] or "", f"{r['bpm']:g} bpm" if r["bpm"] else "", r["key"] or "") if x)
    sc = f"{r['score']:>3}%  " if r["score"] is not None else ""
    return (f"   {i:>3}. {sc}{r['seconds'] or 0:6.2f}s  {by}{tag:<22} {extra:<20} {r['path']}"
            + (f"   [{', '.join(r['aspects'])}]" if r["aspects"] else ""))


def run(a, slug=None, to_ogg=None, loudness_normalize=None, to_flac=None):
    """soundmon --index DIR[,DIR]  /  --similar FILE  /  --find WORDS  /  --describe TEXT  /  --listen  /  --tags"""
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
    k = getattr(a, "similar_k", None) or 20
    if getattr(a, "index", None) is not None:
        folders = [f.strip() for f in a.index.split(",") if f.strip()]
        if not folders:
            sys.exit("--index: give a folder (or several, comma-separated)")
        print(f"🔎 indexing {', '.join(folders)} -> {idx.path}")
        last, t0 = [0.0, ""], time.time()

        def prog(phase, i, n, p):
            if time.time() - last[0] > 2 or i == n or phase != last[1]:
                last[0], last[1] = time.time(), phase
                if phase == "listing":
                    print(f"   listing… {i} files so far", flush=True)
                else:
                    print(f"   {phase} {i}/{n}  {os.path.basename(p)}", flush=True)
        try:
            st = idx.update(folders, prog, workers=getattr(a, "workers", None))
        except ValueError as e:
            sys.exit(f"--index: {e}")
        except (OSError, sqlite3.Error) as e:
            sys.exit(f"--index: can't write the index {idx.path}: {e}")
        print(f"   {'stopped' if st['stopped'] else 'all done'}  |  {idx.count()} sound(s) indexed  "
              f"(new or changed {st['added']}, unchanged {st['kept']}, gone {st['removed']}, "
              f"unreadable {st['failed']}, silent or too short {st['short']}"
              + (f", skipped {st['skipped']}: dangling links or folders that couldn't be read" if st["skipped"] else "")
              + f")  in {time.time() - t0:.0f} s"
              + (f" (resting {st['cooled']} s of it to keep the CPU under {MAX_TEMP:.0f} °C)" if st.get("cooled") else ""))
        for p in st["problems"][:5]:
            print(f"      skipped: {p}")
    if getattr(a, "listen", False):
        import listen
        ok, why = listen.available()
        if not ok:
            sys.exit(f"--listen: {why}")
        if not idx.count():
            sys.exit("--listen: the index is empty — build it first with  soundmon --index DIR[,DIR]")
        print(f"👂 listening to the index ({listen.MODEL})")
        last = [0.0]

        def lprog(phase, i, n, p):
            if time.time() - last[0] > 2 or i == n:
                last[0] = time.time()
                print(f"   {phase} {i}/{n}  {os.path.basename(p or '')}", flush=True)
        st = listen.listen(idx, lprog, device=getattr(a, "listen_device", None))
        print(f"   heard {st['heard']} new, {st['failed']} unreadable, on {st['device']}; "
              f"learned from {st['learned_from']} sound(s) tagged by their words"
              + (f" (on ones held back it agreed with their words {st['agree']:.0%} of the time)"
                 if st["agree"] is not None else "")
              + f"; {st['tagged']} more tagged by ear (where it's at least {st['bar']:.0%} sure)"
              + (f"  (rested {st['cooled']} s to keep the CPU under {MAX_TEMP:.0f} °C)" if st.get("cooled") else ""))
    if getattr(a, "tags", False):
        if not idx.count():
            sys.exit("--tags: the index is empty — build it first with  soundmon --index DIR[,DIR]")
        s = idx.summary(_filters(a))
        print(f"🏷  {s['total']} sound(s): tagged by their words {s['by'].get('words', 0)}, "
              f"by ear {s['by'].get('sound', 0)}, by you {s['by'].get('you', 0)}, untagged {s['by'].get('none', 0)}")
        print("   " + ", ".join(f"{k_} {v}" for k_, v in sorted(s["kinds"].items(), key=lambda x: -x[1])))
        for fam in ("drums", "tonal", "fx"):
            row = [(t, s["tags"].get(t, 0)) for t in tags.IDS if tags.FAMILY[t] == fam]
            print(f"   {fam:6} " + ", ".join(f"{tags.LABEL[t]} {n}" for t, n in row))
    if getattr(a, "find", None) is not None or getattr(a, "describe", None):
        f = _filters(a)
        if a.find:
            f["text"] = a.find
        listener = None
        if a.describe:
            import listen
            ok, why = listen.available()
            if not ok:
                sys.exit(f"--describe: {why}")
            listener = listen.text_query(device=getattr(a, "listen_device", None))
        try:
            res = idx.browse(f, k, describe=a.describe, listener=listener)
        except ValueError as e:
            sys.exit(f"--describe: {e}")
        what = " and ".join(x for x in (f"named {a.find!r}" if a.find else "", f"sounding like {a.describe!r}"
                                        if a.describe else "") if x)
        print(f"🔎 {res['total']} sound(s) {what}" + (f" (the first {len(res['results'])})" if res["total"] > k else ""))
        for i, r in enumerate(res["results"], 1):
            print(_line(i, r))
    if getattr(a, "similar", None) is not None:
        q = os.path.expanduser(a.similar)
        if not a.similar.strip() or not os.path.isfile(q):
            sys.exit(f"--similar: no such file: {a.similar!r}")
        if not idx.count():
            sys.exit("--similar: the index is empty — build it first with  soundmon --index DIR[,DIR]")
        if k < 1:
            sys.exit(f"--similar: -n must be 1 or more, got {k}")
        f = _filters(a)
        if f.get("within"):
            rw = _prefix(os.path.realpath(f["within"]))
            if not any(p.startswith(rw) for p in idx.table()["paths"]):
                sys.exit(f"--similar: nothing indexed inside {a.within} — index it first:  soundmon --index {a.within}")
        weights = {}
        for part in (getattr(a, "aspects", None) or "").split(","):
            if "=" in part:
                name, _, v = part.partition("=")
                if name.strip() not in ASPECTS:
                    sys.exit(f"--aspects: {name.strip()} isn't one of {', '.join(ASPECTS)}")
                try:
                    weights[name.strip()] = float(v)
                except ValueError:
                    sys.exit(f"--aspects: {part} — give name=number")
        try:
            res = idx.search(q, k, filters=f, weights=weights)
        except ValueError as e:
            sys.exit(f"--similar: {e}")
        print(f"🔎 sounds like {os.path.basename(q)}  ({idx.count()} indexed)")
        if not res:
            print("   no other sounds to compare with" + (" that the filters leave" if f else ""))
        for i, r in enumerate(res, 1):
            print(_line(i, r))


def _worker():
    """soundmon's indexing worker: a job a line on stdin (path, root, need), a row a line on stdout."""
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    for line in sys.stdin:
        try:
            row = _work(tuple(json.loads(line)))
        except Exception:
            row = {"path": json.loads(line)[0], "gone": True}
        sys.stdout.write(json.dumps(row) + "\n")
        sys.stdout.flush()


if __name__ == "__main__" and sys.argv[1:2] == ["--worker"]:
    _worker()
