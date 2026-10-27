"""Sets: a list of game sounds (a manifest), many takes per key, one pick per key, exported into the game.

This is the workflow neon-caster's Sound Picker and qb64-dungeon's generate-from-manifest.sh each built
by hand. A set lives in <sets>/<id>/set.json:

    {"id", "name", "created", "updated",
     "theme": {"engine": "sfx", "params": {...form params...}, "takes": 10,
               "game_folder": "~/git/game/sfx", "export": {"encode": "wav", "rate": 0, "mono": false}},
     "rows":  [{"key", "prompt", "seconds", "group", "extra": {...}}],     # frozen manifest snapshot
     "picks": {key: "<job>/<file>" | "now"},  "flags": {key: ["<job>/<file>", ...]},
     "notes": {key: "..."},  "exported": {key: {"at", "path", "from"}}}

Takes are NOT stored here: every render job records {"set": id, "key": key} in its params, so a key's
takes are the gallery's jobs for it. Nothing can get out of step, and a set survives a GUI restart.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import threading
import time

AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3", ".opus")
LOCK = threading.Lock()


def _read(path, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def slug(text, n=48):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:n] or "sound"


KEY_RE = re.compile(r"[\w.\-/]{1,120}")


def _secs(field):
    """'2' / '1.5s' / '30-60s loop' -> (seconds, loop?)  ('' -> (None, False))."""
    f = field.strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?))?\s*s?\s*(loop)?", f)
    if not m:
        return None
    lo, hi = float(m.group(1)), float(m.group(2)) if m.group(2) else None
    return (hi or lo), bool(m.group(3))


def parse_manifest(text):
    """Every manifest shape the games used, one row per sound:

        key | text                        narration (narrate.parse_lines' format)
        key | secs | prompt               qb64-dungeon sfx/   (secs may be '30-60s loop')
        name | secs | bpm | key | mood    qb64-dungeon tracks.txt (chip / AdLib music)
        a line of words                   key = its slug
        [Group name]                      the rows below go in that group
        # comment                         skipped
    """
    rows, seen, group = [], set(), ""
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"\[(.+)\]", line)
        if m:
            group = m.group(1).strip()
            continue
        parts = [p.strip() for p in line.split("|")]
        row = {"group": group, "extra": {}}
        if len(parts) == 1:
            row["key"], row["prompt"] = slug(parts[0]), parts[0]
        elif len(parts) == 2:
            row["key"], row["prompt"] = parts[0], parts[1]
        elif len(parts) >= 5 and _secs(parts[1]) and re.fullmatch(r"\d+(\.\d+)?", parts[2]):
            row["key"], row["prompt"] = parts[0], parts[0].replace("-", " ")
            row["seconds"] = _secs(parts[1])[0]
            row["extra"] = {"bpm": parts[2], "key": parts[3], "mood": parts[4]}
        elif _secs(parts[1]):
            secs, loop = _secs(parts[1])
            row["key"], row["prompt"], row["seconds"] = parts[0], " | ".join(parts[2:]), secs
            if loop:
                row["extra"]["loop"] = True
        else:
            row["key"], row["prompt"] = parts[0], " | ".join(parts[1:])
        if re.fullmatch(r"\d+", row["key"]):            # numeric keys say nothing: name it from the text
            row["key"] = slug(row["prompt"])
        if not KEY_RE.fullmatch(row["key"]):
            raise ValueError(f"{row['key']!r} can't be a file name — keys use letters, digits, . - _ /")
        if row["key"] in seen:
            raise ValueError(f"key {row['key']!r} appears twice")
        if not row["group"]:                             # combat.mon.death.x -> combat · sfx/door -> sfx
            g = re.split(r"[./]", row["key"], maxsplit=1)
            row["group"] = g[0] if len(g) > 1 else ""
        seen.add(row["key"])
        rows.append(row)
    if not rows:
        raise ValueError("no sounds in that list — one per line: `key | text`, `key | secs | prompt`, or plain words")
    return rows


class Sets:
    def __init__(self, root, gallery, home_path):
        self.root, self.gallery, self.home_path = root, gallery, home_path
        os.makedirs(root, exist_ok=True)

    # ---------------- storage ----------------
    def path(self, sid):
        sid = re.sub(r"[^\w-]", "", str(sid or ""))[:64]
        if not sid:
            raise ValueError("no set given")
        return os.path.join(self.root, sid, "set.json")

    def load(self, sid):
        s = _read(self.path(sid), None)
        if not s:
            raise ValueError("that set doesn't exist")
        return s

    def save(self, s):
        s["updated"] = time.time()
        _write(self.path(s["id"]), s)
        return s

    def list(self):
        out = []
        for f in sorted(glob.glob(os.path.join(self.root, "*", "set.json"))):
            s = _read(f, None)
            if s:
                out.append({"id": s["id"], "name": s["name"], "keys": len(s["rows"]),
                            "engine": s["theme"].get("engine"), "updated": s.get("updated")})
        return sorted(out, key=lambda x: -(x["updated"] or 0))

    def create(self, body, validate):
        name = str(body.get("name") or "").strip()[:80]
        if not name:
            raise ValueError("give the set a name")
        rows = parse_manifest(body.get("manifest"))
        theme = self._theme(body.get("theme") or {}, validate)
        sid = time.strftime("%Y%m%d-") + slug(name, 40)
        with LOCK:
            if os.path.exists(self.path(sid)):
                sid += time.strftime("-%H%M%S")
            s = {"id": sid, "name": name, "created": time.time(), "theme": theme, "rows": rows,
                 "picks": {}, "flags": {}, "notes": {}, "exported": {}}
            return self.save(s)

    def _theme(self, t, validate):
        eng = t.get("engine") or "sfx"
        params = dict(t.get("params") or {}, engine=eng)
        for k in ("prompt", "seed", "n", "name", "set", "from_midi", "from_audio"):
            params.pop(k, None)
        validate(dict(params, prompt="test", n=1))          # a theme the CLI would refuse is caught now
        # blips and narration are deterministic (same text, same sound): one take per key, or one per voice
        takes = 1 if eng in ("narrate", "blip") else int(t.get("takes") or 10)
        if not 1 <= takes <= 50:
            raise ValueError("takes per key must be 1..50")
        folder = str(t.get("game_folder") or "").strip()
        if folder:
            self.home_path(folder, must_exist=False)
        ex = t.get("export") or {}
        enc = ex.get("encode") or "wav"
        if enc not in ("wav", "flac", "ogg", "same"):
            raise ValueError("export as wav, flac, ogg, or the take's own format")
        rate = int(ex.get("rate") or 0)
        if rate and not 4000 <= rate <= 192000:
            raise ValueError("export sample rate must be 4000..192000 Hz (or blank)")
        voices = [v for v in (t.get("voices") or []) if v][:12]
        mst = ex.get("master") or "off"
        if mst not in ("off", "sfx", "speech", "like"):
            raise ValueError("master on export: off, sfx, speech or like the reference")
        if mst == "like" and not params.get("ref"):
            raise ValueError("“match the reference” needs a reference sound in the theme (Reference, on the form)")
        return {"engine": eng, "params": params, "takes": takes, "game_folder": folder, "voices": voices,
                "export": {"encode": enc, "rate": rate, "mono": bool(ex.get("mono")), "master": mst}}

    def update(self, body, validate):
        with LOCK:
            s = self.load(body.get("id"))
            if "name" in body:
                s["name"] = str(body["name"] or s["name"]).strip()[:80] or s["name"]
            if "theme" in body:
                s["theme"] = self._theme(body["theme"], validate)
            if "manifest" in body:                         # re-import: keep picks/flags for keys that remain
                rows = parse_manifest(body["manifest"])
                keys = {r["key"] for r in rows}
                for d in ("picks", "flags", "notes"):
                    s[d] = {k: v for k, v in s[d].items() if k in keys}
                s["rows"] = rows
            if isinstance(body.get("row"), dict):          # edit one row's prompt / seconds
                r = body["row"]
                row = next((x for x in s["rows"] if x["key"] == r.get("key")), None)
                if not row:
                    raise ValueError("no such key in this set")
                if "prompt" in r:
                    p = str(r["prompt"] or "").strip()
                    if not p or "\n" in p or len(p) > 2000:
                        raise ValueError("a prompt is one line of up to 2000 characters")
                    row["prompt"] = p
                if "seconds" in r:
                    row["seconds"] = float(r["seconds"]) if r["seconds"] not in (None, "") else None
            return self.save(s)

    def mark(self, body):
        """pick / unpick / keep-old / flag / unflag / note."""
        with LOCK:
            s = self.load(body.get("id"))
            key = body.get("key")
            if key not in {r["key"] for r in s["rows"]}:
                raise ValueError("no such key in this set")
            take = body.get("take")
            if take not in (None, "now") and not self._take_ok(s["id"], key, take):
                raise ValueError("that take isn't one of this key's")
            act = body.get("action")
            if act == "pick":
                if take is None:
                    s["picks"].pop(key, None)
                else:
                    s["picks"][key] = take
            elif act in ("flag", "unflag"):
                fl = set(s["flags"].get(key) or [])
                fl.add(take) if act == "flag" else fl.discard(take)
                s["flags"][key] = sorted(fl)
                if not s["flags"][key]:
                    del s["flags"][key]
            elif act == "note":
                s["notes"][key] = str(body.get("note") or "")[:500]
                if not s["notes"][key]:
                    del s["notes"][key]
            else:
                raise ValueError(f"unknown action {act!r}")
            return self.save(s)

    # ---------------- takes (from the gallery) ----------------
    def takes(self, sid):
        """{key: [{"take": "<job>/<file>", "job", "file", "seed", "voice", "created"}]} oldest first."""
        out = {}
        for jf in sorted(glob.glob(os.path.join(self.gallery, "*", "job.json"))):
            job = _read(jf, None)
            ref = (job or {}).get("params", {}).get("set") or {}
            if ref.get("id") != sid or not job.get("outputs"):
                continue
            for o in job["outputs"]:
                out.setdefault(ref.get("key"), []).append({
                    "take": f"{job['dir']}/{o['file']}", "job": job["dir"], "file": o["file"], "seed": o.get("seed"),
                    "voice": job["params"].get("voice"), "created": job.get("created"), "command": job.get("command")})
        return out

    def pending(self, sid, jobs):
        """{key: n} renders queued or running for this set right now."""
        out = {}
        for j in jobs:
            ref = j["params"].get("set") or {}
            if ref.get("id") == sid and j["status"] in ("queued", "running"):
                out[ref.get("key")] = out.get(ref.get("key"), 0) + 1
        return out

    def _take_ok(self, sid, key, take):
        return any(t["take"] == take for t in self.takes(sid).get(key, []))

    def take_path(self, take):
        full = os.path.realpath(os.path.join(self.gallery, str(take or "")))
        if not full.startswith(os.path.realpath(self.gallery) + os.sep) or not os.path.isfile(full):
            raise ValueError("that take's file is gone")
        return full

    # ---------------- what the game has today ----------------
    def game_dir(self, s):
        if not s["theme"].get("game_folder"):
            return None
        return self.home_path(s["theme"]["game_folder"], must_exist=False)

    def now_files(self, s):
        """{key: [file names in the game folder with that key's stem]} — more than one = two formats."""
        d = self.game_dir(s)
        if not d or not os.path.isdir(d):
            return {}
        keys, out = {r["key"] for r in s["rows"]}, {}
        for r in s["rows"]:
            sub = os.path.join(d, os.path.dirname(r["key"]))
            stem = os.path.basename(r["key"])
            if os.path.isdir(sub):
                hits = sorted(f for f in os.listdir(sub) if os.path.splitext(f)[0] == stem and f.lower().endswith(AUDIO_EXT))
                if hits:
                    out[r["key"]] = [os.path.join(os.path.dirname(r["key"]), h) for h in hits]
        return {k: v for k, v in out.items() if k in keys}

    def now_path(self, s, key):
        """The file the game plays today for key (its first format), confined to the game folder."""
        files = self.now_files(s).get(key)
        if not files:
            raise ValueError("the game has no file for that key yet")
        d = self.game_dir(s)
        full = os.path.realpath(os.path.join(d, files[0]))
        if not full.startswith(os.path.realpath(d) + os.sep):
            raise ValueError("outside the game folder")
        return full

    # ---------------- render params for one key ----------------
    def render_params(self, s, key, n=None, seed=None, voice=None):
        row = next((r for r in s["rows"] if r["key"] == key), None)
        if not row:
            raise ValueError("no such key in this set")
        t = s["theme"]
        p = dict(t["params"], engine=t["engine"], prompt=row["prompt"], set={"id": s["id"], "key": key})
        p["name"] = re.sub(r"[^\w.\- ]", "_", os.path.basename(key))[:64]
        if row.get("seconds") and t["engine"] not in ("narrate", "blip"):
            p["seconds"] = row["seconds"]
        ex = row.get("extra") or {}
        if ex.get("loop") and t["engine"] == "music":
            p["loop"] = True
        if t["engine"] in ("chip", "opl", "song"):
            for k in ("bpm", "key", "mood"):
                if ex.get(k):
                    p[k] = ex[k]
        if t["engine"] in ("narrate", "blip"):
            p["n"] = 1
            if voice:
                p["voice"] = voice
        else:
            p["n"] = int(n or t["takes"])
        if seed is not None:
            p["seed"], p["n"] = int(seed), 1
        return p

    # ---------------- export ----------------
    def export_plan(self, s):
        d = self.game_dir(s)
        if not d:
            raise ValueError("this set has no game folder — set one in the set's theme")
        ex, now, plan = s["theme"]["export"], self.now_files(s), []
        for r in s["rows"]:
            k, pick = r["key"], s["picks"].get(r["key"])
            if not pick or pick == "now":
                continue
            src = self.take_path(pick)
            ext = os.path.splitext(src)[1].lower() if ex["encode"] == "same" else "." + ex["encode"]
            rel = k + ext
            dest = os.path.realpath(os.path.join(d, rel))
            if not dest.startswith(os.path.realpath(d) + os.sep):
                raise ValueError(f"key {k!r} points outside the game folder")
            others = [f for f in now.get(k, []) if f != rel]
            done = s["exported"].get(k)
            mst = ex.get("master") or "off"
            plan.append({"key": k, "from": pick, "to": rel, "replaces": os.path.exists(dest), "other_formats": others,
                         "convert": ext != os.path.splitext(src)[1].lower() or bool(ex["rate"]) or ex["mono"],
                         "master": mst,
                         "unchanged": bool(done and done.get("from") == pick and (done.get("master") or "off") == mst
                                           and done.get("path") == rel and os.path.exists(dest))})
        return plan

    def export(self, s, only_changed=True, master=None):
        """master: the master.master function (soundmon's mastering chain), needed when the theme masters."""
        d, ex = self.game_dir(s), s["theme"]["export"]
        plan = self.export_plan(s)
        backup = os.path.join(d, ".soundmon-backup", time.strftime("%Y-%m-%d_%H%M%S"))
        done, errors = [], []
        for item in plan:
            if only_changed and item["unchanged"]:
                continue
            src, dest = self.take_path(item["from"]), os.path.join(d, item["to"])
            try:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if os.path.exists(dest):                     # nothing is replaced without a copy kept
                    b = os.path.join(backup, item["to"])
                    os.makedirs(os.path.dirname(b), exist_ok=True)
                    shutil.copy2(dest, b)
                if item["master"] != "off":           # neon-caster's chain, then the format the game wants
                    if master is None:
                        raise RuntimeError("mastering isn't available (master.py didn't load)")
                    mst = os.path.join(os.path.dirname(src), "_mastered", item["master"], os.path.splitext(os.path.basename(src))[0] + ".wav")
                    # "like" depends on the theme's reference too, which can change: always redo it
                    if item["master"] == "like" or not os.path.exists(mst) or os.path.getmtime(mst) < os.path.getmtime(src):
                        master(src, mst, item["master"])
                    src = mst
                convert(src, dest, ex)
                s["exported"][item["key"]] = {"at": time.time(), "path": item["to"], "from": item["from"], "master": item["master"]}
                done.append(item)
            except Exception as e:
                errors.append({"key": item["key"], "error": str(e)[:300]})
        with LOCK:
            fresh = self.load(s["id"])
            fresh["exported"].update(s["exported"])
            self.save(fresh)
        return {"exported": done, "errors": errors, "backup": backup if os.path.isdir(backup) else None,
                "skipped": sum(1 for i in plan if i["unchanged"]) if only_changed else 0}


def convert(src, dest, ex):
    """Copy a take into the game, converting format / rate / channels with ffmpeg only when needed."""
    s_ext, d_ext = os.path.splitext(src)[1].lower(), os.path.splitext(dest)[1].lower()
    if s_ext == d_ext and not ex.get("rate") and not ex.get("mono"):
        shutil.copy2(src, dest)
        return
    if not shutil.which("ffmpeg"):
        raise RuntimeError("converting needs ffmpeg on PATH")
    opts = (["-ar", str(ex["rate"])] if ex.get("rate") else []) + (["-ac", "1"] if ex.get("mono") else [])
    tmp = dest + ".part" + d_ext
    r = _encode(src, tmp, d_ext, opts)
    if r.returncode or not os.path.exists(tmp):
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError("ffmpeg: " + (r.stderr or "conversion failed").strip()[-200:])
    if d_ext == ".ogg":
        # A lossy encode rebuilds the waveform, and its inter-sample peaks can land above a WAV that was
        # under -1 dBTP (a mastered take came back at -0.5). Measure the OGG itself; if it's over, encode
        # again that much quieter — the same guard as soundmon's --true-peak.
        tp = true_peak(tmp)
        if tp is not None and tp > -1.0:
            again = dest + ".part2" + d_ext
            r = _encode(src, again, d_ext, opts + ["-af", "volume=%.2fdB" % (-1.0 - tp - 0.2)])
            if not r.returncode and os.path.exists(again):
                os.replace(again, tmp)
            elif os.path.exists(again):
                os.remove(again)
    os.replace(tmp, dest)


def _encode(src, out, ext, opts):
    """ffmpeg src -> out in ext's codec; OGG falls back to oggenc when ffmpeg has no libvorbis
    (qb64-dungeon's NIGHT-NOTES)."""
    base = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-map_metadata", "-1"] + opts
    codec = {".wav": ["-c:a", "pcm_s16le"], ".flac": ["-c:a", "flac"], ".ogg": ["-c:a", "libvorbis", "-q:a", "6"]}.get(ext, [])
    r = subprocess.run(base + codec + [out], capture_output=True, text=True)
    if r.returncode and ext == ".ogg" and shutil.which("oggenc"):
        wav = out + ".wav"
        r = subprocess.run(base + ["-c:a", "pcm_s16le", wav], capture_output=True, text=True)
        if not r.returncode:
            r = subprocess.run(["oggenc", "-Q", "-q", "6", "-o", out, wav], capture_output=True, text=True)
        if os.path.exists(wav):
            os.remove(wav)
    return r


def true_peak(path):
    """A file's true peak in dBTP (ffmpeg's EBU R128 scanner), or None."""
    r = subprocess.run(["ffmpeg", "-nostdin", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.findall(r"Peak:\s*(-?[\d.]+) dBFS", r.stderr)
    return float(m[-1]) if m else None
