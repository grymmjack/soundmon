#!/usr/bin/env python3
"""soundmon-gui — a local web front end for the soundmon CLI.

Serves a single-page UI, queues render jobs, and runs them one at a time by
shelling out to bin/soundmon (always --no-open --output-to <job folder>). Each
job renders into its own folder under the gallery dir with a job.json recording
the exact settings, the command and every take's seed, so any take can be made
again exactly without ComfyUI's history.

Sibling of pixelmon-gui (same skeleton, same look). The render server is a
servers.json alias, host[:port] or URL: --server NAME, else $SOUNDMON_SERVER,
else the one picked in the Setup tab, else servers.json "_default", else "local".
CPU engines (chiptune, AdLib, retro FX, blips, narration) never touch it.

Standard library only. usage: soundmon-gui [--port 8191] [--lan] [--server NAME] [--gallery DIR]
"""
import argparse
import ast
import glob
import http.server
import importlib.util
import json
import mimetypes
import os
import queue
import random
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import sets as setsmod                          # noqa: E402  (gui/sets.py)

HERE = os.path.dirname(os.path.realpath(__file__))
REPO = os.path.dirname(HERE)
SOUNDMON = os.path.join(REPO, "bin", "soundmon")
SERVER_ARG = ""          # --server / $SOUNDMON_SERVER: overrides the Setup tab's choice
GALLERY_HOME = os.path.expanduser("~/soundmon-gallery")
INPUTS = os.path.join(GALLERY_HOME, "inputs")         # MIDI / audio / banks dropped on the form
SETUP_FILE = os.path.join(GALLERY_HOME, "gui-setup.json")
SETS_ROOT = os.path.join(GALLERY_HOME, "sets")         # one folder per sound set (gui/sets.py)
PRESETS = os.path.join(GALLERY_HOME, "gui-presets")    # saved form snapshots, one .json each
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
DONE_LINE = re.compile(r"✅\s*\[(\d+)/(\d+)\]")
SEED = re.compile(r"seed=(\d+)")
AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3", ".opus")
SCORE_EXT = (".mod", ".rad", ".mid")
INPUT_EXT = (".mid", ".midi", ".wav", ".flac", ".ogg", ".mp3", ".opus", ".aif", ".aiff", ".m4a", ".sbi", ".txt")
HOME = os.path.realpath(os.path.expanduser("~"))
LAN = False          # --lan: other devices can reach the page, so disk browsing / opening stays with this machine
IS_WIN, IS_MAC = sys.platform.startswith("win"), sys.platform == "darwin"

# Engines: what the form calls them -> how soundmon is told, and what they need.
ENGINES = {
    "sfx":     {"flags": [],            "gpu": True,  "label": "SFX"},
    "music":   {"flags": ["--music"],   "gpu": True,  "label": "Music loop"},
    "song":    {"flags": ["--song"],    "gpu": True,  "label": "Song"},
    "chip":    {"flags": ["--chip"],    "gpu": False, "label": "Chiptune"},
    "opl":     {"flags": ["--opl"],     "gpu": False, "label": "AdLib"},
    "fx":      {"flags": [],            "gpu": False, "label": "Retro FX"},     # --chipfx / --oplfx
    "blip":    {"flags": ["--blip"],    "gpu": False, "label": "Blips"},
    "narrate": {"flags": ["--narrate"], "gpu": False, "label": "Narrate"},
}
ENCODES = {"wav": [], "flac": ["--flac"], "ogg": ["--ogg"], "mp3": ["--mp3"], "opus": ["--opus"]}


# ---------------------------------------------------------------------------
# Metadata, read live from soundmon's own files so the GUI never drifts
# ---------------------------------------------------------------------------
def load_module(name):
    """A fresh import of one of soundmon's engine modules (stdlib-only at the top), so a
    `git pull` shows up without restarting the GUI."""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    spec = importlib.util.spec_from_file_location(f"_sm_{name}", os.path.join(REPO, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def source_constant(path, name, default=None):
    """Evaluate one top-level assignment of a source file on its own (literals, dict(...),
    comprehensions) without importing the file — soundmon.py pulls in torch via the node."""
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                ns = {}
                exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"),
                     {"__builtins__": {"dict": dict, "range": range, "list": list, "tuple": tuple}}, ns)
                return ns[name]
    except Exception:
        pass
    return default


def cli_spec():
    """Every soundmon flag's default / choices / help, read from the add_argument calls in
    soundmon.py — the GUI's tooltips and validation come from the CLI itself."""
    out = {}
    try:
        tree = ast.parse(open(os.path.join(REPO, "soundmon.py"), encoding="utf-8").read())
    except OSError:
        return out
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if not flags:
            continue
        rec = {"flags": flags}
        for kw in node.keywords:
            if kw.arg in ("default", "choices", "help", "action", "metavar"):
                try:
                    rec[kw.arg] = ast.literal_eval(kw.value)
                except Exception:
                    if kw.arg == "help" and isinstance(kw.value, ast.JoinedStr):
                        rec["help"] = "".join(v.value for v in kw.value.values if isinstance(v, ast.Constant))
            elif kw.arg == "type":
                rec["type"] = getattr(kw.value, "id", None)
        out[flags[0]] = rec
    return out


def load_styles():
    """sounds.json in its three sections (SFX / Music / Modifiers), in file order."""
    try:
        with open(os.path.join(REPO, "sounds.json"), encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {"sections": [], "styles": {}}
    sections, cur = [{"name": "SFX", "keys": []}], None
    cur = sections[0]
    for k, v in raw.items():
        if k == "_music":
            cur = {"name": "Music", "keys": []}; sections.append(cur)
        elif k == "_modifiers":
            cur = {"name": "Modifiers", "keys": []}; sections.append(cur)
        elif not k.startswith("_"):
            cur["keys"].append(k)
    return {"sections": sections, "styles": {k: v for k, v in raw.items() if not k.startswith("_")}}


def load_formats():
    fm = source_constant(os.path.join(REPO, "custom_nodes", "retro_sfx", "nodes.py"), "FORMATS", {}) or {}
    return [{"name": k, **(v or {})} for k, v in fm.items()] or [{"name": "none"}]


def load_meta():
    sm = os.path.join(REPO, "soundmon.py")
    spec = cli_spec()
    meta = {"engines": {k: {"label": v["label"], "gpu": v["gpu"]} for k, v in ENGINES.items()},
            "styles": load_styles(), "formats": load_formats(), "cli": spec,
            "keys": source_constant(sm, "KEYS", []),
            "negatives": {k: source_constant(sm, k, "") for k in ("SFX_NEGATIVE", "MUSIC_NEGATIVE", "SONG_NEGATIVE")},
            "scales": (spec.get("--scale") or {}).get("choices") or [],
            "chippy": (spec.get("--chippy") or {}).get("choices") or [],
            "waves": (spec.get("--blip-wave") or {}).get("choices") or [],
            "timesigs": (spec.get("--timesig") or {}).get("choices") or [],
            "moods": [], "archetypes": [], "opl_instruments": [], "voices": {}}
    for mod, key, fn in (("chip", "moods", lambda m: list(m.MOODS)),
                         ("chipfx", "archetypes", lambda m: list(m.ARCHETYPES)),
                         ("opl", "opl_instruments", lambda m: list(m.INSTRUMENTS)),
                         ("narrate", "voices", lambda m: dict(m.VOICES))):
        try:
            meta[key] = fn(load_module(mod))
        except Exception as e:          # an engine that can't load just has an empty list
            meta.setdefault("load_errors", {})[mod] = str(e)[:200]
    return meta


# ---------------------------------------------------------------------------
# Render servers (same rules as soundmon / pixelmon-gui)
# ---------------------------------------------------------------------------
def _servers_json():
    try:
        with open(os.path.join(REPO, "servers.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def known_servers():
    """servers.json aliases (keys starting with _ are comments) plus the built-in 'local'."""
    out = {k: v for k, v in _servers_json().items() if not k.startswith("_") and isinstance(v, str)}
    out.setdefault("local", "http://127.0.0.1:8188")
    return out


def default_server():
    return re.sub(r"\s+", "", str(_servers_json().get("_default") or "")) or "local"


def render_server():
    """Where GPU renders go: --server / $SOUNDMON_SERVER, else the Setup tab's pick, else servers.json's default."""
    return SERVER_ARG or load_setup().get("server") or default_server()


def server_url(name=None):
    """The URL of a render server name (a farm list -> its first server)."""
    name = (name or render_server()).split(",")[0].strip()
    url = known_servers().get(name, name)
    if "://" not in url:
        url = "http://" + url
    p = urllib.parse.urlparse(url)
    return f"{p.scheme}://{p.hostname}:{p.port or 8188}" if p.hostname else None


def server_up(name=None):
    url = server_url(name)
    if not url:
        return False
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/system_stats", timeout=3):
            return True
    except Exception:
        return False


def _combo(info, node, field):
    try:
        spec = info[node]["input"]["required"][field]
    except (KeyError, TypeError):
        return []
    if isinstance(spec[0], list):
        return list(spec[0])
    if len(spec) > 1 and isinstance(spec[1], dict):
        return list(spec[1].get("options") or [])
    return []


# What each engine needs on a box. Names come from soundmon.py's defaults.
NEEDS = {"sa3": "stable_audio_3_medium.safetensors", "ace": "ace_step_1.5_turbo_aio.safetensors",
         "sao": "stable-audio-open-1.0.safetensors"}
_MODELS = {}


def server_models(name):
    """{'sa3': bool, 'ace': bool, 'sao': bool, 'qwen': bool, 't5gemma': bool} for one box (cached a minute)."""
    hit = _MODELS.get(name)
    if hit and time.time() - hit[0] < 60:
        return hit[1]
    url, info = server_url(name), {}
    for node in ("CheckpointLoaderSimple", "CLIPLoader"):
        try:
            with urllib.request.urlopen(f"{url}/object_info/{node}", timeout=4) as r:
                info.update(json.load(r))
        except Exception:
            pass
    if not info:
        return None
    ck = _combo(info, "CheckpointLoaderSimple", "ckpt_name")
    clips = _combo(info, "CLIPLoader", "clip_name")
    v = {k: f in ck for k, f in NEEDS.items()}
    v["qwen"] = any("qwen" in c.lower() for c in clips)
    v["t5gemma"] = any("t5gemma" in c.lower() for c in clips)
    _MODELS[name] = (time.time(), v)
    return v


SERVER_STATE = {"server": None, "url": None, "up": None, "stuck": None, "restart": None}
STUCK_AFTER, DOWN_AFTER, RESTART_EVERY = 20, 30, 180
# errors that leave ComfyUI failing every prompt until it's restarted (seen after Stable Audio
# jobs swap Qwen / T5Gemma / SA3 through an 8 GB card)
POISONED = ("object has no attribute 'model_size'", "object has no attribute 'is_dynamic'")


def restart_command(name):
    cmds = _servers_json().get("_restart")
    return cmds.get(name) if isinstance(cmds, dict) and isinstance(cmds.get(name), str) else None


def restart_server(name, why):
    cmd = restart_command(name)
    state = {"server": name, "why": why, "at": time.time(), "status": "running", "output": ""}
    SERVER_STATE["restart"] = state
    print(f"[soundmon-gui] render server {name} {why}: restarting it ({cmd})", flush=True)

    def run():
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=90, stdin=subprocess.DEVNULL)
            state.update(status="done" if r.returncode == 0 else "failed", output=(r.stdout + r.stderr).strip()[-400:])
        except Exception as e:
            state.update(status="failed", output=str(e))
    threading.Thread(target=run, daemon=True).start()


def server_queue(name=None):
    try:
        with urllib.request.urlopen(server_url(name).rstrip("/") + "/queue", timeout=3) as r:
            return json.load(r)
    except Exception:
        return None


def server_poisoned(name=None):
    try:
        with urllib.request.urlopen(server_url(name).rstrip("/") + "/history?max_items=1", timeout=3) as r:
            hist = json.load(r)
    except Exception:
        return None
    for pid, h in hist.items():
        for kind, msg in (h.get("status") or {}).get("messages") or []:
            err = str((msg or {}).get("exception_message") or "") if kind == "execution_error" else ""
            if any(p in err for p in POISONED):
                return {"prompt": pid, "error": err.strip()[:120]}
    return None


def _gpu_work():
    jobs = getattr(Handler, "jobs", None)
    if not jobs:
        return False
    with jobs.lock:
        return any(j["status"] in ("queued", "running") and ENGINES.get(j["engine"], {}).get("gpu") for j in jobs.jobs.values())


def watch_server():
    """Check the render server every few seconds so polls never wait on a dead one; restart it
    (servers.json "_restart") when it is down, stuck, or poisoned."""
    head, down_since, last_restart = None, None, {}
    while True:
        name = render_server()
        up = server_up(name)
        down_since = None if up else (down_since or time.time())
        stuck = None
        q = server_queue(name) if up else None
        if q and not q.get("queue_running") and q.get("queue_pending"):
            first = min(q["queue_pending"], key=lambda p: p[0])[1]
            if not head or head[:2] != (name, first):
                head = (name, first, time.time())
            elif time.time() - head[2] >= STUCK_AFTER:
                stuck = {"pending": len(q["queue_pending"]), "since": head[2]}
        else:
            head = None
        poisoned = server_poisoned(name) if up and not stuck else None
        SERVER_STATE.update(server=name, url=server_url(name), up=up, stuck=stuck or poisoned)
        why = ("is stuck: prompts waiting, nothing running" if stuck
               else f"fails every prompt ({poisoned['error']})" if poisoned
               else f"hasn't answered for {DOWN_AFTER}s" if down_since and time.time() - down_since >= DOWN_AFTER else None)
        # only while this GUI has GPU work for it: a box shut down on purpose isn't restarted forever
        if why and restart_command(name) and _gpu_work() and time.time() - last_restart.get(name, 0) >= RESTART_EVERY:
            last_restart[name] = time.time()
            head = None
            restart_server(name, why)
        time.sleep(4)


def server_is_up():
    if SERVER_STATE["server"] == render_server() and SERVER_STATE["up"] is not None:
        return SERVER_STATE["up"]
    return server_up()


def servers_report():
    """Every known server with its state and the models soundmon needs (checked in parallel)."""
    names = sorted(known_servers())
    out = {n: {"name": n, "url": server_url(n), "up": False, "models": None} for n in names}

    def check(n):
        out[n]["up"] = server_up(n)
        if out[n]["up"]:
            out[n]["models"] = server_models(n)
    ts = [threading.Thread(target=check, args=(n,), daemon=True) for n in names]
    for t in ts:
        t.start()
    for t in ts:
        t.join(8)
    cur = render_server().split(",")
    for n in names:
        out[n]["default"] = n == default_server()
        out[n]["selected"] = n in cur
        out[n]["restart"] = bool(restart_command(n))
    return {"servers": list(out.values()), "render": render_server(), "restart": SERVER_STATE["restart"]}


# ---------------------------------------------------------------------------
# Setup (this machine)
# ---------------------------------------------------------------------------
def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def load_setup():
    s = _read_json(SETUP_FILE, {}) if os.path.exists(SETUP_FILE) else {}
    s.setdefault("server", "")          # render server; "" = servers.json "_default"
    s.setdefault("audio_apps", [])      # [{name, exec, args}] — "{}" = the file, else appended
    s.setdefault("folder_apps", [])
    s.setdefault("game_folders", [])    # [{name, path}] — where Sets export picks
    s.setdefault("similar_folders", []) # folders the Find tab indexes
    return s


def _clean_app(a):
    return {"name": str(a.get("name") or "")[:60], "exec": str(a.get("exec") or "")[:500], "args": str(a.get("args") or "{}")[:300]}


def save_setup(body):
    s = load_setup()
    for k in ("audio_apps", "folder_apps"):
        if isinstance(body.get(k), list):
            s[k] = [_clean_app(a) for a in body[k][:40] if isinstance(a, dict) and (a.get("exec") or "").strip()]
    if isinstance(body.get("game_folders"), list):
        s["game_folders"] = [{"name": str(g.get("name") or "")[:60], "path": str(g.get("path") or "")[:500]}
                             for g in body["game_folders"][:40] if isinstance(g, dict) and str(g.get("path") or "").strip()]
    if isinstance(body.get("similar_folders"), list):
        s["similar_folders"] = [str(f).strip()[:500] for f in body["similar_folders"][:40] if str(f).strip()]
    if "server" in body:
        s["server"] = re.sub(r"\s+", "", str(body["server"] or ""))[:300]
    _write_json(SETUP_FILE, s)
    return s


def home_path(path, must_exist=True):
    """Expand ~ and confine to the user's home dir."""
    full = os.path.realpath(os.path.expanduser(str(path or "")))
    if full != HOME and not full.startswith(HOME + os.sep):
        raise ValueError("path must be inside your home folder")
    if must_exist and not os.path.exists(full):
        raise ValueError(f"not found: {path}")
    return full


OPENABLE = (".wav", ".flac", ".ogg", ".mp3", ".opus", ".aif", ".aiff", ".m4a", ".mid", ".midi", ".mod", ".rad", ".xm", ".it",
            ".s3m", ".sbi", ".txt", ".csv", ".json", ".md", ".zip", ".png", ".gif", ".pdf", ".bwpreset", ".multisample", ".xrni")
DISK_AUDIO = (".wav", ".flac", ".ogg", ".mp3", ".opus", ".aif", ".aiff", ".m4a")


def browse(path, kind="any", exts=None, hidden=False, gallery=None):
    """One folder of the user's home for the Browse dialog: sub-folders, then files (filtered by
    extension), breadcrumbs back to home, and shortcuts. Home-confined like every other path here."""
    full = home_path(path or "~")
    if os.path.isfile(full):
        full = os.path.dirname(full)
    dirs, files = [], []
    try:
        with os.scandir(full) as it:
            for e in it:
                if not hidden and e.name.startswith("."):
                    continue
                try:
                    if e.is_dir():
                        dirs.append(e.name)
                    elif kind != "dir" and e.is_file() and (not exts or e.name.lower().endswith(exts)):
                        st = e.stat()
                        files.append({"name": e.name, "size": st.st_size, "mtime": int(st.st_mtime)})
                except OSError:
                    continue
    except OSError as e:
        raise ValueError(f"can't read {full}: {e.strerror or e}")
    dirs.sort(key=str.lower)
    files.sort(key=lambda f: f["name"].lower())
    crumbs, p = [], full
    while True:
        crumbs.append({"name": "~" if p == HOME else os.path.basename(p), "path": p})
        if p == HOME or os.path.dirname(p) == p:
            break
        p = os.path.dirname(p)
    places = [{"name": "Home", "path": HOME}]
    for label, d in (("Gallery", gallery), ("Dropped files", INPUTS), ("Music", "~/Music"), ("Downloads", "~/Downloads"),
                     ("Desktop", "~/Desktop"), ("Documents", "~/Documents"), ("git", "~/git")):
        if d and os.path.isdir(os.path.expanduser(d)):
            places.append({"name": label, "path": os.path.realpath(os.path.expanduser(d))})
    setup = load_setup()
    for f in setup.get("similar_folders", []) + [g.get("path", "") for g in setup.get("game_folders", [])]:
        try:
            d = home_path(f)
        except ValueError:
            continue
        if os.path.isdir(d) and all(x["path"] != d for x in places):
            places.append({"name": os.path.basename(d) or d, "path": d})
    cap = 3000
    return {"path": full, "home": HOME, "parent": None if full == HOME else os.path.dirname(full), "crumbs": crumbs[::-1],
            "dirs": dirs[:cap], "files": files[:cap], "more": max(0, len(dirs) + len(files) - 2 * cap), "places": places,
            "sep": os.sep}


def _desktop_entries():
    """Linux: every installed app's .desktop entry (pixelmon-gui's scan)."""
    dirs = [os.path.expanduser("~/.local/share/applications"), "/usr/local/share/applications", "/usr/share/applications",
            os.path.expanduser("~/.local/share/flatpak/exports/share/applications"),
            "/var/lib/flatpak/exports/share/applications", "/var/lib/snapd/desktop/applications"]
    seen, out = set(), []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.desktop"))):
            if os.path.basename(f) in seen:
                continue
            seen.add(os.path.basename(f))
            e, sect = {}, None
            try:
                with open(f, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        line = line.strip()
                        if line.startswith("["):
                            sect = line
                        elif sect == "[Desktop Entry]" and "=" in line and not line.startswith("#"):
                            k, v = line.split("=", 1)
                            e.setdefault(k.strip(), v.strip())
            except OSError:
                continue
            if e.get("Type") == "Application" and e.get("Exec") and e.get("NoDisplay", "").lower() != "true" \
                    and e.get("Hidden", "").lower() != "true":
                out.append(e)
    return out


def _desktop_app(e):
    try:
        parts = shlex.split(e["Exec"])
    except ValueError:
        return None
    parts = [x for x in parts if x not in ("%i", "%c", "%k")]
    args = ["{}" if x in ("%f", "%F", "%u", "%U") else x for x in parts[1:]]
    if "{}" not in args:
        args.append("{}")
    return {"name": e.get("Name", parts[0]), "exec": parts[0], "args": shlex.join(args).replace("'{}'", "{}")}


def _mac_app(name, bundle):
    for root in ("/Applications", os.path.expanduser("~/Applications"), "/System/Applications"):
        for app in glob.glob(os.path.join(root, bundle)):
            return {"name": name, "exec": "open", "args": f"-a {shlex.quote(os.path.splitext(os.path.basename(app))[0])} {{}}"}
    return None


def _win_app(name, *patterns):
    roots = [os.environ.get(v, "") for v in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA", "ProgramW6432")]
    for pat in patterns:
        if not any(ch in pat for ch in "\\/*"):
            w = shutil.which(pat)
            if w:
                return {"name": name, "exec": w, "args": "{}"}
            continue
        for r in [x for x in roots if x]:
            hits = sorted(glob.glob(os.path.join(r, pat)))
            if hits:
                return {"name": name, "exec": hits[-1], "args": "{}"}
    return None


_DETECTED = {}


def detect_programs():
    """Sound programs and file managers installed here, offered in Setup's "Add a program" (cached)."""
    if "v" in _DETECTED:
        return _DETECTED["v"]
    snd, fold = [], []
    if IS_MAC:
        for n, b in (("Audacity", "Audacity.app"), ("ocenaudio", "ocenaudio.app"), ("VLC", "VLC.app"), ("REAPER", "REAPER*.app"),
                     ("Logic Pro", "Logic Pro*.app"), ("GarageBand", "GarageBand.app"), ("Bitwig Studio", "Bitwig Studio.app"),
                     ("Renoise", "Renoise*.app"), ("OpenMPT", "OpenMPT*.app"), ("MuseScore", "MuseScore*.app"), ("QuickTime Player", "QuickTime Player.app")):
            a = _mac_app(n, b)
            if a:
                snd.append(a)
        fold.append({"name": "Finder", "exec": "open", "args": "{}"})
    elif IS_WIN:
        for n, *pats in (("Audacity", r"Audacity\Audacity.exe"), ("ocenaudio", r"ocenaudio\ocenaudio.exe"), ("VLC", r"VideoLAN\VLC\vlc.exe"),
                         ("REAPER", r"REAPER (x64)\reaper.exe", r"REAPER\reaper.exe"), ("Bitwig Studio", r"Bitwig Studio\Bitwig Studio.exe"),
                         ("Renoise", r"Renoise*\Renoise.exe"), ("OpenMPT", r"OpenMPT\OpenMPT.exe"), ("MilkyTracker", r"MilkyTracker\milkytracker.exe"),
                         ("MuseScore", r"MuseScore*\bin\MuseScore*.exe")):
            a = _win_app(n, *pats)
            if a:
                snd.append(a)
        fold.append({"name": "Explorer", "exec": "explorer", "args": "{}"})
    else:
        for e in _desktop_entries():
            mimes = [m for m in e.get("MimeType", "").split(";") if m]
            a = _desktop_app(e)
            if not a:
                continue
            if any(m.startswith("audio/") or m in ("application/ogg", "audio/x-wav", "audio/midi") for m in mimes):
                snd.append(a)
            if "inode/directory" in mimes or "FileManager" in e.get("Categories", ""):
                fold.append(a)
        have = {a["exec"] for a in snd}
        for n, x, args in (("Audacity", "audacity", "{}"), ("mpv", "mpv", "--force-window {}"), ("ffplay", "ffplay", "-autoexit {}"),
                           ("sox play", "play", "{}"), ("MilkyTracker", "milkytracker", "{}"), ("SchismTracker", "schismtracker", "{}")):
            if x not in have and shutil.which(x):
                snd.append({"name": n, "exec": x, "args": args})
    key = lambda a: a["name"].lower()     # noqa: E731
    _DETECTED["v"] = (sorted({a["name"]: a for a in snd}.values(), key=key), sorted({a["name"]: a for a in fold}.values(), key=key))
    return _DETECTED["v"]


# ---------------------------------------------------------------------------
# Backups of ~/soundmon-gallery (pixelmon-gui's, for sound)
# ---------------------------------------------------------------------------
BACKUP_SKIP_TOP = {"backups"}
BACKUP_SKIP_ANY = {"_analysis", "_mastered"}          # measurements and mastered previews: remade on demand
STORED_EXT = (".ogg", ".mp3", ".opus", ".flac", ".m4a", ".zip", ".png", ".jpg")


def backups_dir():
    return os.path.join(GALLERY_HOME, "backups")


def make_backup(keyword, gallery):
    """Zip ~/soundmon-gallery (takes, sets, presets, inputs, the Find index) into
    backups/soundmon-gallery-<keyword>-<date>.zip. Returns (path, files, bytes, skipped)."""
    kw = re.sub(r"[^\w-]+", "-", str(keyword or "").strip()).strip("-")[:40] or "backup"
    bk = backups_dir()
    os.makedirs(bk, exist_ok=True)
    base = f"soundmon-gallery-{kw}-{time.strftime('%Y-%m-%d')}"
    path, i = os.path.join(bk, base + ".zip"), 2
    while os.path.exists(path):
        path, i = os.path.join(bk, f"{base}-{i}.zip"), i + 1
    roots = [GALLERY_HOME] + ([gallery] if not os.path.realpath(gallery).startswith(os.path.realpath(GALLERY_HOME) + os.sep) else [])
    n = skipped = 0
    try:
        with zipfile.ZipFile(path + ".part", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for root in roots:
                top = os.path.basename(os.path.normpath(root))
                for d, dirs, files in os.walk(root):
                    rel = os.path.relpath(d, root)
                    dirs[:] = [x for x in dirs if x not in BACKUP_SKIP_ANY and not (rel == "." and x in BACKUP_SKIP_TOP)]
                    for f in files:
                        if f.endswith((".part", ".tmp")):
                            continue
                        full = os.path.join(d, f)
                        if not os.path.isfile(full):
                            skipped += 1
                            continue
                        try:
                            z.write(full, os.path.normpath(os.path.join(top, rel, f)),
                                    compress_type=zipfile.ZIP_STORED if f.lower().endswith(STORED_EXT) else zipfile.ZIP_DEFLATED)
                            n += 1
                        except OSError:
                            skipped += 1
        os.replace(path + ".part", path)
    except BaseException:
        if os.path.exists(path + ".part"):
            os.remove(path + ".part")
        raise
    return path, n, os.path.getsize(path), skipped


def list_backups():
    bk = backups_dir()
    if not os.path.isdir(bk):
        return []
    out = []
    for f in sorted(os.listdir(bk), key=lambda f: -os.path.getmtime(os.path.join(bk, f))):
        full = os.path.join(bk, f)
        if f.endswith(".zip") and os.path.isfile(full):
            try:
                with zipfile.ZipFile(full) as z:
                    n = sum(1 for i in z.infolist() if not i.is_dir())
            except zipfile.BadZipFile:
                continue
            out.append({"name": f, "bytes": os.path.getsize(full), "mtime": os.path.getmtime(full), "files": n})
    return out


def backup_path(name):
    full = os.path.realpath(os.path.join(backups_dir(), os.path.basename(str(name or ""))))
    if not full.endswith(".zip") or not full.startswith(os.path.realpath(backups_dir()) + os.sep) or not os.path.isfile(full):
        raise ValueError("no such backup")
    return full


def restore_backup(name, gallery):
    """Put back everything from a backup that's missing now. Never overwrites an existing file."""
    full = backup_path(name)
    roots = {os.path.basename(os.path.normpath(GALLERY_HOME)): GALLERY_HOME, os.path.basename(os.path.normpath(gallery)): gallery}
    restored = kept = 0
    with zipfile.ZipFile(full) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            top, _, rest = info.filename.partition("/")
            root = roots.get(top)
            if not root or not rest:
                continue
            dest = os.path.realpath(os.path.join(root, rest))
            if not dest.startswith(os.path.realpath(root) + os.sep) or os.path.relpath(dest, root).split(os.sep)[0] == "backups":
                continue
            if os.path.exists(dest):
                kept += 1
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with z.open(info) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            restored += 1
    return {"restored": restored, "kept": kept}


def clear_gallery(gallery):
    """Delete every job folder in the gallery. Sets, presets, inputs, the Find index and backups live
    elsewhere and are untouched."""
    gallery = os.path.realpath(gallery)
    if gallery in (HOME, os.path.realpath(GALLERY_HOME), "/"):
        raise ValueError("refusing to clear that folder")
    n = 0
    for f in os.listdir(gallery):
        p = os.path.join(gallery, f)
        if os.path.isdir(p) and not os.path.islink(p) and os.path.isfile(os.path.join(p, "job.json")):
            shutil.rmtree(p)
            n += 1
    return n


def os_opener(is_dir):
    if IS_MAC:
        return {"name": "Finder" if is_dir else "system default", "exec": "open", "args": "{}"}
    if IS_WIN:
        return {"name": "Explorer" if is_dir else "system default", "exec": "explorer", "args": "{}"}
    return {"name": "file manager" if is_dir else "system default", "exec": "xdg-open", "args": "{}"}


def launch(app, path, cwd=None):
    """Start app on path. args template: '{}' marks the path, else it's appended. Never waits."""
    exe = os.path.expanduser(app["exec"])
    try:
        args = shlex.split(app.get("args") or "{}", posix=not IS_WIN)
    except ValueError as e:
        raise ValueError(f"bad arguments for {app.get('name')}: {e}")
    args = [a.replace("{}", path) for a in args] if any("{}" in a for a in args) else args + [path]
    if os.sep in exe and not os.path.isfile(exe):
        raise ValueError(f"{app.get('name') or exe}: program not found at {exe} — fix it in the Setup tab")
    if os.sep not in exe and not shutil.which(exe):
        raise ValueError(f"{app.get('name') or exe}: '{exe}' isn't installed or isn't on PATH")
    subprocess.Popen([exe] + args, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    return exe


# ---------------------------------------------------------------------------
# Form -> soundmon argv
# ---------------------------------------------------------------------------
def shq(s):
    """Shell-quote for display/copy: bare when safe, double quotes for text with apostrophes."""
    s = str(s)
    if s and re.fullmatch(r"[\w@%+=:,./-]+", s):
        return s
    if "'" not in s:
        return "'" + s + "'"
    if not re.search(r'[$`"\\!]', s):
        return '"' + s + '"'
    return shlex.quote(s)


def shjoin(argv):
    return " ".join(shq(a) for a in argv)


def safe_name(s, maxlen=64):
    return re.sub(r"[^\w.-]+", "-", str(s)).strip("-.")[:maxlen]


def input_path(name):
    """A file dropped on the form (MIDI, reference audio, OPL bank) -> its path in INPUTS."""
    full = os.path.realpath(os.path.join(INPUTS, os.path.basename(str(name or ""))))
    if not full.startswith(os.path.realpath(INPUTS) + os.sep) or not os.path.isfile(full):
        raise ValueError(f"input file not found: {name} — drop it on the form again")
    return full


# Advanced tab: key -> (flag, kind, extra). Blank / missing = soundmon's own default.
ADV = {
    "engine":         ("--engine", "choice", ["sa3", "sao"]),
    "sa3_size":       ("--sa3-size", "choice", ["medium", "small"]),
    "steps":          ("--steps", "int", (1, 200)),
    "cfg":            ("--cfg", "float", (0, 30)),
    "fast":           ("--fast", "flag", None),
    "sampler":        ("--sampler", "name", None),
    "scheduler":      ("--scheduler", "name", None),
    "base":           ("--base", "name", None),
    "text_encoder":   ("--text-encoder", "name", None),
    "negative":       ("--negative", "text", 600),
    "reprompt_model": ("--reprompt-model", "name", None),
    "reprompt_temp":  ("--reprompt-temp", "float", (0, 2)),
    "llm_cfg":        ("--llm-cfg", "float", (0, 20)),
    "temperature":    ("--temperature", "float", (0, 2)),
    "top_p":          ("--top-p", "float", (0, 1)),
    "top_k":          ("--top-k", "int", (0, 500)),
    "min_p":          ("--min-p", "float", (0, 1)),
    "no_audio_codes": ("--no-audio-codes", "flag", None),
    "song_base":      ("--song-base", "name", None),
    "trim":           (None, "choice", ["trim", "no-trim"]),
    "max_seconds":    ("--max-seconds", "float", (0, 1000)),
    "threshold_db":   ("--threshold-db", "float", (-120, 0)),
    "normalize_db":   ("--normalize-db", "float", (-60, 0)),
    "fade_ms":        ("--fade-ms", "float", (0, 5000)),
    "loop_crossfade": ("--loop-crossfade", "float", (0, 30)),
    "true_peak":      ("--true-peak", "float", (-20, 0)),
    "ogg_quality":    ("--ogg-quality", "int", (0, 10)),
    "keep_wav":       ("--keep-wav", "flag", None),
    "midi_start":     ("--midi-start", "float", (0, 30)),
    "transpose":      ("--transpose", "int", (-48, 48)),
}


_GPU, _SA = ("sfx", "music", "song"), ("sfx", "music")
ADV_SCOPE = {   # which engines each Advanced override means something to (others leave it out)
    **{k: _SA for k in ("engine", "sa3_size", "text_encoder", "reprompt_model", "reprompt_temp")},
    **{k: _GPU for k in ("steps", "cfg", "fast", "sampler", "scheduler", "base", "negative", "trim", "max_seconds",
                         "threshold_db", "fade_ms")},
    **{k: ("song",) for k in ("llm_cfg", "temperature", "top_p", "top_k", "min_p", "no_audio_codes", "song_base")},
    "loop_crossfade": ("music", "song"), "midi_start": ("chip", "opl"), "transpose": ("chip", "opl"),
}


def adv_argv(adv, eng=None):
    out = []
    if not isinstance(adv, dict):
        raise ValueError("advanced settings must be an object")
    for k, v in adv.items():
        if k not in ADV or v in (None, "", False):
            continue
        if eng and k in ADV_SCOPE and eng not in ADV_SCOPE[k]:
            continue
        flag, kind, extra = ADV[k]
        if kind == "flag":
            out.append(flag)
        elif kind == "choice":
            if str(v) not in extra:
                raise ValueError(f"advanced: {k} must be one of {', '.join(extra)}")
            out += [f"--{v}"] if flag is None else [flag, str(v)]
        elif kind in ("int", "float"):
            try:
                x = int(v) if kind == "int" else float(v)
            except (TypeError, ValueError):
                raise ValueError(f"advanced: {k} must be a number")
            if not extra[0] <= x <= extra[1]:
                raise ValueError(f"advanced: {k} must be {extra[0]}..{extra[1]}")
            out += [flag, f"{x:g}" if kind == "float" else str(x)]
        elif kind == "name":
            s = str(v).strip()
            if not re.fullmatch(r"[\w.\-/ ]{1,160}", s):
                raise ValueError(f"advanced: {k} has characters a model name can't")
            out += [flag, s]
        elif kind == "text":
            out += [flag, str(v).strip()[:extra]]
    return out


def preflight(p):
    """Things that will render, but probably not the way you meant (pixelmon-gui's setup check):
    [{"msg", "fix": {field: value} | None, "adv": {field: value} | None}]."""
    eng, adv, out = p.get("engine") or "sfx", p.get("adv") if isinstance(p.get("adv"), dict) else {}, []
    gpu = ENGINES.get(eng, {}).get("gpu")
    sao = adv.get("engine") == "sao"
    try:
        secs = float(p.get("seconds") or 0)
    except (TypeError, ValueError):
        secs = 0
    if eng in ("sfx", "music") and sao and secs > 47:
        out.append({"msg": "Stable Audio Open stops at 47 s.", "fix": {"seconds": "47"}})
    if eng in ("sfx", "music") and not sao and (adv.get("steps") or adv.get("cfg")):
        out.append({"msg": "SA3 is distilled for 8 steps at CFG 1; other values tend to sound worse, not better.", "adv": {"steps": "", "cfg": ""}})
    if eng in ("sfx", "music") and not sao and p.get("reprompt") is False:
        out.append({"msg": "SA3 was trained on rewritten descriptions: without the Qwen rewrite it usually sounds flatter.", "fix": {"reprompt": True}})
    if eng in ("chip", "opl") and p.get("loop"):
        out.append({"msg": "Chiptune and AdLib loop by construction; the loop crossfade isn't applied to them.", "fix": {"loop": False}})
    if p.get("ref") and not gpu:
        out.append({"msg": "The reference only steers SFX, Music and Song; this engine ignores it.", "fix": {"ref": ""}})
    if (p.get("ref_match") or p.get("match_tone")) and not p.get("ref") and gpu:
        out.append({"msg": "Tempo/key and tone matching need a reference sound — drop one in Reference.", "fix": {"ref_match": False, "match_tone": False}})
    if gpu and SERVER_STATE.get("up"):
        m = server_models(render_server().split(",")[0]) or {}
        need = "ace" if eng == "song" else ("sao" if sao else "sa3")
        if m and m.get(need) is False:
            out.append({"msg": f"{render_server().split(',')[0]} doesn't have the {'ACE-Step' if need == 'ace' else 'SAO' if need == 'sao' else 'SA3 medium'} model — "
                               f"provision it (./download-models.sh {'--song' if need == 'ace' else '--sa3' if need == 'sa3' else ''}) or pick another server in Setup.", "fix": None})
    return out


def is_manifest(text):
    """More than one line, or `key | text` rows: one file per line."""
    lines = [ln for ln in str(text or "").splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return len(lines) > 1 or any("|" in ln for ln in lines)


def build_argv(p, meta, jdir=None):
    """Validated form params -> (argv for bin/soundmon, files to write into the job folder, takes expected).
    jdir None = preview only (the command shown under RENDER)."""
    eng = p.get("engine") or "sfx"
    if eng not in ENGINES:
        raise ValueError(f"unknown engine {eng!r}")
    E, argv, files = ENGINES[eng], [SOUNDMON], {}
    prompt = str(p.get("prompt") or "").strip()
    if len(prompt) > 20000:
        raise ValueError("that text is too long (20,000 characters max)")

    def num(key, typ, lo, hi, default=None):
        v = p.get(key, default)
        if v in (None, ""):
            return default
        try:
            v = typ(v)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number")
        if not lo <= v <= hi:
            raise ValueError(f"{key} must be between {lo:g} and {hi:g}")
        return v

    def choice(key, options, default=None):
        v = p.get(key, default)
        if v in (None, ""):
            return default
        if v not in options:
            raise ValueError(f"{key}: {v!r} isn't one of {', '.join(map(str, options[:12]))}…")
        return v

    def jfile(name, text):
        files[name] = text
        return os.path.join(jdir, name) if jdir else name

    takes = 1
    if eng in ("blip", "narrate"):
        if not prompt:
            raise ValueError("type a line to " + ("speak" if eng == "narrate" else "blip") + ", or paste `key | text` lines")
        argv += E["flags"]
        if is_manifest(prompt):
            argv += ["--narrate-file" if eng == "narrate" else "--blip-file", jfile("lines.txt", prompt + "\n")]
            takes = len([ln for ln in prompt.splitlines() if ln.strip() and not ln.strip().startswith("#")])
        else:
            argv.insert(1, prompt)
        if eng == "narrate" or p.get("blip_style") == "voice":
            v = choice("voice", list(meta["voices"]) or ["bm_george"])
            if v:
                argv += ["--voice", v]
        if eng == "narrate":
            sp, pt = num("speed", float, 0.5, 2.0), num("pitch", float, -12, 12)
            if sp is not None and sp != 1.0:
                argv += ["--speed", f"{sp:g}"]
            if pt:
                argv += ["--pitch", f"{pt:g}"]
        else:
            st = choice("blip_style", ["synth", "voice"])
            if st and st != "synth":
                argv += ["--blip-style", st]
            w = choice("blip_wave", meta["waves"] or ["square"])
            if w and w != "square":
                argv += ["--blip-wave", w]
            for key, flag, typ, lo, hi in (("blip_rate", "--blip-rate", float, 1, 60), ("blip_pitch", "--blip-pitch", float, -24, 24),
                                           ("blip_jitter", "--blip-jitter", float, 0, 12), ("blip_duty", "--blip-duty", float, 0.05, 0.95),
                                           ("blip_speed", "--blip-speed", float, 0.5, 6)):
                v = num(key, typ, lo, hi)
                if v is not None:
                    argv += [flag, f"{v:g}"]
    else:
        if eng == "fx":
            argv += ["--oplfx" if p.get("fx_kind") == "fm" else "--chipfx"]
            a = choice("archetype", meta["archetypes"] or [])
            if a:
                argv += ["--fx-archetype", a]
        else:
            argv += E["flags"]
        has_src = eng in ("chip", "opl") and (p.get("from_midi") or p.get("from_audio"))
        if not prompt and not has_src:
            raise ValueError("describe the sound first" if eng in ("sfx", "fx") else
                             "describe the piece, or give it a MIDI / audio file to follow" if eng in ("chip", "opl") else
                             "describe the music first" if eng == "music" else "give the song some tags")
        if prompt:
            if "\n" in prompt:
                raise ValueError("one description per render — use a Set for a list of sounds")
            argv.insert(1, prompt)
        lim = {"sfx": (0.03, 120), "music": (1, 120), "song": (1, 1000), "chip": (2, 600), "opl": (2, 600), "fx": (0.03, 10)}[eng]
        if (p.get("adv") or {}).get("engine") == "sao" and eng in ("sfx", "music"):
            lim = (lim[0], 47)
        secs = num("seconds", float, *lim)
        if secs is not None:
            argv += ["--seconds", f"{secs:g}"]
        takes = num("n", int, 1, 200, 1)
        if takes > 1:
            argv += ["-n", str(takes)]
        if E["gpu"]:
            styles = p.get("styles") or []
            if not isinstance(styles, list):
                raise ValueError("styles must be a list")
            styles = [str(s) for s in styles if s]
            bad = [s for s in styles if s not in meta["styles"]["styles"]]
            if bad:
                raise ValueError(f"unknown style guide(s): {', '.join(bad)}")
            if styles:
                argv += ["--style", ",".join(styles)]
            if eng in ("sfx", "music") and p.get("reprompt") is False:
                argv.append("--no-reprompt")
            if eng == "music" and p.get("loop"):
                argv.append("--loop")
        if eng == "song":
            lyr = str(p.get("lyrics") or "").strip()
            if len(lyr) > 20000:
                raise ValueError("lyrics are too long (20,000 characters max)")
            if lyr:
                argv += ["--lyrics-file", jfile("lyrics.txt", lyr + "\n")]
            ts = choice("timesig", [str(t) for t in meta["timesigs"]] or ["4"], None)
            if ts and ts != "4":
                argv += ["--timesig", ts]
            lang = str(p.get("lang") or "").strip()
            if lang and lang != "en":
                if not re.fullmatch(r"[a-z]{2}", lang):
                    raise ValueError("language must be a two-letter code like en")
                argv += ["--lang", lang]
        if eng in ("song", "chip", "opl"):
            bpm = num("bpm", float, 10, 300)
            if bpm is not None:
                argv += ["--bpm", f"{bpm:g}"]
            key = choice("key", meta["keys"] or ["C minor"])
            if key:
                argv += ["--key", key]
        if eng in ("chip", "opl"):
            mood = choice("mood", ["auto"] + meta["moods"])
            if mood and mood != "auto":
                argv += ["--mood", mood]
            sc = choice("scale", meta["scales"])
            if sc:
                argv += ["--scale", sc]
            ch = choice("chippy", meta["chippy"])
            if ch and ch != "off":
                argv += ["--chippy", ch]
            arp = num("arp", int, 0, 16)
            if arp:
                argv += ["--arp", str(arp)]
            for key, flag in (("from_midi", "--from-midi"), ("from_audio", "--from-audio")):
                if p.get(key):
                    src = input_path(p[key])
                    argv += [flag, src]
            if eng == "chip" and p.get("write_mod"):
                argv.append("--write-mod")
            if eng == "opl" and p.get("write_rad"):
                argv.append("--write-rad")
            if p.get("write_midi") and not p.get("from_midi"):
                argv.append("--write-midi")
        if eng == "opl":
            for key, flag in (("opl_lead", "--opl-lead"), ("opl_arp", "--opl-arp"), ("opl_bass", "--opl-bass")):
                v = choice(key, meta["opl_instruments"])
                if v:
                    argv += [flag, v]
            if p.get("opl_bank"):
                argv += ["--opl-bank", input_path(p["opl_bank"])]
        if E["gpu"] and p.get("ref"):              # steer with a reference sound (audio-to-audio)
            ref = input_path(p["ref"])
            argv += ["--ref", ref]
            st = num("ref_strength", float, 0.05, 1.0)
            if st is not None and abs(st - 0.6) > 1e-9:
                argv += ["--ref-strength", f"{st:g}"]
            if p.get("ref_match"):
                argv.append("--ref-match")
            if p.get("match_tone"):
                argv += ["--match-tone", ref]
        if E["gpu"]:
            srv = p.get("servers") or []
            if not isinstance(srv, list):
                raise ValueError("servers must be a list")
            srv = [str(s).strip() for s in srv if str(s).strip()]
            for s in srv:
                if not re.fullmatch(r"[\w.:/\-\[\]]{1,200}", s):
                    raise ValueError(f"bad render server {s!r}")
            argv += ["--server", ",".join(srv) if srv else render_server()]

    name = str(p.get("name") or "").strip()
    if name:
        if not re.fullmatch(r"[\w.\- ]{1,64}", name):
            raise ValueError("name: letters, digits, spaces, - _ and . only (64 max)")
        argv += ["--name", name]
    seed = num("seed", int, -1, 2**63 - 1)
    if seed is not None and seed >= 0:
        argv += ["--seed", str(seed)]
    fmt = choice("format", [f["name"] for f in meta["formats"]])
    if fmt and fmt != "none":
        argv += ["--format", fmt]
    bits = choice("bits", ["8", "16", "24"], None) if str(p.get("bits") or "") else None
    if bits and bits != "16":
        argv += ["--bits", bits]
    lufs = str(p.get("lufs") if p.get("lufs") is not None else "").strip()
    if lufs:
        if lufs != "off":
            try:
                if not -40 <= float(lufs) <= 0:
                    raise ValueError
            except ValueError:
                raise ValueError("loudness must be 'off' or -40..0 LUFS")
        if lufs != "-16" or eng in ("chip", "opl"):      # chip/opl default to no ceiling: say it
            argv += ["--lufs", lufs]
    enc = choice("encode", list(ENCODES), "wav")
    if enc in ("mp3", "opus") and not E["gpu"]:
        raise ValueError(f"{enc.upper()} is only made by the GPU engines — pick WAV, FLAC or OGG")
    argv += ENCODES[enc]
    argv += adv_argv(p.get("adv") or {}, eng)
    argv.append("--no-open")
    return argv, files, takes


def command_text(argv):
    return shjoin(["soundmon"] + argv[1:])


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
def _seed_of(fname):
    m = re.search(r"_s(\d+)(?:_\d+_)?\.", fname)
    return int(m.group(1)) if m else None


def scan_outputs(d):
    """The sounds a job folder holds, newest-run order, each with its seed and any tracker / MIDI score."""
    found = []
    for f in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
        rel = os.path.relpath(f, d)
        if not os.path.isfile(f) or rel.startswith(("takes" + os.sep, "_")):
            continue
        if f.lower().endswith(AUDIO_EXT) and not os.path.basename(f).startswith("input-"):
            found.append(rel)
    out = []
    for rel in found:
        stem = os.path.splitext(rel)[0]
        o = {"file": rel, "seed": _seed_of(os.path.basename(rel)), "bytes": os.path.getsize(os.path.join(d, rel))}
        scores = [os.path.basename(stem) + e for e in SCORE_EXT
                  if os.path.isfile(os.path.join(d, os.path.dirname(rel), os.path.basename(stem) + e))]
        if scores:
            o["scores"] = [os.path.join(os.path.dirname(rel), s) for s in scores]
        out.append(o)
    return out


class JobQueue:
    def __init__(self, gallery):
        self.gallery = gallery
        self.jobs, self.order = {}, []
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.proc, self.current = None, None
        self.made = 0                       # sounds made this session (status bar)
        threading.Thread(target=self._worker, daemon=True).start()

    def add(self, params, group=None, label=None):
        meta = load_meta()
        ref = params.get("set")
        if ref is not None:                 # a Sets render: which set and key the takes belong to
            if not (isinstance(ref, dict) and re.fullmatch(r"[\w-]{1,64}", str(ref.get("id") or ""))
                    and setsmod.KEY_RE.fullmatch(str(ref.get("key") or ""))):
                raise ValueError("bad set reference")
            params["set"] = {"id": ref["id"], "key": ref["key"]}
        build_argv(params, meta)            # validate before touching disk
        jid = time.strftime("%Y%m%d-%H%M%S-") + f"{random.randrange(16**4):04x}"
        jdir = os.path.join(self.gallery, jid)
        os.makedirs(jdir, exist_ok=True)
        argv, files, takes = build_argv(params, meta, jdir)
        # the job keeps its own copy of every input it USES (an engine that ignores a stale reference
        # left in the form mustn't need it): reproducible later, whatever happens to inputs/
        for key in ("from_midi", "from_audio", "opl_bank", "ref"):
            if params.get(key):
                try:
                    src = input_path(params[key])
                except ValueError:
                    continue
                if src in argv:
                    dst = os.path.join(jdir, "input-" + os.path.basename(src))
                    shutil.copy2(src, dst)
                    argv = [dst if a == src else a for a in argv]
        for name, text in files.items():
            with open(os.path.join(jdir, name), "w", encoding="utf-8") as fh:
                fh.write(text)
        job = {"id": jid, "params": params, "argv": argv, "command": command_text(argv),
               "engine": params.get("engine") or "sfx", "status": "queued", "log": [], "outputs": [],
               "done": 0, "total": takes, "group": group, "label": label, "ratings": {},
               "created": time.time(), "started": None, "finished": None, "dir": jid}
        with self.lock:
            self.jobs[jid] = job
            self.order.append(jid)
        self._save(job)
        self.q.put(jid)
        return job

    def gallery_path(self, d, f):
        full = os.path.realpath(os.path.join(self.gallery, str(d or ""), str(f or "")))
        if not full.startswith(os.path.realpath(self.gallery) + os.sep) or not os.path.isfile(full):
            raise ValueError("that sound isn't in the gallery")
        return full

    def cancel(self, jid):
        with self.lock:
            job = self.jobs.get(jid)
            if not job:
                return False
            if job["status"] == "queued":
                job["status"], job["finished"] = "cancelled", time.time()
            elif job["status"] == "running" and self.current == jid and self.proc:
                # SIGTERM: soundmon takes its own prompts off ComfyUI's queue and interrupts the one
                # rendering if it is ours, so the GPU stops too (and other tools' prompts survive)
                job["status"] = "cancelled"
                job["log"].append("⛔ cancelled from the GUI")
                self.proc.terminate()
            else:
                return True
        self._save(job)
        return True

    def clear(self):
        """Clear finished jobs from the Queue tab (their files stay in the gallery). Waiting and
        running jobs are left alone: cancelling is Cancel's job, not Clear's."""
        removed = 0
        with self.lock:
            for jid in list(self.order):
                if self.jobs[jid]["status"] in ("done", "failed", "cancelled"):
                    self.order.remove(jid)
                    del self.jobs[jid]
                    removed += 1
        return {"cancelled": 0, "removed": removed}

    def abort(self, whole_server=False):
        """⛔ CANCEL AND ABORT: cancel every waiting job and stop the running one. The running soundmon
        removes its own prompts from ComfyUI on the way out (SIGTERM), so a ComfyUI shared with
        pixelmon or another soundmon keeps their work. whole_server=True also empties the render
        servers' entire queues and interrupts whatever they're sampling (the old, blunt behaviour)."""
        servers = set(render_server().split(","))
        proc, cancelled, touched = None, 0, []
        with self.lock:
            for jid in self.order:
                job = self.jobs[jid]
                if job["status"] == "queued":
                    job["status"], job["finished"] = "cancelled", time.time()
                    touched.append(job)
                    cancelled += 1
                elif job["status"] == "running":
                    argv = job["argv"]
                    if "--server" in argv:
                        servers.update(argv[argv.index("--server") + 1].split(","))
                    if self.current == jid and self.proc:
                        job["status"] = "cancelled"
                        proc = self.proc
                        proc.terminate()
                        cancelled += 1
        for job in touched:
            self._save(job)
        if proc:                                         # let soundmon clean up and exit first
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
        comfy = {}
        if not whole_server:
            return {"cancelled": cancelled, "servers": comfy, "scoped": True}
        for name in sorted(s.strip() for s in servers if s.strip()):
            url = server_url(name)
            if not url or url in comfy.values():
                continue
            try:
                for path, body in (("/queue", {"clear": True}), ("/interrupt", {})):
                    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), method="POST",
                                                 headers={"Content-Type": "application/json"})
                    urllib.request.urlopen(req, timeout=4).close()
                comfy[name] = url
            except Exception as e:
                comfy[name] = f"unreachable ({e})"
        return {"cancelled": cancelled, "servers": comfy}

    def snapshot(self, every=False):
        """The Queue tab's view: every waiting/running job plus the latest 60 finished ones
        (every=True: all of them, for questions like "which keys of this set are pending?")."""
        with self.lock:
            ids = self.order if every else [j for j in self.order if self.jobs[j]["status"] in ("queued", "running")] + \
                [j for j in self.order if self.jobs[j]["status"] not in ("queued", "running")][-60:]
            ids = sorted(ids, key=self.order.index)
            return [dict(self.jobs[j], log=self.jobs[j]["log"][-40:]) for j in ids]

    def recover(self):
        """At start-up: jobs a previous soundmon-gui left "queued" or "running" (it was stopped or
        crashed) are marked as interrupted, and whatever takes they finished are kept."""
        n = 0
        for f in glob.glob(os.path.join(self.gallery, "*", "job.json")):
            job = _read_json(f, None)
            if not isinstance(job, dict) or job.get("status") not in ("queued", "running"):
                continue
            outs = scan_outputs(os.path.dirname(f))
            job.update(status="done" if outs and job["status"] == "running" and len(outs) >= (job.get("total") or 1) else "failed",
                       outputs=outs, finished=job.get("finished") or time.time(),
                       error=None if outs and len(outs) >= (job.get("total") or 1) else
                       f"soundmon-gui stopped while this job was {job['status']}" + (f" ({len(outs)} take{'s' if len(outs) != 1 else ''} finished)" if outs else ""))
            _write_json(f, job)
            n += 1
        return n

    def _save(self, job):
        d = os.path.join(self.gallery, job["dir"])
        os.makedirs(d, exist_ok=True)
        _write_json(os.path.join(d, "job.json"), {k: v for k, v in job.items() if k != "log"})
        with open(os.path.join(d, "log.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(job["log"]) + "\n")

    def _worker(self):
        while True:
            jid = self.q.get()
            with self.lock:
                job = self.jobs.get(jid)
                if not job or job["status"] == "cancelled":
                    continue
                job["status"], job["started"] = "running", time.time()
                self.current = jid
            d = os.path.join(self.gallery, job["dir"])
            self._save(job)
            argv = job["argv"] + ["--output-to", d]
            rc = -1
            try:
                self.proc = subprocess.Popen(argv, cwd=d, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, bufsize=1, stdin=subprocess.DEVNULL,
                                             env=dict(os.environ, NO_COLOR="1", PYTHONUNBUFFERED="1"))
                for line in self.proc.stdout:
                    line = ANSI.sub("", line.rstrip())
                    if not line.strip():
                        continue
                    with self.lock:
                        job["log"].append(line)
                        m = DONE_LINE.search(line)
                        if m:
                            job["done"] = int(m.group(1))
                            job["total"] = max(job["total"], int(m.group(2)))
                rc = self.proc.wait()
            except Exception as e:
                job["log"].append(f"GUI error: {e}")
            with self.lock:
                outs = scan_outputs(d)                      # trust the folder over the log
                seeds = [int(s) for s in SEED.findall("\n".join(ln for ln in job["log"] if "✅" in ln))]
                for o in outs:                              # files without a seed in the name (blips) keep the log's
                    if o["seed"] is None and seeds:
                        o["seed"] = seeds[min(len(seeds) - 1, outs.index(o))]
                job["outputs"] = outs
                self.made += len(outs)
                if job["status"] != "cancelled":
                    job["status"] = "done" if rc == 0 and outs else "failed"
                    if job["status"] == "failed":           # the node's own message beats "N jobs failed"
                        last = next((ln for ln in reversed(job["log"]) if ln.strip()), f"exit {rc}").strip()
                        why = next((ln.strip() for ln in job["log"]
                                    if "ComfyUI failed" in ln or re.search(r"failed seed=\d+:", ln)), "")
                        job["error"] = (why + ("  —  " + last if why and last not in why else "") if why else last)[:600]
                job["finished"] = time.time()
                self.current, self.proc = None, None
            self._save(job)


def list_presets():
    out = []
    for f in sorted(glob.glob(os.path.join(PRESETS, "*.json"))):
        p = _read_json(f, None)
        if p and isinstance(p.get("snapshot"), dict):
            out.append({"name": p.get("name") or os.path.splitext(os.path.basename(f))[0], "saved": p.get("saved"),
                        "engine": p["snapshot"].get("engine"), "snapshot": p["snapshot"]})
    return sorted(out, key=lambda p: p["name"].lower())


def save_preset(body):
    name = re.sub(r"\s+", " ", str(body.get("name") or "")).strip()[:60]
    if not name:
        raise ValueError("give the preset a name")
    snap = body.get("snapshot")
    if not isinstance(snap, dict) or snap.get("engine") not in ENGINES:
        raise ValueError("nothing to save")
    if len(json.dumps(snap)) > 200000:
        raise ValueError("that preset is too big")
    _write_json(os.path.join(PRESETS, safe_name(name, 60) + ".json"), {"name": name, "saved": time.time(), "snapshot": snap})
    return list_presets()


def delete_preset(name):
    f = os.path.join(PRESETS, safe_name(name, 60) + ".json")
    if not os.path.isfile(f):
        raise ValueError("no preset by that name")
    os.remove(f)
    return list_presets()


def mastering():
    """soundmon's master.py (neon-caster's chain), loaded fresh; ValueError when numpy / ffmpeg are missing."""
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise LookupError("measuring and mastering need numpy — run soundmon-gui with ComfyUI's venv")
    if not shutil.which("ffmpeg"):
        raise LookupError("measuring and mastering need ffmpeg on PATH")
    return load_module("master")


def analysis(path, cache_dir):
    """master.analyse() of one file, cached beside it (keyed by the file's mtime)."""
    os.makedirs(cache_dir, exist_ok=True)
    cf = os.path.join(cache_dir, os.path.basename(path) + ".json")
    hit = _read_json(cf, None)
    if hit and hit.get("mtime") == os.path.getmtime(path):
        return hit["a"]
    a = mastering().analyse(path, sonogram=True)
    _write_json(cf, {"mtime": os.path.getmtime(path), "a": a})
    return a


# ---------------------------------------------------------------------------
# Find: sounds on disk like this one (similar.py)
# ---------------------------------------------------------------------------
FIND = {"index": None, "mtime": None, "job": None}


def find_index():
    """similar.Index, reloaded only when its file changed."""
    sim = load_module("similar")
    p = sim.INDEX
    mt = os.path.getmtime(p) if os.path.exists(p) else None
    if FIND["index"] is None or FIND["mtime"] != mt:
        FIND["index"], FIND["mtime"] = sim.Index(p), mt
    return FIND["index"]


def find_start(folders):
    """Index folders in the background; FIND["job"] reports progress."""
    if FIND["job"] and FIND["job"]["running"]:
        raise ValueError("already indexing — wait for it to finish")
    folders = [home_path(f) for f in folders]
    if not folders:
        raise ValueError("add a folder to index first")
    mastering()                                         # numpy + ffmpeg, or a clear 501
    job = {"running": True, "done": 0, "total": 0, "current": "", "stats": None, "error": None, "started": time.time()}
    FIND["job"] = job

    def prog(i, n, path):
        job.update(done=i, total=n, current=os.path.basename(path))

    def run():
        try:
            idx = load_module("similar").Index()
            job["stats"] = idx.update(folders, prog)
        except Exception as e:
            job["error"] = str(e)[:300]
        job["running"] = False
        FIND["index"] = None
    threading.Thread(target=run, daemon=True).start()
    return job


def gallery_items(gallery, limit=400):
    items = []
    for jf in sorted(glob.glob(os.path.join(gallery, "*", "job.json")), reverse=True)[:limit]:
        job = _read_json(jf, None)
        if job and job.get("outputs"):
            items.append(job)
    return items


def rate(gallery, d, f, stars):
    jf = os.path.join(gallery, safe_name(d, 64), "job.json")
    job = _read_json(jf, None)
    if not job:
        raise ValueError("that job isn't in the gallery")
    if f not in {o["file"] for o in job.get("outputs") or []}:
        raise ValueError("that sound isn't in the job")
    stars = int(stars)
    if not 0 <= stars <= 5:
        raise ValueError("stars must be 0..5")
    job.setdefault("ratings", {})[f] = stars
    _write_json(jf, job)
    return job["ratings"]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
AUDIO_TYPES = {".wav": "audio/wav", ".flac": "audio/flac", ".ogg": "audio/ogg", ".mp3": "audio/mpeg",
               ".opus": "audio/ogg", ".mod": "audio/x-mod", ".rad": "application/octet-stream",
               ".mid": "audio/midi", ".txt": "text/plain; charset=utf-8", ".json": "application/json"}


class QuietServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


class Handler(http.server.BaseHTTPRequestHandler):
    jobs = None
    gallery = None
    sets = None

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _trusted(self):
        """Refuse requests that didn't come from this GUI's own page.

        Host must be this server by IP or localhost: a web page can point its own domain at
        127.0.0.1 (DNS rebinding) and would otherwise be "same origin" with the GUI. A browser's
        Origin/Referer, when present, must match Host: another site's page can't drive the GUI.
        Tools without those headers (curl) are fine."""
        host = (self.headers.get("Host") or "").strip().lower()
        name = host.rsplit(":", 1)[0].strip("[]") if host.count(":") <= 1 or host.startswith("[") else host
        ip_literal = bool(re.fullmatch(r"[\d.]+|[0-9a-f:]+", name))
        if not (name in ("localhost", "127.0.0.1", "::1") or (LAN and (ip_literal or name in (socket.gethostname().lower(), socket.gethostname().lower() + ".local")))):
            return False
        src = self.headers.get("Origin") or self.headers.get("Referer")
        if src and src != "null":
            p = urllib.parse.urlparse(src)
            if (p.netloc or "").lower() != host:
                return False
        return True

    def _local(self):
        """Is the browser on this machine? Always true unless --lan opened the page to other devices."""
        return not LAN or self.client_address[0] in ("127.0.0.1", "::1") or self.client_address[0].startswith("127.")

    def _file(self, path, ctype, download=None):
        """Serve a file, honouring Range so the browser can seek long music."""
        try:
            size = os.path.getsize(path)
            fh = open(path, "rb")
        except OSError:
            return self._json({"error": "not found"}, 404)
        with fh:
            start, end = 0, size - 1
            rng = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
            if rng and size:
                if rng.group(1):
                    start = int(rng.group(1))
                    end = min(int(rng.group(2)), size - 1) if rng.group(2) else size - 1
                else:
                    start = max(0, size - int(rng.group(2) or 0))
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            self.send_response(206 if rng and size else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1 if size else 0))
            if rng and size:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if download:                                   # ASCII fallback + RFC 5987 for any other name
                plain = re.sub(r'[^\x20-\x7e]|["\\]', "_", download)
                self.send_header("Content-Disposition", f"attachment; filename=\"{plain}\"; filename*=UTF-8''{urllib.parse.quote(download)}")
            self.end_headers()
            fh.seek(start)
            left = end - start + 1 if size else 0
            while left > 0:
                chunk = fh.read(min(1 << 16, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def do_GET(self):
        if not self._trusted():
            return self._json({"error": "forbidden: open soundmon-gui from its own address"}, 403)
        u = urllib.parse.urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._file(os.path.join(HERE, "index.html"), "text/html; charset=utf-8")
        if u.path == "/api/meta":
            meta = load_meta()
            meta.update(server=render_server(), server_up=server_is_up(), known_servers=sorted(known_servers()),
                        default_server=default_server(), setup=load_setup(), gallery=self.gallery)
            return self._json(meta)
        if u.path == "/api/jobs":
            return self._json({"jobs": self.jobs.snapshot(), "made": self.jobs.made,
                               "render": {"server": render_server(), "url": server_url(), "up": server_is_up(),
                                          "stuck": SERVER_STATE["stuck"] if SERVER_STATE["server"] == render_server() else None,
                                          "restart": SERVER_STATE["restart"]}})
        if u.path == "/api/servers":
            return self._json(servers_report())
        if u.path == "/api/gallery":
            return self._json({"items": gallery_items(self.gallery)})
        if u.path == "/api/backups":
            return self._json({"backups": list_backups(), "folder": backups_dir()})
        if u.path == "/api/backup/download":
            try:
                full = backup_path(urllib.parse.parse_qs(u.query).get("name", [""])[0])
            except ValueError as e:
                return self._json({"error": str(e)}, 404)
            return self._file(full, "application/zip", os.path.basename(full))
        if u.path == "/api/setup":
            snd, fold = detect_programs()
            return self._json(dict(load_setup(), detected={"audio_apps": snd, "folder_apps": fold}, locked=SERVER_ARG or None))
        if u.path == "/api/sets":
            return self._json({"sets": self.sets.list()})
        if u.path == "/api/presets":
            return self._json({"presets": list_presets()})
        if u.path == "/api/similar/status":
            try:
                idx = find_index()
                return self._json({"count": len(idx.data["files"]), "folders": idx.data["folders"],
                                   "updated": idx.data.get("updated"), "job": FIND["job"],
                                   "setup_folders": load_setup()["similar_folders"]})
            except Exception as e:
                return self._json({"error": str(e)}, 500)
        if u.path in ("/api/browse", "/api/disk/file") and not self._local():
            return self._json({"error": "browsing this computer's files only works on the computer running soundmon-gui"}, 403)
        if u.path == "/api/browse":                    # the Browse dialog: one folder under home
            q = urllib.parse.parse_qs(u.query)
            exts = tuple(e.strip().lower() for e in (q.get("ext") or [""])[0].split(",") if e.strip()) or None
            try:
                return self._json(browse((q.get("path") or ["~"])[0], (q.get("kind") or ["any"])[0], exts,
                                         (q.get("hidden") or [""])[0] == "1", self.gallery))
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
        if u.path == "/api/disk/file":                 # hear an audio file picked in Browse (audio only, under home)
            try:
                p = home_path(urllib.parse.parse_qs(u.query).get("path", [""])[0])
            except ValueError as e:
                return self._json({"error": str(e)}, 404)
            ext = os.path.splitext(p)[1].lower()
            if not os.path.isfile(p) or ext not in DISK_AUDIO:
                return self._json({"error": "only audio files can be played"}, 404)
            return self._file(p, AUDIO_TYPES.get(ext) or mimetypes.guess_type(p)[0] or "application/octet-stream")
        if u.path == "/api/input":                     # hear a file dropped on the form (reference, recording)
            try:
                p = input_path(urllib.parse.parse_qs(u.query).get("name", [""])[0])
            except ValueError as e:
                return self._json({"error": str(e)}, 404)
            ext = os.path.splitext(p)[1].lower()
            return self._file(p, AUDIO_TYPES.get(ext) or mimetypes.guess_type(p)[0] or "application/octet-stream")
        if u.path == "/api/similar/file":              # play a Find result: only files in the index
            p = urllib.parse.parse_qs(u.query).get("path", [""])[0]
            if p not in find_index().data["files"] or not os.path.isfile(p):
                return self._json({"error": "not in the index"}, 404)
            ext = os.path.splitext(p)[1].lower()
            return self._file(p, AUDIO_TYPES.get(ext) or mimetypes.guess_type(p)[0] or "application/octet-stream")
        if u.path in ("/api/set", "/api/set/now"):
            q = urllib.parse.parse_qs(u.query)
            try:
                s = self.sets.load(q.get("id", [""])[0])
                if u.path == "/api/set/now":             # the file the game plays today, for ▶ NOW
                    full = self.sets.now_path(s, q.get("key", [""])[0])
                    ext = os.path.splitext(full)[1].lower()
                    return self._file(full, AUDIO_TYPES.get(ext) or "application/octet-stream")
                return self._json({"set": s, "takes": self.sets.takes(s["id"]), "now": self.sets.now_files(s),
                                   "pending": self.sets.pending(s["id"], self.jobs.snapshot(every=True))})
            except ValueError as e:
                return self._json({"error": str(e)}, 404 if u.path == "/api/set/now" else 400)
        if u.path == "/api/analyse":
            # measure a take (?take=<job>/<file>) or the game's current file (?set=&key=): loudness, tilt,
            # energy above 2.5 kHz, harsh seconds, sonogram — neon-caster's Sound Audit numbers
            q = urllib.parse.parse_qs(u.query)
            try:
                if q.get("take"):
                    path = self.sets.take_path(q["take"][0])
                    cache = os.path.join(os.path.dirname(path), "_analysis")
                else:
                    s = self.sets.load(q.get("set", [""])[0])
                    path = self.sets.now_path(s, q.get("key", [""])[0])
                    cache = os.path.join(SETS_ROOT, s["id"], "_analysis")
                return self._json({"analysis": analysis(path, cache)})
            except LookupError as e:
                return self._json({"error": str(e)}, 501)
            except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as e:
                return self._json({"error": str(e)}, 400)
        if u.path.startswith("/files/"):
            rel = urllib.parse.unquote(u.path[len("/files/"):])
            if "\x00" in rel:
                return self._json({"error": "not found"}, 404)
            full = os.path.realpath(os.path.join(self.gallery, rel))
            if not full.startswith(os.path.realpath(self.gallery) + os.sep) or not os.path.isfile(full):
                return self._json({"error": "not found"}, 404)
            ext = os.path.splitext(full)[1].lower()
            dl = os.path.basename(full) if "download" in urllib.parse.parse_qs(u.query, keep_blank_values=True) else None
            return self._file(full, AUDIO_TYPES.get(ext) or mimetypes.guess_type(full)[0] or "application/octet-stream", dl)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self._trusted():
            return self._json({"error": "forbidden: open soundmon-gui from its own address"}, 403)
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/upload":                      # raw bytes; ?filename=
            return self._upload(urllib.parse.parse_qs(u.query))
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            return self._json({"error": "expected application/json"}, 415)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json({"error": "bad json"}, 400)
        if not isinstance(body, dict):
            return self._json({"error": "expected a JSON object"}, 400)
        try:
            if u.path == "/api/command":                 # the exact command RENDER would run
                try:                                     # half-typed forms are normal here: answer, don't fail
                    argv, _, takes = build_argv(body, load_meta())
                except ValueError as e:
                    return self._json({"invalid": str(e)})
                return self._json({"command": command_text(argv), "takes": takes, "warnings": preflight(body)})
            if u.path == "/api/render":
                group, label = body.pop("group", None), body.pop("label", None)
                job = self.jobs.add(body, group=group, label=label)
                return self._json({"jobs": [job["id"]], "command": job["command"]})
            if u.path == "/api/sweep":
                # same seed, one setting varied over a few values: one labelled job per value, one queue card
                base, field, values = dict(body.get("params") or {}), str(body.get("field") or ""), body.get("values")
                if not re.fullmatch(r"(adv\.)?[a-z_]{1,30}", field):
                    raise ValueError("pick a setting to sweep")
                if not isinstance(values, list) or not 2 <= len(values) <= 24:
                    raise ValueError("give 2 to 24 values, separated by commas")
                seed = str(base.get("seed") or "").strip()
                if not seed.isdigit():
                    seed = str(random.randrange(1, 2 ** 31))
                base.update(seed=seed, n=1)
                variants = []
                for v in values:
                    q = json.loads(json.dumps(base))
                    v = v.strip() if isinstance(v, str) else v
                    if field.startswith("adv."):
                        q.setdefault("adv", {})[field[4:]] = v
                    elif field == "styles":
                        q["styles"] = [x for x in str(v).split("+") if x]
                    else:
                        q[field] = v
                    try:
                        build_argv(q, load_meta())
                    except ValueError as e:
                        raise ValueError(f"{field} = {v}: {e}")
                    variants.append((q, f"{field.split('.')[-1]} = {v}"))
                group = f"sweep:{field.split('.')[-1]}:{time.strftime('%H%M%S')}"
                ids = [self.jobs.add(q, group=group, label=lab)["id"] for q, lab in variants]
                return self._json({"group": group, "jobs": ids, "seed": seed})
            if u.path == "/api/input/from-take":         # 🧬 vary: a gallery take becomes a reference
                src = self.jobs.gallery_path(body.get("dir"), body.get("file"))
                return self._json({"file": store_input(safe_name(os.path.basename(src), 96), src=src)})
            if u.path.startswith("/api/cancel/"):
                return self._json({"ok": self.jobs.cancel(u.path.rsplit("/", 1)[-1])})
            if u.path == "/api/jobs/clear":
                return self._json(self.jobs.clear())
            if u.path == "/api/backup":
                path, n, size, skipped = make_backup(body.get("keyword"), self.gallery)
                return self._json({"name": os.path.basename(path), "files": n, "bytes": size, "skipped": skipped})
            if u.path == "/api/backups/restore":
                return self._json(restore_backup(body.get("name"), self.gallery))
            if u.path == "/api/gallery/clear":
                if any(j["status"] in ("queued", "running") for j in self.jobs.snapshot()):
                    raise ValueError("renders are still queued or running — wait for them (or clear the queue) first")
                if str(body.get("confirm") or "") != "CLEAR":
                    raise ValueError("type CLEAR to confirm")
                backup = None
                if body.get("backup", True):
                    backup = os.path.basename(make_backup(body.get("keyword") or "before-clear", self.gallery)[0])
                removed = clear_gallery(self.gallery)
                self.jobs.clear()
                return self._json({"removed": removed, "backup": backup})
            if u.path == "/api/jobs/abort":
                return self._json(self.jobs.abort(bool(body.get("whole_server"))))
            if u.path == "/api/rate":
                return self._json({"ratings": rate(self.gallery, body.get("dir"), body.get("file"), body.get("stars", 0))})
            if u.path == "/api/setup":
                snd, fold = detect_programs()
                return self._json(dict(save_setup(body), detected={"audio_apps": snd, "folder_apps": fold}, locked=SERVER_ARG or None))
            if u.path == "/api/presets":
                return self._json({"presets": save_preset(body)})
            if u.path == "/api/similar/index":
                folders = body.get("folders")
                if isinstance(folders, list):
                    for f in folders:                          # check before saving: a bad path never sticks
                        if not os.path.isdir(home_path(f)):
                            raise ValueError(f"not a folder: {f}")
                    save_setup({"similar_folders": folders})
                try:
                    return self._json({"job": find_start(load_setup()["similar_folders"])})
                except LookupError as e:
                    return self._json({"error": str(e)}, 501)
            if u.path == "/api/similar/search":
                # the query: a dropped file (input), a gallery take (take), or an indexed file (path)
                idx = find_index()
                if body.get("take"):
                    q = self.sets.take_path(body["take"])
                elif body.get("input"):
                    q = input_path(body["input"])
                elif body.get("path") in idx.data["files"]:
                    q = body["path"]
                else:
                    raise ValueError("drop a sound to search with")
                if not idx.data["files"]:
                    raise ValueError("nothing is indexed yet — add your sound folders and press Index")
                within = str(body.get("within") or "").strip()
                if within:
                    within = home_path(within)
                k = max(1, min(200, int(body.get("k") or 30)))
                try:
                    mastering()
                except LookupError as e:
                    return self._json({"error": str(e)}, 501)
                return self._json({"query": os.path.basename(q), "results": idx.search(q, k, within=within or None)})
            if u.path == "/api/input/from-path":        # Browse picked a file on disk: copy it in, like a drop
                if not self._local():
                    raise ValueError("picking files from this computer only works on the computer running soundmon-gui")
                p = home_path(body.get("path"))
                if not os.path.isfile(p):
                    raise ValueError("pick a file, not a folder")
                name = safe_name(os.path.basename(p), 96)
                if not name.lower().endswith(INPUT_EXT):
                    raise ValueError(f"not a file soundmon reads: {os.path.basename(p)} (MIDI, WAV/FLAC/OGG/MP3, .sbi or .txt)")
                if os.path.getsize(p) > 200 * 1024 * 1024:
                    raise ValueError("file must be under 200 MB")
                res = {"file": store_input(name, src=p), "from": p}
                if name.lower().endswith(".txt"):
                    with open(p, encoding="utf-8", errors="replace") as fh:
                        res["text"] = fh.read(200000)
                return self._json(res)
            if u.path == "/api/similar/use":            # a Find result becomes a reference (copied into inputs)
                p = body.get("path")
                if p not in find_index().data["files"] or not os.path.isfile(p):
                    raise ValueError("that file isn't in the index")
                name = safe_name(os.path.basename(p), 96)
                if not name.lower().endswith(INPUT_EXT):
                    name += ".wav"
                return self._json({"file": store_input(name, src=p)})
            if u.path == "/api/presets/delete":
                return self._json({"presets": delete_preset(body.get("name"))})
            if u.path == "/api/voices":
                # one line in several voices: one narrate job per voice, grouped so the Voices tab can find them
                line = str(body.get("line") or "").strip()
                if not line or "\n" in line or "|" in line or len(line) > 400:
                    raise ValueError("type one line (up to 400 characters) to hear in each voice")
                meta = load_meta()
                voices = [v for v in (body.get("voices") or list(meta["voices"])) if v in meta["voices"]]
                if not voices:
                    raise ValueError("Kokoro's voice list didn't load (narrate.py) — is soundmon installed?")
                group = "voices:" + time.strftime("%Y%m%d-%H%M%S")
                ids = [self.jobs.add({"engine": "narrate", "prompt": line, "voice": v, "name": v, "speed": body.get("speed"),
                                      "pitch": body.get("pitch")}, group=group, label=v)["id"] for v in voices]
                return self._json({"group": group, "jobs": ids})
            if u.path == "/api/master":
                # hear a take mastered before exporting it: writes <job>/_mastered/<kind>/<take>.wav (cached)
                kind = body.get("kind") or "sfx"
                src = self.sets.take_path(body.get("take"))
                rel = os.path.relpath(src, self.gallery)
                job_dir = rel.split(os.sep)[0]
                dst = os.path.join(self.gallery, job_dir, "_mastered", safe_name(kind, 10),
                                   os.path.splitext(os.path.basename(src))[0] + ".wav")
                try:
                    m = mastering()
                    if kind not in m.KINDS:
                        raise ValueError("kind must be sfx or speech")
                    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
                        m.master(src, dst, kind, analysis(src, os.path.join(os.path.dirname(src), "_analysis")))
                    after = analysis(dst, os.path.join(os.path.dirname(dst), "_analysis"))
                except LookupError as e:
                    return self._json({"error": str(e)}, 501)
                return self._json({"url": "/files/" + urllib.parse.quote(os.path.relpath(dst, self.gallery)), "analysis": after})
            if u.path.startswith("/api/sets/"):
                return self._json(self._sets_post(u.path.rsplit("/", 1)[-1], body))
            if u.path == "/api/open":
                # open a sound or a job folder in a program saved in Setup, or the OS default.
                # Only programs saved in Setup can be launched: the request names one, it can't supply a command.
                setup, key, t = load_setup(), str(body.get("app") or "os"), body.get("target") or {}
                is_dir = bool(t.get("folder")) or bool(t.get("indexed"))
                if t.get("backups"):
                    os.makedirs(backups_dir(), exist_ok=True)
                    t = {"path": backups_dir()}
                if t.get("path") or t.get("input") or t.get("set_now"):
                    if t.get("path") and not self._local():
                        raise ValueError("opening paths only works on the computer running soundmon-gui")
                    # any file or folder under home (a path field, a Browse pick, a Find result, the game's file)
                    if t.get("input"):
                        path = input_path(t["input"])
                    elif t.get("set_now"):
                        path = self.sets.now_path(self.sets.load(t["set_now"].get("id")), t["set_now"].get("key"))
                    else:
                        path = home_path(t["path"])
                    if t.get("reveal") and os.path.isfile(path):
                        path = os.path.dirname(path)
                    is_dir = os.path.isdir(path)
                    if not is_dir and not path.lower().endswith(OPENABLE):
                        raise ValueError("only sound, music and text files open from here (never programs or scripts)")
                elif t.get("indexed"):                        # a Find result: open the folder it is in
                    if t["indexed"] not in find_index().data["files"]:
                        raise ValueError("that file isn't in the index")
                    path = os.path.dirname(t["indexed"])
                elif is_dir:
                    path = os.path.join(self.gallery, safe_name(t.get("dir"), 64)) if t.get("dir") else self.gallery
                    if not os.path.isdir(path):
                        raise ValueError("that folder doesn't exist any more")
                else:
                    path = self.jobs.gallery_path(t.get("dir"), t.get("file"))
                if key == "os":
                    app = os_opener(is_dir)
                elif key.startswith(("snd:", "dir:")):
                    lst = setup["folder_apps" if key.startswith("dir:") else "audio_apps"]
                    i = int(key[4:]) if key[4:].isdigit() else -1
                    if not 0 <= i < len(lst):
                        raise ValueError("that program isn't in the Setup tab any more")
                    app = lst[i]
                else:
                    raise ValueError(f"unknown program {key!r}")
                launch(app, path)
                return self._json({"opened": path, "with": app.get("name")})
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        except (TypeError, AttributeError, KeyError) as e:         # a field of the wrong type or shape
            return self._json({"error": f"bad request: {type(e).__name__}: {e}"}, 400)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        return self._json({"error": "not found"}, 404)

    def _sets_post(self, act, body):
        validate = lambda p: build_argv(p, load_meta())     # noqa: E731
        if act == "create":
            return {"set": self.sets.create(body, validate)}
        if act == "update":
            return {"set": self.sets.update(body, validate)}
        if act == "mark":
            return {"set": self.sets.mark(body)}
        if act == "render":
            # render takes for some keys ("missing" = every key with no takes and nothing queued)
            s = self.sets.load(body.get("id"))
            keys = body.get("keys")
            if keys == "missing":
                have, busy = self.sets.takes(s["id"]), self.sets.pending(s["id"], self.jobs.snapshot(every=True))
                gone = self.sets.now_files(s) if body.get("skip_existing") else {}
                keys = [r["key"] for r in s["rows"] if not have.get(r["key"]) and not busy.get(r["key"]) and not gone.get(r["key"])]
                if not keys:
                    raise ValueError(f"nothing to render — every key has takes or is rendering ({sum(busy.values())} render{'s' if sum(busy.values()) != 1 else ''} queued)"
                                     if busy else "nothing to render — every key already has takes")
            if not isinstance(keys, list) or not keys:
                raise ValueError("nothing to render — pick some keys")
            if len(keys) > 1000:
                raise ValueError("at most 1000 keys at once")
            jobs, voices = [], s["theme"].get("voices") if s["theme"]["engine"] == "narrate" and body.get("seed") is None else None
            todo, bad = [], []
            for k in keys:                                # check every render first: all of them queue, or none
                for v in (voices or [None]):
                    p = self.sets.render_params(s, k, n=body.get("n"), seed=body.get("seed"), voice=v)
                    try:
                        build_argv(json.loads(json.dumps(p)), load_meta())
                    except ValueError as e:
                        bad.append(f"{k}: {e}")
                        break
                    todo.append((p, k))
            if bad:
                raise ValueError(f"nothing queued: {len(bad)} key{'s' if len(bad) > 1 else ''} won't render — " + "; ".join(bad[:6]) + (" …" if len(bad) > 6 else ""))
            for p, k in todo:
                jobs.append(self.jobs.add(p, group=f"set:{s['id']}", label=k)["id"])
            return {"jobs": jobs}
        if act == "export":
            s = self.sets.load(body.get("id"))
            if body.get("preview"):
                return {"plan": self.sets.export_plan(s), "folder": self.sets.game_dir(s)}
            mfn, mode = None, s["theme"]["export"].get("master") or "off"
            if mode != "off":
                try:
                    m = mastering()
                except LookupError as e:
                    raise ValueError(str(e))
                if mode == "like":                    # match the tone of the set's reference sound
                    ref = (s["theme"]["params"] or {}).get("ref")
                    if not ref:
                        raise ValueError("“match the reference” needs a reference sound in the set's theme")
                    rp = input_path(ref)
                    mfn = lambda src, dst, kind: m.match_tone(src, dst, rp)      # noqa: E731
                else:
                    mfn = m.master
            return self.sets.export(s, only_changed=not body.get("all"), master=mfn)
        raise ValueError(f"unknown sets action {act!r}")

    def _upload(self, q):
        """A MIDI / reference audio / OPL bank / manifest dropped on the form -> INPUTS (never overwrites)."""
        fname = safe_name((q.get("filename") or ["input.mid"])[0], 96)
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._json({"error": "bad Content-Length"}, 400)
        if not fname.lower().endswith(INPUT_EXT):
            return self._json({"error": f"not a file soundmon reads: {fname} (MIDI, WAV/FLAC/OGG/MP3, .sbi or .txt)"}, 400)
        if n <= 0:
            return self._json({"error": f"{fname} is empty"}, 400)
        if n > 200 * 1024 * 1024:
            return self._json({"error": "file must be under 200 MB"}, 400)
        data = self.rfile.read(n)
        if len(data) != n:
            return self._json({"error": f"the upload of {fname} was cut short ({len(data)} of {n} bytes)"}, 400)
        if fname.lower().endswith((".mid", ".midi")) and data[:4] != b"MThd":
            return self._json({"error": f"{fname} isn't a MIDI file"}, 400)
        res = {"file": store_input(fname, data=data)}
        if fname.lower().endswith(".txt"):
            res["text"] = data.decode("utf-8", "replace")[:200000]
        return self._json(res)


def store_input(name, data=None, src=None):
    """Put a dropped/picked file in INPUTS under a safe name. If the very same bytes are already
    there (the same file dropped again, a Find result used twice), reuse that one instead of
    piling up name-1, name-2 copies."""
    import hashlib
    os.makedirs(INPUTS, exist_ok=True)
    if data is None:
        with open(src, "rb") as fh:
            data = fh.read()
    digest, size = hashlib.sha1(data).hexdigest(), len(data)
    stem, ext = os.path.splitext(name)
    for f in os.listdir(INPUTS):
        full = os.path.join(INPUTS, f)
        if f.lower().endswith(ext.lower()) and os.path.isfile(full) and os.path.getsize(full) == size:
            with open(full, "rb") as fh:
                if hashlib.sha1(fh.read()).hexdigest() == digest:
                    return f
    out = unique_path(INPUTS, name)
    with open(out, "wb") as fh:
        fh.write(data)
    return os.path.basename(out)


def unique_path(d, fname):
    """d/fname, or d/fname-1, -2 … so nothing is ever overwritten."""
    stem, ext = os.path.splitext(fname)
    out, k = os.path.join(d, fname), 1
    while os.path.exists(out):
        out, k = os.path.join(d, f"{stem}-{k}{ext}"), k + 1
    return out


def main():
    global SERVER_ARG, GALLERY_HOME, INPUTS, SETUP_FILE, SETS_ROOT, PRESETS
    ap = argparse.ArgumentParser(prog="soundmon-gui", description="Web GUI for soundmon.")
    ap.add_argument("--port", type=int, default=8191)
    ap.add_argument("--server", default=os.environ.get("SOUNDMON_SERVER", ""),
                    help="render server: a servers.json alias, host[:port] or URL (default: $SOUNDMON_SERVER, "
                         "else the Setup tab's choice, else servers.json \"_default\", else 'local')")
    ap.add_argument("--lan", action="store_true", help="listen on all interfaces so other devices on the LAN can use it")
    ap.add_argument("--gallery", default=os.path.expanduser("~/soundmon-gallery/gui"),
                    help="where renders + job.json records go (default ~/soundmon-gallery/gui)")
    args = ap.parse_args()
    # The gallery's state (inputs, sets, presets, setup) lives beside it only in the standard layout
    # <home>/gui; any other --gallery keeps ~/soundmon-gallery, instead of littering the folder's parent
    g = os.path.normpath(os.path.expanduser(args.gallery))
    GALLERY_HOME = os.path.dirname(g) if os.path.basename(g) == "gui" else os.path.expanduser("~/soundmon-gallery")
    INPUTS = os.path.join(GALLERY_HOME, "inputs")
    SETUP_FILE = os.path.join(GALLERY_HOME, "gui-setup.json")
    SETS_ROOT = os.path.join(GALLERY_HOME, "sets")
    PRESETS = os.path.join(GALLERY_HOME, "gui-presets")
    SERVER_ARG = re.sub(r"\s+", "", args.server or "")
    os.makedirs(args.gallery, exist_ok=True)
    Handler.gallery = os.path.realpath(os.path.expanduser(args.gallery))
    Handler.jobs = JobQueue(Handler.gallery)
    n = Handler.jobs.recover()
    if n:
        print(f"[soundmon-gui] marked {n} job(s) a previous run left unfinished", flush=True)
    Handler.sets = setsmod.Sets(SETS_ROOT, Handler.gallery, home_path)
    threading.Thread(target=watch_server, daemon=True).start()
    global LAN
    LAN = bool(args.lan)
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    try:
        srv = QuietServer((host, args.port), Handler)
    except OSError as e:
        if e.errno != 98:                        # EADDRINUSE
            raise
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/api/meta", timeout=2) as r:
                json.load(r)
            sys.exit(f"soundmon-gui is already running: http://127.0.0.1:{args.port}\n"
                     f"  (stop it with:  pkill -f soundmon/gui/server.py   or pick another --port)")
        except (OSError, ValueError):
            sys.exit(f"port {args.port} is in use by another program; try  soundmon-gui --port {args.port + 1}")
    print(f"soundmon-gui on http://{'<this-ip>' if args.lan else '127.0.0.1'}:{args.port}  "
          f"(GPU renders on '{render_server()}' {server_url() or ''}, gallery {Handler.gallery})  Ctrl-C to quit")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
