#!/usr/bin/env python3
"""soundmon — make a game-ready sound effect from a text description.

This is the brains; run it through the `soundmon` wrapper, which makes sure the
ComfyUI server is running first. It talks to ComfyUI's HTTP API (so the visual
node-graph happens behind the scenes — you just describe the sound).

Sibling project to pixelmon; same architecture, one dimension over.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

# soundmon can render on a REMOTE ComfyUI (e.g. a faster box on the LAN). Choose a
# target with `--server NAME` (an alias from servers.json) or `--server host[:port]`/URL,
# or the SOUNDMON_SERVER env var. Default is local. When the target is remote, results
# are fetched back over HTTP (/view) — no shared filesystem needed.
SERVER = "http://127.0.0.1:8188"
REMOTE = False
POOL = []   # >1 entry (--server a,b,c) turns on render-farm mode (jobs fan across GPUs)
COMFY = os.path.expanduser("~/ComfyUI")
OUTPUT = os.path.join(COMFY, "output")
SFX_DIR = os.path.join(COMFY, "custom_nodes", "retro_sfx")

# Pull the format names straight from the node's registry so the two never drift
# apart (and so --list-formats reflects formats you add yourself).
sys.path.insert(0, SFX_DIR)
try:
    import nodes as _sfx
    FORMATS = list(_sfx.FORMATS.keys())
except Exception:
    _sfx = None
    FORMATS = ["none", "amiga", "sb", "sb22", "gameboy", "nes", "snes", "psx", "cd"]

# Style guides (prompt snippets) loaded from sounds.json next to this script.
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
try:
    with open(os.path.join(_SCRIPT_DIR, "sounds.json"), encoding="utf-8") as _sf:
        STYLES = {k: v for k, v in json.load(_sf).items() if not k.startswith("_")}
except Exception:
    STYLES = {}

# Named ComfyUI targets for `--server NAME`. Your personal servers.json (gitignored)
# is loaded if present; otherwise just the built-in 'local'.
try:
    with open(os.path.join(_SCRIPT_DIR, "servers.json"), encoding="utf-8") as _svf:
        SERVERS = {k: v for k, v in json.load(_svf).items() if not k.startswith("_")}
except Exception:
    SERVERS = {}
SERVERS.setdefault("local", "http://127.0.0.1:8188")

# Default negatives. The SFX one pushes *music and speech* away, because an SFX
# request drifts into a little musical phrase surprisingly often — but that same
# negative sabotages --music, which is why the two are separate. Either can be
# overridden with an explicit --negative.
SFX_NEGATIVE = ("music, melody, song, speech, voice, vocals, "
                "low quality, distorted, clipping, hiss, background noise")
MUSIC_NEGATIVE = ("sound effect, foley, speech, spoken word, silence, "
                  "low quality, distorted, clipping, hiss, muffled")


def resolve_server(value):
    """Resolve a --server value (a servers.json alias, or host[:port]/full URL) to a URL."""
    import urllib.parse
    url = SERVERS.get(value, value)
    if "://" not in url:
        url = "http://" + url
    parsed = urllib.parse.urlparse(url)
    if not parsed.port:
        url = f"{parsed.scheme}://{parsed.hostname}:8188"
    return url.rstrip("/")


def _colors():
    """ANSI color codes — auto-disabled when piped or NO_COLOR is set."""
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return {k: "" for k in ("b", "dim", "cyan", "grn", "yel", "mag", "rst")}
    return {"b": "\033[1m", "dim": "\033[2m", "cyan": "\033[36m", "grn": "\033[32m",
            "yel": "\033[33m", "mag": "\033[35m", "rst": "\033[0m"}


C = _colors()


def print_help():
    c = C

    def opt(flag, desc, default=""):
        tail = f"  {c['dim']}[{default}]{c['rst']}" if default else ""
        return f"  {c['grn']}{flag:<19}{c['rst']} {desc}{tail}"

    def ex(cmd, note):
        return f"  {c['yel']}{cmd:<48}{c['rst']}{c['dim']}{note}{c['rst']}"

    print("\n".join([
        f"{c['b']}{c['mag']}soundmon{c['rst']} — generate sound effects from a text description",
        "",
        f"{c['b']}{c['cyan']}USAGE{c['rst']}",
        f"  {c['mag']}soundmon{c['rst']} {c['yel']}\"a description\"{c['rst']} [options]",
        "",
        f"{c['b']}{c['cyan']}EXAMPLES{c['rst']}",
        ex('soundmon "a heavy wooden door creaking open"', "best quality (the default)"),
        ex('soundmon "a sword hitting metal" --style impact', "steer with a style guide"),
        ex('soundmon "footsteps on gravel" -n 4', "4 variations to pick from"),
        ex('soundmon "a lo-fi hip hop piano loop" --music', "musical loops/beds/stings"),
        ex('soundmon "laser blast" --format sb --bits 8', "SoundBlaster-era 8-bit crunch"),
        ex('soundmon --batch "door,glass,fire" -n 8', "8 of each → own folders"),
        ex('soundmon "rain" --seconds 30 --server rtx,titan', "long, fanned across GPUs"),
        "",
        f"{c['b']}{c['cyan']}OPTIONS{c['rst']}",
        opt("description", "the sound you want (in quotes)"),
        opt("-n, --number N", "how many to make, each a different seed", "1"),
        opt('--batch "a,b,c"', "round-robin subjects → a folder each (N of each)"),
        opt("--seconds N", "length in seconds (model max 47)", "10"),
        opt("--style NAMES", "append proven style guide(s) — see --list-styles"),
        opt("--music", "music mode: loops/beds/stings (no full songs or vocals)"),
        opt("--format NAME", "hardware format lock — see --list-formats", "none"),
        opt("--bits N", "WAV bit depth: 8 / 16 / 24", "16"),
        opt("--seed N", "lock / repeat a result (re-run a favorite)", "random"),
        opt("--steps N", "sampling steps (more = better, slower)", "50"),
        opt("--cfg N", "prompt adherence (higher = stricter)", "5.0"),
        opt("--fast", "16 steps: ~3x faster, rougher"),
        opt("--no-trim", "keep the model's leading/trailing silence"),
        opt("--normalize-db N", "peak level after generation", "-1.0"),
        opt("--fade-ms N", "de-click fade on both ends", "5"),
        opt("--list-formats", "show every hardware format"),
        opt("--list-styles", "show every style guide"),
        opt("-h, --help", "show this help"),
        "",
        f"{c['b']}{c['cyan']}ADVANCED{c['rst']}",
        opt("--server NAMES", "remote ComfyUI (alias/host/URL); comma-list = render farm", "local"),
        opt("--flac / --mp3 / --opus", "save compressed instead of WAV"),
        opt('--negative "..."', "negative prompt (what to avoid)"),
        opt("--name NAME", "output filename base", "from description"),
        opt("--sampler NAME", "ksampler sampler", "dpmpp_3m_sde_gpu"),
        opt("--scheduler NAME", "ksampler scheduler", "exponential"),
        opt("--threshold-db N", "silence floor for trimming", "-45"),
        opt("--base FILE", "Stable Audio checkpoint", "stable-audio-open-1.0"),
        opt("--text-encoder FILE", "T5 text encoder", "t5_base"),
        opt("--no-open", "don't auto-play the result"),
        opt("--output-to DIR", "move outputs into DIR (relative to cwd)"),
        opt("--move-to-dirs", "put a run in its own ./<description>/ folder"),
        opt("--create-dirs", "create output folders if missing"),
        opt("--no-subdirs", "with --batch/--output-to: dump all into one flat folder"),
        "",
        f"{c['b']}{c['cyan']}OUTPUT{c['rst']}",
        f"  {c['dim']}{OUTPUT}/soundmon/{c['rst']}",
        f"  {c['dim']}44.1 kHz stereo WAV, trimmed and normalized{c['rst']}",
        "",
        f"  {c['b']}{c['yel']}TIP{c['rst']}  generation is cheap — run {c['grn']}-n 8{c['rst']} and keep the one you like. "
        f"The seed is in every\n       filename, so re-run it later without {c['grn']}--fast{c['rst']} for the full-quality take.",
        "",
    ]))


def print_formats():
    c = C
    specs = getattr(_sfx, "FORMATS", {}) or {}
    print(f"{c['b']}{c['cyan']}Hardware formats{c['rst']}  {c['dim']}(use with --format NAME){c['rst']}\n")
    print(f"  {c['grn']}{'none':<10}{c['rst']} {c['dim']}keep the model's own 44.1 kHz stereo output{c['rst']}")
    for name in FORMATS:
        spec = specs.get(name)
        if not spec:
            continue
        ch = "mono" if spec["mono"] else "stereo"
        print(f"  {c['grn']}{name:<10}{c['rst']} "
              f"{c['dim']}{spec['rate']} Hz · {spec['bits']}-bit · {ch}{c['rst']}")


def print_styles():
    c = C
    print(f"{c['b']}{c['cyan']}Style guides{c['rst']}  "
          f"{c['dim']}(append with --style NAME[,NAME2] — combine freely){c['rst']}\n")
    if not STYLES:
        print("  (none — sounds.json not found)")
        return
    for name, spec in STYLES.items():
        prm = spec.get("prompt", "")
        prm = prm if len(prm) <= 58 else prm[:57] + "…"
        print(f"  {c['grn']}{name:<11}{c['rst']} {c['dim']}{prm}{c['rst']}")


def slug(text):
    out = "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")
    return out[:40] or "sound"


def build_graph(a, seed, subject=None, server=None):
    subject = subject if subject is not None else a.prompt
    parts = [subject]
    if a.style_add:
        parts.append(a.style_add)
    # The tail nudges the model toward the right *kind* of audio. "sound effect"
    # actively hurts a music request, so --music swaps it out along with the
    # negative prompt (see MUSIC_NEGATIVE).
    parts.append("high quality, clean recording"
                 if a.music else "sound effect, high quality, clean recording")
    prompt = ", ".join(parts)
    negative = a.negative + ((", " + a.style_neg) if a.style_neg else "")

    name = slug(subject) if a.batch else (a.name or slug(subject))
    prefix = f"soundmon/{name}_{a.seconds:g}s_{a.format}_s{seed}"

    # NOTE: the Stable Audio Open checkpoint bundles the DiT + VAE + the two
    # seconds-embedders, but NOT the text encoder — it has zero T5 tensors. So
    # CheckpointLoaderSimple's CLIP output (slot 1) is unusable and we load T5
    # separately via CLIPLoader with type=stable_audio. Wiring slot 1 instead is
    # the #1 way to get a confusing failure here.
    g = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": a.base}},
        "5": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": a.text_encoder, "type": "stable_audio"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["5", 0], "text": negative}},
        # Tells the model where in a clip this sits and how long it runs. Stable
        # Audio conditions on duration, so this is not cosmetic — it changes the
        # shape of what gets generated, not just how much is kept.
        "8": {"class_type": "ConditioningStableAudio",
              "inputs": {"positive": ["6", 0], "negative": ["7", 0],
                         "seconds_start": 0.0, "seconds_total": a.seconds}},
        "9": {"class_type": "EmptyLatentAudio",
              "inputs": {"seconds": a.seconds, "batch_size": 1}},
        "3": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": a.steps, "cfg": a.cfg,
                         "sampler_name": a.sampler, "scheduler": a.scheduler, "denoise": 1.0,
                         "model": ["4", 0], "positive": ["8", 0],
                         "negative": ["8", 1], "latent_image": ["9", 0]}},
        "10": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "11": {"class_type": "RetroSFX",
               "inputs": {"audio": ["10", 0], "format": a.format,
                          "trim_silence": not a.no_trim, "threshold_db": a.threshold_db,
                          "normalize_db": a.normalize_db, "fade_ms": a.fade_ms}},
    }

    if a.flac:
        g["12"] = {"class_type": "SaveAudio",
                   "inputs": {"audio": ["11", 0], "filename_prefix": prefix}}
    elif a.mp3:
        g["12"] = {"class_type": "SaveAudioMP3",
                   "inputs": {"audio": ["11", 0], "filename_prefix": prefix, "quality": "V0"}}
    elif a.opus:
        g["12"] = {"class_type": "SaveAudioOpus",
                   "inputs": {"audio": ["11", 0], "filename_prefix": prefix, "quality": "128k"}}
    else:
        g["12"] = {"class_type": "SaveSFX",
                   "inputs": {"audio": ["11", 0], "filename_prefix": prefix,
                              "bit_depth": str(a.bits)}}
    return g


def submit(graph, server=None):
    server = server or SERVER
    data = json.dumps({"prompt": graph}).encode()
    req = urllib.request.Request(f"{server}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        sys.exit("ComfyUI rejected the request:\n" + e.read().decode()[:1200])
    except urllib.error.URLError:
        sys.exit("Couldn't reach ComfyUI at " + server + " — is the server running?")


def wait(pid, server=None, timeout=1800):
    server = server or SERVER
    for _ in range(timeout):
        with urllib.request.urlopen(f"{server}/history/{pid}", timeout=30) as r:
            hist = json.loads(r.read())
        if pid in hist and hist[pid].get("outputs"):
            return hist[pid]["outputs"]
        time.sleep(1)
    sys.exit("Timed out waiting for the sound.")


def poll(pid, server):
    """One non-blocking /history check; returns the outputs dict, or None if not ready."""
    with urllib.request.urlopen(f"{server}/history/{pid}", timeout=30) as r:
        hist = json.loads(r.read())
    if pid in hist and hist[pid].get("outputs"):
        return hist[pid]["outputs"]
    return None


def server_up(server):
    try:
        urllib.request.urlopen(f"{server}/system_stats", timeout=5).read()
        return True
    except Exception:
        return False


def _short(url):
    return url.split("//", 1)[-1]


def audio_outs(outs):
    """Flatten ComfyUI's outputs dict to the list of saved audio files.
    Save nodes report under 'audio' (SaveAudio, SaveAudioMP3/Opus, and our SaveSFX)."""
    return [f for node in outs.values() for f in node.get("audio", [])]


def fetch_audio(item, dest_dir, server=None):
    """Download one server-side output file via /view into dest_dir; return local path."""
    import urllib.parse
    server = server or SERVER
    q = urllib.parse.urlencode({"filename": item["filename"],
                                "subfolder": item.get("subfolder", ""),
                                "type": item.get("type", "output")})
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, item["filename"])
    with urllib.request.urlopen(f"{server}/view?{q}", timeout=300) as r, open(out, "wb") as f:
        shutil.copyfileobj(r, f)
    return out


def run_farm(a, work):
    """Render farm: distribute jobs across POOL with dynamic dispatch (feed the free GPU).
    Faster boxes naturally pull more jobs; results are fetched back from whichever GPU made them."""
    live = [s for s in POOL if server_up(s)]
    down = [s for s in POOL if s not in live]
    if down:
        print(f"   ⚠ skipping unreachable: {', '.join(_short(s) for s in down)}")
    if not live:
        sys.exit("render farm: no reachable servers in the pool.")
    print(f"   \U0001f69c render farm: {len(live)} GPU(s) — {', '.join(_short(s) for s in live)}")
    pending = list(work)        # (subject, seed, dest)
    inflight = {}               # server -> (subject, seed, dest, pid)
    total = len(work)
    done = 0

    def launch(srv):
        """Submit the next pending job to srv. False = server unusable (drop it)."""
        while pending:
            subj, seed, d = pending.pop(0)
            try:
                pid = submit(build_graph(a, seed, subject=subj, server=srv), srv)
            except SystemExit:
                pending.insert(0, (subj, seed, d))   # couldn't submit; keep the job
                return False
            inflight[srv] = (subj, seed, d, pid)
            return True
        return True             # nothing left to do

    for srv in list(live):
        if launch(srv) is False:
            live.remove(srv)

    while inflight:
        advanced = False
        for srv, (subj, seed, d, pid) in list(inflight.items()):
            try:
                outs = poll(pid, srv)
            except Exception:
                print(f"   ⚠ {_short(srv)} unreachable — requeueing its job")
                pending.append((subj, seed, d))
                del inflight[srv]
                advanced = True
                continue
            if outs is None:
                continue
            advanced = True
            dest_dir = d or os.path.join(OUTPUT, "soundmon")
            files = [fetch_audio(it, dest_dir, srv) for it in audio_outs(outs)]
            done += 1
            sj = f"{subj}  " if a.batch else ""
            print(f"   ✅ [{done}/{total}] {_short(srv):<20} {sj}seed={seed}  ->  "
                  f"{files[0] if files else '(no file)'}")
            del inflight[srv]
            launch(srv)          # feed the now-free GPU its next job
        if not advanced:
            time.sleep(1)

    if pending:
        print(f"   ⚠ {len(pending)} job(s) left undone (all GPUs dropped).")


def main():
    p = argparse.ArgumentParser(prog="soundmon", add_help=False)
    p.add_argument("-h", "--help", action="store_true", dest="show_help")
    p.add_argument("prompt", nargs="?", help='the sound you want, e.g. "a heavy door creaking"')
    p.add_argument("-n", "--number", type=int, default=1,
                   help="how many to generate, each with a different seed. default 1 "
                        "(with --batch: how many of EACH subject)")
    p.add_argument("--batch", default=None, metavar="SUBJECTS",
                   help='comma-separated subjects to round-robin, one of each per pass; '
                        'each goes to its own folder')
    p.add_argument("--seconds", type=float, default=10.0,
                   help="length in seconds (model max 47). default 10. The model conditions "
                        "on this, so it shapes the sound — not just its length.")
    p.add_argument("--style", default="",
                   help="style guide(s) to append, comma-separated. see --list-styles")
    p.add_argument("--format", default="none",
                   help="hardware format lock (sample rate + bit depth + channels). "
                        "'none' = the model's own 44.1 kHz stereo. see --list-formats")
    p.add_argument("--bits", type=int, default=16, choices=[8, 16, 24],
                   help="WAV bit depth. default 16")
    p.add_argument("--steps", type=int, default=None, help="sampling steps (default 50, or 16 with --fast)")
    p.add_argument("--cfg", type=float, default=5.0, help="prompt adherence. default 5.0")
    p.add_argument("--fast", action="store_true", help="16 steps: ~3x faster, rougher")
    p.add_argument("--seed", type=int, default=-1, help="-1 = random each run")
    p.add_argument("--music", action="store_true",
                   help="music mode: drops the anti-music negative prompt and the "
                        "'sound effect' prompt tail. Good for loops, beds, stings and "
                        "riffs — the model does NOT do full songs or vocals.")
    p.add_argument("--negative", default=None,
                   help="what to avoid (defaults differ for SFX vs --music)")
    p.add_argument("--name", default=None, help="output filename base (default: from description)")
    # --- post-processing (the RetroSFX node) ---
    p.add_argument("--no-trim", dest="no_trim", action="store_true",
                   help="keep the model's leading/trailing silence (default: trim it)")
    p.add_argument("--threshold-db", dest="threshold_db", type=float, default=-45.0,
                   help="silence floor for trimming, in dB. default -45")
    p.add_argument("--normalize-db", dest="normalize_db", type=float, default=-1.0,
                   help="peak level after generation, in dB. default -1.0")
    p.add_argument("--fade-ms", dest="fade_ms", type=int, default=5,
                   help="de-click fade on both ends, in ms. default 5")
    # --- output format ---
    p.add_argument("--flac", action="store_true", help="save FLAC instead of WAV")
    p.add_argument("--mp3", action="store_true", help="save MP3 instead of WAV")
    p.add_argument("--opus", action="store_true", help="save Opus instead of WAV")
    # --- where finished files go ---
    p.add_argument("--output-to", dest="output_to", default=None, metavar="DIR",
                   help="move finished files into DIR (relative to your current directory)")
    p.add_argument("--move-to-dirs", dest="move_to_dirs", action="store_true",
                   help="organize each run into its own subdir named after the description")
    p.add_argument("--create-dirs", dest="create_dirs", action="store_true",
                   help="create the output dir(s) if they don't exist")
    p.add_argument("--no-subdirs", dest="no_subdirs", action="store_true",
                   help="with --batch/--output-to: dump everything flat into the one folder")
    # --- model / engine ---
    p.add_argument("--server", default=None, metavar="NAME|HOST[,...]",
                   help="render on a remote ComfyUI: a servers.json alias or host[:port]/URL. "
                        "comma-list = RENDER FARM, jobs fan across all GPUs. "
                        "default: local (also honors $SOUNDMON_SERVER)")
    p.add_argument("--base", default="stable-audio-open-1.0.safetensors",
                   help="Stable Audio checkpoint")
    p.add_argument("--text-encoder", dest="text_encoder", default="t5_base.safetensors",
                   help="T5 text encoder (models/text_encoders/)")
    p.add_argument("--sampler", default="dpmpp_3m_sde_gpu", help="ksampler sampler_name")
    p.add_argument("--scheduler", default="exponential", help="ksampler scheduler")
    p.add_argument("--list-formats", action="store_true", help="list hardware formats and exit")
    p.add_argument("--list-styles", action="store_true", help="list style guides and exit")
    p.add_argument("--no-open", action="store_true", help="don't auto-play the result")
    a = p.parse_args()

    global SERVER, REMOTE, POOL
    _target = a.server or os.environ.get("SOUNDMON_SERVER")
    if _target:
        POOL = [resolve_server(s.strip()) for s in _target.split(",") if s.strip()]
        SERVER = POOL[0]
        REMOTE = not any(h in SERVER for h in ("127.0.0.1", "localhost", "[::1]"))

    if a.show_help or (not a.prompt and not a.batch and not a.list_formats and not a.list_styles):
        print_help()
        return
    if a.list_formats:
        print_formats()
        return
    if a.list_styles:
        print_styles()
        return
    if a.format not in FORMATS:
        p.error(f"unknown format {a.format!r}. See --list-formats.")
    if a.seconds <= 0 or a.seconds > 47:
        p.error("--seconds must be between 0 and 47 (the model's trained maximum)")

    # Resolve --style guide(s) into prompt/negative additions (used by build_graph).
    a.style_add, a.style_neg = "", ""
    if a.style:
        names = [s.strip() for s in a.style.replace(",", " ").split() if s.strip()]
        adds, negs = [], []
        for nm in names:
            if nm not in STYLES:
                p.error(f"unknown style {nm!r}. See --list-styles.")
            adds.append(STYLES[nm].get("prompt", ""))
            if STYLES[nm].get("negative"):
                negs.append(STYLES[nm]["negative"])
        a.style_add = ", ".join(x for x in adds if x)
        a.style_neg = ", ".join(negs)

    a.steps = (16 if a.fast else 50) if a.steps is None else a.steps
    if a.negative is None:
        a.negative = MUSIC_NEGATIVE if a.music else SFX_NEGATIVE

    n = max(1, a.number)
    subjects = [s.strip() for s in a.batch.split(",") if s.strip()] if a.batch else [a.prompt]
    base = os.path.abspath(os.path.expanduser(a.output_to)) if a.output_to else os.getcwd()

    def dest_for(subject):
        if a.no_subdirs:
            d = base if (a.output_to or a.batch or a.move_to_dirs) else None
        elif a.batch:
            d = os.path.join(base, slug(subject))
        elif a.move_to_dirs:
            d = os.path.join(base, a.name or slug(a.prompt))
        elif a.output_to:
            d = base
        else:
            d = None
        if d is None:
            return None
        if not os.path.isdir(d):
            if a.create_dirs or a.move_to_dirs or a.batch:
                os.makedirs(d, exist_ok=True)
            else:
                p.error(f"output dir does not exist: {d}\n  (add --create-dirs to make it)")
        return d

    dests = {subj: dest_for(subj) for subj in subjects}

    total = n * len(subjects)
    # Rough seconds/clip for the ETA. Measured on an RX 6600 (ROCm, --lowvram):
    # ~6.4s/clip at 50 steps, ~4s at 16. Audio is far cheaper than SDXL — a
    # 4-second SFX is a much smaller latent than a 1024x1024 image.
    per = 4 if a.fast else 7
    style_label = f"  |  style: {a.style}" if a.style else ""
    fmt_label = a.format if a.format != "none" else "44.1kHz stereo"
    subj_label = f"{len(subjects)} subjects: {', '.join(subjects)}" if a.batch else repr(a.prompt)
    count_label = f"{n} each = {total} total" if a.batch else f"{n} clip(s)"
    if len(POOL) > 1:
        print(f"🚜 render farm: {len(POOL)} servers — {', '.join(_short(s) for s in POOL)}")
    elif REMOTE:
        print(f"🌐 rendering on remote server {SERVER} (results fetched back here)")
    print(f"🔊 {subj_label}  |  {a.seconds:g}s  |  {fmt_label}{style_label}  |  "
          f"{'FAST' if a.fast else 'quality'} {a.steps}st  |  {count_label}")
    if total > 1:
        eta = total * per
        tip = "" if a.fast else "  (tip: add --fast for quick variations)"
        print(f"   ~{eta // 60}m{eta % 60:02d}s estimated{tip}")

    # Build the work list. --batch round-robins (one of each subject per pass).
    work = []  # (subject, seed, dest)
    k = 0
    for _ in range(n):
        for subj in subjects:
            seed = (a.seed + k) if a.seed >= 0 else random.randint(0, 2**31 - 1)
            work.append((subj, seed, dests[subj]))
            k += 1

    t0 = time.time()
    first_open = None
    if len(POOL) > 1:
        run_farm(a, work)
    else:
        jobs = [(subj, seed, d, submit(build_graph(a, seed, subject=subj, server=SERVER)))
                for (subj, seed, d) in work]
        if total > 1:
            print(f"   queued {total} jobs; generating...")
        for i, (subj, seed, d, pid) in enumerate(jobs, 1):
            outs = wait(pid)
            items = audio_outs(outs)
            if REMOTE:
                dest_dir = d or os.path.join(OUTPUT, "soundmon")
                files = [fetch_audio(it, dest_dir) for it in items]
            else:
                files = [os.path.join(OUTPUT, it.get("subfolder", ""), it["filename"])
                         for it in items]
                if d:  # move finished files out of ComfyUI's output into the target folder
                    moved = []
                    for f in files:
                        if os.path.exists(f):
                            tgt = os.path.join(d, os.path.basename(f))
                            shutil.move(f, tgt)
                            moved.append(tgt)
                    files = moved
            clip = files[0] if files else None
            first_open = first_open or clip
            tag = f"[{i}/{total}] " if total > 1 else ""
            subj_note = f"{subj}  " if a.batch else ""
            print(f"   ✅ {tag}{subj_note}seed={seed}  ->  {clip}")

    where = ", ".join(sorted({str(x) for x in dests.values() if x})) or f"{OUTPUT}/soundmon/"
    print(f"   all done in {time.time() - t0:.1f}s  |  files in {where}")
    if first_open and not a.no_open and shutil.which("xdg-open"):
        try:
            subprocess.Popen(["xdg-open", first_open],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


if __name__ == "__main__":
    main()
