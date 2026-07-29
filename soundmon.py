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

# Stable Audio 3's prompt-rewriter system prompts, lifted verbatim from
# ComfyUI's official SA3 workflow (JsonExtractString node). SA3 is trained on
# richly structured prompts — named instrumentation, arrangement, and a
# "BPM: X. Length: Y seconds" tail — and these turn a loose description into
# that shape. Skipping this stage yields technically clean but musically weak
# results; it is part of the pipeline, not a nicety.
try:
    with open(os.path.join(_SCRIPT_DIR, "sa3_reprompt.json"), encoding="utf-8") as _rf:
        SA3_REPROMPT = json.load(_rf)
except Exception:
    SA3_REPROMPT = {}

# Default negatives. The SFX one pushes *music and speech* away, because an SFX
# request drifts into a little musical phrase surprisingly often — but that same
# negative sabotages --music, which is why the two are separate. Either can be
# overridden with an explicit --negative.
SFX_NEGATIVE = ("music, melody, song, speech, voice, vocals, "
                "low quality, distorted, clipping, hiss, background noise")
MUSIC_NEGATIVE = ("sound effect, foley, speech, spoken word, silence, "
                  "low quality, distorted, clipping, hiss, muffled")
# What --song pushes away. ACE-Step takes genre TAGS rather than a sentence, so
# the negative is a tag list too.
SONG_NEGATIVE = "low quality, noisy, distorted, clipping, muffled, off-key, amateur"

# Musical keys accepted by --key, mirroring ComfyUI's TextEncodeAceStepAudio1.5
# combo exactly (17 roots x 2 qualities). Order matters only for --list-keys.
KEYS = [f"{root} {quality}"
        for quality in ("major", "minor")
        for root in ("C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb",
                     "G", "G#", "Ab", "A", "A#", "Bb", "B")]


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
        ex("soundmon --narrate-file lines.txt --ogg", "speak a script with a TTS voice"),
        ex("soundmon --record-file lines.txt --ogg", "record that same script yourself"),
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
        opt("--loop", "crossfade tail over head so the track loops seamlessly"),
        opt("--chip", "real 2A03 chiptune synthesis — no model, loops perfectly"),
        opt("--chip-arp N", "arpeggio speed in 16ths (1 = classic buzz)", "1"),
        opt("--opl", "real AdLib/OPL3 FM via the Nuked core — no model"),
        opt("--opl-bank F", "external patch bank (.sbi); omit for built-in"),
        opt("--chipfx", "synthesize an 8-bit SFX (PSG) — sfxr-style, no model"),
        opt("--oplfx", "synthesize an SFX through the OPL3 FM core"),
        opt("--fx-archetype N", "force the archetype (hit/boom/coin/creak/...)"),
        opt("--blip", "JRPG text-box narration (Undertale / Animal Crossing)"),
        opt("--blip-style S", "synth = no model at all | voice = Animalese", "synth"),
        opt("--blip-wave W", "square / triangle / sine / saw / noise", "square"),
        opt("--blip-rate N", "characters per second (typing speed)", "14"),
        opt("--loop-crossfade SEC", "crossfade length for --loop", "2.0"),
        opt("--normalize-db N", "peak level after generation", "-1.0"),
        opt("--fade-ms N", "de-click fade on both ends", "5"),
        opt("--list-formats", "show every hardware format"),
        opt("--list-styles", "show every style guide"),
        opt("-h, --help", "show this help"),
        "",
        f"{c['b']}{c['cyan']}ADVANCED{c['rst']}",
        opt("--server NAMES", "remote ComfyUI (alias/host/URL); comma-list = render farm", "local"),
        opt("--lufs N", "loudness CEILING in LUFS, attenuate-only ('off')", "-16"),
        opt("--true-peak N", "true-peak ceiling in dBTP", "-1.0"),
        opt("--ogg", "compress to OGG Vorbis (~25x smaller) — off by default"),
        opt("--ogg-quality N", "OGG quality 0-10 (accuracy, not bandwidth)", "8"),
        opt("--flac / --mp3 / --opus", "save compressed instead of WAV"),
        opt('--negative "..."', "negative prompt (what to avoid)"),
        opt("--name NAME", "output filename base", "from description"),
        opt("--sampler NAME", "ksampler sampler", "dpmpp_3m_sde_gpu"),
        opt("--scheduler NAME", "ksampler scheduler", "exponential"),
        opt("--threshold-db N", "silence floor for trimming", "-45"),
        opt("--engine NAME", "sa3 (full-band, fast) / sao (2024, 16kHz cut)", "sa3"),
        opt("--sa3-size N", "small (specialist) / medium (generalist)", "small"),
        opt("--base FILE", "checkpoint override", "per --engine"),
        opt("--text-encoder FILE", "T5 text encoder", "t5_base"),
        opt("--no-open", "don't auto-play the result"),
        opt("--output-to DIR", "move outputs into DIR (relative to cwd)"),
        opt("--move-to-dirs", "put a run in its own ./<description>/ folder"),
        opt("--create-dirs", "create output folders if missing"),
        opt("--no-subdirs", "with --batch/--output-to: dump all into one flat folder"),
        "",
        f"{c['b']}{c['cyan']}SONG{c['rst']}  {c['dim']}(--song: full songs with real vocals, via ACE-Step 1.5. "
        f"needs ./download-models.sh --song){c['rst']}",
        opt("--song", "full-song mode; the description becomes genre TAGS"),
        opt('--lyrics "..."', "lyrics to sing — use [verse] / [chorus] markers"),
        opt("--lyrics-file F", "read lyrics from a file"),
        opt("--bpm N", "tempo, 10-300", "120"),
        opt('--key "A minor"', "musical key — see --list-keys", "C minor"),
        opt("--timesig N", "time signature: 2 / 3 / 4 / 6", "4"),
        opt("--lang CODE", "lyrics language (51 supported)", "en"),
        opt("--no-audio-codes", "skip the quality LLM pass — much faster"),
        opt("--llm-cfg N", "ACE text-encoder guidance", "2.0"),
        opt("--temperature N", "LLM temperature", "0.85"),
        opt("--list-keys", "show every musical key"),
        "",
        f"{c['b']}{c['cyan']}VOICE{c['rst']}  {c['dim']}(spoken narration — two ways to make the same pack: "
        f"a TTS voice, or your own){c['rst']}",
        opt("--narrate", "speak the text with a TTS voice (Kokoro, local CPU)"),
        opt("--record", "record it YOURSELF from a mic, line by line"),
        opt("--narrate-file F", "narrate every 'key | text' row of a file"),
        opt("--record-file F", "record every row instead — resumable, one take each"),
        opt("--voice NAME", "TTS voice — see --list-voices", "bm_george"),
        opt("--pitch N", "semitones, NEGATIVE = deeper (duration kept)", "0"),
        opt("--speed N", "speech rate", "1.0"),
        opt("--device NAME", "microphone to record from — see --list-devices", "default"),
        opt("--record-rate N", "capture sample rate", "48000"),
        opt("--list-voices", "show every TTS voice"),
        opt("--list-devices", "show every microphone this machine can see"),
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


def _reprompt_nodes(a, text, seconds):
    """Qwen 3.5 rewrites `text` into the form Stable Audio 3 expects.

    Returns (nodes, text_ref) where text_ref is a graph reference usable as a
    CLIPTextEncode `text` input. The theme's own wording goes INTO the LLM
    input rather than being appended afterwards — the rewriter otherwise has no
    idea what pack it is serving and will cheerfully put electric guitars in an
    orchestral cue.

    The nested sampling params must be dot-namespaced (`sampling_mode.seed`);
    `sampling_mode` is a COMFY_DYNAMICCOMBO_V3, which is only discoverable from
    /object_info.
    """
    category = "Music" if a.music else "SFX"
    sysp = SA3_REPROMPT.get(category, "")
    if not sysp:
        return {}, None
    full = (f"{sysp}\n\nInput: {text}\n"
            f"Target audio length: {max(1, int(round(seconds)))} seconds.\nOutput:")
    return {
        "20": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": a.reprompt_model, "type": "stable_diffusion"}},
        "21": {"class_type": "TextGenerate",
               "inputs": {"clip": ["20", 0], "prompt": full, "max_length": 256,
                          "sampling_mode": "on",
                          "sampling_mode.temperature": a.reprompt_temp,
                          "sampling_mode.top_k": 64, "sampling_mode.top_p": 0.95,
                          "sampling_mode.min_p": 0.05,
                          "sampling_mode.repetition_penalty": 1.05,
                          "sampling_mode.seed": 0,
                          "thinking": False, "use_default_template": True}},
    }, ["21", 0]


def _song_nodes(a, seed, subject):
    """ACE-Step 1.5 graph — full songs with vocals, lyrics, BPM and key.

    A different engine from the SFX path, but it lands its AUDIO on node "10"
    just like _sfx_nodes does, so the shared RetroSFX + save tail is identical.
    (Same trick as pixelmon's --animate: swap the pipeline, keep the CLI.)

    The all-in-one checkpoint bundles the DiT, the VAE and the Qwen text encoder
    — ComfyUI's ACEStep15 declares vae_key_prefix/text_encoder_key_prefix — so a
    single CheckpointLoaderSimple supplies all three, unlike Stable Audio which
    needs T5 loaded separately.
    """
    def encode(tags, lyrics, codes):
        return {"class_type": "TextEncodeAceStepAudio1.5",
                "inputs": {"clip": ["4", 1], "tags": tags, "lyrics": lyrics,
                           "seed": seed, "bpm": a.bpm, "duration": a.seconds,
                           "timesignature": str(a.timesig), "language": a.lang,
                           "keyscale": a.key, "generate_audio_codes": codes,
                           "cfg_scale": a.llm_cfg, "temperature": a.temperature,
                           "top_p": a.top_p, "top_k": a.top_k, "min_p": a.min_p}}

    return {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": a.song_base}},
        # generate_audio_codes runs an LLM pass that markedly improves quality but
        # is slow; it's pure waste on the negative branch, so only the positive
        # conditioning pays for it.
        "6": encode(subject, a.lyrics, not a.no_audio_codes),
        "7": encode(a.negative, "", False),
        "9": {"class_type": "EmptyAceStep1.5LatentAudio",
              "inputs": {"seconds": a.seconds, "batch_size": 1}},
        "3": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": a.steps, "cfg": a.cfg,
                         "sampler_name": a.sampler, "scheduler": a.scheduler,
                         "denoise": 1.0, "model": ["4", 0], "positive": ["6", 0],
                         "negative": ["7", 0], "latent_image": ["9", 0]}},
        "10": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    }


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
    tag = f"{a.bpm}bpm" if a.song else a.format
    prefix = f"soundmon/{name}_{a.seconds:g}s_{tag}_s{seed}"

    if a.song:
        # Different engine, same tail — _song_nodes also lands its AUDIO on "10".
        g = _song_nodes(a, seed, subject)
        g["11"] = {"class_type": "RetroSFX",
                   "inputs": {"audio": ["10", 0], "format": a.format,
                              "trim_silence": not a.no_trim, "max_seconds": a.max_seconds,
                          "threshold_db": a.threshold_db,
                              "normalize_db": a.normalize_db, "fade_ms": a.fade_ms}}
        return _attach_save(a, g, prefix)

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
                          "trim_silence": not a.no_trim, "max_seconds": a.max_seconds,
                          "threshold_db": a.threshold_db,
                          "normalize_db": a.normalize_db, "fade_ms": a.fade_ms}},
    }

    # Prompt rewriter: replace the literal positive text with the LLM's output.
    # Only for SA3 — Stable Audio Open 1.0 was not trained this way.
    if a.engine == "sa3" and not a.no_reprompt and not a.song:
        rp, ref = _reprompt_nodes(a, prompt, a.seconds)
        if ref:
            g.update(rp)
            g["6"]["inputs"]["text"] = ref

    return _attach_save(a, g, prefix)


def _attach_save(a, g, prefix):
    """Attach the save node. Shared by both engines — they agree that the
    finished AUDIO is on node "11", so everything downstream is identical."""
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


def server_has_ckpt(server, ckpt):
    """Does this box actually have `ckpt` installed?

    A mixed fleet is the normal case: the SFX models are ~5.3 GB and the song
    model another ~9.3 GB, so boxes get provisioned at different times. Without
    this check, a --song job dispatched to an SFX-only box comes back as an
    opaque ComfyUI validation error partway through a long batch. Cheaper to ask
    once at startup and route around it.
    """
    try:
        with urllib.request.urlopen(f"{server}/object_info/CheckpointLoaderSimple",
                                    timeout=10) as r:
            info = json.loads(r.read())
        node = info.get("CheckpointLoaderSimple", info)
        return ckpt in node["input"]["required"]["ckpt_name"][0]
    except Exception:
        return False


def _short(url):
    return url.split("//", 1)[-1]


def loop_wrap(path, crossfade=2.0):
    """Make a track loop seamlessly by mixing its own tail back over its head.

    WHY THIS EXISTS, and why the obvious fix was the wrong one.

    Music that plays continuously has to survive its last sample running into
    its first. Generated music does not, for a reason that has nothing to do
    with this tool: **the model composes an ending.** Asked for 60 seconds it
    writes a 60-second piece of music, with a ritardando and a decay, because
    that is what its training data does. Measured on raw output, the final two
    seconds fall ~40 dB.

    The tempting fix is to stop post-processing from touching the endpoints —
    `--no-trim --fade-ms 0` — on the theory that trimming and de-click fades are
    what flattened the tail. That was measured here and it is **backwards**:

        tail level relative to body, souls pack, same models, same prompts
          trim on,  5 ms fade   ->  -30.4 dB   (trim cuts into the model's decay)
          trim off, no fade     ->  -66.6 dB   (the model's full decay survives)

    Trimming *helped*, because near-silence trimming eats most of the composed
    fade-out. It just cannot finish the job: a decay is only "silence" at the
    very end, so trimming leaves the quiet part it never crossed the threshold
    for. No endpoint policy fixes this, because the hole is musical content, not
    processing damage.

    So: construct the loop instead of hoping for one. Given source S of length
    L and a crossfade of X, the output is T = L - X samples:

        O[i]      = S[i]                              for i in [X, T)
        O[0:X]    = S[0:X]*fade_in + S[T:L]*fade_out

    The seam is then continuous *by construction*, not by luck. Play O[T-1] into
    O[0] and you get S[T-1] into S[T] — adjacent samples of the original take.
    The composed ending is still there; it now lands underneath the opening
    instead of on top of a hard cut.

    Equal-power (sin/cos) curves rather than linear: the two halves are
    uncorrelated material, so linear curves lose ~3 dB in the middle of the
    blend and you hear a dip pass by once per loop.

    Costs X seconds of length — a 60 s render becomes a 58 s loop. Ask the model
    for the longer number if the exact duration matters.

    Returns (path, seconds_of_output) — or (path, None) if it declined to act.
    """
    try:
        import numpy as np
        import soundfile as sf
    except ImportError:
        # Same treatment --narrate gets: the dependency is real but optional, so
        # a box without it degrades to "not looped" rather than failing the run.
        return path, None

    try:
        info = sf.info(path)
        audio, sr = sf.read(path, always_2d=True)
    except Exception:
        return path, None

    n = len(audio)
    x = int(sr * crossfade)
    # Refuse rather than mangle. Below 4x the crossfade there is not enough
    # material left over for the loop to be anything but the crossfade itself.
    if x < 1 or n < 4 * x:
        return path, None

    t = np.linspace(0.0, 1.0, x, endpoint=False, dtype=audio.dtype)[:, None]
    out = audio[:n - x].copy()
    out[:x] = audio[:x] * np.sin(t * np.pi / 2) + audio[n - x:] * np.cos(t * np.pi / 2)

    tmp = os.path.splitext(path)[0] + ".loop.wav"
    try:
        sf.write(tmp, out, sr, subtype=info.subtype)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        return path, None
    return path, len(out) / sr


def loudness_normalize(path, lufs=-16.0, true_peak=-1.0):
    """Enforce a loudness CEILING and a true-peak ceiling. Attenuate-only.

    The RetroSFX node peak-normalizes, which makes every file's tallest sample
    match — but not how loud it *sounds*. Measured across a generated pack,
    peaks sat within 0.8 dB of each other while integrated loudness spanned
    12.3 dB (alarm -8.1 LUFS vs treasure -20.4 LUFS). A dense sustained sound
    at -1 dBFS peak is far louder to the ear than a sparse transient one.

    So this is the gain stage: measure EBU R128 integrated loudness, and if the
    file is louder than `lufs` (or peaks above `true_peak`), pull it down by a
    single fixed gain. Nothing is ever boosted — see the note at the gain
    calculation for why targeting a level instead of capping one is wrong here.

    True peak rather than sample peak, because lossy decode reconstructs
    inter-sample peaks above the original samples: ogg files here measured
    -0.4 dBTP from a -1.0 dBFS source, which is how you get playback clipping
    from a file that looks compliant.
    """
    if shutil.which("ffmpeg") is None:
        return path
    # loudnorm needs enough signal to integrate over. Measured on real output:
    # it reports cleanly at 2.97s and 0.60s, and returns nothing at 0.20s and
    # 0.12s. So the floor is ~0.4s, NOT the 3s an EBU R128 window suggests — a
    # 3s guard silently skipped almost the whole SFX pack, since generated
    # effects land at 2.97s. Anything shorter keeps the node's peak
    # normalization, and the JSON-parse fallback below catches stragglers.
    try:
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True).stdout.strip())
    except (ValueError, FileNotFoundError):
        return path
    if dur < 0.4:
        return path

    meas = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", path,
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    try:
        blob = json.loads(meas[meas.rindex("{"):meas.rindex("}") + 1])
        cur_i, cur_tp = float(blob["input_i"]), float(blob["input_tp"])
    except (ValueError, KeyError):
        return path

    # A CEILING, not a target. Only ever attenuate.
    #
    # Targeting a fixed loudness would mean BOOSTING sparse sounds, and they
    # can't take it: a coin-jingle measured 34.4 dB crest factor with its peak
    # already at -0.97 dBFS, so reaching -16 LUFS needs +14 dB and would put
    # the peak at +13 dBFS. The only ways to get there are clipping it or
    # compressing it flat — both destroy exactly what makes a transient read as
    # a transient. Quiet-but-punchy is a legitimate sound; too loud is not.
    #
    # So: pull down anything above the ceiling, leave everything else alone,
    # and separately guarantee no true peak exceeds the limit.
    gain = min(0.0, lufs - cur_i, true_peak - cur_tp)
    if gain > -0.1:                       # already compliant; don't re-encode
        return path

    tmp = os.path.splitext(path)[0] + ".ln.wav"
    m = f"volume={gain:.2f}dB"
    r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", path, "-af", m, tmp],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
        os.replace(tmp, path)
    elif os.path.exists(tmp):
        os.remove(tmp)
    return path


def to_ogg(path, quality=5, keep=False):
    """Transcode a finished file to OGG Vorbis. Returns the new path (or the
    original, unchanged, if conversion isn't possible).

    Done CLIENT-SIDE, on purpose. The obvious alternative — having the ComfyUI
    save node emit .ogg — would need an encoder installed on every farm box and
    a node update + restart across the fleet. Converting after the file lands
    means unmodified farm boxes keep working and the CLI alone decides the
    output format. Encoding a 60s track takes well under a second, so there is
    nothing to gain by pushing it upstream.

    ffmpeg's *native* vorbis encoder is used with `-strict -2` because this
    build ships no libvorbis, and python-soundfile segfaults writing OGG here
    (libsndfile 1.2.2, truncates then SIGSEGVs). Measured on a 60s 48kHz track:
    11 MB -> 436 KB at q=5, a 26x reduction with the full duration intact.
    """
    if shutil.which("ffmpeg") is None:
        print("   ⚠ --ogg needs ffmpeg on PATH; leaving WAV")
        return path
    out = os.path.splitext(path)[0] + ".ogg"
    # `-ac 2` is not optional: ffmpeg's native vorbis encoder is STEREO-ONLY
    # ("Current FFmpeg Vorbis encoder only supports 2 channels"), and narration
    # comes out of Kokoro as 24kHz mono. Upmixing costs almost nothing because
    # Vorbis joint-stereo codes two identical channels efficiently.
    r = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", path,
         "-c:a", "vorbis", "-strict", "-2", "-ac", "2", "-q:a", str(quality), out],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) == 0:
        print(f"   ⚠ ogg encode failed for {os.path.basename(path)}; keeping WAV")
        print("     " + r.stderr.decode().strip().splitlines()[0][:160]
              if r.stderr else "")
        if os.path.exists(out):
            os.remove(out)        # don't leave a 0-byte .ogg that shadows the WAV
        return path
    if not keep:
        os.remove(path)
    return out


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

    # Route around boxes that lack the model this run needs, rather than letting
    # them fail jobs mid-batch.
    need = a.song_base if a.song else a.base
    ready = [s for s in live if server_has_ckpt(s, need)]
    missing = [s for s in live if s not in ready]
    if missing:
        print(f"   ⚠ skipping (no {need}): {', '.join(_short(s) for s in missing)}")
        hint = "--song" if a.song else ""
        print(f"     provision them with:  ./download-models.sh {hint}".rstrip())
    if not ready:
        sys.exit(f"render farm: no server in the pool has {need}.")
    live = ready
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
            # Before loudness: the wrap MIXES two signals, so it changes level.
            # Anything that changes gain after normalizing invalidates it.
            if a.loop:
                for f in files:
                    loop_wrap(f, a.loop_crossfade)
            if a.lufs_target is not None:
                files = [loudness_normalize(f, a.lufs_target, a.true_peak) for f in files]
            if a.ogg:
                files = [to_ogg(f, a.ogg_quality, a.keep_wav) for f in files]
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
    # --- chip mode: a real PSG synthesizer. No model, no GPU, no ComfyUI. ---
    p.add_argument("--chip", action="store_true",
                   help="CHIPTUNE mode — synthesize real 2A03 chiptune (2 pulse + "
                        "triangle + noise). Authentic by construction; loops seamlessly.")
    p.add_argument("--chip-scale", dest="chip_scale", default=None,
                   choices=["minor", "harmonic", "dorian", "phrygian", "major",
                            "mixolydian", "pentatonic"],
                   help="override the scale implied by --key")
    p.add_argument("--chip-arp", dest="chip_arp", type=int, default=1, metavar="N",
                   help="arpeggio speed in 16ths — 1 is the classic buzzy chord (default: 1)")
    # --- opl mode: real AdLib / Sound Blaster FM via the Nuked OPL3 core ---
    p.add_argument("--opl", action="store_true",
                   help="ADLIB/OPL3 mode — drive a cycle-accurate Nuked OPL3 core. "
                        "Real FM synthesis, not an imitation. Build it with "
                        "./download-models.sh --opl")
    p.add_argument("--opl-bank", dest="opl_bank", default=None, metavar="FILE",
                   help="load an external patch bank (.sbi). Omit for the built-in bank")
    p.add_argument("--opl-lead", dest="opl_lead", default="brass",
                   help="instrument for the lead voice (default: brass)")
    p.add_argument("--opl-arp", dest="opl_arp", default="organ",
                   help="instrument for the arpeggio voice (default: organ)")
    p.add_argument("--opl-bass", dest="opl_bass", default="bass",
                   help="instrument for the bass voice (default: bass)")
    p.add_argument("--opl-lib", dest="opl_lib", default=None, metavar="PATH",
                   help="explicit path to libopl3.so (default: auto-detect)")
    p.add_argument("--chipfx", action="store_true",
                   help="CHIP SFX mode — synthesize an 8-bit sound effect (PSG: "
                        "pulse/triangle/LFSR noise), sfxr-style. No model.")
    p.add_argument("--oplfx", action="store_true",
                   help="OPL SFX mode — the same effect recipes rendered through "
                        "the Nuked OPL3 FM core instead of a PSG.")
    p.add_argument("--fx-archetype", dest="fx_archetype", default=None, metavar="NAME",
                   help="force the effect archetype instead of inferring it from "
                        "the name/description (e.g. hit, boom, coin, creak)")
    # --- blip mode: JRPG text-box narration, CPU, and 'synth' needs no model ---
    p.add_argument("--blip", action="store_true",
                   help="BLIP mode — JRPG text-box narration (Undertale / Animal "
                        "Crossing style). 'synth' needs no model at all.")
    p.add_argument("--blip-file", dest="blip_file", default=None, metavar="FILE",
                   help="blip every line of a file — same 'key | text' rows --narrate-file reads")
    p.add_argument("--blip-style", dest="blip_style", default="synth",
                   choices=["synth", "voice"],
                   help="synth = one oscillator blip per character (Undertale, no model); "
                        "voice = Kokoro sped up, i.e. Animalese (Animal Crossing)")
    p.add_argument("--blip-wave", dest="blip_wave", default="square",
                   choices=["square", "triangle", "sine", "saw", "noise"],
                   help="oscillator for --blip-style synth (default: square)")
    p.add_argument("--blip-rate", dest="blip_rate", type=float, default=14.0, metavar="N",
                   help="characters per second — the text-box typing speed (default: 14)")
    p.add_argument("--blip-pitch", dest="blip_pitch", type=float, default=0.0, metavar="ST",
                   help="base pitch in semitones from A4. Lower = bigger character")
    p.add_argument("--blip-jitter", dest="blip_jitter", type=float, default=1.5, metavar="ST",
                   help="per-character pitch spread. 0 = flat (authentic Undertale)")
    p.add_argument("--blip-duty", dest="blip_duty", type=float, default=0.55, metavar="F",
                   help="fraction of each character slot that sounds (default: 0.55)")
    p.add_argument("--blip-speed", dest="blip_speed", type=float, default=3.0, metavar="X",
                   help="playback speed-up for --blip-style voice (default: 3.0)")
    # --- narration mode (Kokoro TTS): spoken lines, runs on CPU, no ComfyUI ---
    p.add_argument("--narrate", action="store_true",
                   help="NARRATION mode — speak the text with a TTS voice (Kokoro). "
                        "Runs locally on CPU; no ComfyUI, no GPU, no farm.")
    p.add_argument("--narrate-file", dest="narrate_file", default=None, metavar="FILE",
                   help="narrate every line of a file, one WAV each. Understands "
                        "'key | text' rows (and '#' comments), so game string tables "
                        "work directly; the key becomes the filename.")
    p.add_argument("--voice", default="bm_george",
                   help="TTS voice (--narrate). default bm_george (British male). --list-voices")
    p.add_argument("--speed", type=float, default=1.0,
                   help="speech rate (--narrate). 1.0 = normal, 0.85 = slow and grave")
    p.add_argument("--pitch", type=float, default=0.0,
                   help="pitch shift in semitones (--narrate). NEGATIVE = deeper, "
                        "e.g. -3 for a booming dungeon master. Duration is preserved.")
    p.add_argument("--list-voices", action="store_true", help="list TTS voices and exit")
    # --- recording mode: you are the narrator; a booth around the same manifest ---
    p.add_argument("--record", action="store_true",
                   help="RECORDING mode — record narration in your own voice, one "
                        "line at a time, from a mic. Same manifest and same output "
                        "filenames as --narrate, so packs are interchangeable.")
    p.add_argument("--record-file", dest="record_file", default=None, metavar="FILE",
                   help="record every line of a file, one take each. Reads the same "
                        "'key | text' rows --narrate-file does; the key becomes the "
                        "filename. Resumable — rerun to continue where you stopped.")
    p.add_argument("--device", default=None, metavar="NAME",
                   help="microphone to record from (--record). See --list-devices. "
                        "default: the system default input")
    p.add_argument("--list-devices", dest="list_devices", action="store_true",
                   help="list microphones this machine can record from, and exit")
    p.add_argument("--record-rate", dest="record_rate", type=int, default=48000,
                   metavar="N", help="capture sample rate (--record). default 48000")
    # --- song mode (ACE-Step 1.5): full songs with vocals, lyrics, BPM, key ---
    p.add_argument("--song", action="store_true",
                   help="FULL SONG mode via ACE-Step 1.5 — real vocals and lyrics. "
                        "The description becomes the genre/style TAGS "
                        "(e.g. \"dark fantasy, orchestral, choir\"). Needs "
                        "./download-models.sh --song")
    p.add_argument("--lyrics", default="",
                   help="lyrics to sing (--song). Use [verse] / [chorus] markers; "
                        "leave empty for an instrumental")
    p.add_argument("--lyrics-file", dest="lyrics_file", default=None, metavar="FILE",
                   help="read --lyrics from a file")
    p.add_argument("--bpm", type=int, default=120, help="tempo, 10-300 (--song). default 120")
    p.add_argument("--key", default="C minor",
                   help='musical key (--song), e.g. "A minor", "F# major". --list-keys')
    p.add_argument("--timesig", default="4", choices=["2", "3", "4", "6"],
                   help="time signature (--song). default 4")
    p.add_argument("--lang", default="en", help="lyrics language code (--song). default en")
    p.add_argument("--no-audio-codes", dest="no_audio_codes", action="store_true",
                   help="skip the audio-code LLM pass (--song): much faster, lower quality")
    p.add_argument("--llm-cfg", dest="llm_cfg", type=float, default=2.0,
                   help="ACE text-encoder LLM guidance (--song). default 2.0")
    p.add_argument("--temperature", type=float, default=0.85, help="LLM temperature (--song)")
    p.add_argument("--top-p", dest="top_p", type=float, default=0.9, help="LLM top_p (--song)")
    p.add_argument("--top-k", dest="top_k", type=int, default=0, help="LLM top_k (--song)")
    p.add_argument("--min-p", dest="min_p", type=float, default=0.0, help="LLM min_p (--song)")
    p.add_argument("--song-base", dest="song_base",
                   default="ace_step_1.5_turbo_aio.safetensors",
                   help="ACE-Step checkpoint (all-in-one: DiT + VAE + Qwen encoder)")
    p.add_argument("--list-keys", action="store_true", help="list musical keys and exit")
    p.add_argument("--music", action="store_true",
                   help="music mode: drops the anti-music negative prompt and the "
                        "'sound effect' prompt tail. Good for loops, beds, stings and "
                        "riffs — the model does NOT do full songs or vocals.")
    p.add_argument("--negative", default=None,
                   help="what to avoid (defaults differ for SFX vs --music)")
    p.add_argument("--loop", action="store_true",
                   help="make the result loop seamlessly by crossfading its tail "
                        "over its head. Costs --loop-crossfade seconds of length. "
                        "For music that plays continuously — the model composes an "
                        "ENDING, which no trim/fade setting can undo.")
    p.add_argument("--loop-crossfade", dest="loop_crossfade", type=float, default=2.0,
                   metavar="SEC",
                   help="crossfade length for --loop (default: 2.0). Longer hides "
                        "a bigger composed ending but blurs more of the opening; "
                        "shorter keeps the opening crisp but can let the decay show.")
    p.add_argument("--name", default=None, help="output filename base (default: from description)")
    # --- post-processing (the RetroSFX node) ---
    p.add_argument("--no-trim", dest="no_trim", action="store_true",
                   help="keep the model's leading/trailing silence "
                        "(SFX default: trim. --song default: keep)")
    p.add_argument("--trim", action="store_true",
                   help="force silence-trimming on in --song mode (off there by default, "
                        "because trimming breaks bar alignment)")
    p.add_argument("--max-seconds", dest="max_seconds", type=float, default=0.0,
                   metavar="N",
                   help="hard length cap in seconds (0 = none). Unlike trimming, which only "
                        "removes silence, this truncates — use it for one-shot UI blips")
    p.add_argument("--threshold-db", dest="threshold_db", type=float, default=-45.0,
                   help="silence floor for trimming, in dB. default -45")
    p.add_argument("--normalize-db", dest="normalize_db", type=float, default=-1.0,
                   help="peak level after generation, in dB. default -1.0")
    p.add_argument("--fade-ms", dest="fade_ms", type=int, default=5,
                   help="de-click fade on both ends, in ms. default 5")
    # --- output format ---
    p.add_argument("--lufs", default="-16",
                   help="loudness CEILING in LUFS ('off' to disable). Anything louder is "
                        "pulled down; nothing is ever boosted. Peak normalization matches "
                        "the tallest sample, not how loud a sound seems. default -16")
    p.add_argument("--true-peak", dest="true_peak", type=float, default=-1.0,
                   metavar="N",
                   help="true-peak ceiling in dBTP (not sample peak — lossy decode "
                        "overshoots). default -1.0")
    p.add_argument("--ogg", action="store_true",
                   help="compress the finished audio to OGG Vorbis (~25x smaller). "
                        "Off by default. Done client-side with ffmpeg, so farm boxes "
                        "need nothing extra.")
    p.add_argument("--ogg-quality", dest="ogg_quality", type=int, default=8, metavar="N",
                   help="OGG Vorbis quality 0-10, higher = bigger/better. default 8. "
                        "Quality does NOT extend bandwidth — measured identical 17.3 kHz "
                        "rolloff from q=3 to q=10 — it buys accuracy BELOW the rolloff: "
                        "19.6 dB signal-to-error at q=5 vs 22.7 dB at q=8 for 23%% more size.")
    p.add_argument("--keep-wav", dest="keep_wav", action="store_true",
                   help="with --ogg, keep the original WAV alongside the .ogg")
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
    p.add_argument("--no-reprompt", dest="no_reprompt", action="store_true",
                   help="skip the Qwen prompt rewriter (--engine sa3). On by default: SA3 "
                        "is trained on structured prompts and a loose description yields "
                        "technically clean but musically weaker results.")
    p.add_argument("--reprompt-model", dest="reprompt_model",
                   default="qwen3.5_2b_bf16.safetensors",
                   help="LLM used by the prompt rewriter (models/text_encoders/)")
    p.add_argument("--reprompt-temp", dest="reprompt_temp", type=float, default=0.7,
                   help="rewriter sampling temperature. default 0.7")
    p.add_argument("--engine", default="sa3", choices=["sa3", "sao"],
                   help="which model makes SFX and --music: 'sa3' = Stable Audio 3 "
                        "(default; full-band to 22kHz, ~20x faster), 'sao' = Stable Audio "
                        "Open 1.0 (the 2024 model; its VAE hard-cuts at 16kHz)")
    p.add_argument("--sa3-size", dest="sa3_size", default="medium",
                   choices=["small", "medium"],
                   help="Stable Audio 3 variant. 'medium' (default) is the only one with an "
                        "official ComfyUI workflow and the only one verified to sound right; "
                        "the 'small' sfx/music specialists produced scrambled audio here.")
    p.add_argument("--base", default=None,
                   help="checkpoint override (else chosen by --engine)")
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

    if a.show_help or (not a.prompt and not a.batch and not a.list_formats
                       and not a.list_styles and not a.list_keys and not a.list_voices
                       and not a.narrate_file and not a.record_file
                       and not a.blip_file and not a.chip and not a.opl
                       and not a.chipfx and not a.oplfx
                       and not a.list_devices):
        print_help()
        return
    if a.list_formats:
        print_formats()
        return
    if a.list_styles:
        print_styles()
        return
    if a.list_voices:
        c = C
        sys.path.insert(0, _SCRIPT_DIR)
        import narrate
        print(f"{c['b']}{c['cyan']}Narration voices{c['rst']}  "
              f"{c['dim']}(use with --voice, --narrate) — [grade] is Kokoro's own rating{c['rst']}")
        groups = [("bm_", "British male"), ("bf_", "British female"),
                  ("am_", "American male"), ("af_", "American female")]
        for pre, label in groups:
            names = [v for v in narrate.VOICES if v.startswith(pre)]
            if not names:
                continue
            print(f"\n  {c['b']}{c['yel']}{label}{c['rst']} {c['dim']}({len(names)}){c['rst']}")
            for v in names:
                print(f"    {c['grn']}{v:<12}{c['rst']} {c['dim']}{narrate.VOICES[v]}{c['rst']}")
        print(f"\n  {c['dim']}{len(narrate.VOICES)} English voices. Kokoro also ships 26 "
              f"non-English (ja/zh/es/pt/hi/it/fr) — pass one explicitly with a matching --lang.{c['rst']}")
        print(f"\n  {c['b']}{c['yel']}TIP{c['rst']}  a booming dungeon master: "
              f"{c['grn']}--voice bm_george --pitch -3 --speed 0.9{c['rst']}")
        return
    if a.list_keys:
        c = C
        print(f"{c['b']}{c['cyan']}Musical keys{c['rst']}  {c['dim']}(use with --key, --song only){c['rst']}\n")
        for q in ("major", "minor"):
            ks = [k for k in KEYS if k.endswith(q)]
            print(f"  {c['grn']}{q:<6}{c['rst']} {c['dim']}{', '.join(k.rsplit(' ',1)[0] for k in ks)}{c['rst']}")
        return
    # --lufs is parsed HERE, not with the rest of the diffusion-side setup below.
    # The narrate/record hand-offs return before that setup ever runs, so leaving
    # the parse down there meant a.lufs_target did not exist yet and the loudness
    # cap silently did nothing on the two engines that ask for it by getattr.
    # Engine selection. The GRAPH is identical for Stable Audio 1 and 3 — same
    # CLIPLoader, ConditioningStableAudio, EmptyLatentAudio, VAEDecodeAudio —
    # so switching models is just different files and sampler settings.
    # ComfyUI detects T5-Gemma vs T5-base from the weights themselves, which is
    # why one CLIPLoader serves both.
    if not a.song:
        if a.engine == "sa3":
            if a.sa3_size == "medium":
                _ck = "stable_audio_3_medium.safetensors"   # covers sfx and music
            else:
                _ck = ("stable_audio_3_small_music.safetensors" if a.music
                       else "stable_audio_3_small_sfx.safetensors")
            a.base = a.base or _ck
            if a.text_encoder == "t5_base.safetensors":
                a.text_encoder = "t5gemma_b_b_ul2.safetensors"
            # ComfyUI's official Stable Audio 3 workflow: lcm / simple / 8 steps
            # / cfg 1. Verified by ear — 50 steps at cfg 7 produces clacking and
            # popping, and euler produces noise. These are not tunables.
            if a.sampler == "dpmpp_3m_sde_gpu":
                a.sampler = "lcm"
            if a.scheduler == "exponential":
                a.scheduler = "simple"
            if a.steps is None:
                a.steps = 8
            if a.cfg == 5.0:
                a.cfg = 1.0
        else:
            a.base = a.base or "stable-audio-open-1.0.safetensors"

    a.lufs_target = None
    if str(a.lufs).lower() not in ("off", "none", "no", ""):
        try:
            a.lufs_target = float(a.lufs)
        except ValueError:
            p.error(f"--lufs must be a number or 'off', got {a.lufs!r}")

    if a.list_devices:
        sys.path.insert(0, _SCRIPT_DIR)
        import record
        record.list_devices()
        return
    # Recording is its own pipeline too — a terminal booth around a microphone,
    # no model of any kind. It shares --narrate's manifest parser and the whole
    # output tail, so a hand-voiced pack and a generated one are interchangeable.
    if a.record or a.record_file:
        sys.path.insert(0, _SCRIPT_DIR)
        import record
        record.run(a, slug, to_ogg, loudness_normalize)
        return
    # Narration is its own pipeline (Kokoro on CPU) — it never touches ComfyUI,
    # so it short-circuits before any of the diffusion-side setup below.
    # Same shape as pixelmon handing --animate off to animate.py.
    # --blip is checked BEFORE --narrate so `--blip --narrate-file lines.txt`
    # reads the manifest in blip mode rather than being captured by narration.
    # The synth style needs no model at all, so this must also short-circuit
    # ahead of anything that would load one.
    # --opl and --chip are pure synthesis — short-circuit before anything
    # contacts ComfyUI or loads a checkpoint.
    if a.chipfx or a.oplfx:
        sys.path.insert(0, _SCRIPT_DIR)
        import chipfx
        chipfx.run(a, slug, to_ogg, loudness_normalize)
        return
    if a.opl:
        sys.path.insert(0, _SCRIPT_DIR)
        import opl
        opl.run(a, slug, to_ogg, loudness_normalize)
        return
    if a.chip:
        sys.path.insert(0, _SCRIPT_DIR)
        import chip
        chip.run(a, slug, to_ogg, loudness_normalize)
        return
    if a.blip or a.blip_file:
        sys.path.insert(0, _SCRIPT_DIR)
        import blip
        blip.run(a, slug, to_ogg, loudness_normalize)
        return
    if a.narrate or a.narrate_file:
        sys.path.insert(0, _SCRIPT_DIR)
        import narrate
        narrate.run(a, slug, to_ogg, loudness_normalize)
        return

    if a.format not in FORMATS:
        p.error(f"unknown format {a.format!r}. See --list-formats.")

    # --song is a different model with a very different length range: Stable
    # Audio tops out at 47s, ACE-Step goes to 1000s (~16 min). The default
    # differs too — 10s is right for an SFX, absurd for a song.
    if a.song:
        if a.seconds == 10.0:              # untouched SFX default -> song default
            a.seconds = 120.0
        if a.seconds <= 0 or a.seconds > 1000:
            p.error("--song --seconds must be between 0 and 1000")
        if a.key not in KEYS:
            p.error(f"unknown --key {a.key!r}. See --list-keys.")
        if not 10 <= a.bpm <= 300:
            p.error("--bpm must be between 10 and 300")
        if a.lyrics_file:
            try:
                with open(os.path.expanduser(a.lyrics_file), encoding="utf-8") as f:
                    a.lyrics = f.read()
            except OSError as e:
                p.error(f"--lyrics-file: {e}")
    else:
        # 47s is Stable Audio Open 1.0's trained maximum. Stable Audio 3 goes
        # further — verified 60s and 90s generating correctly (and in 3-4s), so
        # the old cap would needlessly block the manifest's 60s loops.
        _max = 120.0 if a.engine == "sa3" else 47.0
        if a.seconds <= 0 or a.seconds > _max:
            p.error(f"--seconds must be between 0 and {_max:g} for --engine {a.engine}")

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

    # Sampler defaults differ per engine. The ACE checkpoint we ship is the
    # *turbo* (distilled) build, which wants few steps and cfg ~1 — feeding it
    # Stable Audio's 50 steps / cfg 5 wastes minutes and oversaturates.
    if a.song:
        a.steps = (8 if a.fast else 16) if a.steps is None else a.steps
        if a.cfg == 5.0:
            a.cfg = 1.0
        if a.sampler == "dpmpp_3m_sde_gpu":
            a.sampler = "euler"
        if a.scheduler == "exponential":
            a.scheduler = "simple"
    else:
        a.steps = (16 if a.fast else 50) if a.steps is None else a.steps

    if a.negative is None:
        a.negative = (SONG_NEGATIVE if a.song
                      else MUSIC_NEGATIVE if a.music else SFX_NEGATIVE)

    # Trimming is essential for SFX (the model pads short effects with dead air)
    # but harmful for songs: --seconds 30 at --bpm 120 in 4/4 is exactly 60 bars,
    # and shaving the tail leaves you with a clip that no longer lines up to a
    # bar — useless for looping or scoring to picture. Measured: a 30s request
    # came back 22.84s. So --song keeps silence unless you ask for --trim.
    if a.song and not a.trim:
        a.no_trim = True

    # --loop dictates its own endpoint policy, because the two settings people
    # reach for here are each half-wrong on their own:
    #
    #   trim  MUST stay on  — it removes the model's silent lead-in/run-out, so
    #                         the crossfade blends music with music. Wrapping an
    #                         untrimmed take just crossfades two silences and
    #                         leaves the hole exactly where it was. (Measured:
    #                         --no-trim made the tail 36 dB WORSE, not better.)
    #   fade  MUST be off   — a de-click ramp at the endpoints lands inside the
    #                         blend region and puts a dip at the seam, which is
    #                         the one artifact the wrap exists to remove.
    #
    # The seam is a butt-join between adjacent source samples after wrapping, so
    # there is nothing left to de-click.
    if a.loop:
        a.no_trim = False
        a.fade_ms = 0

    # --chip and --opl compose in whole bars, so they already loop
    # sample-accurately. Crossfading would shorten the track to hide an ending
    # neither of them has.
    if a.chip or a.opl:
        if a.seconds == 10:                  # the SFX default; wrong for music
            a.seconds = 60
        if a.loop:
            mode = "--chip" if a.chip else "--opl"
            print(f"   ℹ --loop ignored: {mode} already loops by construction")
            a.loop = False

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
    if a.song:
        lyr = f"{len(a.lyrics.split())} words of lyrics" if a.lyrics.strip() else "instrumental"
        print(f"🎵 tags: {subj_label}  |  {a.seconds:g}s  |  {a.bpm}bpm  |  {a.key}  |  "
              f"{a.timesig}/4  |  {lyr}  |  {a.steps}st  |  {count_label}")
    else:
        eng = "SA3" if a.engine == "sa3" else "SAO1"
        print(f"🔊 {subj_label}  |  {a.seconds:g}s  |  {fmt_label}{style_label}  |  "
              f"{eng} {'FAST' if a.fast else 'quality'} {a.steps}st  |  {count_label}")
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

    # Capability check for the single-server path too. run_farm() already routes
    # around boxes that lack the model; without this, a single --server pointed at
    # a box missing the checkpoint dumped a raw ComfyUI validation blob instead of
    # saying which file was missing where.
    if len(POOL) <= 1 and (REMOTE or POOL):
        need = a.song_base if a.song else a.base
        if need and not server_has_ckpt(SERVER, need):
            sys.exit(f"{_short(SERVER)} doesn't have {need}\n"
                     f"  provision it:  ./download-models.sh"
                     f"{' --song' if a.song else (' --sa3' if a.engine == 'sa3' else '')}")

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
            # Before loudness: the wrap MIXES two signals, so it changes level.
            # Anything that changes gain after normalizing invalidates it.
            if a.loop:
                for f in files:
                    _, sec = loop_wrap(f, a.loop_crossfade)
                    if sec:
                        print(f"   ↻ looped  {os.path.basename(f)}  "
                              f"({sec:.1f}s, {a.loop_crossfade:g}s crossfade)")
            if a.lufs_target is not None:
                files = [loudness_normalize(f, a.lufs_target, a.true_peak) for f in files]
            if a.ogg:
                files = [to_ogg(f, a.ogg_quality, a.keep_wav) for f in files]
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
