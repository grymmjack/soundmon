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
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

__version__ = "0.1.0"

# soundmon can render on a REMOTE ComfyUI (e.g. a faster box on the LAN). Choose a
# target with `--server NAME` (an alias from servers.json) or `--server host[:port]`/URL,
# or the SOUNDMON_SERVER env var. When the target is remote, results are fetched back
# over HTTP (/view) — no shared filesystem needed.
#
# THE DEFAULT IS THE `local` ALIAS, NOT A HARDCODED 127.0.0.1. That literal was the
# real default before, so repointing the alias in servers.json changed `--server local`
# and left a bare `soundmon "..."` still aimed at a machine with no ComfyUI on it —
# two defaults that could disagree, which is exactly the kind of split that makes a
# config change look like it did nothing. Now there is one place to edit.
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
    # The node needs torch, which a CLI-only box (or one rendering on a remote server) may not have.
    # chip.py carries the same registry for the CPU engines (stdlib-only to import), so --list-formats
    # and validation fall back to it instead of to a hand-kept list that had already fallen behind.
    _sfx = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
        import chip as _chipreg
        _sfx = type("_Reg", (), {"FORMATS": _chipreg.FORMATS})
        FORMATS = list(_chipreg.FORMATS.keys())
    except Exception:
        FORMATS = ["none", "amiga", "sb", "sb22", "gameboy", "nes", "snes", "psx", "cd",
                   "mod8", "mod8s", "mod6", "mod6s", "crush"]

# Style guides (prompt snippets) loaded from sounds.json next to this script.
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
try:
    with open(os.path.join(_SCRIPT_DIR, "sounds.json"), encoding="utf-8") as _sf:
        STYLES = {k: v for k, v in json.load(_sf).items() if not k.startswith("_")}
except Exception:
    STYLES = {}

# Named ComfyUI targets for `--server NAME`. Your personal servers.json (gitignored)
# is loaded if present; otherwise just the built-in 'local'.
# "_default": "NAME[,...]" is the target when neither --server nor $SOUNDMON_SERVER is given. It is
# the same key pixelmon reads, and install.sh copies pixelmon's servers.json — ignoring it here meant
# a user whose pixelmon defaults to `rtx` got a soundmon that silently rendered on `local`, while
# soundmon's own GUI (which did read it) rendered on rtx. One file, one answer, in all three.
try:
    with open(os.path.join(_SCRIPT_DIR, "servers.json"), encoding="utf-8") as _svf:
        _svjson = json.load(_svf)
    SERVERS = {k: v for k, v in _svjson.items() if not k.startswith("_")}
    DEFAULT_SERVER = str(_svjson.get("_default") or "").strip()
except Exception:
    SERVERS, DEFAULT_SERVER = {}, ""
SERVERS.setdefault("local", "http://127.0.0.1:8188")

SERVER = SERVERS.get("local", "http://127.0.0.1:8188")
REMOTE = not any(_h in SERVER for _h in ("127.0.0.1", "localhost", "[::1]"))   # refined in main() by is_local()

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


def given(*flags, argv=None):
    """Was one of these flags actually typed (`--flag V` or `--flag=V`)? The ONLY honest test for
    "the user didn't choose": comparing a value to its default can't tell a typed `--seconds 10`
    from no --seconds at all, and treated the typed 10 as "pick the song default" (120 s)."""
    argv = sys.argv[1:] if argv is None else argv
    return any(x == f or x.startswith(f + "=")
               or (len(f) == 2 and x.startswith(f) and not x.startswith("--"))   # -n5
               for x in argv for f in flags)


def pick_target(cli=None, env=None, default=None):
    """Which --server value applies: --server > $SOUNDMON_SERVER > servers.json "_default" > "" (the
    `local` alias). bin/soundmon makes the same choice, so the wrapper starts ComfyUI only for the
    box this will actually talk to."""
    env = os.environ.get("SOUNDMON_SERVER") if env is None else env
    default = DEFAULT_SERVER if default is None else default
    return (cli or "").strip() or (env or "").strip() or (default or "").strip()


def resolve_server(value):
    """Resolve a --server value (a servers.json alias, or host[:port]/full URL) to a URL."""
    import urllib.parse
    url = SERVERS.get(value, value)
    if "://" not in url:
        url = "http://" + url
    parsed = urllib.parse.urlparse(url)
    if not parsed.hostname:
        raise ValueError(f"--server {value!r}: not an alias in servers.json, a host, or a URL")
    if not parsed.port:     # ComfyUI's port; keep any path (a ComfyUI behind a reverse proxy at /comfy)
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        url = f"{parsed.scheme}://{host}:8188{parsed.path}"
    return url.rstrip("/")


def _colors():
    """ANSI color codes — auto-disabled when piped or NO_COLOR is set."""
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return {k: "" for k in ("b", "dim", "cyan", "grn", "yel", "mag", "rst")}
    return {"b": "\033[1m", "dim": "\033[2m", "cyan": "\033[36m", "grn": "\033[32m",
            "yel": "\033[33m", "mag": "\033[35m", "rst": "\033[0m"}


C = _colors()


def print_help():
    """Every flag appears here (tools/selfcheck.py checks it against build_parser())."""
    c = C

    def opt(flag, desc, default=""):
        tail = f"  {c['dim']}[{default}]{c['rst']}" if default else ""
        return f"  {c['grn']}{flag:<19}{c['rst']} {desc}{tail}"

    def ex(cmd, note):
        # at least two spaces between a long command and its note ("--style impact steer with" ran together)
        return f"  {c['yel']}{cmd:<48}{c['rst']}  {c['dim']}{note}{c['rst']}"

    def head(title, note=""):
        return f"{c['b']}{c['cyan']}{title}{c['rst']}" + (f"  {c['dim']}{note}{c['rst']}" if note else "")

    print("\n".join([
        f"{c['b']}{c['mag']}soundmon{c['rst']} {__version__} — generate sound effects from a text description",
        "",
        head("USAGE"),
        f"  {c['mag']}soundmon{c['rst']} {c['yel']}\"a description\"{c['rst']} [options]",
        "",
        head("EXAMPLES"),
        ex('soundmon "a heavy wooden door creaking open"', "best quality (the default)"),
        ex('soundmon "a sword hitting metal" --style impact', "steer with a style guide"),
        ex('soundmon "footsteps on gravel" -n 4', "4 variations to pick from"),
        ex('soundmon "a lo-fi hip hop piano loop" --music', "musical loops/beds/stings"),
        ex('soundmon "laser blast" --format sb --bits 8', "SoundBlaster-era 8-bit crunch"),
        ex('soundmon --batch "door,glass,fire" -n 8', "8 of each → own folders"),
        ex("soundmon --batch-file sfx.txt --output-to pack", "one file per 'key | prompt' row"),
        ex('soundmon "rain" --seconds 30 --server rtx,titan', "long, fanned across GPUs"),
        ex("soundmon --narrate-file lines.txt --ogg", "speak a script with a TTS voice"),
        ex("soundmon --record-file lines.txt --ogg", "record that same script yourself"),
        "",
        head("OPTIONS"),
        opt("description", "the sound you want (in quotes)"),
        opt("-n, --number N", "how many to make, each a different seed", "1"),
        opt('--batch "a,b,c"', "round-robin subjects → a folder each (N of each)"),
        opt("--batch-file F", "every 'key | prompt' or 'key | seconds | prompt' row → key.wav"),
        opt("--skip-existing", "with --batch-file: skip keys that already have a file (resume)"),
        opt("--seconds N", "length in seconds (SA3 max 120, SAO max 47)", "10"),
        opt("--style NAMES", "append proven style guide(s) — see --list-styles"),
        opt("--music", "music mode: loops/beds/stings (no full songs or vocals)"),
        opt("--format NAME", "hardware format lock — see --list-formats", "none"),
        opt("--bits N", "WAV bit depth: 8 / 16 / 24", "16"),
        opt("--seed N", "lock / repeat a result (re-run a favorite)", "random"),
        opt("--steps N", "sampling steps (more = better, slower)", "8 on SA3"),
        opt("--cfg N", "prompt adherence (higher = stricter)", "1.0 on SA3"),
        opt("--fast", "fewer steps on --engine sao / --song (no effect on SA3)"),
        opt("--no-trim", "keep the model's leading/trailing silence"),
        opt("--trim", "trim silence in --song mode too (off there: it breaks bars)"),
        opt("--max-seconds N", "hard length cap — truncates, unlike trimming", "0 = none"),
        opt("--loop", "crossfade tail over head so the track loops seamlessly"),
        opt("--loop-crossfade SEC", "crossfade length for --loop", "2.0"),
        opt("--normalize-db N", "peak level after generation", "-1.0"),
        opt("--fade-ms N", "de-click fade on both ends", "5"),
        opt("--list-formats", "show every hardware format"),
        opt("--list-styles", "show every style guide"),
        opt("--version", "print the version"),
        opt("-h, --help", "show this help"),
        "",
        head("PROMPTS", "(see and control exactly what the model is sent)"),
        opt("--show-prompt", "print the final positive/negative prompt before rendering"),
        opt("--exact-prompt", "send the text as typed: no tail, default negative or Qwen rewrite"),
        opt("--no-reprompt", "skip SA3's Qwen rewrite of the description (on by default)"),
        opt("--reprompt-model F", "the rewriting LLM (models/text_encoders/)", "qwen3.5_2b_bf16"),
        opt("--reprompt-temp N", "rewriter temperature", "0.7"),
        opt('--negative "..."', "negative prompt (what to avoid)", "per mode"),
        opt("--dry-run", "print the ComfyUI graph(s) as JSON and exit — no server"),
        "",
        head("ADVANCED"),
        opt("--server NAMES", "remote ComfyUI (alias/host/URL); comma-list = render farm",
            DEFAULT_SERVER or "local"),
        opt("--lufs N", "loudness CEILING in LUFS, attenuate-only ('off')", "-16"),
        opt("--true-peak N", "true-peak ceiling in dBTP", "-1.0"),
        opt("--ogg", "compress to OGG Vorbis (~25x smaller) — off by default"),
        opt("--ogg-quality N", "OGG quality 0-10 (accuracy, not bandwidth)", "8"),
        opt("--keep-wav", "with --ogg/--flac: keep the WAV too"),
        opt("--flac / --mp3 / --opus", "save compressed instead of WAV"),
        opt("--no-recipe", "don't embed how it was made (prompt, seed, command) in the file"),
        opt("--name NAME", "output filename base", "from description"),
        opt("--sampler NAME", "ksampler sampler", "lcm on SA3"),
        opt("--scheduler NAME", "ksampler scheduler", "simple on SA3"),
        opt("--threshold-db N", "silence floor for trimming", "-45"),
        opt("--engine NAME", "sa3 (full-band, fast) / sao (2024, 16kHz cut)", "sa3"),
        opt("--sa3-size N", "medium (the only verified one) / small", "medium"),
        opt("--base FILE", "checkpoint override", "per --engine"),
        opt("--text-encoder FILE", "text encoder", "t5gemma on SA3"),
        opt("--no-open", "don't auto-play the result"),
        opt("--output-to DIR", "move outputs into DIR (relative to cwd)"),
        opt("--move-to-dirs", "put a run in its own ./<description>/ folder"),
        opt("--create-dirs", "create output folders if missing"),
        opt("--no-subdirs", "with --batch/--output-to: dump all into one flat folder"),
        "",
        head("SONG", "(--song: full songs with real vocals, via ACE-Step 1.5. needs ./download-models.sh --song)"),
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
        opt("--top-p / --top-k N", "LLM nucleus / top-k sampling", "0.9 / 0"),
        opt("--min-p N", "LLM min-p sampling", "0.0"),
        opt("--song-base FILE", "ACE-Step checkpoint (all-in-one)", "ace_step_1.5_turbo_aio"),
        opt("--list-keys", "show every musical key"),
        "",
        head("CHIP & FM", "(real chip synthesis on the CPU — no model, no server)"),
        opt("--chip", "2A03 chiptune (2 pulse + triangle + noise), loops perfectly"),
        opt("--opl", "AdLib/OPL3 FM via the Nuked core (./download-models.sh --opl)"),
        opt("--mood NAME", "heroic/ominous/eerie/... or auto — see --list-moods", "auto"),
        opt("--arp N", "arpeggio speed in 16ths, 0 = the mood chooses (alias --chip-arp)", "0"),
        opt("--scale NAME", "override the key's scale (alias --chip-scale)"),
        opt("--from-midi F", "play a MIDI file on the chip — the best input"),
        opt("--midi-start F", "skip this fraction into the MIDI first", "0.15"),
        opt("--transpose N", "shift --from-midi by N semitones", "0"),
        opt("--from-audio F", "transcribe a recording and play it on the chip"),
        opt("--chippy L", "off / some / lots / max: arps, slides, echo (--from-midi)", "off"),
        opt("--write-midi", "also write a .mid (--chip/--opl)"),
        opt("--write-mod", "also write a ProTracker .MOD (--chip)"),
        opt("--write-rad", "also write a Reality AdLib Tracker .RAD (--opl)"),
        opt("--opl-bank F", "external patch bank (.sbi); omit for built-in"),
        opt("--opl-lead / --opl-arp / --opl-bass NAME", "instrument per voice"),
        opt("", "", "brass / organ / bass"),
        opt("--opl-lib PATH", "explicit path to libopl3.so", "auto-detect"),
        opt("--chipfx", "synthesize an 8-bit SFX (PSG) — sfxr-style, no model"),
        opt("--oplfx", "synthesize an SFX through the OPL3 FM core"),
        opt("--fx-archetype N", "force the archetype (hit/boom/coin/creak/...)"),
        opt("--list-moods", "show every mood and what it changes"),
        "",
        head("BLIP", "(JRPG text-box narration — Undertale / Animal Crossing)"),
        opt("--blip", "blip the description, one blip per character"),
        opt("--blip-file F", "blip every 'key | text' row of a file"),
        opt("--blip-style S", "synth = no model at all | voice = Animalese", "synth"),
        opt("--blip-wave W", "square / triangle / sine / saw / noise", "square"),
        opt("--blip-rate N", "characters per second (typing speed)", "14"),
        opt("--blip-pitch ST", "base pitch in semitones from A4 (lower = bigger)", "0"),
        opt("--blip-jitter ST", "per-character pitch spread, 0 = flat", "1.5"),
        opt("--blip-duty F", "fraction of each character slot that sounds", "0.55"),
        opt("--blip-speed X", "speed-up for --blip-style voice", "3.0"),
        "",
        head("VOICE", "(spoken narration — two ways to make the same pack: a TTS voice, or your own)"),
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
        head("MASTERING", "(make a set of sounds sit together — no model)"),
        opt("--master F|DIR", "master a sound or a folder into --output-to (default DIR/mastered/)"),
        opt("--master-kind K", "sfx / speech / auto (speech for voice folders)", "auto"),
        opt("--master-like REF", "match tone + loudness to REF instead"),
        "",
        head("STEERING & FINDING", "(make it like a sound you have, or find ones like it)"),
        opt("--ref FILE", "start from FILE instead of silence (SFX, --music, --song)"),
        opt("--ref-strength N", "0.3 close variation … 0.9 loosely inspired", "0.6"),
        opt("--ref-match", "also take its tempo and key"),
        opt("--match-tone REF", "EQ + level each take toward REF"),
        opt("--index DIR[,DIR]", "index folders of sounds: fingerprint + tags (incremental)"),
        opt("--workers N", "with --index: files read at once", "half the cores, ≤ 8"),
        opt("--similar FILE", "list indexed sounds most like FILE (-n, --in DIR, --aspects)"),
        opt("--find WORDS", "list indexed sounds named so (--tag, --kind, --in narrow it)"),
        opt("--listen", "let an audio model hear the index: tags by ear, --describe"),
        opt("--listen-device D", "where it listens: auto, cpu, cuda, cuda:1…", "auto"),
        opt("--describe TEXT", "list sounds that sound like the words (after --listen)"),
        opt("--tags", "how the index is tagged (kick, pad, riser…; loop / one-shot)"),
        "",
        head("OUTPUT"),
        f"  {c['dim']}AI engines:  {OUTPUT}/soundmon/ (or --output-to) — 44.1 kHz stereo WAV,{c['rst']}",
        f"  {c['dim']}             trimmed and normalized{c['rst']}",
        f"  {c['dim']}CPU engines: the current folder (or --output-to) — each at its own rate and channels{c['rst']}",
        f"  {c['dim']}             (chip 44.1 kHz mono, OPL 49716 Hz, blips/narration 24 kHz; --format overrides){c['rst']}",
        f"  {c['dim']}Every file carries its recipe (prompt, seed, engine, the full command) in its comment{c['rst']}",
        f"  {c['dim']}tag — ffprobe shows it; --no-recipe leaves it out.{c['rst']}",
        "",
        f"  {c['b']}{c['yel']}TIP{c['rst']}  generation is cheap — run {c['grn']}-n 8{c['rst']} and keep the one you like. "
        f"The seed is in every\n       filename, so {c['grn']}--seed N{c['rst']} re-makes exactly that take later.",
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


def reprompts(a):
    """Does Qwen rewrite the description on the server? SA3 only — Stable Audio Open 1.0 and
    ACE-Step were not trained on that structured form — and never with --exact-prompt."""
    return (a.engine == "sa3" and not a.song and not a.no_reprompt
            and not getattr(a, "exact_prompt", False) and bool(SA3_REPROMPT))


def _song_nodes(a, seed, subject, negative=None):
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
        "7": encode(a.negative if negative is None else negative, "", False),
        "9": {"class_type": "EmptyAceStep1.5LatentAudio",
              "inputs": {"seconds": a.seconds, "batch_size": 1}},
        "3": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": a.steps, "cfg": a.cfg,
                         "sampler_name": a.sampler, "scheduler": a.scheduler,
                         "denoise": 1.0, "model": ["4", 0], "positive": ["6", 0],
                         "negative": ["7", 0], "latent_image": ["9", 0]}},
        "10": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    }


def final_prompts(a, subject):
    """(positive, negative) exactly as the graph sends them — shared by build_graph and --show-prompt,
    so what is shown can't drift from what is sent. For SA3 the positive is then rewritten by Qwen
    on the server (_reprompt_nodes) unless --no-reprompt / --exact-prompt."""
    if getattr(a, "exact_prompt", False):         # nothing added, nothing changed
        return subject, a.negative or ""
    if a.song:
        # ACE-Step takes genre TAGS: a style guide's words are added as tags (they were dropped).
        # Its NEGATIVE is not: the guides are written for Stable Audio, and several push away
        # "vocals, singing" — the one thing a song with lyrics must have.
        return ", ".join(x for x in (subject, a.style_add) if x), a.negative
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
    if getattr(a, "ref_tail", ""):
        prompt = f"{prompt}, {a.ref_tail}"
    return prompt, negative


def build_graph(a, seed, subject=None, server=None, seconds=None):
    """The ComfyUI graph for one take. `seconds` overrides --seconds for this job (a --batch-file
    row that names its own length)."""
    if seconds is not None and seconds != a.seconds:
        import copy
        a = copy.copy(a)
        a.seconds = seconds
    subject = subject if subject is not None else a.prompt
    prompt, negative = final_prompts(a, subject)

    name = slug(subject) if (a.batch or getattr(a, "batch_file", None)) else (a.name or slug(subject))
    tag = f"{a.bpm}bpm" if a.song else a.format
    steer = getattr(a, "ref_name", None)
    prefix = f"soundmon/{name}_{a.seconds:g}s_{tag}{'_ref' if steer else ''}_s{seed}"

    if a.song:
        # Different engine, same tail — _song_nodes also lands its AUDIO on "10".
        g = _song_nodes(a, seed, prompt, negative)
        if steer:
            _apply_ref(a, g)
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
    if reprompts(a):
        rp, ref = _reprompt_nodes(a, prompt, a.seconds)
        if ref:
            g.update(rp)
            g["6"]["inputs"]["text"] = ref
    if steer:
        _apply_ref(a, g)

    return _attach_save(a, g, prefix)


def _attach_save(a, g, prefix):
    """Attach the save node. Shared by both engines — they agree that the
    finished AUDIO is on node "11", so everything downstream is identical.

    ALWAYS a WAV (SaveSFX), whatever --flac/--mp3/--opus say; the CLI encodes last (encode()).
    These flags used to pick ComfyUI's SaveAudio/MP3/Opus nodes, and everything after the server
    (--loop, the loudness ceiling, --match-tone) then worked on a compressed file and wrote WAV data
    back under its .flac/.mp3/.opus name — nearly every SA3 render, since the -16 LUFS ceiling
    almost always attenuates. Processing a WAV and encoding once at the end is also one lossy
    generation instead of two, and farm boxes need no encoders."""
    g["12"] = {"class_type": "SaveSFX",
               "inputs": {"audio": ["11", 0], "filename_prefix": prefix,
                          "bit_depth": str(a.bits)}}
    return g


# --- steering with a reference sound (--ref) -----------------------------------
# pixelmon starts an image from your picture instead of noise ("keep composition");
# this is the audio version. The reference is encoded by the checkpoint's own VAE and
# sampled from at --ref-strength (KSampler denoise): 0.3 stays close, 0.9 only borrows.
REF_CACHE = os.path.expanduser("~/.cache/soundmon/refs")
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _ffprobe_seconds(path):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                           capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


def _channels(path):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels",
                            "-of", "csv=p=0", path], capture_output=True, text=True, timeout=30)
        return int(r.stdout.strip().splitlines()[0])
    except (ValueError, IndexError, OSError, subprocess.SubprocessError):
        return 0


def prepare_ref(src, seconds):
    """The reference as the model wants it: 44.1 kHz stereo WAV, exactly `seconds` long (trimmed, or
    padded with silence). Cached by content + length, so a batch uploads one file."""
    import hashlib
    if not shutil.which("ffmpeg"):
        sys.exit("--ref needs ffmpeg on PATH")
    h = hashlib.sha1()
    with open(src, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    h.update(f"{seconds:.3f}".encode())
    os.makedirs(REF_CACHE, exist_ok=True)
    out = os.path.join(REF_CACHE, f"soundmon_ref_{h.hexdigest()[:12]}.wav")
    # Reuse a cached copy only if it is whole: the name promises `seconds` of audio. A Ctrl-C during
    # the conversion used to leave a 172 s file under a 999 s name, reused for ever after.
    if os.path.exists(out):
        got = _ffprobe_seconds(out)
        if got is None or abs(got - seconds) > 0.05:
            os.remove(out)
    if not os.path.exists(out):
        tmp = f"{out}.{os.getpid()}.part.wav"       # written aside, renamed only once complete
        # pan, not -ac 2: ffmpeg's mono->stereo upmix is -3 dB, which made a mono reference quieter
        # than itself (and --ref-strength steer toward a quieter sound)
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-af",
                            f"apad=whole_dur={seconds:.3f},pan=stereo|c0=c0|c1=c0" if _channels(src) == 1
                            else f"apad=whole_dur={seconds:.3f}",
                            "-t", f"{seconds:.3f}", "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", tmp],
                           capture_output=True, text=True)
        if r.returncode:
            if os.path.exists(tmp):
                os.remove(tmp)
            sys.exit(f"--ref: couldn't read {src}: {r.stderr.strip()[-300:]}")
        os.replace(tmp, out)
    return out


def upload_ref(server, path):
    """Put the reference in a ComfyUI box's input folder (the endpoint its own audio widget uses).
    Top level, not a subfolder: LoadAudio only lists files directly in input/."""
    boundary = "----soundmon" + os.urandom(8).hex()
    name = os.path.basename(path)
    with open(path, "rb") as f:
        data = f.read()
    body = b"".join([
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
        f"Content-Type: audio/wav\r\n\r\n".encode(), data, b"\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
        f"--{boundary}--\r\n".encode()])
    req = urllib.request.Request(f"{server}/upload/image", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        got = json.loads(r.read() or b"{}")
    return got.get("name") or name


def ref_summary(path):
    """(bpm, 'D minor', seconds) of a reference, or None when it isn't musical enough to read."""
    try:
        import numpy as np
        import soundfile as sf
        sys.path.insert(0, _SCRIPT_DIR)
        import transcribe
        info = transcribe.analyze(path, np, sf)
    except Exception:
        return None
    if not info:
        return None
    return int(round(info["bpm"])), f"{NOTE_NAMES[info['root'] % 12]} {info['mode']}", info["duration"]


def _apply_ref(a, g):
    """Swap the silent starting latent (node "9") for the encoded reference."""
    g.pop("9", None)
    g["30"] = {"class_type": "LoadAudio", "inputs": {"audio": a.ref_name}}
    g["31"] = {"class_type": "VAEEncodeAudio", "inputs": {"audio": ["30", 0], "vae": ["4", 2]}}
    g["3"]["inputs"]["latent_image"] = ["31", 0]
    g["3"]["inputs"]["denoise"] = a.ref_strength
    return g


def match_tone_files(files, ref):
    """--match-tone: EQ + level each finished file toward ref (master.match_tone), in place."""
    sys.path.insert(0, _SCRIPT_DIR)
    import master
    out = []
    for f in files:
        try:
            what = master.match_tone(f, f, ref)
            print(f"   🎚 {os.path.basename(f)}: tone matched, gain {what['gain']:+.1f} dB")
        except Exception as e:
            print(f"   ⚠ couldn't match tone on {os.path.basename(f)}: {e}")
        out.append(f)
    return out


# Prompts this run has sent and not yet seen finish, per server. On Ctrl-C or a stop signal (the GUI's
# Cancel) they are taken off ComfyUI's queue, and the one rendering is interrupted only if it is ours:
# a shared ComfyUI may be busy with pixelmon or another soundmon, and those must keep going.
SENT = {}


def _forget(server, pid):
    SENT.get(server, set()).discard(pid)


def cancel_sent(why="stopped"):
    """Remove this run's prompts from every server's queue; interrupt the running one if it is ours."""
    told = []
    for server, pids in list(SENT.items()):
        if not pids:
            continue
        ids = sorted(pids)
        try:
            req = urllib.request.Request(f"{server}/queue", data=json.dumps({"delete": ids}).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=3).close()
            with urllib.request.urlopen(f"{server}/queue", timeout=3) as r:
                running = {e[1] for e in json.loads(r.read()).get("queue_running", []) if len(e) > 1}
            if running & pids:
                urllib.request.urlopen(urllib.request.Request(f"{server}/interrupt", data=b"{}", method="POST",
                                                              headers={"Content-Type": "application/json"}), timeout=3).close()
            told.append(f"{_short(server)} ({len(ids)} prompt{'s' if len(ids) > 1 else ''})")
        except Exception:
            told.append(f"{_short(server)} (unreachable; its queue may still hold {len(ids)})")
        pids.clear()
    if told:
        print(f"   ⛔ {why}: cleared {', '.join(told)}", flush=True)


def _on_stop(signum, frame):
    import signal
    signal.signal(signum, signal.SIG_IGN)             # a second Ctrl-C while cleaning up shouldn't cut it short
    cancel_sent("cancelled")
    sys.exit(130 if signum == signal.SIGINT else 143)


def _validation_text(body):
    """ComfyUI's 400 answer to /prompt as one readable line per node ("node 5 (CLIPLoader): Value not
    in list: clip_name …"), instead of 1200 characters of JSON. Falls back to the raw text."""
    try:
        j = json.loads(body)
    except ValueError:
        return body[:1200]
    lines = []
    for nid, ne in (j.get("node_errors") or {}).items():
        for er in (ne or {}).get("errors") or []:
            msg = er.get("message", "")
            det = er.get("details", "")
            lines.append(f"  node {nid} ({ne.get('class_type', '?')}): {msg}{': ' + det if det and det not in msg else ''}")
    head = (j.get("error") or {}).get("message", "") if isinstance(j.get("error"), dict) else str(j.get("error") or "")
    return "\n".join(([head] if head else []) + lines)[:2000] or body[:1200]


def submit(graph, server=None):
    server = server or SERVER
    data = json.dumps({"prompt": graph}).encode()
    req = urllib.request.Request(f"{server}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        pid = json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
        SENT.setdefault(server, set()).add(pid)
        return pid
    except urllib.error.HTTPError as e:
        why = _validation_text(e.read().decode(errors="replace"))
        # One line when it fits: the GUI keeps only the LAST line of a failed run as the job's error.
        lines = [ln.strip() for ln in why.splitlines() if ln.strip()]
        if len(lines) <= 2:
            sys.exit(f"ComfyUI on {_short(server)} rejected the graph: {' — '.join(lines)}")
        sys.exit(f"ComfyUI on {_short(server)} rejected the graph:\n{why}\n"
                 f"({len(lines) - 1} node errors on {_short(server)}; first: {lines[1]})")
    except (OSError, ValueError, KeyError):
        sys.exit("Couldn't reach ComfyUI at " + server + " — is the server running?")


class JobFailed(Exception):
    """ComfyUI finished a job without audio: a node raised, or nothing was saved.
    .kind is what classify() called it: error / poisoned / transient / cached / lost / timeout."""

    def __init__(self, msg, kind="error"):
        super().__init__(msg)
        self.kind = kind


class CachedNoOutput(JobFailed):
    """An identical, fully cached prompt that ComfyUI answered with 'success' and no files."""

    def __init__(self, msg="identical cached prompt returned no audio"):
        super().__init__(msg, "cached")


class MissingOutput(JobFailed):
    """The server says the job is done but the file it names is gone (404). ComfyUI's cache answers an
    identical prompt with the PREVIOUS run's file names — which a local run already moved away."""

    def __init__(self, msg):
        super().__init__(msg, "missing")


class ServerGone(Exception):
    """The server stopped answering for longer than SERVER_GRACE: the whole run, not one job, is over."""


def _job_error(entry):
    """The reason a /history entry failed, or None while it is still fine.

    /history only grows an "outputs" dict on success. A job that raises inside a
    node (a missing model in CLIPLoader, say) gets status_str "error" and NO
    outputs, so waiting for outputs alone waits out the full timeout: 30 minutes
    of a dead run that looks busy. Read the status too.

    "Completed with no outputs" is NOT an error here: ComfyUI answers a prompt it has already run,
    unchanged, from its cache, with success and nothing saved (see _cached_empty). Calling that a
    failure made the help's own tip ("re-run that seed later") fail, and on the farm it fed the
    circuit breaker, so re-running a favourite seed twice dropped a healthy box.
    """
    st = entry.get("status") or {}
    if st.get("status_str") == "error":
        for kind, data in st.get("messages") or []:
            if kind == "execution_error" and isinstance(data, dict):
                node = data.get("node_type") or f"node {data.get('node_id', '?')}"
                msg = " ".join(str(data.get("exception_message") or "").split())
                et = data.get("exception_type") or ""
                if et and et.split(".")[-1] not in msg:
                    msg = f"{et.split('.')[-1]}: {msg}" if msg else et
                return f"{node}: {msg}"[:600] if msg else node
        for kind, _ in st.get("messages") or []:
            if kind == "execution_interrupted":
                return "interrupted on the server (its Cancel/Interrupt, or a restart)"
        return "ComfyUI reported an error (no details)"
    return None


def _cached_empty(entry):
    st = (entry or {}).get("status") or {}
    return (st.get("status_str") == "success" and bool(st.get("completed"))
            and not audio_outs((entry or {}).get("outputs") or {}))


# Errors after which a server fails EVERY prompt until its ComfyUI is restarted (a broken model
# manager). The GUIs restart such a box when servers.json has a "_restart" command for it; the CLI
# waits for that restart and sends the job again, instead of failing it at once.
POISONED = ("object has no attribute 'model_size'", "object has no attribute 'is_dynamic'")


def _transient(err):
    """A server error worth one automatic retry: ComfyUI's model manager occasionally trips over its own
    loaded-model list (an IndexError in a loader) or runs short of VRAM while swapping models."""
    err = err or ""
    return ("Loader" in err.split(":")[0] and "IndexError" in err) or "out of memory" in err.lower()


def classify(entry):
    """What a /history entry means, as (kind, detail):
    ok (detail = outputs) · error · poisoned · transient (detail = the message) · cached · pending."""
    if entry is None:
        return "pending", None
    outs = entry.get("outputs") or {}
    if audio_outs(outs):
        return "ok", outs
    err = _job_error(entry)
    if err:
        if any(p in err for p in POISONED):
            return "poisoned", err
        return ("transient" if _transient(err) else "error"), err
    if _cached_empty(entry):
        return "cached", "the server finished but saved no audio (an identical, cached prompt)"
    return "pending", None


def fresh_copy(graph):
    """The same graph, saved under a unique subfolder. ComfyUI then re-runs only the save step (every
    node upstream is still cached, so it is instant and the audio identical) and reports the files.
    The FILE NAME does not change — the seed in it is how a take is found again."""
    import copy
    import uuid
    g = copy.deepcopy(graph)
    tag = uuid.uuid4().hex[:8]
    for node in g.values():
        pre = (node.get("inputs") or {}).get("filename_prefix")
        if isinstance(pre, str):
            head, _, tail = pre.rpartition("/")
            node["inputs"]["filename_prefix"] = f"{head + '/' if head else ''}rerun-{tag}/{tail}"
    return g


def _queue(server, timeout=10):
    with urllib.request.urlopen(f"{server}/queue", timeout=timeout) as r:
        return json.loads(r.read())


def _queued(pid, server):
    """True while the prompt is still waiting in the server's queue (not started yet)."""
    try:
        return any(len(it) > 1 and it[1] == pid for it in _queue(server).get("queue_pending", []))
    except Exception:
        return False


def _known(pid, server):
    """True while the server still has the prompt waiting or running. True when unsure (no answer),
    so a slow server never looks like one that forgot the prompt."""
    try:
        q = _queue(server)
        return any(len(it) > 1 and it[1] == pid for it in q.get("queue_running", []) + q.get("queue_pending", []))
    except Exception:
        return True


def _wait_restart(pid, server, limit=120):
    """After a POISONED error: wait (up to `limit` s) for someone (the GUI's _restart, a person) to
    restart the server, which shows as the prompt dropping out of its history. True once it has."""
    t = time.time()
    while time.time() - t < limit:
        time.sleep(3)
        try:
            with urllib.request.urlopen(f"{server}/history/{pid}", timeout=10) as r:
                if pid not in json.loads(r.read()):
                    return True
        except (OSError, ValueError):             # down: restarting
            pass
    return False


def _history(pid, server, timeout=30):
    import http.client
    try:
        with urllib.request.urlopen(f"{server}/history/{pid}", timeout=timeout) as r:
            return json.loads(r.read())
    except http.client.HTTPException as e:        # a half-answered request mid-restart: same as no answer
        raise OSError(str(e)) from e


SERVER_GRACE = 240    # seconds wait() rides out a server that stops answering (a restart takes ~10-30 s)
RESTARTS = {}         # server -> restarts wait() has seen this run (main() then re-sends the rest of the queue)


def _restarted(server):
    RESTARTS[server] = RESTARTS.get(server, 0) + 1


def wait(pid, server=None, timeout=1800, resubmit=None):
    """Wait for a prompt's audio. `timeout` counts seconds of rendering: time spent queued behind other
    jobs (pixelmon's, another soundmon's) doesn't count. Raises JobFailed for a job that cannot be
    saved, ServerGone when the server is unreachable for SERVER_GRACE seconds. With resubmit(fresh)
    — which sends the job again and returns the new prompt id — it also recovers on its own:

      server restarting        waited out (logged when it goes and comes back)
      prompt forgotten         sent again, at most twice (a restart empties ComfyUI's queue)
      POISONED error           wait for the restart, then send again
      transient error          (loader IndexError, out of memory) retried once
      identical cached re-run  re-saved once under a fresh folder (fresh=True), see fresh_copy()
    """
    server = server or SERVER
    waited, refreshed, retried = 0, False, False
    relost, lost, n, down = 2, 0, 0, None
    back = False                                  # just came back after being down: check at once

    def again(fresh):
        _forget(server, pid)
        return resubmit(fresh)

    while waited < timeout:
        try:
            hist = _history(pid, server)
        except (OSError, ValueError) as e:        # refused / reset / timed out: restarting?
            if down is None:
                down = time.time()
                print(f"   … {_short(server)} stopped answering ({getattr(e, 'reason', e)}) — waiting for it", flush=True)
            elif time.time() - down > SERVER_GRACE:
                raise ServerGone(f"ComfyUI at {_short(server)} has been unreachable for {SERVER_GRACE}s "
                                 f"({getattr(e, 'reason', e)}).")
            time.sleep(3)
            continue
        if down is not None:
            print(f"   … {_short(server)} is back after {time.time() - down:.0f}s", flush=True)
            down, lost = None, 0
            _restarted(server)
            back = True
        n += 1
        kind, info = classify(hist.get(pid))
        if kind == "pending" and (n % 5 == 0 or back):   # every ~5 s: does the server still know it?
            # right after an outage, one "unknown" is enough: the downtime already says it restarted
            lost = 0 if _known(pid, server) else (3 if back else lost + 1)
            back = False
            if lost >= 3:                         # forgotten three checks running: the server restarted
                if not (resubmit and relost):
                    _forget(server, pid)
                    raise JobFailed(f"{_short(server)} restarted and lost this job", "lost")
                print("   ↻ the server restarted and lost this job — sending it again", flush=True)
                _restarted(server)
                pid, relost, lost, waited = again(False), relost - 1, 0, 0
                continue
        if kind == "ok":
            _forget(server, pid)
            return info
        if kind == "poisoned" and resubmit and relost:
            print(f"   ↻ {_short(server)} broke and needs a ComfyUI restart — waiting for it, then sending "
                  "this again", flush=True)
            if not _wait_restart(pid, server):
                _forget(server, pid)
                raise JobFailed(f"{info} — its ComfyUI needs a restart", "poisoned")
            _restarted(server)
            pid, relost, lost, waited = again(False), relost - 1, 0, 0
            continue
        if kind == "transient" and resubmit and not retried:
            print(f"   ↻ server hiccup ({info.split(': ', 1)[-1][:60]}) — retrying once", flush=True)
            pid, retried, waited = again(False), True, 0
            continue
        if kind == "cached":
            if resubmit and not refreshed:
                print("   ↻ identical re-run answered from the server's cache without files — re-saving it", flush=True)
                pid, refreshed, waited = again(True), True, 0
                continue
            _forget(server, pid)
            raise JobFailed(info, "cached")
        if kind in ("error", "poisoned", "transient"):
            _forget(server, pid)
            raise JobFailed(info, kind)
        time.sleep(1)
        if waited % 10 or not _queued(pid, server):
            waited += 1
    _forget(server, pid)
    raise JobFailed(f"timed out after {timeout}s of rendering", "timeout")


def poll(pid, server):
    """One non-blocking /history check for the farm; returns the outputs dict, or None if not ready.
    Raises CachedNoOutput for a cached identical re-run, JobFailed (with .kind) when ComfyUI gave up,
    and OSError when the box didn't answer (the farm decides how long that may last)."""
    kind, info = classify(_history(pid, server, timeout=10).get(pid))
    if kind == "ok":
        _forget(server, pid)
        return info
    if kind == "cached":
        _forget(server, pid)
        raise CachedNoOutput()
    if kind != "pending":
        _forget(server, pid)
        raise JobFailed(info, kind)
    return None


def resend_lost(jobs, server):
    """[(job, pid, graph)] with every prompt the server no longer has (not queued, not running, not
    in its history) submitted again. Unsure (no answer) = leave it to wait()."""
    try:
        q = _queue(server)
        known = {it[1] for it in q.get("queue_running", []) + q.get("queue_pending", []) if len(it) > 1}
    except Exception:
        return jobs
    out, n = [], 0
    for job, pid, g in jobs:
        if pid not in known:
            try:
                done = pid in _history(pid, server, timeout=10)
            except (OSError, ValueError):
                return out + jobs[len(out):]
            if not done:
                _forget(server, pid)
                pid, n = submit(g, server), n + 1
        out.append((job, pid, g))
    if n:
        print(f"   ↻ sent {n} queued job(s) again — the server forgot them when it restarted", flush=True)
    return out


def server_up(server):
    try:
        urllib.request.urlopen(f"{server}/system_stats", timeout=5).read()
        return True
    except Exception:
        return False


_ARGV = {}


def server_argv(server):
    """The server's own launch arguments (ComfyUI's /system_stats reports sys.argv as system.argv),
    or None when it doesn't say (an older ComfyUI, or no answer). Asked once per server per run."""
    if server not in _ARGV:
        try:
            with urllib.request.urlopen(f"{server}/system_stats", timeout=5) as r:
                argv = (json.loads(r.read()).get("system") or {}).get("argv")
        except Exception:
            argv = None
        _ARGV[server] = argv if isinstance(argv, list) else None
    return _ARGV[server]


def fp32_warning(server):
    """A warning when an SA3 render goes to a ComfyUI that wasn't started with --force-fp32, else "".

    In fp16 Stable Audio 3 returns a WAV of the right length full of static (AGENTS.md gotcha 11),
    quickly and without any error, so nothing downstream notices. A ComfyUI started for pixelmon is
    the usual cause: whichever wrapper starts the shared server first decides its dtype. A warning,
    not a refusal — a box can run fp32 some other way, and the person may know better."""
    argv = server_argv(server)
    if argv is None or "--force-fp32" in argv:
        return ""
    return (f"   ⚠ {_short(server)}: ComfyUI was started WITHOUT --force-fp32 — Stable Audio 3 renders static in fp16.\n"
            f"     Restart it there with  ~/launch-comfyui.sh --force-fp32  (fp32 costs pixelmon VRAM; see AGENTS.md)")


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
    # Equal-power assumes the two halves are UNcorrelated. When head and tail are in phase (a drone,
    # a sustained chord) the blend reaches x1.41 and the integer write hard-clipped it. Bring the
    # whole loop back to the source's own peak instead; the loudness stage runs after this.
    src_pk, pk = float(np.abs(audio).max()), float(np.abs(out).max())
    if pk > src_pk > 0:
        out *= src_pk / pk

    root, ext = os.path.splitext(path)
    tmp = f"{root}.loop{ext}"                 # same container as the source, never WAV-in-.flac
    try:
        sf.write(tmp, out, sr, subtype=info.subtype, format=info.format)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        return path, None
    return path, len(out) / sr


def _true_peak(path):
    """True peak in dBTP via ffmpeg's ebur128 (4x oversampled), or None. Works on any length."""
    import re
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    m = re.findall(r"True peak:\s*\n\s*Peak:\s*(-?[\d.]+|-inf)", r.stderr)
    try:
        v = float(m[-1]) if m else None
    except ValueError:
        return None
    return v if v is not None and v > -200 else None


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
    # loudnorm needs enough signal to INTEGRATE loudness over. Measured on real output:
    # it reports cleanly at 2.97s and 0.60s, and returns nothing at 0.20s and
    # 0.12s. So the floor is ~0.4s, NOT the 3s an EBU R128 window suggests — a
    # 3s guard silently skipped almost the whole SFX pack, since generated
    # effects land at 2.97s. Below it the LUFS ceiling is skipped and the node's
    # peak normalization stands.
    #
    # The TRUE-PEAK ceiling is NOT skipped. This guard used to return early, which
    # dropped both: a 0.3 s chipfx blip (a square wave peak-normalized to -1 dBFS)
    # shipped at +1.1 dBTP, because a square's band-limited reconstruction
    # overshoots its samples. True peak needs no integration window.
    try:
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True).stdout.strip())
    except (ValueError, FileNotFoundError):
        return path
    short = dur < 0.4

    meas = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-i", path,
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    cur_i = cur_tp = None
    try:
        blob = json.loads(meas[meas.rindex("{"):meas.rindex("}") + 1])
        cur_i, cur_tp = float(blob["input_i"]), float(blob["input_tp"])
    except (ValueError, KeyError):
        pass
    import math
    if short or cur_i is None or not math.isfinite(cur_i):
        cur_i = None                       # nothing to integrate: no loudness ceiling for this file
    if cur_tp is None or not math.isfinite(cur_tp):
        cur_tp = _true_peak(path)          # loudnorm said nothing: ask ebur128 for the peak alone
    if cur_tp is None:
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
    gain = min(0.0, (lufs - cur_i) if cur_i is not None else 0.0, true_peak - cur_tp)
    if gain > -0.1:                       # already compliant; don't re-encode
        return path

    root, ext = os.path.splitext(path)
    tmp = f"{root}.ln{ext or '.wav'}"         # same container as the source: never WAV data under .flac
    m = f"volume={gain:.2f}dB"
    # PRESERVE THE SOURCE'S PCM WIDTH. ffmpeg defaults to 16-bit, so re-encoding
    # here silently promoted format-locked 8-bit audio back to 16-bit — throwing
    # away the honesty of the format lock and doubling the file for no benefit.
    codec = []
    try:
        import soundfile as _sf
        sub = _sf.info(path).subtype
        codec = {"PCM_U8": ["-c:a", "pcm_u8"],
                 "PCM_S8": ["-c:a", "pcm_s8"],
                 "PCM_16": ["-c:a", "pcm_s16le"],
                 "PCM_24": ["-c:a", "pcm_s24le"],
                 "PCM_32": ["-c:a", "pcm_s32le"]}.get(sub, [])
    except Exception:
        codec = []
    if ext.lower() == ".ogg":
        codec = ["-c:a", "libvorbis", "-q:a", "8"] if _has_encoder("libvorbis") else \
                ["-c:a", "vorbis", "-strict", "-2", "-ac", "2", "-q:a", "8"]
    r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", path, "-af", m] + codec + [tmp],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
        os.replace(tmp, path)
    elif os.path.exists(tmp):
        os.remove(tmp)
    return path


_ENCODERS = None


def _has_encoder(name):
    """Does this ffmpeg build have the encoder? (Asked once.)"""
    global _ENCODERS
    if _ENCODERS is None:
        try:
            _ENCODERS = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-encoders"], capture_output=True,
                                       text=True, timeout=30).stdout
        except Exception:
            _ENCODERS = ""
    return f" {name} " in _ENCODERS


def to_lossy(path, kind, keep=False):
    """MP3 or Opus from a finished WAV, client-side — for the CPU engines, which write WAV themselves
    (on the diffusion path --mp3/--opus are ComfyUI save nodes). Returns the new path, or the original
    when it can't. Same never-onto-itself rule as to_flac."""
    if shutil.which("ffmpeg") is None:
        print(f"   ⚠ --{kind} needs ffmpeg on PATH; leaving WAV")
        return path
    out = os.path.splitext(path)[0] + "." + kind
    if os.path.abspath(out) == os.path.abspath(path):
        return path
    if kind == "mp3":
        codec = ["-c:a", "libmp3lame", "-q:a", "0"] if _has_encoder("libmp3lame") else ["-b:a", "256k"]
    else:   # Opus only runs at 48 kHz; libopus resamples by itself, the native encoder needs -ar
        codec = (["-c:a", "libopus", "-b:a", "128k"] if _has_encoder("libopus")
                 else ["-c:a", "opus", "-strict", "-2", "-ar", "48000", "-b:a", "128k"])
    r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", path] + codec + [out],
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) == 0:
        print(f"   ⚠ {kind} encode failed for {os.path.basename(path)}; keeping WAV")
        if os.path.exists(out):
            os.remove(out)
        return path
    if not keep:
        os.remove(path)
    return out


def to_flac(path, keep=False):
    """Transcode to FLAC losslessly, client-side. Returns the new path.

    Every engine's FLAC comes from here: the CPU engines write WAV themselves, and the
    diffusion path now always saves WAV on the server and encodes last (_attach_save).

    FLAC is the right container for format-locked audio. Measured on a 6-bit
    11 kHz render, distinct amplitude levels surviving the encode:

        WAV source   57      OGG q=8   96775      FLAC   57

    A bit-crushed signal is maximally hostile to lossy compression: the
    quantization steps ARE broadband high-frequency content, which is exactly what
    a psychoacoustic model is built to discard. So the crush cannot survive Vorbis
    or MP3 at any bitrate — it is not a quality setting, it is the codec working
    as designed. FLAC keeps it bit-for-bit, keeps mono mono, and still shrinks the
    file.
    """
    if shutil.which("ffmpeg") is None:
        return path
    out = os.path.splitext(path)[0] + ".flac"
    # NEVER transcode a file onto itself. ffmpeg refuses, and the cleanup at the
    # bottom of this function then removes `path` — which is the same file. That
    # deleted every ComfyUI-saved .flac after a successful download.
    if os.path.abspath(out) == os.path.abspath(path):
        return path
    r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", path, "-c:a", "flac", "-compression_level", "8", out],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) == 0:
        if os.path.exists(out):
            os.remove(out)
        return path
    if not keep:
        os.remove(path)
    return out


def to_ogg(path, quality=5, keep=False):
    """Transcode a finished file to OGG Vorbis. Returns the new path (or the
    original, unchanged, if conversion isn't possible).

    Done CLIENT-SIDE, on purpose. The obvious alternative — having the ComfyUI
    save node emit .ogg — would need an encoder installed on every farm box and
    a node update + restart across the fleet. Converting after the file lands
    means unmodified farm boxes keep working and the CLI alone decides the
    output format. Encoding a 60s track takes well under a second, so there is
    nothing to gain by pushing it upstream.

    libvorbis when this ffmpeg has it; otherwise ffmpeg's *native* vorbis encoder
    with `-strict -2` (the dev box's build ships no libvorbis). python-soundfile
    segfaults writing OGG here (libsndfile 1.2.2, truncates then SIGSEGVs). Measured on a 60s 48kHz track:
    11 MB -> 436 KB at q=5, a 26x reduction with the full duration intact.
    """
    if shutil.which("ffmpeg") is None:
        print("   ⚠ --ogg needs ffmpeg on PATH; leaving WAV")
        return path
    out = os.path.splitext(path)[0] + ".ogg"
    if os.path.abspath(out) == os.path.abspath(path):
        return path            # same trap as to_flac: would delete its own input
    if _has_encoder("libvorbis"):
        # The real encoder when this ffmpeg has it: any channel count, and every quality works.
        codec = ["-c:a", "libvorbis", "-q:a", str(quality)]
    else:
        # ffmpeg's native encoder is STEREO-ONLY ("Current FFmpeg Vorbis encoder only supports
        # 2 channels"), and narration and the chip engines are mono. `-ac 2` upmixed with a -3 dB pan
        # law, so every mono OGG came out 3 dB quieter than its WAV; duplicating the channel keeps the
        # level. Vorbis joint-stereo codes two identical channels almost for free.
        # It also writes SILENCE at -q:a 0 (measured -91 dB mean on a chip track), so 1 is its floor.
        try:
            import soundfile as _sf
            mono = _sf.info(path).channels == 1
        except Exception:
            mono = False
        codec = (["-af", "pan=stereo|c0=c0|c1=c0"] if mono else ["-ac", "2"]) + \
                ["-c:a", "vorbis", "-strict", "-2", "-q:a", str(max(1, quality))]
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", path] + codec + [out],
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


def encode(path, a):
    """Encode one finished WAV to the requested delivery format. Returns the path.

    ONE place, because this decision was previously made in seven, and --flac was
    wired into only three of them. --chip, --opl and --chipfx honoured it; the
    diffusion path, --blip, --narrate and --record each had their own
    `if a.ogg: to_ogg(...)` and no flac branch at all. Worse than a missing
    feature: blip.py and narrate.py both ACCEPT a `to_flac` callable and never
    call it, so the flag was accepted, plumbed, and silently dropped.

    That is the same failure mode as "two renderers drifting apart because a fix
    landed in only one", which is why selfcheck.py exists. So: one function, and
    an assertion per engine that it is reached.

    FLAC wins when both are asked for. It is lossless, and for the bit-crushed
    output these engines produce a lossy codec destroys the very thing that makes
    it sound like hardware.

    IDEMPOTENT, and that is load-bearing rather than tidy. When the diffusion path
    still had --flac attach a ComfyUI SaveAudio node, the server already returned a
    .flac; re-encoding it resolved the output to the SAME path, ffmpeg refused to
    write it, and the `if not keep: os.remove(path)` at the end deleted the file.
    Every asset downloaded fine and was then destroyed, reported as "nothing
    produced". So: if the file is already in the requested format, leave it alone.
    """
    want = next((k for k in ("flac", "ogg", "mp3", "opus") if getattr(a, k, False)), None)
    if want is None or path.lower().endswith("." + want):
        return path
    if want == "flac":
        return to_flac(path, getattr(a, "keep_wav", False))
    if want == "ogg":
        return to_ogg(path, getattr(a, "ogg_quality", 5),
                      getattr(a, "keep_wav", False))
    return to_lossy(path, want, getattr(a, "keep_wav", False))


def encode_all(paths, a):
    """encode() over a list, for the diffusion path's batches."""
    return [encode(p, a) for p in paths]


def audio_outs(outs):
    """Flatten ComfyUI's outputs dict to the list of saved audio files.
    Save nodes report under 'audio' (SaveAudio, SaveAudioMP3/Opus, and our SaveSFX)."""
    return [f for node in outs.values() for f in node.get("audio", [])]


def unique_path(path):
    """`path`, or path with _2, _3 … when a file is already there. The same seed re-rendered into the
    same folder (another box, another run, other flags) used to overwrite the earlier take silently."""
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    sep = "" if root.endswith("_") else "_"      # ComfyUI names end in "_": x_00001_2.wav, not x_00001__2.wav
    k = 2
    while os.path.exists(f"{root}{sep}{k}{ext}"):
        k += 1
    return f"{root}{sep}{k}{ext}"


def fetch_audio(item, dest_dir, server=None, tries=3):
    """Download one server-side output file via /view into dest_dir; return the local path.
    Written to a .part file and renamed when complete (a Ctrl-C mid-download left half a WAV under
    the real name), checked against Content-Length (a cut connection was accepted as a 5 s file for
    a 10 s render), and retried a couple of times. A 404 is final."""
    import http.client
    import urllib.parse
    server = server or SERVER
    q = urllib.parse.urlencode({"filename": item["filename"],
                                "subfolder": item.get("subfolder", ""),
                                "type": item.get("type", "output")})
    os.makedirs(dest_dir, exist_ok=True)
    out = unique_path(os.path.join(dest_dir, os.path.basename(item["filename"])))
    tmp = out + ".part"
    last = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(f"{server}/view?{q}", timeout=300) as r, open(tmp, "wb") as f:
                want = r.headers.get("Content-Length")
                shutil.copyfileobj(r, f)
            got = os.path.getsize(tmp)
            if want is not None and int(want) != got:
                raise OSError(f"short download: {got} of {want} bytes")
            if got == 0:
                raise OSError("empty download")
            os.replace(tmp, out)
            return out
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 404:
                break
        except (OSError, http.client.HTTPException, ValueError) as e:
            last = e
        time.sleep(1 + attempt)
    if os.path.exists(tmp):
        os.remove(tmp)
    why = f"couldn't download {item['filename']} from {_short(server)}: {getattr(last, 'reason', last)}"
    if isinstance(last, urllib.error.HTTPError) and last.code == 404:
        raise MissingOutput(why)
    raise OSError(why)


def is_local(url):
    """Is this server THIS machine? By hostname, not substring: 127.0.0.12 is not 127.0.0.1."""
    import urllib.parse
    try:
        return urllib.parse.urlparse(url).hostname in ("127.0.0.1", "localhost", "::1")
    except ValueError:
        return False


def collect(items, dest, server):
    """The saved files of a finished job, on this machine. A local server's files are MOVED out of
    ComfyUI's output folder (a farm that included the local box used to leave copies behind); when
    they aren't where expected — a local ComfyUI whose output folder isn't ~/ComfyUI/output — they
    are downloaded instead. Remote servers: /view. Never an empty success."""
    files = []
    for it in items:
        local = os.path.join(OUTPUT, it.get("subfolder", ""), it["filename"]) if is_local(server) else None
        if local and os.path.isfile(local):
            if dest:
                tgt = unique_path(os.path.join(dest, os.path.basename(local)))
                shutil.move(local, tgt)
                files.append(tgt)
            else:
                files.append(local)
        else:
            files.append(fetch_audio(it, dest or os.path.join(OUTPUT, "soundmon"), server))
    if not files:
        raise JobFailed("the server reported success but no audio file")
    return files


# --- provenance: every file says how it was made -------------------------------
# The seed in a file name re-runs a take only if you also remember the prompt, the style, the engine
# and the flags. The GUI keeps a job.json; the CLI kept nothing, so a good take found weeks later in a
# game folder could not be made again. The recipe goes INSIDE the file instead of beside it, so it
# survives being copied, renamed or shipped: a RIFF LIST/INFO ICMT comment in WAV, a comment tag
# (Vorbis comment / ID3) in FLAC, OGG, Opus and MP3. The audio data is not touched. --no-recipe
# leaves it out (shipped game assets that shouldn't carry prompts or local paths).
def command_line():
    """The command as typed, ready to paste back (the wrapper's name, not soundmon.py)."""
    return shlex.join(["soundmon"] + sys.argv[1:])


def recipe(a, subject=None, seed=None, server=None, seconds=None, engine=None):
    """How a take was made, as a dict (stored as compact JSON). Diffusion takes get the full recipe;
    the CPU engines (engine=NAME) get their engine and the command, which reproduces them exactly."""
    r = {"tool": "soundmon", "version": __version__}
    if engine:
        r["engine"] = engine
        if subject:
            r["prompt"] = subject
    else:
        song = bool(getattr(a, "song", False))
        r["engine"] = "song" if song else a.engine
        r["model"] = a.song_base if song else a.base
        r["prompt"] = subject
        if song:
            r.update(bpm=a.bpm, key=a.key, timesig=a.timesig, lang=a.lang,
                     lyrics=(a.lyrics or "")[:400], negative=a.negative)
        else:
            pos, neg = final_prompts(a, subject)
            r.update(positive=pos, negative=neg)
            if reprompts(a):
                r["reprompt"] = f"rewritten on the server by {a.reprompt_model} (temp {a.reprompt_temp:g})"
            if a.style:
                r["style"] = a.style
        r.update(seed=seed, seconds=seconds if seconds is not None else a.seconds, steps=a.steps, cfg=a.cfg,
                 sampler=a.sampler, scheduler=a.scheduler, format=a.format, bits=a.bits)
        if getattr(a, "ref", None):
            r.update(ref=os.path.basename(a.ref), ref_strength=a.ref_strength, ref_match=bool(a.ref_match))
        if getattr(a, "match_tone", None):
            r["match_tone"] = os.path.basename(a.match_tone)
        if server:
            r["server"] = _short(server)
    r["command"] = command_line()
    return r


def _tag_wav(path, text, software):
    """Put (or replace) a LIST/INFO chunk with ICMT + ISFT at the end of a WAV. Pure byte surgery:
    every other chunk, the fmt header and the samples are copied unchanged. Not done with ffmpeg on
    purpose — remuxing rewrites the header (24-bit becomes WAVE_FORMAT_EXTENSIBLE, which some old
    tools and game engines refuse)."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return False
    pos, kept = 12, []
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        end = pos + 8 + size
        if end > len(data):                       # a streamed/unfinished header: don't touch it
            return False
        if not (cid == b"LIST" and data[pos + 8:pos + 12] == b"INFO"):
            kept.append(data[pos:end] + (b"\0" if size & 1 else b""))
        pos = end + (size & 1)

    def sub(cid, value):
        b = value.encode("utf-8") + b"\0"
        return cid + len(b).to_bytes(4, "little") + b + (b"\0" if len(b) & 1 else b"")
    info = b"INFO" + sub(b"ICMT", text) + sub(b"ISFT", software)
    body = b"WAVE" + b"".join(kept) + b"LIST" + len(info).to_bytes(4, "little") + info
    tmp = path + ".tag"
    with open(tmp, "wb") as f:
        f.write(b"RIFF" + len(body).to_bytes(4, "little") + body)
    os.replace(tmp, path)
    return True


def tag_file(path, rec):
    """Embed `rec` (the recipe) in a finished file. Never fails the run: a file that can't be tagged
    is left exactly as it was."""
    text = json.dumps(rec, ensure_ascii=True, separators=(",", ":"))
    software = f"soundmon {__version__}"
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".wav":
            return _tag_wav(path, text, software)
        if ext in (".flac", ".ogg", ".opus", ".mp3") and shutil.which("ffmpeg"):
            root, _ = os.path.splitext(path)
            tmp = f"{root}.tagging{ext}"
            r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path, "-map", "0", "-c", "copy",
                                "-map_metadata", "0", "-metadata", f"comment={text}",
                                "-metadata", f"encoder={software}", tmp],
                               capture_output=True, text=True)
            if r.returncode == 0 and os.path.getsize(tmp) > 0:
                os.replace(tmp, path)
                return True
            if os.path.exists(tmp):
                os.remove(tmp)
    except Exception:
        pass
    return False


def read_recipe(path):
    """The recipe a soundmon file carries, or None (selfcheck, and anyone scripting re-runs)."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json", path],
                           capture_output=True, text=True, timeout=30)
        j = json.loads(r.stdout or "{}")
        # WAV/FLAC/MP3 keep it as a format tag; OGG and Opus as a tag of the audio stream
        tags = [j.get("format", {}).get("tags", {})] + [st.get("tags", {}) for st in j.get("streams", [])]
        c = next((v for t in tags for k, v in t.items() if k.lower() == "comment"), None)
        return json.loads(c) if c else None
    except Exception:
        return None


class WriteWatch:
    """Which files a CPU engine wrote. The engines (chip, opl, chipfx, blip, narrate) own their file
    writes and all go through soundfile.write, so rather than threading a recorder through five
    modules, soundmon watches that one call while the engine runs, then tags whatever survived
    (the WAV, or the .ogg/.flac it was encoded to)."""

    def __init__(self):
        self.paths = []

    def __enter__(self):
        try:
            import soundfile as sf
        except Exception:
            self.sf = None
            return self
        self.sf, self.orig = sf, sf.write

        def write(file, *args, **kw):
            out = self.orig(file, *args, **kw)
            if isinstance(file, (str, os.PathLike)):
                self.paths.append(os.path.abspath(os.fspath(file)))
            return out
        sf.write = write
        return self

    def __exit__(self, *exc):
        if self.sf is not None:
            self.sf.write = self.orig
        return False

    def finals(self):
        out = []
        for p in dict.fromkeys(self.paths):
            for cand in [p] + [os.path.splitext(p)[0] + e for e in (".flac", ".ogg", ".mp3", ".opus")]:
                if os.path.isfile(cand) and cand not in out:
                    out.append(cand)
        return out


def run_engine(a, name, module):
    """Hand off to a CPU engine and tag what it made (unless --no-recipe)."""
    sys.path.insert(0, _SCRIPT_DIR)
    import importlib
    mod = importlib.import_module(module)
    with WriteWatch() as ww:
        made = mod.run(a, slug, to_ogg, loudness_normalize, to_flac)
    # --mp3 / --opus: the engines only know FLAC and OGG, so they hand back a WAV; convert it here,
    # in one place, rather than teaching five engines two more codecs.
    if (a.mp3 or a.opus) and not (a.flac or a.ogg):
        kind = "mp3" if a.mp3 else "opus"
        for f in [f for f in ww.finals() if f.lower().endswith(".wav")]:
            out = to_lossy(f, kind, a.keep_wav)
            if out != f:
                ww.paths.append(out)
                print(f"   ↪ {os.path.basename(out)}")
    if not getattr(a, "no_recipe", False):
        subj = a.prompt or a.from_midi or a.from_audio or getattr(a, "narrate_file", None) or getattr(a, "blip_file", None)
        rec = recipe(a, subject=subj, engine=name)
        for f in ww.finals():
            tag_file(f, rec)
    return made


# --- --batch-file: one file per manifest row, named by its key -------------------
from collections import namedtuple  # noqa: E402

# One take to render. `name` is the final file name without extension (a --batch-file key) or None
# (keep ComfyUI's <subject>_<len>_<fmt>_s<seed>_00001_ name); `seconds` overrides --seconds or None.
Job = namedtuple("Job", "subj seed dest name seconds")
AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".opus")


def parse_batch_file(path, slugify=None):
    """Read a --batch-file manifest into [(key, seconds|None, prompt)].

        key | prompt                  one sound, saved as key.wav
        key | seconds | prompt        with its own length
        a plain line                  named after itself
        # comment, blank lines        skipped

    The same `key | text` shape --narrate-file and --blip-file read, so one game string table can
    drive narration, blips and sound effects. Unlike --batch, prompts may contain commas — that is
    the point: a real description ("a goblin, red cloak, laughing") can't go in a comma list."""
    import re
    slugify = slugify or slug
    out, seen = [], {}
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            secs = None
            if "|" in line:
                parts = [x.strip() for x in line.split("|")]
                if len(parts) > 3:
                    raise ValueError(f"line {ln}: expected 'key | prompt' or 'key | seconds | prompt', "
                                     f"got {len(parts)} fields")
                key, text = parts[0], parts[-1]
                if len(parts) == 3:
                    try:
                        secs = float(parts[1].rstrip("s"))
                    except ValueError:
                        raise ValueError(f"line {ln}: the middle field must be seconds, got {parts[1]!r}") from None
            else:
                key, text = "", line
            if not text:
                raise ValueError(f"line {ln}: no prompt")
            key = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("._") or slugify(text)
            if key in seen:
                raise ValueError(f"line {ln}: key {key!r} is already used on line {seen[key]}")
            seen[key] = ln
            out.append((key, secs, text))
    return out


def existing_output(dest, name):
    """The file a --batch-file key already has in dest (any audio format), or None."""
    for ext in AUDIO_EXTS:
        p = os.path.join(dest, name + ext)
        if os.path.isfile(p):
            return p
    return None


def finish_files(a, files, job, server=None):
    """Everything after the server: loop, loudness, tone match, encode, the --batch-file name, the
    recipe. ONE place for the single-server path and the farm, which had drifted (only one of them
    reported loops)."""
    # Order: tone match (EQ the take) -> loop (the wrap MIXES two signals, so it changes level) ->
    # the loudness/true-peak ceiling LAST before encoding: anything that changes gain after it
    # invalidates it, and --match-tone used to run after it and could lift a take over the ceiling.
    # Everything here works on the WAV the server saved; encode() makes the FLAC/OGG/MP3/Opus last.
    if getattr(a, "match_tone", None):
        files = match_tone_files(files, a.match_tone)
    if a.loop:
        for f in files:
            _, sec = loop_wrap(f, a.loop_crossfade)
            if sec:
                print(f"   ↻ looped  {os.path.basename(f)}  ({sec:.1f}s, {a.loop_crossfade:g}s crossfade)")
            else:
                print(f"   ⚠ not looped: {os.path.basename(f)} is shorter than 4x --loop-crossfade "
                      f"({a.loop_crossfade:g}s), or unreadable")
    if a.lufs_target is not None:
        files = [loudness_normalize(f, a.lufs_target, a.true_peak) for f in files]
    files = encode_all(files, a)
    if job.name:
        named = []
        for i, f in enumerate(files):
            stem = job.name if i == 0 else f"{job.name}_{i + 1}"
            tgt = os.path.join(job.dest or os.path.dirname(f), stem + os.path.splitext(f)[1])
            os.replace(f, tgt)
            named.append(tgt)
        files = named
    if not getattr(a, "no_recipe", False):
        rec = recipe(a, job.subj, job.seed, server, job.seconds)
        for f in files:
            tag_file(f, rec)
    return files


FARM_GRACE = 30      # seconds a farm box may stop answering before its job goes to another box


def run_farm(a, work):
    """Render farm: distribute jobs across POOL with dynamic dispatch (feed the free GPU).
    Faster boxes naturally pull more jobs; results are fetched back from whichever GPU made them.
    Returns the finished files."""
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
        print(f"     provision them with:  {download_hint(a)}")
    if not ready:
        sys.exit(f"render farm: no server in the pool has {need} — provision one:  {download_hint(a)}")
    live = ready
    if not a.song and a.engine == "sa3":         # a box in fp16 renders static, quickly: say so per box
        for s in live:
            w = fp32_warning(s)
            if w:
                print(w)
    if getattr(a, "ref_name", None):           # every box needs its own copy of the reference
        for s in list(live):
            try:
                upload_ref(s, a.ref_file)
            except Exception as e:
                print(f"   ⚠ skipping {_short(s)}: couldn't send it the reference ({e})")
                live.remove(s)
        if not live:
            sys.exit("render farm: no server accepted the reference.")
    print(f"   \U0001f69c render farm: {len(live)} GPU(s) — {', '.join(_short(s) for s in live)}")
    pending = list(work)        # Job tuples
    inflight = {}               # server -> [job, pid, graph, polls, lost, resends]
    total = len(work)
    done = 0
    failed = 0
    made = []
    fails = {}                  # server -> consecutive failed jobs
    nfail, nok = {}, {}         # server -> failed / finished jobs this run (the failure-rate rule)
    tries = {}                  # job -> failed attempts
    avoid = {}                  # job -> the box that failed it (its retry should go elsewhere)
    refreshed = set()           # jobs already re-saved after a cached-empty answer
    silent = {}                 # server -> when it stopped answering polls

    def drop(srv, why):
        if srv in live:
            live.remove(srv)
            print(f"   ⚠ dropping {_short(srv)}: {why}", flush=True)

    def launch(srv):
        """Submit the next pending job to srv. False = server unusable (drop it).
        A job that failed on srv goes to another box when one is live: the box that just failed is
        free first, so without this the retry landed right back on it."""
        if not pending:
            return True             # nothing left to do
        others = [x for x in live if x != srv]
        pick = next((j for j in pending if avoid.get(j) != srv), None)
        if pick is None:
            if others:
                return True         # leave it for another box
            pick = pending[0]
        pending.remove(pick)
        g = build_graph(a, pick.seed, subject=pick.subj, server=srv, seconds=pick.seconds)
        try:
            pid = submit(g, srv)
        except SystemExit as e:     # refused (a 400 says why) or unreachable: keep the job, say why
            pending.insert(0, pick)
            drop(srv, str(e).replace("\n", " | ")[:400])
            return False
        inflight[srv] = [pick, pid, g, 0, 0, 0]
        return True

    def job_failed(srv, job, why, breaker=True):
        """A job that came back without audio: requeue it once (on whichever box frees up), and feed the
        circuit breaker (AGENTS gotcha 16): a box that fails fast is free again first, so dynamic
        dispatch would hand it most of the queue."""
        nonlocal failed
        inflight.pop(srv, None)
        tries[job] = tries.get(job, 0) + 1
        avoid[job] = srv
        print(f"   ⚠ {_short(srv)} failed seed={job.seed}: {why}")
        if tries[job] >= 2:
            print(f"   ✖ seed={job.seed} failed twice — giving up on it")
            failed += 1
        else:
            pending.append(job)
        if breaker:
            fails[srv] = fails.get(srv, 0) + 1
            nfail[srv] = nfail.get(srv, 0) + 1
            if fails[srv] >= 2:
                drop(srv, f"{fails[srv]} failures in a row")
            elif nfail[srv] >= 3 and nfail[srv] > nok.get(srv, 0):
                # a box failing every other job never fails twice IN A ROW, and kept half the queue
                drop(srv, f"{nfail[srv]} of its {nfail[srv] + nok.get(srv, 0)} jobs failed")

    for srv in list(live):
        launch(srv)

    while inflight or (pending and live):
        advanced = False
        for srv, slot in list(inflight.items()):
            job, pid, g = slot[0], slot[1], slot[2]
            try:
                outs = poll(pid, srv)
            except CachedNoOutput:
                # ComfyUI served an identical re-run from its cache and saved nothing. The box is fine:
                # re-save it under a fresh folder (instant), and never count it toward the breaker.
                advanced = True
                if job not in refreshed:
                    refreshed.add(job)
                    print(f"   ↻ {_short(srv)}: identical re-run of seed={job.seed} answered from its cache — re-saving it",
                          flush=True)
                    g2 = fresh_copy(g)
                    try:
                        inflight[srv] = [job, submit(g2, srv), g2, 0, 0, slot[5]]
                    except SystemExit:
                        job_failed(srv, job, "couldn't re-send the cached job", breaker=False)
                    continue
                job_failed(srv, job, "the server returned no audio for an identical cached prompt", breaker=False)
                continue
            except JobFailed as e:
                advanced = True
                silent.pop(srv, None)
                why = str(e)
                if e.kind == "poisoned":
                    why += "  (its ComfyUI needs a restart)"
                job_failed(srv, job, why)
                continue
            except Exception as e:
                # One unanswered poll is not a dead box: a busy server can miss a 10 s deadline, or
                # answer one /history with a 500. Requeueing at once sent the job to another box (or
                # back to the same one) while the original was still rendering — 75 prompts for one
                # job on a flaky box. So: keep polling for FARM_GRACE; after that the box is out of
                # this run (never re-fed), and its job goes to the others.
                if srv not in silent:
                    print(f"   … {_short(srv)} not answering ({getattr(e, 'reason', e)}) — waiting up to {FARM_GRACE}s",
                          flush=True)
                first = silent.setdefault(srv, time.time())
                if time.time() - first < FARM_GRACE:
                    continue
                silent.pop(srv, None)
                _forget(srv, pid)
                inflight.pop(srv, None)
                pending.append(job)
                avoid[job] = srv
                drop(srv, f"no answer for {FARM_GRACE}s ({getattr(e, 'reason', e)}) — its job goes to another box")
                advanced = True
                continue
            if silent.pop(srv, None) is not None:
                print(f"   … {_short(srv)} answering again")
            if outs is None:
                # Every ~5 polls: does the box still know the job? A restarted ComfyUI forgets its whole
                # queue, and without this the farm would wait on that prompt forever.
                slot[3] += 1
                if slot[3] % 5 == 0:
                    slot[4] = 0 if _known(pid, srv) else slot[4] + 1
                    if slot[4] >= 3:
                        _forget(srv, pid)
                        advanced = True
                        if slot[5] < 2:
                            print(f"   ↻ {_short(srv)} restarted and lost seed={job.seed} — sending it again")
                            try:
                                inflight[srv] = [job, submit(g, srv), g, 0, 0, slot[5] + 1]
                            except SystemExit:
                                job_failed(srv, job, "it restarted and won't take the job again")
                        else:
                            job_failed(srv, job, "it keeps losing the job")
                continue
            advanced = True
            try:
                files = collect(audio_outs(outs), job.dest, srv)
            except MissingOutput as e:
                if job not in refreshed:        # a cache answer naming a file that's gone: re-save, no breaker
                    refreshed.add(job)
                    print(f"   ↻ {_short(srv)}: seed={job.seed} came from its cache, file gone — re-saving it", flush=True)
                    g2 = fresh_copy(g)
                    try:
                        inflight[srv] = [job, submit(g2, srv), g2, 0, 0, slot[5]]
                    except SystemExit:
                        job_failed(srv, job, "couldn't re-send the cached job", breaker=False)
                    continue
                job_failed(srv, job, str(e))
                continue
            except Exception as e:
                job_failed(srv, job, f"couldn't fetch the result ({getattr(e, 'reason', e)})")
                continue
            fails[srv] = 0
            nok[srv] = nok.get(srv, 0) + 1
            del inflight[srv]
            try:
                files = finish_files(a, files, job, srv)
            except Exception as e:     # local post-processing: not the box's fault, and re-rendering won't fix it
                failed += 1
                print(f"   ✗ seed={job.seed}: rendered, but finishing it here failed: {e}")
                launch(srv)
                continue
            done += 1
            made.extend(files)
            sj = f"{job.name or job.subj}  " if (a.batch or a.batch_file) else ""
            print(f"   ✅ [{done}/{total}] {_short(srv):<20} {sj}seed={job.seed}  ->  "
                  f"{files[0] if files else '(no file)'}", flush=True)
            launch(srv)          # feed the now-free GPU its next job
        # Requeued work must reach a box that already went idle, or it is lost.
        for srv in list(live):
            if pending and srv not in inflight:
                launch(srv)
        if not advanced:
            time.sleep(1)

    if pending:
        print(f"   ⚠ {len(pending)} job(s) left undone (all GPUs dropped).")
    if failed or pending:
        sys.exit(f"render farm: {failed + len(pending)} of {total} job(s) produced no audio.")
    return made


def download_hint(a):
    """The download-models.sh line that provisions this run's model."""
    if a.song:
        return "./download-models.sh --song"
    return "./download-models.sh --sa3" if a.engine == "sa3" else "./download-models.sh --sao"


# --- argument checks, all BEFORE the engine hand-offs ------------------------------
# The CPU engines `return` from main() early (AGENTS gotcha 10), so a check placed with the diffusion
# setup never ran for them: --mood bogus played "mysterious", --bpm 0 and --blip-rate 0 rendered
# (a 270 s blip), and two engine flags were settled silently by whichever hand-off came first.
CPU_ENGINES = ("--chip", "--opl", "--chipfx", "--oplfx", "--blip", "--narrate", "--record")


def _engine_flags(a):
    """Every engine this command line asks for, by its flag."""
    out = []
    for on, flag in ((a.chip, "--chip"), (a.opl, "--opl"), (a.chipfx, "--chipfx"), (a.oplfx, "--oplfx"),
                     (a.blip or a.blip_file, "--blip"),
                     # `--blip --narrate-file lines.txt` blips that manifest (the GUI sends it): one engine
                     ((a.narrate or a.narrate_file) and not (a.blip or a.blip_file), "--narrate"),
                     (a.record or a.record_file, "--record"), (a.song, "--song"), (a.music, "--music"),
                     (a.master is not None, "--master"), (_finding(a), "--index/--similar")):
        if on:
            out.append(flag)
    return out


def _bad_key(text):
    """Why a --key can't be read, or None. Song keys must be one of ComfyUI's 34; the chip engines read
    'D minor', 'f#', 'Bb dorian' — but 'H major' used to become C minor without a word."""
    import re
    t = (text or "").strip()
    m = re.fullmatch(r"([A-Ga-g])([#sb]?)(?:\s+([A-Za-z]+))?", t)
    if not m:
        return f"--key {text!r}: expected a note A-G (with # or b) and a scale, e.g. \"D minor\""
    word = (m.group(3) or "").lower()
    if word:
        try:
            sys.path.insert(0, _SCRIPT_DIR)
            import chip
            scales = set(chip.SCALES)
        except Exception:
            scales = {"major", "minor"}
        if word not in scales and not word.startswith(("maj", "min")):
            return f"--key {text!r}: unknown scale {word!r} (one of: {', '.join(sorted(scales))})"
    return None


def safe_name(name):
    """--name as a file name and nothing else: no directories (`../x` wrote outside --output-to,
    `a/b/c` crashed libsndfile), no leading dots, at most 80 characters. The same rule as every
    manifest key (narrate.safe_key), so `--name a/b` and a manifest row `a/b` both give `a_b`."""
    sys.path.insert(0, _SCRIPT_DIR)
    import narrate
    return narrate.safe_key(name) or "sound"


def check_args(a, p):
    """Refuse what can't work, in one place, before any engine runs. Returns the engine flag (or
    "sfx" for the default), or None when nothing at all was asked for (main shows the help)."""
    import math
    eng = _engine_flags(a)
    if len(eng) > 1:
        p.error(f"pick one engine: {' and '.join(eng)} can't be combined")
    engine = eng[0] if eng else None
    cpu = engine in CPU_ENGINES
    has_input = bool(a.prompt or a.batch or a.batch_file or a.narrate_file or a.record_file or a.blip_file
                     or a.from_midi or a.from_audio or a.master is not None or _finding(a))
    if engine is None and not has_input:
        return None
    # An engine with nothing to work on was answered with the full help and exit 0.
    if engine in ("--blip", "--narrate", "--record") and not has_input:
        p.error(f"{engine} needs text: soundmon \"Hello there\" {engine}   (or {engine}-file FILE)")
    if engine in ("--song", "--music") and not (a.prompt or a.batch or a.batch_file):
        p.error(f"{engine} needs a description: soundmon \"…\" {engine}")
    if (a.from_midi or a.from_audio) and engine not in ("--chip", "--opl"):
        p.error(f"{'--from-midi' if a.from_midi else '--from-audio'} plays a file on a chip: add --chip or --opl")
    if cpu:
        for on, flag in ((a.batch, "--batch"), (a.batch_file, "--batch-file"), (a.move_to_dirs, "--move-to-dirs"),
                         (a.no_subdirs, "--no-subdirs"), (a.style, "--style"), (a.ref, "--ref"),
                         (a.match_tone, "--match-tone"), (a.ref_match, "--ref-match"),
                         (given("--ref-strength"), "--ref-strength"),
                         (a.show_prompt, "--show-prompt"), (a.exact_prompt, "--exact-prompt"),
                         (a.dry_run, "--dry-run")):
            if on:
                p.error(f"{flag} is for the AI engines (SFX, --music, --song); {engine} doesn't use it"
                        + ("  (for a list of lines use --narrate-file / --blip-file)" if flag.startswith("--batch")
                           else ""))
    # numbers that would write destructive or absurd audio
    def finite(name, v, lo=None, hi=None):
        if v is None:
            return
        if not math.isfinite(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
            rng = f"{lo:g}..{hi:g}" if lo is not None and hi is not None else (f"≥ {lo:g}" if lo is not None else f"≤ {hi:g}")
            p.error(f"{name} must be a number {rng}, got {v}")
    if a.number < 1:
        p.error(f"-n must be at least 1, got {a.number}")
    if not math.isfinite(a.seconds) or a.seconds <= 0:
        p.error(f"--seconds must be more than 0, got {a.seconds}")
    if engine in ("--chip", "--opl"):
        finite("--seconds", a.seconds, 0.5, 600)
    if not cpu and engine not in ("--master", "--index/--similar") and a.seconds < 1.0:
        # ComfyUI's EmptyLatentAudio has min 1.0: say it here, not as a validation blob from the server
        p.error(f"--seconds must be at least 1 for the AI engines, got {a.seconds:g} "
                f"(use --max-seconds to cut a take shorter)")
    enc = [f for f, on in (("--flac", a.flac), ("--ogg", a.ogg), ("--mp3", a.mp3), ("--opus", a.opus)) if on]
    if len(enc) > 1:
        p.error(f"pick one output format: {' and '.join(enc)}  (one container per run)")
    if a.loop and not cpu and (a.ogg or a.mp3 or a.opus):
        print(f"   ⚠ --loop with {enc[0]}: lossy encoders pad the start, so the seam can click — "
              "--flac keeps a loop sample-exact")
    if a.match_tone and engine not in CPU_ENGINES:
        mt = os.path.expanduser(a.match_tone)
        if not os.path.isfile(mt) or not (_ffprobe_seconds(mt) or 0) > 0:
            p.error(f"--match-tone {a.match_tone}: not a readable sound file")
    finite("--normalize-db", a.normalize_db, -60, 0)
    finite("--true-peak", a.true_peak, -30, 0)
    finite("--fade-ms", a.fade_ms, 0, 10000)
    finite("--threshold-db", a.threshold_db, -120, 0)
    finite("--max-seconds", a.max_seconds, 0, None)
    finite("--loop-crossfade", a.loop_crossfade, 0.01, 60)
    finite("--ogg-quality", a.ogg_quality, 0, 10)
    finite("--bpm", a.bpm, 10, 300)
    finite("--blip-rate", a.blip_rate, 1, 60)
    finite("--blip-duty", a.blip_duty, 0.05, 1)
    finite("--blip-speed", a.blip_speed, 0.25, 8)
    finite("--blip-pitch", a.blip_pitch, -36, 36)
    finite("--blip-jitter", a.blip_jitter, 0, 24)
    finite("--midi-start", a.midi_start, 0, 0.99)
    finite("--transpose", a.transpose, -48, 48)
    finite("--speed", a.speed, 0.25, 4)
    finite("--pitch", a.pitch, -24, 24)
    finite("--record-rate", a.record_rate, 8000, 192000)
    finite("--reprompt-temp", a.reprompt_temp, 0, 2)
    if a.cfg is not None:
        finite("--cfg", a.cfg, 0, 30)
    if a.steps is not None:
        finite("--steps", a.steps, 1, 500)
    if a.lufs is not None and str(a.lufs).lower() not in ("off", "none", "no", ""):
        try:
            finite("--lufs", float(a.lufs), -70, 0)
        except ValueError:
            p.error(f"--lufs must be a number or 'off', got {a.lufs!r}")
    if a.mood != "auto" and engine in ("--chip", "--opl"):
        try:
            sys.path.insert(0, _SCRIPT_DIR)
            import chip
            moods = list(chip.MOODS)
        except Exception:
            moods = None
        if moods and a.mood not in moods:
            p.error(f"unknown --mood {a.mood!r}. One of: auto, {', '.join(moods)}  (see --list-moods)")
    if given("--key") or engine == "--song":
        if engine == "--song" and a.key not in KEYS:
            p.error(f"unknown --key {a.key!r}. See --list-keys.")
        why = _bad_key(a.key)
        if why:
            p.error(why)
    if a.format not in FORMATS:
        p.error(f"unknown format {a.format!r}. See --list-formats.")
    if a.style and engine not in CPU_ENGINES:
        for nm in [x.strip() for x in a.style.replace(",", " ").split() if x.strip()]:
            if nm not in STYLES:
                p.error(f"unknown style {nm!r}. See --list-styles.")
    if a.name is not None:
        a.name = safe_name(a.name)
    if a.output_to:
        out = os.path.abspath(os.path.expanduser(a.output_to))
        if os.path.exists(out) and not os.path.isdir(out):
            p.error(f"--output-to {a.output_to}: that is a file, not a folder")
    return engine or "sfx"


def build_parser():
    """Every flag. A function of its own so tools/selfcheck.py can check that --help mentions each one."""
    # allow_abbrev=False: `--sec 3` must not quietly mean --seconds — given() (which decides whether a
    # value was typed or defaulted) only knows full flag names.
    p = argparse.ArgumentParser(prog="soundmon", add_help=False, allow_abbrev=False,
                                # a short usage line: argparse's own listed every flag (50 lines) above each error
                                usage='soundmon "a description" [options]   (soundmon -h for all of them)')
    p.add_argument("-h", "--help", action="store_true", dest="show_help")
    p.add_argument("--version", action="version", version=f"soundmon {__version__}",
                   help="print the version and exit")
    p.add_argument("prompt", nargs="?", help='the sound you want, e.g. "a heavy door creaking"')
    p.add_argument("-n", "--number", type=int, default=1,
                   help="how many to generate, each with a different seed. default 1 "
                        "(with --batch: how many of EACH subject)")
    p.add_argument("--batch", default=None, metavar="SUBJECTS",
                   help='comma-separated subjects to round-robin, one of each per pass; '
                        'each goes to its own folder')
    p.add_argument("--batch-file", dest="batch_file", default=None, metavar="FILE",
                   help="render every row of a manifest: 'key | prompt' or 'key | seconds | prompt' "
                        "(prompts may contain commas). One file per key, named by the key, in --output-to "
                        "(default: the current directory). Works with the farm.")
    p.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                   help="with --batch-file: skip keys that already have a file in the destination "
                        "(resume an interrupted pack)")
    p.add_argument("--show-prompt", dest="show_prompt", action="store_true",
                   help="print the final positive/negative prompt (with the style and reference tails) "
                        "before rendering")
    p.add_argument("--exact-prompt", dest="exact_prompt", action="store_true",
                   help="send the description exactly as typed: no 'sound effect' tail, no default negative, "
                        "no Qwen rewrite (implies --no-reprompt; can't be combined with --style)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="build the ComfyUI graph(s), print them as JSON and exit — no server is contacted")
    p.add_argument("--no-recipe", dest="no_recipe", action="store_true",
                   help="don't embed the recipe (prompt, seed, engine, command) in the output files' "
                        "comment tag")
    p.add_argument("--seconds", type=float, default=10.0,
                   help="length in seconds (SA3 max 120, SAO max 47). default 10. The model conditions "
                        "on this, so it shapes the sound — not just its length.")
    p.add_argument("--style", default="",
                   help="style guide(s) to append, comma-separated. see --list-styles")
    p.add_argument("--format", default="none",
                   help="hardware format lock (sample rate + bit depth + channels). "
                        "'none' = the model's own 44.1 kHz stereo. see --list-formats")
    p.add_argument("--bits", type=int, default=16, choices=[8, 16, 24],
                   help="WAV bit depth. default 16")
    p.add_argument("--steps", type=int, default=None,
                   help="sampling steps (default: 8 on SA3, 50 on SAO (16 with --fast), 16 on --song)")
    p.add_argument("--cfg", type=float, default=5.0, help="prompt adherence. default 1.0 on SA3 and --song, 5.0 on SAO")
    p.add_argument("--fast", action="store_true",
                   help="fewer steps, rougher: 16 instead of 50 on --engine sao, 8 instead of 16 on --song. "
                        "No effect on SA3, which already renders in 8 steps")
    p.add_argument("--seed", type=int, default=-1, help="-1 = random each run")
    # --- chip mode: a real PSG synthesizer. No model, no GPU, no ComfyUI. ---
    p.add_argument("--chip", action="store_true",
                   help="CHIPTUNE mode — synthesize real 2A03 chiptune (2 pulse + "
                        "triangle + noise). Authentic by construction; loops seamlessly.")
    # --scale, not --chip-scale: opl.py shares chip.compose(), so this drives
    # AdLib as well. The old name stays as an alias — it is in scripts already.
    p.add_argument("--scale", "--chip-scale", dest="chip_scale", default=None,
                   choices=["minor", "harmonic", "dorian", "phrygian", "major",
                            "mixolydian", "pentatonic"],
                   help="override the scale implied by --key")
    p.add_argument("--write-mod", dest="write_mod", action="store_true",
                   help="ALSO write a ProTracker .MOD (--chip). Real samples from "
                        "the PSG synth plus pattern data — editable in OpenMPT or "
                        "Schism, and tiny.")
    p.add_argument("--write-rad", dest="write_rad", action="store_true",
                   help="ALSO write a Reality AdLib Tracker .RAD (--opl). OPL "
                        "instruments from the bank plus patterns — QB64 plays it "
                        "natively.")
    p.add_argument("--write-midi", dest="write_midi", action="store_true",
                   help="ALSO write a .mid beside the audio (--chip/--opl). A few "
                        "KB, plays natively in QB64, and can be paired with any "
                        "soundfont — the audio render is unaffected.")
    p.add_argument("--from-midi", dest="from_midi", default=None, metavar="FILE",
                   help="play a MIDI FILE on the chip. The BEST input: exact "
                        "pitches, real note durations, explicit tempo and time "
                        "signature, drums already separated. No inference at all.")
    p.add_argument("--midi-start", dest="midi_start", type=float, default=0.15,
                   metavar="F",
                   help="skip this fraction into the file before taking bars "
                        "(game MIDIs often open with empty bars)")
    p.add_argument("--transpose", type=int, default=0, metavar="N",
                   help="shift --from-midi by N semitones")
    p.add_argument("--from-audio", dest="from_audio", default=None, metavar="FILE",
                   help="TRANSCRIBE a reference recording and play it on the chip. "
                        "Takes the tempo, key, chords, melody, bass and drums from "
                        "FILE — the composition is the reference's, the sound is the "
                        "chip's. Mood still chooses the voices.")
    p.add_argument("--mood", default="auto",
                   help="mood for --chip/--opl: heroic, ominous, eerie, melancholy, "
                        "solemn, mysterious, tense, frantic, driving, playful, serene, "
                        "grand, wondrous, triumphant. 'auto' reads it from your "
                        "description. --list-moods")
    p.add_argument("--list-moods", dest="list_moods", action="store_true",
                   help="show every mood and what it changes")
    p.add_argument("--chippy", default="off",
                   choices=["off", "some", "lots", "max"],
                   help="how CHIP the chip sounds (--chip --from-midi). Chords "
                        "become fast arpeggios, leaps get pitch slides, and at "
                        "'max' a subtle tracker-style echo. off = exact polyphony.")
    p.add_argument("--arp", "--chip-arp", dest="chip_arp", type=int, default=0, metavar="N",
                   help="arpeggio speed in 16ths — 1 is the classic buzzy chord (default: 0, the mood chooses)")
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
    # --- mastering: neon-caster's chain, so a set of sounds sits together (master.py) ---
    p.add_argument("--master", default=None, metavar="FILE|DIR",
                   help="master a sound or a folder of them into --output-to (default <dir>/mastered/): "
                        "tilt toward -3 dB/octave, tame 2.5-5 kHz and sibilance, back to its own loudness, "
                        "never past -1 dBTP. CPU + ffmpeg; no ComfyUI")
    # --- find similar sounds on disk (similar.py) ---
    p.add_argument("--index", default=None, metavar="DIR[,DIR]",
                   help="index the sounds in these folders for --similar (incremental; ~/soundmon-gallery/similar/)")
    p.add_argument("--similar", default=None, metavar="FILE",
                   help="list the indexed sounds most like FILE (how alike, and in what: timbre, brightness, "
                        "envelope, length). -n sets how many (default 20)")
    p.add_argument("--in", dest="within", default=None, metavar="DIR",
                   help="with --similar / --find / --describe / --tags: only look inside DIR")
    p.add_argument("--find", default=None, metavar="WORDS",
                   help="list the indexed sounds whose path has all these words ('' for all; -n how many)")
    p.add_argument("--tag", dest="find_tag", default=None, metavar="TAG[,TAG]",
                   help="with --similar / --find / --describe / --tags: only these tags (kick, snare, pad, riser... "
                        "or none; soundmon --tags lists them)")
    p.add_argument("--kind", dest="find_kind", default=None, metavar="K",
                   help="with --similar / --find / --describe / --tags: loop, oneshot or unknown")
    p.add_argument("--aspects", default=None, metavar="NAME=W,...",
                   help="with --similar: how much each aspect counts — timbre, brightness, envelope, length, "
                        "character (what --listen heard). default timbre=0.5,brightness=0.2,envelope=0.2,"
                        "length=0.1, and character 70%% of the whole once heard")
    p.add_argument("--workers", type=int, default=None, metavar="N",
                   help="with --index: read N files at once (default: half the cores, at most 8)")
    p.add_argument("--listen", action="store_true",
                   help="let an audio model (LAION CLAP, ~600 MB once) hear every indexed sound: tags the ones "
                        "whose words say nothing, learning from the ones whose words do; adds 'character' to "
                        "--similar and makes --describe work. Needs torch + transformers; a GPU if there is one")
    p.add_argument("--listen-device", dest="listen_device", default=None, metavar="DEV",
                   help="with --listen / --describe: auto (default), cpu, cuda, cuda:1…")
    p.add_argument("--describe", default=None, metavar="TEXT",
                   help="list the indexed sounds that sound most like the words, e.g. 'a dark metallic hit' "
                        "(after --listen)")
    p.add_argument("--tags", action="store_true",
                   help="how the index is tagged: by words, by ear, by you; each tag; loops and one-shots")
    p.add_argument("--master-like", dest="master_like", default=None, metavar="REF",
                   help="with --master: match each file's tone and loudness to REF instead of the standard chain")
    # --- steering with a reference sound (the audio version of pixelmon's init image) ---
    p.add_argument("--ref", default=None, metavar="FILE",
                   help="start from this sound instead of silence (SFX, --music, --song): the result keeps "
                        "its shape and rhythm, the description changes its character. Length defaults to the "
                        "reference's")
    p.add_argument("--ref-strength", dest="ref_strength", type=float, default=0.6, metavar="0..1",
                   help="how far from the reference to go: 0.3 = a close variation, 0.9 = loosely inspired. default 0.6")
    p.add_argument("--ref-match", dest="ref_match", action="store_true",
                   help="also take the reference's tempo and key (--song sets --bpm/--key, --music adds them to the prompt)")
    p.add_argument("--match-tone", dest="match_tone", default=None, metavar="REF",
                   help="after rendering, EQ each take toward REF's spectrum and bring it to REF's loudness "
                        "(never past -1 dBTP)")
    p.add_argument("--master-kind", dest="master_kind", default="auto", choices=["auto", "sfx", "speech"],
                   help="how --master treats each file: sfx (low-pass, de-ess more) or speech (left alone unless "
                        "bright). auto = speech for speech/ narration/ voice/ folders, else sfx")
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
    return p


def _finding(a):
    """--index / --similar / --find / --describe / --listen / --tags: the sound library (similar.py)"""
    return (a.index is not None or a.similar is not None or a.find is not None or bool(a.describe)
            or a.listen or a.tags)


def main():
    p = build_parser()
    a = p.parse_args()

    global SERVER, REMOTE, POOL
    _target = pick_target(a.server)
    if _target:
        try:
            # De-duplicated by resolved URL: `rtx,192.168.1.77` (or `a,a`) is ONE box. Listed twice it
            # got two jobs at once, the second overwrote the first in the farm's bookkeeping, and that
            # job was lost while the run still exited 0.
            POOL = list(dict.fromkeys(resolve_server(s.strip()) for s in _target.split(",") if s.strip()))
        except ValueError as e:
            p.error(str(e))
        SERVER = POOL[0]
    REMOTE = not is_local(SERVER)

    if a.show_help:
        print_help()
        return
    listing = (a.list_formats or a.list_styles or a.list_keys or a.list_voices or a.list_moods
               or a.list_devices)
    if not listing:
        engine = check_args(a, p)
        if engine is None:                       # nothing to make and no engine named: show the way
            print_help()
            return
    if a.list_moods:
        sys.path.insert(0, _SCRIPT_DIR)
        import chip as _c
        print("\n  MOODS — for --chip and --opl\n")
        print(f"  {'mood':<13}{'scales':<24}{'bpm':>5}  {'arp':>3} {'duty':>4} "
              f"{'oct':>3} {'lean':>4}")
        for name, m in _c.MOODS.items():
            print(f"  {name:<13}{'/'.join(m['scales']):<24}{m['bpm']:>5.2f}  "
                  f"{m['arp']:>3} {_c.DUTIES[m['duty']]:>4.3g} {m['octave']:>+3} "
                  f"{('up' if m['contour']>0 else 'down' if m['contour']<0 else '--'):>4}")
        print("\n  'auto' (the default) infers the mood from your description.")
        print("  A mood sets scale, progression, tempo, rhythm density, arpeggio")
        print("  speed, duty cycle, drum pattern, register and melodic contour —")
        print("  changing only the scale gives you minor-key cheerfulness.\n")
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
        print(f"{c['b']}{c['cyan']}Musical keys{c['rst']}  {c['dim']}(use with --key: --song takes exactly these; "
              f"--chip/--opl also take other scales, e.g. \"D dorian\"){c['rst']}\n")
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
            if not given("--text-encoder"):
                a.text_encoder = "t5gemma_b_b_ul2.safetensors"
            # ComfyUI's official Stable Audio 3 workflow: lcm / simple / 8 steps
            # / cfg 1. Verified by ear — 50 steps at cfg 7 produces clacking and
            # popping, and euler produces noise. These are not tunables.
            if not given("--sampler"):
                a.sampler = "lcm"
            if not given("--scheduler"):
                a.scheduler = "simple"
            if a.steps is None:
                a.steps = 8
            if not given("--cfg"):
                a.cfg = 1.0
        else:
            a.base = a.base or "stable-audio-open-1.0.safetensors"

    # No loudness ceiling is +inf, NOT None. The engines skip loudness_normalize() entirely when the
    # target is None, and that function is also the only place --true-peak is enforced — so "no LUFS
    # ceiling" (chip/opl music, --lufs off) silently meant "no true-peak ceiling" too: chip moods
    # shipped at +0.8 dBTP and `--true-peak -3` did nothing. With +inf the loudness term never
    # applies and the true-peak ceiling always does (attenuate-only either way).
    NO_CEILING = float("inf")
    a.lufs_target = NO_CEILING
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
    # Similarity search reads files that already exist — no model, no server.
    if _finding(a):   # `--index ''` is an empty folder list, not "not given"
        # -n defaults to 1 (one take), which similar.py can't tell from a typed `-n 1`; this can.
        a.similar_k = a.number if given("-n", "--number") else 20
        sys.path.insert(0, _SCRIPT_DIR)
        import similar
        similar.run(a)
        return
    # Mastering works on files that already exist — no model, no server.
    if a.master is not None:
        sys.path.insert(0, _SCRIPT_DIR)
        import master
        master.run(a, slug, to_ogg, loudness_normalize, to_flac)
        return
    if a.record or a.record_file:
        sys.path.insert(0, _SCRIPT_DIR)
        import record
        record.run(a, slug, to_ogg, loudness_normalize, to_flac)
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
    # MUST be set before the engine hand-offs below, which `return` — AGENTS.md
    # gotcha 10, which I wrote and then walked straight into: the first version of
    # this sat after them and silently never ran.
    #
    # The -16 LUFS ceiling is a STREAMING target and it audibly pulls chip and FM
    # music down. Peak normalisation to --normalize-db plus the true-peak ceiling
    # already prevent clipping, including inter-sample overs after a lossy encode,
    # so for music the loudness ceiling is redundant and only costs level.
    #
    # SFX keep it: a pack where one effect is deafening is worse than a quiet one,
    # and that consistency is exactly what the ceiling buys. Diffusion keeps it
    # too — SA3 renders vary wildly in loudness, which is why it exists.
    if (a.chip or a.opl) and not given("--lufs"):
        a.lufs_target = NO_CEILING
    # --chip and --opl compose in whole bars, so they already loop
    # sample-accurately. Crossfading would shorten the track to hide an ending
    # neither of them has. This sat BELOW the engine hand-offs (gotcha 10 again),
    # so it never ran: --chip/--opl got the SFX default of 10 s, and a typed
    # `--seconds 10` couldn't have been told apart from no --seconds anyway.
    if (a.chip or a.opl) and not (a.chipfx or a.oplfx):
        if not given("--seconds"):           # 10 s is the SFX default; wrong for music
            a.seconds = 60.0
        if a.loop:
            mode = "--chip" if a.chip else "--opl"
            print(f"   ℹ --loop ignored: {mode} already loops by construction")
            a.loop = False

    if a.chipfx or a.oplfx:
        run_engine(a, "oplfx" if a.oplfx else "chipfx", "chipfx")
        return
    if a.opl:
        run_engine(a, "opl", "opl")
        return
    if a.chip:
        run_engine(a, "chip", "chip")
        return
    if a.blip or a.blip_file:
        run_engine(a, "blip", "blip")
        return
    if a.narrate or a.narrate_file:
        run_engine(a, "narrate", "narrate")
        return

    if a.format not in FORMATS:
        p.error(f"unknown format {a.format!r}. See --list-formats.")
    if a.batch_file and (a.batch or a.prompt):
        p.error("--batch-file replaces the description and --batch; give only the file")
    if a.skip_existing and not a.batch_file:
        p.error("--skip-existing works with --batch-file (where each key has one known file name)")
    if a.exact_prompt and a.style:
        p.error("--exact-prompt sends the text as typed, so it can't add --style guides")
    if a.exact_prompt:
        a.no_reprompt = True

    # --ref / --match-tone: steering with a reference sound. Read it now (its length decides
    # --seconds unless given); it is trimmed/padded and uploaded once the length is final.
    a.ref_name, a.ref_file, a.ref_tail = None, None, ""
    if a.match_tone:
        a.match_tone = os.path.expanduser(a.match_tone)
        if not os.path.isfile(a.match_tone):
            p.error(f"--match-tone: no such file: {a.match_tone}")
    if a.ref:
        a.ref = os.path.expanduser(a.ref)
        if not os.path.isfile(a.ref):
            p.error(f"--ref: no such file: {a.ref}")
        if not 0.05 <= a.ref_strength <= 1.0:
            p.error("--ref-strength must be between 0.05 and 1")
        cap = 1000.0 if a.song else (120.0 if a.engine == "sa3" else 47.0)
        if not given("--seconds"):
            dur = _ffprobe_seconds(a.ref)
            if dur:
                a.seconds = round(min(cap, max(1.0, dur)), 2)      # 1 s: EmptyLatentAudio's minimum
        if a.ref_match:
            got = ref_summary(a.ref)
            if got:
                bpm, key, _dur = got
                if a.song:
                    if not given("--bpm"):
                        a.bpm = bpm
                    if not given("--key") and key in KEYS:
                        a.key = key
                elif a.music:
                    a.ref_tail = f"{bpm} BPM, in {key}"
                used = "" if (a.song or a.music) else "  (a plain SFX uses only its length: tempo and key steer --song / --music)"
                print(f"   🎯 from the reference: {bpm} BPM, {key}, {a.seconds:g}s{used}")
            else:
                print("   ⚠ --ref-match: no steady tempo or key in the reference; only its length is used")

    # --song is a different model with a very different length range: Stable
    # Audio tops out at 47s, ACE-Step goes to 1000s (~16 min). The default
    # differs too — 10s is right for an SFX, absurd for a song.
    if a.song:
        if not given("--seconds") and not a.ref:  # no length asked for -> the song default
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
        if not given("--cfg"):
            a.cfg = 1.0
        if not given("--sampler"):
            a.sampler = "euler"
        if not given("--scheduler"):
            a.scheduler = "simple"
    else:
        a.steps = (16 if a.fast else 50) if a.steps is None else a.steps
    # SA3's 8 lcm steps are already its fast path (gotcha 13: fewer or more is noise or clacking), so
    # --fast has nothing to cut there. Say so instead of printing "FAST" over an unchanged render.
    a.fast_applies = a.fast and (a.song or a.engine != "sa3")
    if a.fast and not a.fast_applies:
        print("   ℹ --fast has no effect on SA3 — it already renders in 8 steps (it speeds up --engine sao and --song)")

    if a.negative is None:
        a.negative = ("" if a.exact_prompt else SONG_NEGATIVE if a.song
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

    if a.ref:                              # the length is final now
        a.ref_file = prepare_ref(a.ref, a.seconds)
        a.ref_name = os.path.basename(a.ref_file)

    n = max(1, a.number)
    rows = None
    if a.batch_file:
        try:
            rows = parse_batch_file(a.batch_file)
        except (OSError, ValueError) as e:
            p.error(f"--batch-file: {e}")
        if not rows:
            p.error(f"--batch-file: no rows in {a.batch_file}")
        cap = 1000.0 if a.song else (120.0 if a.engine == "sa3" else 47.0)
        for key, secs, _ in rows:
            if secs is not None and not 0 < secs <= cap:
                p.error(f"--batch-file: {key}: {secs:g}s is outside 0..{cap:g} for this engine")
        if a.ref and any(secs is not None for _, secs, _ in rows):
            # The reference is cut to ONE length and its length is the render's; a row asking for
            # another length would be conditioned on one length and given another.
            print("   ⚠ --ref sets the length; the seconds column of --batch-file is ignored")
            rows = [(k, None, t) for k, _, t in rows]
        subjects = [t for _, _, t in rows]
    else:
        subjects = [s.strip() for s in a.batch.split(",") if s.strip()] if a.batch else [a.prompt]
    base = os.path.abspath(os.path.expanduser(a.output_to)) if a.output_to else os.getcwd()

    def dest_for(subject):
        if a.batch_file:                     # flat, like --narrate-file: the keys are the file names
            d = base
        elif a.no_subdirs:
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
        if not os.path.isdir(d) and not a.dry_run:
            if a.create_dirs or a.move_to_dirs or a.batch:
                os.makedirs(d, exist_ok=True)
            else:
                p.error(f"output dir does not exist: {d}\n  (add --create-dirs to make it)")
        return d

    dests = {subj: dest_for(subj) for subj in subjects}

    # Build the work list. --batch / --batch-file round-robin (one of each subject per pass).
    work, skipped = [], []
    k = 0
    for take in range(n):
        for idx, subj in enumerate(subjects):
            seed = (a.seed + k) if a.seed >= 0 else random.randint(0, 2**31 - 1)
            k += 1
            name = secs = None
            if rows:
                key, secs = rows[idx][0], rows[idx][1]
                name = key if n == 1 else f"{key}_{take + 1:02d}"
                if a.skip_existing and existing_output(dests[subj], name):
                    skipped.append(name)
                    continue
            work.append(Job(subj, seed, dests[subj], name, secs))
    if skipped:
        print(f"   ⏭ {len(skipped)} already made, skipped: {', '.join(skipped[:8])}{' …' if len(skipped) > 8 else ''}")
    if not work:
        print("   nothing to do — every key already has a file (drop --skip-existing to remake them)")
        return

    total = len(work)
    # Rough seconds/clip for the ETA. Measured on an RX 6600 (ROCm, --lowvram):
    # ~6.4s/clip at 50 steps, ~4s at 16. Audio is far cheaper than SDXL — a
    # 4-second SFX is a much smaller latent than a 1024x1024 image.
    per = 4 if a.fast_applies else 7
    style_label = f"  |  style: {a.style}" if a.style else ""
    fmt_label = a.format if a.format != "none" else "44.1kHz stereo"
    if a.batch_file:
        subj_label = f"{len(subjects)} rows of {os.path.basename(a.batch_file)}"
    else:
        subj_label = f"{len(subjects)} subjects: {', '.join(subjects)}" if a.batch else repr(a.prompt)
    count_label = (f"{n} each = {total} total" if (a.batch or a.batch_file) else f"{n} clip(s)")
    if a.dry_run:
        pass
    elif len(POOL) > 1:
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
              f"{eng} {'FAST' if a.fast_applies else 'quality'} {a.steps}st  |  {count_label}")
    if a.ref:
        print(f"   🎯 steering with {os.path.basename(a.ref)} at strength {a.ref_strength:g}"
              f"  (0.3 close · 0.9 loose)")
    if a.match_tone:
        print(f"   🎚 matching tone to {os.path.basename(a.match_tone)}")
    if a.show_prompt:
        for subj in dict.fromkeys(subjects):
            pos, neg = final_prompts(a, subj)
            if a.song:
                print(f"   📝 tags:     {pos}\n   🚫 negative: {neg}")
                continue
            print(f"   📝 positive: {pos}\n   🚫 negative: {neg}")
        if reprompts(a):
            print(f"   🤖 on the server, {a.reprompt_model} rewrites the positive into Stable Audio 3's structured "
                  f"form before the model sees it\n      (--no-reprompt or --exact-prompt sends it as shown)")
    if total > 1 and not a.dry_run:
        eta = total * per
        tip = "" if (a.fast_applies or (a.engine == "sa3" and not a.song)) else "  (tip: add --fast for quick variations)"
        print(f"   ~{eta // 60}m{eta % 60:02d}s estimated{tip}")

    if a.dry_run:
        # The graphs exactly as they would be POSTed to /prompt, for bug reports, diffing and the GUI's
        # preview. A --ref is prepared locally (its uploaded name is content-hashed, so it is the same
        # name the server would get); nothing is uploaded and no server is asked anything.
        out = [{"subject": j.subj, "seed": j.seed, "name": j.name, "dest": j.dest,
                "graph": build_graph(a, j.seed, subject=j.subj, server=SERVER, seconds=j.seconds)} for j in work]
        print(json.dumps(out, indent=1))
        return

    # Capability check for the single-server path too — including the bare local default, which
    # used to skip it and then dump a raw ComfyUI validation blob when SA3 wasn't downloaded.
    # run_farm() routes around boxes that lack the model; this says which file is missing where.
    if len(POOL) <= 1:
        if not server_up(SERVER):
            sys.exit(f"can't reach ComfyUI at {SERVER} — is it running there? (check the address in servers.json)")
        need = a.song_base if a.song else a.base
        if need and not server_has_ckpt(SERVER, need):
            sys.exit(f"{_short(SERVER)} doesn't have {need} — provision it there:  {download_hint(a)}")
        if not a.song and a.engine == "sa3":
            w = fp32_warning(SERVER)
            if w:
                print(w)

    if a.ref_name and len(POOL) <= 1:
        try:
            upload_ref(SERVER, a.ref_file)
        except Exception as e:
            sys.exit(f"couldn't send the reference to {_short(SERVER)}: {e}")

    t0 = time.time()
    first_open = None
    failed = 0
    if len(POOL) > 1:
        made = run_farm(a, work)
        first_open = made[0] if made else None
    else:
        # Queue everything up front (ComfyUI runs them one at a time) and keep each graph: wait() may
        # have to send a job again — the server restarted and forgot it, or answered an identical
        # re-run from its cache without saving anything (then a fresh_copy() of it).
        jobs = []
        for job in work:
            g = build_graph(a, job.seed, subject=job.subj, server=SERVER, seconds=job.seconds)
            jobs.append((job, submit(g), g))
        if total > 1:
            print(f"   queued {total} jobs; generating...")
        in_a_row = 0
        seen_restarts = RESTARTS.get(SERVER, 0)
        for i, (job, pid, g) in enumerate(jobs, 1):
            if RESTARTS.get(SERVER, 0) != seen_restarts:
                # The server restarted while an earlier job was waited on: it forgot every prompt
                # still queued. Send them all again now, rather than letting each one sit out
                # wait()'s ~15 s "is it lost?" check in turn.
                seen_restarts = RESTARTS[SERVER]
                jobs[i - 1:] = resend_lost(jobs[i - 1:], SERVER)
                job, pid, g = jobs[i - 1]
            tag = f"[{i}/{total}] "              # always, even [1/1]: the GUI's progress reads it
            subj_note = f"{job.name or job.subj}  " if (a.batch or a.batch_file) else ""
            try:
                resend = lambda fresh, g=g: submit(fresh_copy(g) if fresh else g)    # noqa: E731
                outs = wait(pid, resubmit=resend)
                try:
                    got = collect(audio_outs(outs), job.dest, SERVER)
                except MissingOutput:
                    print("   ↻ the server answered from its cache with a file that's gone — re-saving it", flush=True)
                    got = collect(audio_outs(wait(resend(True), resubmit=None)), job.dest, SERVER)
                files = finish_files(a, got, job, SERVER)
            except ServerGone as e:
                sys.exit(str(e))                  # atexit takes the rest of this run off its queue
            except (JobFailed, OSError, ValueError, subprocess.SubprocessError) as e:
                # One bad job doesn't end the batch: the rest are already queued and rendering. But a
                # server failing EVERY job (a missing encoder, a broken install) would only print the
                # same error N times, so three in a row ends the run.
                failed += 1
                in_a_row += 1
                print(f"   ✗ {tag}{subj_note}seed={job.seed} failed on {_short(SERVER)}: {e}", flush=True)
                if in_a_row >= 3 and i < total:
                    sys.exit(f"{_short(SERVER)} failed {in_a_row} jobs in a row — stopping "
                             f"({total - i} not run).")
                continue
            in_a_row = 0
            clip = files[0] if files else None
            first_open = first_open or clip
            print(f"   ✅ {tag}{subj_note}seed={job.seed}  ->  {clip}", flush=True)

    if a.batch_file:
        where = base
    else:
        where = ", ".join(sorted({str(x) for x in dests.values() if x})) or f"{OUTPUT}/soundmon/"
    print(f"   all done in {time.time() - t0:.1f}s  |  files in {where}")
    if first_open and not a.no_open and shutil.which("xdg-open"):
        try:
            subprocess.Popen(["xdg-open", first_open],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    if failed:
        sys.exit(f"{failed} of {total} job(s) produced no audio.")


if __name__ == "__main__":
    import atexit
    import signal
    signal.signal(signal.SIGINT, _on_stop)
    signal.signal(signal.SIGTERM, _on_stop)
    # Any other way out with prompts still queued (a failed job ending a batch, an error) clears them too:
    # without this, the rest of a failed batch kept rendering on the GPU for nobody.
    atexit.register(cancel_sent, "stopped early")
    main()
