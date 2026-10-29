"""soundmon --narrate — text-to-speech narration via Kokoro.

The third engine, and the one that deliberately does NOT go through ComfyUI.
SFX and songs are diffusion problems where ComfyUI earns its keep (model
loading, VRAM management, the render farm). Kokoro is an 82M feed-forward
model that speaks a line in about a second — wrapping it in a node graph and
fanning it across four GPUs would add latency, not remove it. So this runs
in-process and writes through soundmon's normal output/organization path:
one interface, the right mechanism per engine.

Runs on CPU on purpose — see RUN_DEVICE below.
"""
import os
import re
import sys

# Every ENGLISH voice Kokoro ships: 8 British + 20 American. Prefix is
# <accent><gender>_ — b=British, a=American; m=male, f=female.
#
# The grade is Kokoro's own published quality rating (VOICES.md), not ours:
# A is excellent, D is rough. It tracks how much training audio the voice had,
# so a C voice is usually thinner or less stable rather than differently
# characterised. Where a description names a character trait, that voice has
# actually been listened to; the rest carry accent + grade only, because
# inventing personalities for voices nobody has heard is how you end up
# recommending the wrong narrator.
#
# Kokoro also ships 26 non-English voices (ja/zh/es/pt/hi/it/fr). They're
# deliberately not listed — this is a narration tool for an English game — but
# any of them works if you pass it explicitly with a matching --lang.
VOICES = {
    # --- British male ---
    "bm_george":   "British male, deep and measured — the classic DM  [B]",
    "bm_fable":    "British male, warm storyteller  [B]",
    "bm_lewis":    "British male, deep and gravelly — the deepest here  [C]",
    "bm_daniel":   "British male, lighter and brisker  [C]",
    # --- British female ---
    "bf_emma":     "British female  [B] — best-trained British female",
    "bf_isabella": "British female, formal — Souls-style narrator  [B]",
    "bf_alice":    "British female  [C]",
    "bf_lily":     "British female  [C]",
    # --- American male ---
    "am_fenrir":   "American male  [B]",
    "am_michael":  "American male  [B]",
    "am_puck":     "American male  [B]",
    "am_echo":     "American male  [C]",
    "am_eric":     "American male  [C]",
    "am_liam":     "American male  [C]",
    "am_onyx":     "American male  [C]",
    "am_santa":    "American male  [C] — very little training audio",
    "am_adam":     "American male  [D] — roughest voice in the set",
    # --- American female ---
    "af_bella":    "American female  [A] — highest-rated voice Kokoro ships",
    "af_heart":    "American female  [A] — Kokoro's flagship default",
    "af_nicole":   "American female  [B] — tuned for headphones",
    "af_aoede":    "American female  [B]",
    "af_kore":     "American female  [B]",
    "af_sarah":    "American female  [B]",
    "af_alloy":    "American female  [B]",
    "af_nova":     "American female  [B]",
    "af_sky":      "American female  [B] — very little training audio",
    "af_jessica":  "American female  [C]",
    "af_river":    "American female  [C]",
}

SAMPLE_RATE = 24000     # Kokoro's native output rate

# CPU, not GPU. On the RX 6600 (gfx1032 pretending to be gfx1030 via
# HSA_OVERRIDE_GFX_VERSION) Kokoro dies with "HIP error: invalid device
# function" — it uses a kernel the override doesn't cover. At 82M parameters
# CPU inference is ~1s a line anyway, so there is nothing to win on the GPU
# and a whole class of driver problems to lose.
RUN_DEVICE = "cpu"
KOKORO_REPO = "hexgrad/Kokoro-82M"



def kokoro_pipeline(lang):
    """Kokoro's pipeline without the noise it prints on every run: torch's dropout and weight_norm
    warnings (harmless, about the model's own code), the repo_id hint, and the Hub's "unauthenticated
    requests" warning once the model is already downloaded (it is then only checked, never fetched)."""
    import logging
    import warnings
    from kokoro import KPipeline
    warnings.filterwarnings("ignore", message=r"dropout option adds dropout", category=UserWarning)
    warnings.filterwarnings("ignore", message=r".*weight_norm.* is deprecated", category=FutureWarning)
    for name in ("huggingface_hub", "huggingface_hub.utils._http", "kokoro"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from huggingface_hub import try_to_load_from_cache
        if isinstance(try_to_load_from_cache(KOKORO_REPO, "config.json"), str):
            os.environ["HF_HUB_OFFLINE"] = "1"
            import huggingface_hub.constants as hc
            hc.HF_HUB_OFFLINE = True
    except Exception:
        pass
    return KPipeline(lang_code=lang, repo_id=KOKORO_REPO, device=RUN_DEVICE)

def _pitch_shift(audio, semitones, np):
    """Lower (or raise) pitch while keeping the original duration.

    Kokoro has no pitch control, so we get it from the one knob it does have.
    Resampling alone changes pitch AND duration together; asking Kokoro to
    speak faster by the same factor first cancels the duration change out and
    leaves only the pitch move. Negative `semitones` = deeper.

    Called with the audio already generated at speed=2**(-semitones/12).
    """
    if not semitones:
        return audio
    ratio = 2.0 ** (-semitones / 12.0)      # >1 when going deeper
    n = int(round(len(audio) * ratio))
    if n < 2:
        return audio
    # Linear interpolation is plenty for narration; a phase vocoder would be
    # overkill for a voice we're deliberately making unnatural anyway.
    src = np.linspace(0.0, len(audio) - 1.0, n)
    return np.interp(src, np.arange(len(audio)), audio).astype(audio.dtype)


def safe_key(key, maxlen=80):
    """A manifest key or --name as a FILE NAME and nothing else. Returns "" if
    nothing usable is left (the caller decides the fallback).

    The same rule soundmon.safe_name applies to --name: letters, digits, space,
    `.`, `-`, `_`; everything else (path separators above all) becomes `_`; no
    leading or trailing dots/spaces/underscores, so `../x` and `.hidden` cannot
    leave the output folder or hide in it; at most 80 characters, which keeps
    `<key>_s<seed>.wav` far below every filesystem's name limit.

    Separators become `_` rather than being cut at (basename): `sfx/door` and
    `music/door` in one manifest must stay two different files.
    """
    out = re.sub(r"[^\w.\- ]+", "_", str(key or "").replace("\\", "/"))
    return out.strip(" ._")[:maxlen].strip(" ._")


def _read_text(path):
    """A manifest's text. UTF-8 (with or without BOM) first; a file saved in a
    legacy 8-bit code page (Windows Notepad, old editors) is read as cp1252 /
    latin-1 with a notice, instead of dying in a UnicodeDecodeError traceback."""
    p = os.path.expanduser(path)
    if not os.path.exists(p):
        sys.exit(f"{path}: no such file")
    if os.path.isdir(p):
        sys.exit(f"{path}: is a folder, not a `key | text` file")
    try:
        with open(p, "rb") as f:
            raw = f.read()
    except OSError as e:
        sys.exit(f"{path}: can't read it ({e.strerror or e})")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    for enc in ("cp1252", "latin-1"):
        try:
            txt = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        print(f"   ℹ {os.path.basename(p)} is not UTF-8; read it as {enc}", file=sys.stderr)
        return txt
    sys.exit(f"{path}: not a text file")


def parse_lines(path):
    """Read a narration source file into [(name, text), ...].

    Understands the qb64-dungeon data format — `key | text`, '#' comments —
    which covers strings.txt, labels.txt, curios.txt and friends. For a
    pipe-delimited row the LAST field is the prose and the first is the key;
    a plain line is used as its own text with a generated name.

    `[Group]` lines (the GUI's manifest grouping) are headings, not lines to
    speak, and are skipped. Every key goes through safe_key() — the key IS the
    output file name, so `../escape` or `sub/dir` would write outside the pack
    or crash halfway through it. All rows are validated BEFORE anything is
    rendered: a duplicate key with different text stops here, with line
    numbers, rather than silently overwriting the first file.
    """
    out, n = [], 0
    seen = {}                       # key -> (line number, text)
    renamed = []
    for ln, raw in enumerate(_read_text(path).splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if re.fullmatch(r"\[[^\]|]*\]", line):
            continue                # [Group] heading
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            key, text = parts[0], parts[-1]
            # Rows that are pure data (all numeric tail) have no prose to
            # read — e.g. labels.txt is "col | row | TEXT", so the key is a
            # number. Fall back to naming by the text itself.
            if not key or key.replace(".", "").isdigit():
                key = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]
        else:
            n += 1
            text, key = line, f"line{n:03d}"
        if not text:
            continue
        safe = safe_key(key)
        if not safe:
            safe = safe_key(re.sub(r"[^a-z0-9]+", "_", text.lower())[:40]) or f"line{ln:03d}"
        if safe != key:
            renamed.append((ln, key, safe))
        if safe in seen:
            pln, ptext = seen[safe]
            if ptext == text:
                print(f"   ℹ line {ln}: {safe!r} repeats line {pln} word for word; made once",
                      file=sys.stderr)
                continue
            sys.exit(f"{path}: key {safe!r} is used twice with different text (lines {pln} and {ln})"
                     f" — the second would overwrite the first. Give one of them another key.")
        seen[safe] = (ln, text)
        out.append((safe, text))
    for ln, key, safe in renamed[:5]:
        print(f"   ℹ line {ln}: key {key[:40]!r} is not a safe file name; using {safe!r}",
              file=sys.stderr)
    if len(renamed) > 5:
        print(f"   ℹ ...and {len(renamed) - 5} more keys renamed the same way", file=sys.stderr)
    return out


def run(a, slug, to_ogg=None, loudness_normalize=None, to_flac=None):
    """Generate narration for one prompt or a whole file. Called from soundmon.py.

    `to_ogg` is injected rather than imported: soundmon.py runs as __main__, so
    importing it back from here would re-execute the whole module.
    """
    try:
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline
    except ImportError as e:
        sys.exit(f"--narrate needs Kokoro: pip install kokoro soundfile   ({e})")

    if a.voice not in VOICES:
        sys.exit(f"unknown --voice {a.voice!r}. See --list-voices.")

    items = parse_lines(a.narrate_file) if a.narrate_file else [
        (a.name or slug(a.prompt), a.prompt)]
    if not items:
        sys.exit(f"--narrate-file: nothing to read in {a.narrate_file}")

    dest = os.path.abspath(os.path.expanduser(a.output_to)) if a.output_to else os.getcwd()
    if not os.path.isdir(dest):
        if a.create_dirs or a.output_to:
            os.makedirs(dest, exist_ok=True)
        else:
            sys.exit(f"output dir does not exist: {dest} (add --create-dirs)")

    # British vs American English changes the phonemizer, not just the accent.
    lang = "b" if a.voice[0] == "b" else "a"
    print(f"🎙  {a.voice} ({VOICES[a.voice]})  |  {len(items)} line(s)  |  "
          f"speed {a.speed}  |  pitch {a.pitch:+g} st  |  CPU")

    pipeline = kokoro_pipeline(lang)
    # Speaking faster up front is what lets _pitch_shift lower the pitch without
    # also stretching the line — see the docstring there.
    gen_speed = a.speed * (2.0 ** (-a.pitch / 12.0))

    made = []
    for i, (key, text) in enumerate(items, 1):
        chunks = [np.asarray(au) for _, _, au in
                  pipeline(text, voice=a.voice, speed=gen_speed)]
        if not chunks:
            print(f"   ⚠ [{i}/{len(items)}] {key}: nothing generated, skipped")
            continue
        audio = np.concatenate(chunks)
        audio = _pitch_shift(audio, a.pitch, np)

        peak = float(np.abs(audio).max())
        if peak > 1e-6:                      # same -1 dBFS target as the other engines
            audio = audio * ((10.0 ** (a.normalize_db / 20.0)) / peak)

        path = os.path.join(dest, f"{key}.wav")
        sf.write(path, audio, SAMPLE_RATE, subtype=f"PCM_{a.bits}" if a.bits != 8 else "PCM_U8")
        # Narration compresses especially well — it's mono-ish speech with a lot
        # of near-silence between words.
        if getattr(a, "lufs_target", None) is not None and loudness_normalize:
            loudness_normalize(path, a.lufs_target, a.true_peak)
        # FLAC before OGG: lossless, and for a square-wave text-box voice a lossy
        # codec smears exactly the hard edges. This branch was MISSING -- to_flac
        # was accepted as a parameter and never called, so --flac was silently
        # dropped for narration.
        if getattr(a, "flac", False) and to_flac:
            path = to_flac(path, a.keep_wav)
        elif getattr(a, "ogg", False) and to_ogg:
            path = to_ogg(path, a.ogg_quality, a.keep_wav)
        made.append(path)
        print(f"   ✅ [{i}/{len(items)}] {key:<28} {len(audio)/SAMPLE_RATE:5.2f}s  "
              f"{text[:52]}{'…' if len(text) > 52 else ''}")

    print(f"   all done  |  {len(made)} file(s) in {dest}")
    return made
