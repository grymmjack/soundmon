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


def parse_lines(path):
    """Read a narration source file into [(name, text), ...].

    Understands the qb64-dungeon data format — `key | text`, '#' comments —
    which covers strings.txt, labels.txt, curios.txt and friends. For a
    pipe-delimited row the LAST field is the prose and the first is the key;
    a plain line is used as its own text with a generated name.
    """
    out, n = [], 0
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
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
            if text:
                out.append((key, text))
    return out


def run(a, slug, to_ogg=None):
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

    pipeline = KPipeline(lang_code=lang, device=RUN_DEVICE)
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
        if getattr(a, "ogg", False) and to_ogg:
            path = to_ogg(path, a.ogg_quality, a.keep_wav)
        made.append(path)
        print(f"   ✅ [{i}/{len(items)}] {key:<28} {len(audio)/SAMPLE_RATE:5.2f}s  "
              f"{text[:52]}{'…' if len(text) > 52 else ''}")

    print(f"   all done  |  {len(made)} file(s) in {dest}")
    return made
