# soundmon — text-to-sound-effect generator (NVIDIA · AMD · CPU)

Describe a sound, get a game-ready WAV — with **one command**, entirely on your
own machines. Uses ComfyUI + **Stable Audio Open 1.0**, and shares pixelmon's
engine, render farm, and CLI conventions.

```bash
soundmon "a heavy wooden door creaking open"          # full-quality WAV
soundmon "a laser blast" -n 8 --fast                  # 8 quick variations
soundmon --batch "door,glass,fire" -n 16 --server rtx,titan,local
```

This is the local, free, unlimited answer to cloud SFX generators like ElevenLabs
SFX: same "just describe what you want" workflow, no subscription, no upload, no
per-generation cost, and the render farm means you can make 100 variations while
you get coffee.

---

## What this is

A thin, friendly CLI (`soundmon`) over a local **ComfyUI** server, plus a custom
ComfyUI node (`retro_sfx`) that turns raw model output into an actual game asset.
You describe a sound; you get a WAV. The visual node-graph is handled behind the
scenes.

**The single most important lesson** (same shape as pixelmon's): the model has to
be trained on the thing you want. **Stable Audio Open 1.0** was trained on
Freesound + the Free Music Archive and is explicitly built for *sound effects and
production elements* — not songs. A music model asked for "a door creak" gives
you a song *about* a door.

### It shares everything with pixelmon

soundmon deliberately reuses pixelmon's engine rather than duplicating it:

| Shared | Not shared |
|---|---|
| `~/ComfyUI` + its venv + torch | the models (audio, ~5.3 GB) |
| `~/launch-comfyui.sh` (GPU autodetect) | the custom node (`retro_sfx`) |
| port 8188, `servers.json`, the whole farm | the workflow graph |

Any box already running ComfyUI for pixelmon becomes a soundmon box by running
`./download-models.sh` and linking the node. `install.sh` copies pixelmon's
`servers.json` for you.

---

## Install

```bash
git clone <your-remote> ~/git/soundmon && cd ~/git/soundmon
./install.sh            # links files; reuses an existing ComfyUI if present
./download-models.sh    # ~5.3 GB from Hugging Face (no login/token needed)
# restart ComfyUI so it picks up the retro_sfx node
soundmon "a heavy wooden door creaking open"
```

If you don't have ComfyUI yet, `install.sh` will tell you to run pixelmon's
installer first — it builds exactly the engine soundmon needs (ComfyUI + venv +
the right torch for your GPU).

`download-models.sh` fetches into `~/ComfyUI/models/`:

| File | Size | Goes to |
|---|---|---|
| `stable-audio-open-1.0.safetensors` | 4.85 GB | `models/checkpoints/` |
| `t5_base.safetensors` (text encoder) | 438 MB | `models/text_encoders/` |

> **Gating note.** `stabilityai/stable-audio-open-1.0` is a **gated** repo — it
> 401s without an HF token and a license click. `download-models.sh` pulls from
> an ungated mirror so setup stays one-command. Use the official repo with
> `HF_TOKEN=hf_xxx ./download-models.sh --official`.

---

## Usage

Run `soundmon --help` for the full, colorized list. The essentials:

| Flag | What it does | Default |
|---|---|---|
| `-n, --number N` | how many to make, each a different seed | `1` |
| `--seconds N` | length (model max 47). **Conditions the model** — it shapes the sound, not just its length | `10` |
| `--style NAMES` | append proven prompt guide(s), comma-separated — `--list-styles` | — |
| `--format NAME` | hardware format lock (rate + bit depth + channels) — `--list-formats` | `none` |
| `--bits N` | WAV bit depth: 8 / 16 / 24 | `16` |
| `--batch "a,b,c"` | round-robin subjects, one of each per pass, each into its own folder | — |
| `--fast` | 16 steps instead of 50: ~1.6× faster, rougher | off |
| `--seed N` | lock / repeat a result | random |
| `--no-trim` | keep the model's leading/trailing silence | off |
| `--normalize-db N` | peak level after generation | `-1.0` |
| `--fade-ms N` | de-click fade on both ends | `5` |
| `--flac` / `--mp3` / `--opus` | save compressed instead of WAV | WAV |
| `--server NAME[,...]` | remote ComfyUI; comma-list = render farm | local |

The seed is in every filename, so to make a full-quality version of a fast draft
you liked, just re-run that seed:

```bash
soundmon "a laser blast" --fast          # prints e.g. seed=12345
soundmon "a laser blast" --seed 12345    # same blast, full quality
```

### Style guides (`--style`)

Editable prompt snippets in `sounds.json` (`--list-styles`). Combine them freely:

```bash
soundmon "a sword hitting a shield" --style impact,dry
soundmon "wind through trees" --style ambience,big
soundmon "menu confirm" --style ui,small
```

Most carry a **negative** too — `ui`, for example, pushes *reverb, long tail,
ambience* into the negative prompt, which matters more than the positive for SFX.
The global default negative already fights the two most common failure modes:
the model drifting into **music** or **speech**.

### Render farm

Identical to pixelmon's, because it's the same code — the CLI never touches the
GPU, it just POSTs a JSON graph to `/prompt` and pulls results from `/view`.

```bash
soundmon --batch "explosion,glass,fire" -n 30 --server rtx,titan,local
#   -> 90 clips fanned across 3 GPUs; all land in your local output
```

Dynamic dispatch (each GPU gets its next job the moment it's free), unreachable
boxes skipped, in-flight jobs requeued if a box drops. See pixelmon's
[README-RENDER-FARM.md](https://github.com/grymmjack/pixelmon) for LAN/firewall
setup — it applies verbatim.

> **Each farm box needs the audio models.** A box provisioned for pixelmon has
> ComfyUI but not Stable Audio Open. On each one: `./download-models.sh`, link
> `custom_nodes/retro_sfx`, restart ComfyUI.

---

## How it works

```
description ──► ComfyUI API
                 CheckpointLoaderSimple (Stable Audio Open) ─┐
                 CLIPLoader (T5, type=stable_audio) ─► CLIPTextEncode ×2
                   └─ ConditioningStableAudio ─► KSampler ─► VAEDecodeAudio
                        └─► RetroSFX ─► SaveSFX (WAV)
```

The custom **`RetroSFX`** node is the finishing pass that makes output an actual
asset, exactly as `PixelArtPalette` did for pixelmon:

1. **trim** — drop leading/trailing near-silence. Stable Audio pads short SFX out
   to the requested duration, so nearly every render needs this.
2. **format lock** (optional) — force the audio through a hardware ceiling.
3. **fade** — a few ms of ramp on both ends so trimming at a non-zero crossing
   doesn't click.
4. **normalize** — peak-normalize so every generated SFX sits at a predictable
   level.

**`SaveSFX`** then writes a real **WAV** at your chosen bit depth. ComfyUI core
only ships flac/mp3/opus savers, none of which a game engine, tracker, or
DOS-era toolchain will load.

### The format lock — a palette, one dimension over

pixelmon's insight was that a **palette** (a fixed set of colors) is what makes
an image read as a sprite. The audio twin of a palette is **sample rate + bit
depth + channel count**. Amiga MOD doesn't sound old because of the notes — it
sounds old because Paula was 8-bit at ~8 kHz mono. That ceiling *is* a palette.

`--list-formats` shows the registry in `custom_nodes/retro_sfx/nodes.py`
(`amiga`, `sb`, `sb22`, `gameboy`, `nes`, `snes`, `psx`, `cd`); add your own there.
Default is `none` — clean 44.1 kHz stereo that works anywhere.

```bash
soundmon "a laser blast" --format sb --bits 8      # SoundBlaster-era crunch
soundmon "a coin pickup" --format gameboy --bits 8 # DMG 4-bit
```

> ⚠ **`_format_lock()` is currently unimplemented** — see the TODO in
> `custom_nodes/retro_sfx/nodes.py`. Everything at the default `--format none`
> works fully; only the retro formats are gated on it.

---

## Lessons learned (the gotchas)

These cost real time; they're why the setup looks the way it does.

1. **The checkpoint does NOT contain the text encoder.** Stable Audio Open's
   `model.safetensors` has the DiT (374 tensors), the VAE (365), and the two
   seconds-embedders — and **zero T5 tensors**. So `CheckpointLoaderSimple`'s
   CLIP output (slot 1) is unusable; you must load T5 separately with
   `CLIPLoader` + `type=stable_audio`. Wiring slot 1 is the #1 confusing failure.
2. **Normalize LAST.** Stable Audio frequently starts at full amplitude — the
   loudest sample of a render is often *sample 0*. A de-click fade therefore
   lands right on the peak. Normalizing before fading silently undoes the
   normalization (measured −5.94 dBFS on a −1.0 dBFS request). Any gain-changing
   stage after normalize invalidates it.
3. **`--seconds` is conditioning, not just duration.** The model was trained with
   `seconds_total` as an input, so changing it changes *what* gets generated, not
   just how much. A 4-second request is not a 47-second request truncated.
4. **The model repo is gated.** Unlike everything pixelmon downloads, the
   official Stability repo needs a token + license click. Hence the mirror.
5. **8-bit WAV is unsigned.** It's the one PCM width that breaks the signed
   pattern (offset 128). Getting it wrong yields loud static, not quiet audio.
6. **Negative prompts matter more here than in images.** Without pushing *music,
   melody, speech, voice* into the negative, an SFX request drifts into a little
   musical phrase surprisingly often.

---

## Performance

Measured on an **AMD RX 6600** (RDNA2, 8 GB, ROCm, `--lowvram`), 4-second clips:

| Mode | steps | per clip |
|---|---|---|
| default | 50 | ~6.4 s |
| `--fast` | 16 | ~4 s |

Audio is dramatically cheaper than SDXL — a 4-second mono-ish latent is far
smaller than a 1024×1024 image. Generation is cheap enough that the intended
workflow is *make 8 and keep one*, which is exactly what the cloud tools charge
you per-generation for.

---

## Repo layout

```
soundmon/
├── README.md
├── install.sh                 links files; reuses pixelmon's ComfyUI if present
├── download-models.sh         fetch Stable Audio Open 1.0 + T5 (ungated mirror)
├── soundmon.py                the CLI brains (talks to ComfyUI's API)
├── sounds.json                --style guide snippets (edit / add your own)
├── servers.example.json       template for --server aliases (copy to servers.json)
├── bin/soundmon               wrapper: ensures the server is up, then runs soundmon.py
└── custom_nodes/
    └── retro_sfx/             the finishing node (trim→format→fade→normalize) + WAV saver
        └── nodes.py           FORMATS registry — add your own hardware formats here
```

---

## Credits & licenses

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — the engine (GPL-3.0)
- [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) — Stability AI (Stability AI Community License). Trained on Freesound + Free Music Archive.
- [T5](https://huggingface.co/google-t5/t5-base) — Google, the text encoder

The code in this repo (the CLI, wrapper, installer, and custom node) is released
under the MIT License.

> **Licensing note.** Stable Audio Open ships under the *Stability AI Community
> License*, not a plain OSS license — free for research and for commercial use
> under a revenue threshold, with terms above it. Read it before shipping
> generated audio in a commercial game.
