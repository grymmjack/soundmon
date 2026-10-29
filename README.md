# soundmon — text-to-audio generator: SFX, music loops, songs, and narration

Describe audio, get a WAV — with **one command**, entirely on your own machines.
Behind one CLI: **Stable Audio 3** for sound effects and music, **ACE-Step 1.5**
for full songs with sung vocals, **Kokoro** for spoken narration, a recording
booth for when you would rather narrate it yourself — and, where a model is the
wrong tool, **real synthesis**: a NES 2A03, the **Nuked OPL3** core DOSBox uses,
and sfxr-style effect recipes, none of which need a model or a GPU at all.
Shares pixelmon's ComfyUI engine, render farm, and CLI conventions.

```bash
soundmon "a heavy wooden door creaking open"              # a sound effect
soundmon "a lo-fi piano loop" --music --seconds 12        # a musical loop
soundmon "dark fantasy, choir" --song --bpm 90 \
         --key "D minor" --lyrics-file ballad.txt         # a full song, sung
soundmon --narrate-file lines.txt --voice bm_george       # a spoken script
soundmon --record-file lines.txt                          # ...or read it yourself
soundmon "dungeon level 3" --chip --key "A minor" --bpm 125   # a REAL chiptune
soundmon "dungeon level 3" --opl --key "D minor" --bpm 110    # a REAL OPL3 track
soundmon "heavy door" --chipfx --name door                    # a REAL 8-bit SFX
soundmon --batch "door,glass,fire" -n 16 --server rtx,local,mac
```

This is the local, free, unlimited answer to cloud generators like ElevenLabs
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
be trained on the thing you want, and it has to be run the way it was trained. A
music model asked for "a door creak" gives you a song *about* a door — and the
right model run at the wrong precision gives you static. Both failure modes cost
a full day here; see [Lessons learned](#lessons-learned-the-gotchas).

### Which engine makes what

| You want | Flag | Engine | Notes |
|---|---|---|---|
| a sound effect | *(default)* | Stable Audio 3 medium | 44.1 kHz, full-band |
| a music loop / bed / sting | `--music` | Stable Audio 3 medium | same checkpoint, different conditioning |
| a full song with sung vocals | `--song` | ACE-Step 1.5 turbo | lyrics, BPM, key; 48 kHz |
| a spoken line | `--narrate` | Kokoro | CPU, in-process, no ComfyUI |
| a spoken line **in your voice** | `--record` | your microphone | no models at all |
| JRPG text-box blips | `--blip` | an oscillator | no model at all |
| **authentic** NES chiptune | `--chip` | a 2A03 emulation | no model at all |
| **authentic** AdLib / OPL3 FM | `--opl` | Nuked OPL3 (DOSBox's core) | no model at all |
| **authentic** 8-bit sound effects | `--chipfx` / `--oplfx` | PSG or OPL3 | no model at all |

`--engine sao` falls back to the original **Stable Audio Open 1.0** if you want
it, but SA3 supersedes it for both SFX and music.

### It shares everything with pixelmon

soundmon deliberately reuses pixelmon's engine rather than duplicating it:

| Shared | Not shared |
|---|---|
| `~/ComfyUI` + its venv + torch | the models (audio, ~14 GB for SA3) |
| `~/launch-comfyui.sh` (GPU autodetect) | the custom node (`retro_sfx`) |
| port 8188, `servers.json`, the whole farm | the workflow graph |

Any box already running ComfyUI for pixelmon becomes a soundmon box by running
`./download-models.sh` and linking the node. `install.sh` copies pixelmon's
`servers.json` for you.

---

## Install

```bash
git clone <your-remote> ~/git/soundmon && cd ~/git/soundmon
./install.sh                  # links files; reuses an existing ComfyUI if present
./download-models.sh --sa3    # the default engine (~14 GB, ungated)
# restart ComfyUI so it picks up the retro_sfx node:  ~/launch-comfyui.sh --force-fp32
soundmon "a heavy wooden door creaking open"
soundmon --version            # soundmon 0.1.0
```

If you don't have ComfyUI yet, `install.sh` will tell you to run pixelmon's
installer first — it builds exactly the engine soundmon needs (ComfyUI + venv +
the right torch for your GPU). `install.sh` also links `~/launch-comfyui.sh`
(the script `bin/soundmon` starts ComfyUI with) **only if there isn't one** —
soundmon ships the same script pixelmon does, so it never replaces pixelmon's.

`download-models.sh` fetches into `~/ComfyUI/models/` (or `$COMFYUI_DIR/models`).
Flags combine (`--sa3 --song`); files you have are skipped and a cut-off download
resumes. It ends with what is still missing and exits non-zero if anything is:

| Flag | Fetches | Size |
|---|---|---|
| `--sa3` | **Stable Audio 3** medium + specialists + T5Gemma + **Qwen 3.5 rewriter** | ~14 GB |
| `--song` | ACE-Step 1.5 turbo (all-in-one) | ~9.3 GB |
| `--sao` | Stable Audio Open 1.0 + T5-base — only needed for `--engine sao` | ~5.3 GB |
| `--opl` | builds the Nuked OPL3 core + GM banks for `--opl`/`--oplfx` (needs a C compiler) | tiny |
| `--all` | `--sa3 --song --sao` | ~29 GB |
| `--list` | what's installed and what's missing — downloads nothing | — |
| *(none)* | the legacy `--sao` set, as it always did — new boxes want `--sa3` | ~5.3 GB |

> **The Qwen rewriter is not optional.** ComfyUI's official Stable Audio 3
> workflow runs an LLM over your description *before* the audio model ever sees
> it, with different system prompts per category. `--sa3` therefore pulls it
> automatically. Skipping this stage is what produced a batch of technically
> valid, musically worthless audio here — see gotcha 10.

> **Gating note.** `stabilityai/stable-audio-open-1.0` is a **gated** repo — it
> 401s without an HF token and a license click. `download-models.sh` pulls from
> an ungated mirror so setup stays one-command. Use the official repo with
> `HF_TOKEN=hf_xxx ./download-models.sh --official`. The Stable Audio 3 and
> ACE-Step repackages are **ungated** — no token needed.

### ⚠ Stable Audio 3 requires `--force-fp32`

ComfyUI defaults audio models to fp16, and `StableAudio3` declares no
`supported_inference_dtypes` to override that. **In fp16 the model outputs
noise** — buzzing, radio-scramble, no usable audio. Start ComfyUI like this:

```bash
~/launch-comfyui.sh --force-fp32
```

`bin/soundmon` passes it automatically when it starts the server itself. If
ComfyUI is **already running** (e.g. for pixelmon's SDXL), restart it with the
flag before generating audio. It roughly doubles VRAM and halves throughput,
which is why it isn't simply always on. soundmon reads the server's launch
arguments (`/system_stats` → `system.argv`) before an SA3 render and prints a
`⚠ … started WITHOUT --force-fp32` warning for any box that lacks it (it still
renders: the warning is the point).

---

## Usage

Run `soundmon --help` for the full, colorized list. The essentials:

| Flag | What it does | Default |
|---|---|---|
| `-n, --number N` | how many to make, each a different seed | `1` |
| `--seconds N` | length. **Conditions the model** — it shapes the sound, not just its length | `10` |
| `--engine NAME` | `sa3` (Stable Audio 3) or `sao` (Stable Audio Open 1.0) | `sa3` |
| `--sa3-size NAME` | `medium` / `small` — only medium is verified to sound right | `medium` |
| `--no-reprompt` | skip the Qwen rewrite stage — **usually a mistake**, see gotcha 10 | off |
| `--style NAMES` | append proven prompt guide(s), comma-separated — `--list-styles` | — |
| `--format NAME` | hardware format lock (rate + bit depth + channels) — `--list-formats` | `none` |
| `--bits N` | WAV bit depth: 8 / 16 / 24 | `16` |
| `--batch "a,b,c"` | round-robin subjects, one of each per pass, each into its own folder | — |
| `--batch-file F` | every `key \| prompt` / `key \| seconds \| prompt` row → `key.wav` (see below) | — |
| `--skip-existing` | with `--batch-file`: skip keys that already have a file (resume a pack) | off |
| `--show-prompt` | print the final positive/negative prompt before rendering | off |
| `--exact-prompt` | send the text as typed: no tail, no default negative, no Qwen rewrite | off |
| `--dry-run` | print the ComfyUI graph(s) as JSON and exit — no server contacted | off |
| `--no-recipe` | don't embed the recipe in the output files (see below) | off |
| `--fast` | fewer steps on `--engine sao` and `--song`. **No effect on SA3**, which already renders in 8 steps | off |
| `--seed N` | lock / repeat a result | random |
| `--no-trim` | keep the model's leading/trailing silence. **Not** for loops — use `--loop` | off |
| `--max-seconds N` | hard length cap (0 = none). Truncates — trimming only removes *silence* | `0` |
| `--normalize-db N` | peak ceiling after generation | `-1.0` |
| `--lufs N` | EBU R128 loudness ceiling (`off` to disable) | `-16` |
| `--true-peak N` | true-peak ceiling in dBTP | `-1.0` |
| `--fade-ms N` | de-click fade on both ends (`--loop` forces this to 0) | `5` |
| `--ogg` | compress the result to OGG Vorbis (~25× smaller). **Optional, off by default** | off |
| `--ogg-quality N` | OGG quality 0–10 | `8` |
| `--flac` / `--mp3` / `--opus` | save compressed instead of WAV | WAV |
| `--loop` | crossfade the tail over the head so the track loops seamlessly | off |
| `--loop-crossfade N` | crossfade length in seconds for `--loop` | `2.0` |
| `--server NAME[,...]` | remote ComfyUI; comma-list = render farm. Also `$SOUNDMON_SERVER`, then servers.json `"_default"` | local |
| `--version` | print the version | — |

`--help` lists every flag. The seed is in every filename, so a take you liked is
one flag away:

```bash
soundmon "a laser blast" -n 8            # prints e.g. seed=12345 for each
soundmon "a laser blast" --seed 12345    # that blast again
```

**Every file says how it was made.** The prompt, the final prompt with its
style tail, the negative, seed, engine and model, steps/cfg/sampler, `--ref`
and its strength, the server, and the full command line are embedded as a
comment tag — a RIFF `LIST/INFO ICMT` chunk in WAV (samples and header
untouched), a Vorbis comment / ID3 tag in FLAC, OGG, Opus and MP3. The CPU
engines record their engine and command. It survives renaming and copying into a
game folder:

```bash
ffprobe -v error -show_entries format_tags=comment -of default=nw=1:nk=1 door_s42_00001_.wav
# {"tool":"soundmon","version":"0.1.0","engine":"sa3",...,"seed":42,...,"command":"soundmon 'a door' --seed 42"}
```

`--no-recipe` leaves it out (shipped assets that shouldn't carry prompts or paths).

### A pack from a manifest (`--batch-file`)

`--batch` splits on commas, and real descriptions contain commas. A manifest
doesn't care — the same `key | text` shape `--narrate-file` and `--blip-file`
read, so one game string table can drive narration, blips and effects:

```text
# sfx.txt
door_open  | a heavy wooden door, creaking open
explosion  | 3.5 | a distant explosion, rumbling
glass breaking
```

```bash
soundmon --batch-file sfx.txt --output-to pack --create-dirs     # pack/door_open.wav, pack/explosion.wav, …
soundmon --batch-file sfx.txt --output-to pack --skip-existing   # only what's missing
soundmon --batch-file sfx.txt -n 3 --server rtx,titan            # key_01..key_03 each, across the farm
```

A middle column sets that row's length; a plain line names itself. With `--ref`
the reference sets the length, so the column is ignored.

### Stable Audio 3 (the default engine)

SA3 replaced Stable Audio Open as the default for both SFX and music. It is
full-band — measured cutoff at Nyquist (22.1 kHz) on raw WAV output, versus
ACE-Step's hard 16 kHz VAE ceiling — and it is markedly better at music.

**It only works run the way it was trained.** ComfyUI's official SA3 workflow is
not a suggestion; every parameter below was verified by ear, and every wrong
value produced unusable audio rather than merely worse audio:

| Setting | Value | What the wrong value sounds like |
|---|---|---|
| precision | **fp32** (`--force-fp32`) | buzzing / radio static |
| checkpoint | **medium** | scrambled, "record scratching" |
| sampler | **lcm** | noise |
| scheduler | **simple** | — |
| steps | **8** | 50 steps gives clacking and popping |
| cfg | **1.0** | 7.0 gives noise |
| prompt | **Qwen-rewritten** | valid audio, poor musical content |

soundmon applies all of these automatically when `--engine sa3` is active, which
it is by default. The one you must supply yourself is `--force-fp32` on the
ComfyUI server, because that is a server-launch flag, not a graph parameter.

**The rewrite stage.** Your description goes to Qwen 3.5 first, under one of four
system prompts (Music / Instrument / SFX / One-shot, verbatim from the official
workflow, in `sa3_reprompt.json`), and only the rewritten text reaches the audio
model. `--no-reprompt` skips it. Skipping is defensible in exactly one case:
you are batching and have pre-computed the rewrites yourself, so the LLM does
not have to be swapped in and out of VRAM per job.

> **Why you would pre-compute.** On an 8 GB card, SA3 medium in fp32 (~8.6 GB)
> and Qwen (~4.5 GB) do not co-reside, so ComfyUI evicts and reloads a model on
> *every* job — measured 75.6 s per asset, nearly all of it moving weights across
> PCIe. Rewriting a whole batch first, then generating with `--no-reprompt`,
> turns hundreds of model swaps into two model loads. Same output, ~20× faster.

### Seamless loops (`--loop`)

Music that plays continuously has to survive its last sample running into its
first, and generated music does not — for a reason that has nothing to do with
this tool. **The model composes an ending.** Asked for 60 seconds it writes a
60-second *piece of music*, with a decay, because that is what its training data
does. Measured on raw output, the final two seconds fall ~40 dB.

```bash
soundmon "brooding dungeon bed" --music --seconds 60 --loop
```

`--loop` crossfades the track's tail back over its head, so the seam is
contiguous **by construction** rather than by luck:

```
    O[i]   = S[i]                            for i in [X, T)
    O[0:X] = S[0:X]·fade_in + S[T:L]·fade_out
```

Play `O[T-1]` into `O[0]` and you get `S[T-1]` into `S[T]` — adjacent samples of
the original take. The composed ending is still there; it now sits underneath
the opening instead of on top of a hard cut. Measured on a real 39 s track:

| | tail vs body |
|---|---|
| before | **−39.4 dB** |
| after `--loop` | **+0.7 dB** |

Costs `--loop-crossfade` seconds of length (default 2.0), so a 60 s render
becomes a 58 s loop — ask for the longer number if the exact duration matters.

> **The intuitive fix is the wrong one, and it is worse than doing nothing.**
> `--no-trim --fade-ms 0` looks right — stop the post-processing from touching
> the endpoints — and it makes the seam *deeper*, because silence-trimming was
> quietly eating most of the composed fade-out. Measured across one pack, same
> models, same prompts: **−30.4 dB tail with trim on, −66.6 dB with it off.**
> `--loop` therefore forces `trim` on and `fade_ms` to 0 itself; you cannot
> misconfigure it from the outside.

Leave `--lufs` and `--normalize-db` on. They scale the whole file uniformly and
so cannot introduce a seam.

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

### Music (`--music`)

Yes — and with Stable Audio 3 this got a lot better. `--music` makes **loops,
beds, stings, riffs, drum grooves and full instrumental cues**. What it does not
do is **sung vocals**; that's `--song` and a different engine.

```bash
soundmon "a lo-fi hip hop piano loop" --music --seconds 12
soundmon "a driving rock drum beat" --music --style drums
soundmon "a boss battle theme" --music --style chipmusic
soundmon "a village theme" --music --style gbmusic          # Game Boy DMG
soundmon "a driving stage theme" --music --style fmsynth    # Genesis / YM2612
soundmon "a demoscene intro loop" --music --style tracker   # Amiga MOD
```

`--music` matters more than it looks: the **default negative prompt actively
fights music.** For SFX the default pushes *music, melody, song, voice, vocals*
away, because an SFX request drifts into a little musical phrase surprisingly
often. That same negative sabotages a music request. `--music` swaps in
`MUSIC_NEGATIVE` and drops the "sound effect" tail from the prompt. Music style
guides: `musicloop`, `sting`, `drums`, `cinematic`, and the chip family
below (`chipmusic`, `gbmusic`, `fmsynth`, `tracker`).

### Full songs (`--song`)

`--song` swaps in a second engine — **ACE-Step 1.5** — for real songs with sung
vocals, lyrics, tempo and key. Fetch it once with `./download-models.sh --song`
(~9.3 GB), then:

```bash
soundmon "dark fantasy, epic metal, male choir, orchestral" --song \
  --seconds 45 --bpm 100 --key "E minor" --lyrics-file ballad.txt
soundmon "synthwave, driving, retro" --song --seconds 120 --bpm 128 --key "A minor"
```

In song mode the description becomes the **genre tags**, not a sentence — ACE
takes tags and lyrics as separate conditioning. Musical parameters are
first-class:

| Flag | What | Default |
|---|---|---|
| `--lyrics` / `--lyrics-file` | sung lyrics; `[verse]` / `[chorus]` markers work | instrumental |
| `--bpm` | tempo, 10–300 | 120 |
| `--key` | 34 keys (`--list-keys`) | C minor |
| `--timesig` | 2 / 3 / 4 / 6 | 4 |
| `--lang` | 51 languages | en |
| `--seconds` | up to **1000** (~16 min) | 120 |
| `--no-audio-codes` | skip the quality LLM pass — much faster | off |

Output is **48 kHz** stereo here (ACE's native rate), not 44.1 kHz.

> This is the same trick pixelmon's `--animate` uses: an entirely different
> pipeline behind one CLI. `_song_nodes()` lands its AUDIO on the same graph
> node the SFX path does, so trimming, normalizing, format-locking and saving
> are shared verbatim.

**Speed.** ACE-Step is a 3.5 B model — far heavier than Stable Audio's SFX path.
On the RX 6600 (8 GB, ROCm, `--lowvram`): ~190 s for a 30 s instrumental, ~250 s
for 45 s with vocals. A 12 GB+ NVIDIA card is much happier. `--fast` (8 steps)
and `--no-audio-codes` both cut it substantially.

> **Farm boxes need the song model separately** — `./download-models.sh --song`
> on each one. A box with only the SFX models will fail `--song` jobs.

### Narration (`--narrate`)

A third engine, for spoken lines — a dungeon master, menu readouts, item
descriptions. Uses **Kokoro** (82 M params, Apache-2.0), which ships four
British male voices among others:

```bash
pip install kokoro soundfile        # one-time, into ComfyUI's venv
soundmon "The crypt door grinds open." --narrate --voice bm_george --pitch -3
soundmon --narrate-file lines.txt --voice bm_george --pitch -3 --speed 0.92
```

| Flag | What | Default |
|---|---|---|
| `--narrate-file` | narrate every line of a file, one WAV each | — |
| `--voice` | see `--list-voices` (`bm_george`, `bm_lewis`, …) | `bm_george` |
| `--pitch` | semitones; **negative = deeper**, duration preserved | 0 |
| `--speed` | speech rate | 1.0 |

A booming DM is `--voice bm_george --pitch -3 --speed 0.9` (measured: drops the
fundamental from ~107 Hz to ~85 Hz). Kokoro has no pitch control, so `--pitch`
gets it by asking Kokoro to speak *faster* by the same factor and then
resampling back — the duration change cancels and only the pitch move survives.

**`--narrate-file` reads `key | text` rows** (and skips `#` comments), so game
string tables work directly and the key becomes the filename.

> **This engine deliberately skips ComfyUI.** SFX and songs are diffusion
> problems where ComfyUI earns its keep — model loading, VRAM, the farm. Kokoro
> is an 82 M feed-forward model that speaks a line in about a second; wrapping
> it in a node graph and fanning it across four GPUs would *add* latency. So it
> runs in-process and writes through the same output/`--batch` path: one
> interface, the right mechanism per engine.

> **It runs on CPU on purpose.** On the RX 6600 (gfx1032 masquerading as gfx1030
> via `HSA_OVERRIDE_GFX_VERSION`) Kokoro dies with `HIP error: invalid device
> function`. At 82 M params there was nothing to win on the GPU anyway.

### Real chip music (`--chip`, `--opl`)

**The most important thing in this README, if you want retro music.** Asked for
"chiptune", Stable Audio 3 gives you modern chiptune-*influenced* music: made in
a DAW, with reverb, sampled drums and unlimited polyphony. That genre exists and
the model renders it well. It is simply not what a 2A03 sounds like.

Real chip music is not a genre a model imitates — it is a **constraint**:

```
NES 2A03    2 pulse channels (duty 12.5/25/50/75%), 1 triangle, 1 noise
OPL3        FM synthesis: operator ratios, envelopes, feedback
both        no reverb, nothing sampled, a handful of voices, forever
```

No prompt adds a channel limit a model doesn't have. So these two modes
**synthesize the audio directly** — authentic by construction, because there is
no code path that produces reverb or a sampled drum.

```bash
soundmon "dungeon level 3" --chip --key "A minor" --bpm 125 --seconds 60
soundmon "dungeon level 3" --opl  --key "D minor" --bpm 110 --seconds 60
soundmon "boss theme" --chip -n 20        # 20 takes, keep one. Costs nothing.
```

| Mode | Chip | Engine |
|---|---|---|
| `--chip` | NES 2A03 PSG | pure Python oscillators — **nothing to install** |
| `--opl` | Yamaha YMF262 (AdLib / SB Pro 2 / SB16) | **Nuked OPL3**, the cycle-accurate core DOSBox uses |

`--opl` needs its core built once (~200 KB of C, no build system):

```bash
./download-models.sh --opl      # fetches nukeykt/Nuked-OPL3, compiles libopl3
```

> **Why fetched and not committed.** Nuked OPL3 is LGPL-2.1 and soundmon is MIT.
> Building it locally keeps the licences separate and attributed rather than
> quietly relicensing someone's work — same reason the models aren't in git.

**They share a composer, not a synthesizer.** `chip.compose()` produces note
events — bars, chords, a motif, drum hits — with no idea how they'll be voiced.
Both renderers consume the same events. The musical decisions are hard and worth
writing once; "what does a voice sound like" is exactly what differs between a
2A03 and a YMF262. Adding a SID or a YM2612 means writing a renderer, not
another composer.

| Flag | What | Default |
|---|---|---|
| `--key` | e.g. `"D minor"`, `"f# dorian"` — also picks the progression | `C minor` |
| `--bpm` | tempo | `120` |
| `--seconds` | length; rounded to whole bars | `60` |
| `--mood NAME` | **the big one** — see below. `auto` reads it from your description | `auto` |
| `--scale` | override the scale the mood/key would pick | from mood |
| `--arp` | arpeggio speed in 16ths — `1` is the classic buzz, `0` = mood decides | `0` |
| `--opl-bank` | external patch bank (`.sbi`); omit for the built-in | built-in |
| `--opl-lead` / `--opl-arp` / `--opl-bass` | per-voice instrument | brass / organ / bass |

### Mood is the main control (`--mood`)

A mood is a **bundle** of composition parameters, not a label — and that is the
whole point. Changing the scale alone does almost nothing; you get minor-key
cheerfulness, which is the classic failure of procedural music. `ominous` is a
dark scale **and** a slow tempo **and** a thin duty cycle **and** sparse drums
**and** a falling melodic contour, all at once.

```bash
soundmon "the crypt"    --chip --mood solemn
soundmon "boss fight"   --chip --mood frantic
soundmon "found a door" --opl  --mood eerie
soundmon --list-moods
```

| | |
|---|---|
| bright | `heroic` `triumphant` `playful` `wondrous` `serene` |
| dark | `ominous` `eerie` `melancholy` `solemn` `tense` |
| kinetic | `driving` `frantic` `grand` `mysterious` |

**`auto` is the default, and it reads your description.** Mood words are already
in the text you wrote — so `"the TORTURE CHAMBER -- dried blood and remembered
pain"` infers `eerie`, and `"an intense, fast combat loop"` infers `frantic`,
with nothing extra to specify. Run against a real 24-track manifest it picked 12
distinct moods with no hand-tuning.

Inference is not always right — *"a memorial to triumphant champions"* reads as
`triumphant` when it wants to be `solemn`. That is what explicit `--mood` is for.

**Every parameter it touches:** scale, chord progression, tempo multiplier,
rhythm density, arpeggio speed, duty cycle, drum pattern, lead register, melodic
contour, vibrato depth, and how far the motif roams from the chord. The table is
at the top of `chip.py` and is meant to be edited — add your own mood in six
lines.

**Arpeggios are the signature.** With one note per channel you cannot play a
chord, so chip music fakes them by cycling chord tones every frame or two —
fast enough that the ear fuses it into a buzzy, shimmering chord. That fast
arpeggio *is* the sound of the format, more than the square waves are.

**The OPL patch bank is the sound.** An instrument is ~11 bytes of operator
config, and the whole difference between authentic AdLib and awful lives in
those bytes. The built-in bank is hand-authored so this works with nothing to
fetch; `--opl-bank foo.sbi` loads a specific game's voices if you have them.

> **Both loop seamlessly with no crossfade.** Music is composed in whole bars,
> so the last sample runs into the first by construction. `--loop` exists to hide
> a *composed ending* — these don't have one, so `--loop` is ignored (with a
> notice) rather than shortening the track for nothing.

### Synthesized sound effects (`--chipfx`, `--oplfx`)

The same argument as `--chip`, and it applies harder to short effects. A 0.12 s
footstep is 5,300 samples; a diffusion model asked for one gives you a fragment
of a *recording* of a footstep, with a room around it. A 2A03 footstep is a 30 ms
burst of LFSR noise with a fast decay, and there is nothing to record.

```bash
soundmon "heavy door" --chipfx --name door --max-seconds 0.4     # PSG
soundmon "heavy door" --oplfx  --name door --max-seconds 0.4     # FM
soundmon "boom" --chipfx --fx-archetype boom -n 12               # 12 takes, free
```

Every effect is a **recipe** — a pitch contour, a waveform, an envelope — which
is how sfxr/bfxr and the era's actual sound designers worked. There are ~30
archetypes (`click`, `creak`, `boom`, `coin`, `hit`, `growl`, `rattle`, …) and the
archetype is inferred from the asset name, so `door` becomes a creak and `boom`
becomes an explosion with no prompt involved.

| | |
|---|---|
| `--chipfx` | PSG: pulse / triangle / 15-bit LFSR noise — nothing to install |
| `--oplfx` | the same recipes rendered through the Nuked OPL3 FM core |
| `--fx-archetype NAME` | force the archetype instead of inferring it |

**The recipes are the sound design, and they're meant to be edited.** Don't like
`hit`? It's six numbers in `render_psg()`. That's a faster iteration loop than any
prompt, and it's deterministic — the seed is in the filename.

> **FM has no noise channel.** Outside fixed rhythm mode the OPL can't make
> noise, so `--oplfx` fakes it with an extreme modulator ratio plus high
> feedback — an inharmonic wash. That isn't a workaround, it's the period-correct
> technique; it's how AdLib-only games made explosions.

> **Inference is the fallback, not the mechanism.** Archetypes come from an
> explicit name→archetype table. Keyword matching is only used for names not in
> it, because substring matching on short words is a trap: **"dice" contains
> "ice"**, so `diceroll` classified as a frost shimmer until the table existed.
> When the asset list is finite and known, a table beats inference.

### Blippy JRPG narration (`--blip`)

The voice in an Undertale or Animal Crossing text box is not processed speech —
it is a short instrument note fired once per character as the text types. You
"hear" the words because your eye is reading them at the same rate your ear is
getting blips. `--blip` does both traditions:

```bash
soundmon --blip --blip-file lines.txt --output-to assets/narration/blips
soundmon "You found a rusted key!" --blip --blip-wave triangle --blip-pitch -5
soundmon --blip --blip-style voice --blip-file lines.txt --voice bm_george
```

| Style | What it is | Needs |
|---|---|---|
| `synth` *(default)* | one oscillator blip per character — Undertale, Earthbound, Zelda | **nothing** |
| `voice` | Kokoro speech resampled faster, i.e. "Animalese" — Animal Crossing | Kokoro |

| Flag | What | Default |
|---|---|---|
| `--blip-file` | blip every line of a file — the same `key \| text` rows `--narrate-file` reads | — |
| `--blip-style` | `synth` / `voice` | `synth` |
| `--blip-wave` | `square` / `triangle` / `sine` / `saw` / `noise` | `square` |
| `--blip-rate` | characters per second — the text-box typing speed | `14` |
| `--blip-pitch` | base pitch in semitones from A4. Lower = bigger character | `0` |
| `--blip-jitter` | per-character pitch spread. **`0` = flat, authentic Undertale** | `1.5` |
| `--blip-duty` | fraction of each character slot that sounds | `0.55` |
| `--blip-speed` | playback speed-up for `--blip-style voice` | `3.0` |

**Pitch is deterministic per character, not random.** A random offset per blip
sounds like a machine malfunctioning; the same word producing the same little
melody every time is what makes it read as a *voice saying that word*. Vowels
sit ~2.6 semitones above consonants, a crude nod to how vowels carry the
pitched part of real speech. Punctuation drives the rhythm — a comma pauses two
character-slots, a period four — so the timing comes out of your text for free.

> **`--blip-style synth` needs no model, no GPU and no ComfyUI.** It is an
> oscillator and an envelope, so it renders a seventy-line script in well under
> a second on any machine — including one that has never downloaded a checkpoint.
> That also makes it the one engine here you can run on a CI box.

To change the *character* of the voice, change `_char_semitones()` in `blip.py`.
It is the whole personality: `return 0.0` is authentic Undertale, a wider vowel
lift makes it sing, a larger `--blip-jitter` makes it chatter.

### Recording it yourself (`--record`)

Sometimes you want *your* voice, not a model's. `--record` is a recording booth
in the terminal: it reads **the same `key | text` manifest `--narrate` does**,
shows you a line, records you reading it, and writes the file under the same
name a generated line would have used.

```bash
soundmon --list-devices                              # which mics can I see?
soundmon --record-file lines.txt --ogg \
         --output-to ~/game/assets/narration/me --create-dirs
```

```
soundmon record  ·  line 12/71  ·  3 done  ·  ~/game/assets/narration/me
────────────────────────────────────────────────────────────────────────
  key  intro.descent

  Beneath the hills, the old dungeon waits. Nine levels down, and every
  one of them hungry. Others have gone before you. None of them came
  back up. Take your steel, take your nerve, and begin the descent.

  takes  1  ▸2
  take 2  14.28s  peak  -6.2 dBFS
  ▁▁▃▇█▇▅▃▂▁▁▂▅▇█▆▃▁▁▂▆█▇▄▂▁▁▃▇█▆▃▂▁▁▂▅▇█▅▃▁▁▁▂▄▇█▆▄▂▁▁
────────────────────────────────────────────────────────────────────────
  R record   P play   A cycle take   ENTER accept + next   S skip
  B back     D delete take   Q quit (progress is on disk — just rerun)
```

| Key | What |
|---|---|
| `R` | start recording (a live VU meter runs while you read) |
| `SPACE` | stop — `ESC` instead throws the take away |
| `P` | play the selected take back |
| `A` | cycle through takes of this line |
| `D` | delete the selected take |
| `ENTER` | master the selected take, write the pack file, move to the next line |
| `S` / `B` | skip forward / step back |
| `Q` | quit — **progress lives on disk, so rerunning resumes** |

| Flag | What | Default |
|---|---|---|
| `--record-file` | record every `key \| text` row of a file | — |
| `--device` | which microphone — see `--list-devices` | system default |
| `--record-rate` | capture sample rate | 48000 |
| `--threshold-db` | silence floor used to trim each take | -45 |
| `--fade-ms` | de-click fade on both ends | 5 |

**Takes are kept, and the pick becomes the pack file.** Every `R` writes
`takes/<key>.takeNN.wav`; `ENTER` masters whichever take is selected out to
`<key>.ogg`. Record a line three times, `A` between them, `P` to compare, then
accept the good one. Sweeping the rejects afterwards is one `rm -r takes/`.

**Each accepted take goes through the same mastering the generated packs get** —
leading and trailing silence trimmed (that's the gap between hitting record and
actually speaking), a 5 ms de-click fade, peak normalize, then the `--lufs`
loudness ceiling and `--ogg` compression. That is what makes a hand-voiced pack
and a generated one interchangeable *per line*: record the ten lines you care
about most, generate the other sixty, and nothing audibly switches between them.

> **Recording works on Windows, macOS and Linux**, but each needs a different
> capture backend — DirectShow, AVFoundation and PipeWire respectively. Windows
> and macOS need `ffmpeg` on PATH; Linux uses `pw-record` (or falls back to
> ffmpeg/ALSA/`arecord`). Nothing else is required — in particular `--record`
> does **not** need the Kokoro TTS stack installed, so the machine with your
> good microphone does not have to be the machine with the GPU.

### Mastering a set (`--master`)

Each sound can be fine alone and still grate when a fight stacks twenty of them.
`--master` gives every file its own amounts of one chain (it began as NEON-CASTER's
`sfx/mastering/`):
- tilt toward −3 dB/octave;
- lift the body at 180 Hz where there is any;
- cut 3.2 kHz when the 2.5–5 kHz presence band piles up;
- de-ess sharp sibilants;
- low-pass 12 kHz for effects;
- bring each sound back to its own loudest 400 ms, never past −1 dBTP.

It uses no limiter and never normalizes to full scale. Voices are left alone unless
they're bright.

**A sound comes out as loud as it went in, less harsh.** Only a sound with over 40 % of
its energy above 10 kHz (white noise, a whistle) may come out quieter, by at most 6 dB
(`BRIGHT_DROP` in master.py): some of the level the EQ and low-pass took away is the
treatment. White noise goes from −12.7 to −18.7 LUFS at its loudest 400 ms, and the line
says `(bright: up to 6 dB quieter, by design)`. A sound can also be held back by its
headroom: `(kept under -1 dBTP)`. (It used to be at most +3 dB of make-up and none for a
bright sound: a coin or a sword hit, nearly all highs, came out 15–18 dB down.)

```bash
soundmon --master sfx/                       # -> sfx/mastered/… (never overwrites)
soundmon --master assets/ --output-to out/   # speech/ narration/ voice/ folders are treated as voices
soundmon --master hit.wav --master-kind sfx --flac
```

- **Never overwrites a source.** `--output-to` may not be the folder itself or a folder
  inside it (except its own `mastered/`), and a file that would land on a source is
  skipped.
- Output is a 16-bit WAV at the source's rate and channels, named after the source.
  When two sources share a name (`hit.wav` and `hit.flac`), the `.wav` keeps it and the
  others keep their extension in theirs (`hit.flac.wav`).
- Skipped when mastering a folder: `mastered/`, `takes/`, backups, folders starting with
  `_` or `.` (soundmon-gui's `_mastered/` and `_analysis/`), and hidden files. Reads
  WAV, FLAC, OGG, MP3, Opus, M4A and AIFF.
- Voice or effect: the **nearest** folder that says either decides (`speech/`,
  `narration/`, `voice/`, `vo/`, `dialog/` against `sfx/`, `fx/`, `sounds/`,
  `effects/`, `foley/`), so `voice/demo/sfx/hit.wav` is an effect and
  `--master game/speech` masters voices. `--master-kind` overrides it.
- True peak is measured 4× oversampled and the written file is measured again; one
  that lands over −1 dBTP (or an `--ogg` encode that does) is written again quieter.
- A silent file is written as it is; an unreadable one says why (ffmpeg's reason);
  one with NaN samples is refused.

Each line reports what changed, e.g. `above 2.5k 52%→26%  harsh 0.10→0.00s  loudest -14.9→-14.9 LUFS  gain +1.4 dB  peak -1.1 dBTP`.
On NEON-CASTER it took the effects' energy above 2.5 kHz from 34 % to 21 %. Needs
numpy and ffmpeg, but no ComfyUI.

In **soundmon-gui**:
- a set can **master on export**;
- each take in a set shows its measurements (loudest 400 ms, true peak, share above
  2.5 kHz, harsh seconds, tilt) and a sonogram;
- **▶ Mastered** lets you hear it through the chain first.

`--master DIR --master-like REF` matches every file's tone and loudness to a reference
sound instead (see `--match-tone` below).

### Steering with a reference (`--ref`, `--ref-match`, `--match-tone`)

"Make me a sound like this one." It works like pixelmon's image and depth steering:

| pixelmon | soundmon | what it does |
|---|---|---|
| init image (`--init`, strength) | `--ref FILE --ref-strength N` | start from your sound instead of silence |
| depth lock | `--ref-match` | keep its tempo and key (songs and music loops) |
| colour match | `--match-tone REF` | EQ and level each take toward the reference |

```bash
soundmon "heavy iron door slam" --ref door.wav                    # SFX, a variation at 0.6
soundmon "heavy iron door slam" --ref door.wav --ref-strength 0.3 # stays close to door.wav
soundmon --music "dark synth loop" --ref loop.wav --ref-match     # its BPM and key go in the prompt
soundmon --song "chiptune rock anthem" --ref demo.wav --ref-match # --bpm/--key/--seconds from demo.wav
soundmon "laser zap" -n 8 --match-tone our_laser.wav              # sit with the game's other sounds
```

- `--ref` works with the AI engines: SFX, `--music` and `--song`. The reference is
  converted to 44.1 kHz stereo, trimmed or padded to `--seconds`, uploaded to each
  ComfyUI box's `input/` folder, and encoded by the model's own VAE. Sampling starts
  from it with KSampler denoise = `--ref-strength`: **0.3** is a close variation, **0.6**
  (default) keeps its shape, **0.9** only borrows from it. Takes get `_ref` in their name.
- Without `--seconds`, the render is as long as the reference (capped at 120 s for SA3,
  47 s for SAO).
- `--ref-match` reads tempo and key from the reference (the same analysis as
  `--from-audio`). Songs fill `--bpm`, `--key` and `--seconds` unless you set them;
  music loops get "N BPM, in <key>" added to the prompt. For chip and AdLib, use
  `--from-audio` instead.
- `--match-tone REF` runs after rendering. It compares the take's 1/3-octave curve
  with REF's, applies the difference (smoothed, at most ±9 dB), and sets the level to
  REF's loudest 400 ms, never past −1 dBTP. It works with every engine; for CPU engines
  you can also do it afterwards with `--master DIR --master-like REF`. It keeps the file
  sample-exact (same length, nothing shifted — a `--loop` stays seamless) and in its own
  sample format (8-bit stays 8-bit, 24-bit stays 24-bit). A format-locked take (8-bit,
  or under 16 kHz: `--format amiga`, `sb`, `nes`, `gameboy`…) only has its level matched,
  and only ever downwards, so the crush is neither smoothed nor magnified.
- **Not yet proven on a GPU.** The graph is checked here; whether SA3's 8-step sampler
  gives good variations from an encoded reference needs an rtx run. If SA3 does badly,
  try `--engine sao`, the classic Stable Audio Open audio-to-audio.

### Your sound library, searchable (`--index`, `--find`, `--similar`, `--listen`)

Like Sononym: index your sample folders once, then find sounds by what they are (kick,
pad, riser…, loop or one-shot, tempo, key), by name, by a sound you drop, or by
describing one. Everything runs on your machine.

```bash
soundmon --index ~/samples,~/games/neon-caster/sfx     # build or refresh the index (only changed files are read)
soundmon --tags                                        # how it's tagged: kick 3,120, snare 2,840 …
soundmon --find 909 --tag kick --kind oneshot          # by name and tag
soundmon --similar door.wav                            # the 20 closest, best first
soundmon --similar door.wav -n 50 --in ~/samples/foley # only look inside one folder
soundmon --listen                                      # optional: an audio model hears every sound
soundmon --describe "a dark metallic hit"              # …then search by description
```

```
🔎 sounds like door.wav  (1840 indexed)
     1.  91%    0.84s  Foley                  oneshot   /home/me/samples/foley/metal_door_03.wav   [timbre, envelope]
     2.  87%    1.10s  ~Wood & metal hit      oneshot   /home/me/samples/foley/gate_clang.wav      [timbre]
```

**Tags from a sound's own words.** Every file is tagged from its name, the folders it sits
in, and what the file says about itself (WAV/AIFF chunks: acid and Apple Loops flags, smpl,
INFO, bext, iXML; tags in other formats): one of 30 tags in three families — drums (kick,
snare, clap, hi-hat, cymbal, tom, hand drum, shaker, wood & metal hit, percussion, drums),
tonal (bass, synth & lead, keys, pad & drone, pluck & arp, stab & chord, strings/brass/winds,
guitar, bells & mallets, vocal) and fx (riser, whoosh, impact, explosion & gun, zap & glitch,
noise, ambience, foley, scratch); **loop or one-shot** (an acid or Apple Loops flag, then
words like "loop" or "120 bpm" in the name or its folder, then a sampler's loop points — looped
end to end is a loop, a sustain loop part way in is a held note, so a one-shot — then a length
that fits whole bars, then a short length); and **tempo and key**
("Fmin", "C# minor", the acid chunk). The vocabulary (`tags.py`) is the one sample packs use
everywhere, including how they write it — glued ("DryKick"), without vowels ("snrs"),
abbreviated ("SD", "OH") — and knows nothing about any one library. On a 275,000-sound
library it tags 70–94% of each folder by words alone.

**Tags by ear (`--listen`, optional).** An audio model (LAION's open
[CLAP](https://huggingface.co/laion/larger_clap_general), ~600 MB, downloaded once) hears
the first 10 s of every sound. Your library then teaches it: the sounds whose words say what
they are are the lessons, a small classifier learns them from what the model hears (blended
with the model's own guess from each tag's description), and it tags the sounds whose words
say nothing — only where it's reasonably sure. Each library teaches itself; nothing but the
vocabulary is shipped. It also lets `--similar` weigh a sound's **character** (what the
model hears — the best at "the same kind of sound") and makes `--describe` work. It needs
torch and transformers (ComfyUI's venv has them) and uses a GPU when torch has one that
works, else the CPU (~8 sounds a second). An AMD RDNA2 laptop GPU (RX 6600M and the like)
may need `HSA_OVERRIDE_GFX_VERSION=10.3.0` for ROCm; `--listen-device cpu` forces the CPU.

**Similar.** Each file also gets a fingerprint built only from signal processing, from its
audible part:
- **timbre**: the average 32-band spectrum and how much each band moves;
- **brightness**: spectral centroid, flatness (tonal or noisy) and roll-off;
- **envelope**: attack time, hits per second, and the loudness shape;
- **length**.

A search compares every feature on the same scale and weights timbre highest — or as you
say: `--aspects timbre=1,length=0`. Once listened to, character takes 70% of the whole
unless you set it. Results show a 0–100 "how alike" score and which aspects match.
Identical copies (and links to the same file) show once, files deleted since indexing are
left out, and the query itself never comes back. The fingerprint doesn't depend on volume,
and the score's scale comes from the whole index, so a near-duplicate reads high however
few files the filters leave. `--tag`, `--kind` and `--in` narrow `--similar`, `--find`,
`--describe` and `--tags` alike.

**The index** is one SQLite file, `~/soundmon-gallery/similar/index.db` (`$SOUNDMON_INDEX`
picks another; an `index.json` from an older soundmon is taken in once). About 2 KB a
sound with what the model heard. It reads several files at once (`--workers`, default half
your cores, each at low priority on one thread — about 140 sounds a second from a network
share), saves as it goes (stop it any time; the next run carries on), skips `mastered/`,
`__MACOSX`, backups and folders starting with `_` or `.`, and reads the first 30 s of long
files. Its summary counts unreadable files, silent or too-short ones, and anything skipped
(dangling links, folders it can't read) separately; a file that failed isn't read again
until it changes. Two runs at once are safe. Changing the tag rules re-tags from what the
index kept, without reading the files again. It needs numpy and ffmpeg, but no ComfyUI.

Indexing and `--listen` **rest when the CPU runs hot**: above 92 °C they wait till it's 5 °C
cooler (`$SOUNDMON_MAX_TEMP` sets the limit, `0` never rests). A laptop's CPU and GPU often
share their cooling, so a busy GPU heats the CPU too. It reads the CPU's own sensors (Linux
hwmon); where there are none it never rests. `--listen` on a GPU hears 50-100 sounds a second
(ffmpeg and the model's input are made on several threads while the GPU works).

**Run the heavy part on another machine.** The index is one file: copy it to the machine that has
the sounds on its own disk (and a GPU), index or listen there, and copy it back. If that machine
sees them under another path, say so with `SOUNDMON_PATH_MAP="INDEXED=HERE"` (several joined with
`;`), e.g. `SOUNDMON_PATH_MAP="/mnt/nas/music=/srv/music"`: the index keeps its own paths, only the
reading goes through the map. A file's date and size are what tell the index it's unchanged, so a
share and the disk it serves agree.

### Compressing output (`--ogg`)

WAV is the default because it is lossless and every tool reads it. When size
matters — shipping a game's asset folder, say — `--ogg` transcodes the finished
audio to OGG Vorbis. Measured on a 60 s 48 kHz music track: **11 MB → 436 KB at
q=5, a 26× reduction** with the full duration intact.

```bash
soundmon "a laser blast" --ogg                     # ~25x smaller
soundmon "dark fantasy" --song --ogg --ogg-quality 7
soundmon --narrate-file lines.txt --ogg --keep-wav # keep both
```

It happens **client-side, after the file lands** — not in the ComfyUI save node.
That matters: a node-side encoder would have to be installed on every farm box
and kept in sync, whereas this way unmodified farm boxes keep working and the
CLI alone decides the output format. Encoding is well under a second.

> **Two footguns this walks around.** `python-soundfile` segfaults writing OGG
> here (libsndfile 1.2.2 — truncates, then SIGSEGV), so ffmpeg does the work.
> And ffmpeg's *native* vorbis encoder is **stereo-only**, so mono sources (all
> narration — Kokoro emits 24 kHz mono) are upmixed with `-ac 2` or the encode
> fails outright. Vorbis joint-stereo codes two identical channels cheaply, so
> this costs almost nothing.

### Render farm

Built like pixelmon's — the CLI never touches the GPU, it just POSTs a JSON
graph to `/prompt` and pulls results from `/view` — though the two have since
diverged: soundmon's adds a circuit breaker and the robustness below.

```bash
soundmon --batch "explosion,glass,fire" -n 30 --server rtx,local
#   -> 90 clips fanned across 3 GPUs; all land in your local output
```

Dynamic dispatch (each GPU gets its next job the moment it's free), unreachable
boxes skipped, boxes without the model skipped with the `download-models.sh`
line that fixes them, and a `⚠` for any box whose ComfyUI isn't running
`--force-fp32`. While it runs:

- a box that fails **2 jobs in a row** (or most of its jobs) is dropped, and a
  failed job is retried once **on another box**;
- a box that stops answering gets **30 s** before its job moves elsewhere (one
  missed poll is not a dead box); a box that **restarted** and forgot its job
  gets it again;
- an identical re-run that ComfyUI answers from its cache is re-saved — it is
  not a failure and never counts against the box;
- a download or post-processing error fails that one job, not the farm;
- the same box listed twice (`rtx,192.168.1.77`) is used once.

The same robustness is on a single server: a ComfyUI that restarts mid-batch is
waited out (up to 4 minutes) and everything it forgot is sent again; a
"poisoned" server (`object has no attribute 'model_size'`) is waited on until
someone restarts it; one out-of-memory / loader hiccup is retried; time spent
queued behind other people's jobs doesn't count toward the timeout; one failed
job doesn't end the batch (the run exits non-zero at the end). Ctrl-C (or the
GUI's Cancel) takes this run's jobs off the server's queue.

With no `--server`, the target is `$SOUNDMON_SERVER`, then servers.json's
`"_default"` (the key pixelmon reads too), then the `local` alias. See
pixelmon's [README-RENDER-FARM.md](https://github.com/grymmjack/pixelmon) for
LAN/firewall setup — it applies verbatim.

#### Provisioning a farm box

A box already running pixelmon has ComfyUI but *not* the audio models. It needs
the models and the node — **not** the CLI (the client drives it). From your main
box, per farm member:

```bash
rsync -az --exclude '.git' --exclude 'servers.json' ~/git/soundmon/ box:git/soundmon/
ssh box 'cd ~/git/soundmon && ./install.sh && ./download-models.sh --sa3'      # + --song for songs
ssh box 'tmux new-session -d -s comfy "~/launch-comfyui.sh --force-fp32 2>&1 | tee ~/comfyui.log"'
```

Then confirm the node and models registered:

```bash
curl -s http://<box>:8188/object_info | grep -o RetroSFX | head -1
```

Per-platform notes from provisioning a real mixed fleet:

| Box | Gotcha |
|---|---|
| **macOS** | No `tmux` by default — use `nohup caffeinate -i ./launch-comfyui.sh &` instead (`caffeinate` also stops the Mac sleeping mid-run). Ships **bash 3.2**, so scripts must avoid bash-4-only syntax. |
| **WSL2** | The distro **shuts down when idle**, which takes sshd *and* ComfyUI with it — the box still pings (Windows is up) while every WSL port reads `filtered`. Wake it with `wsl` in PowerShell, then `sudo service ssh start`. |
| **Linux** | `loginctl enable-linger "$USER"` once, or systemd kills tmux on logout. |

> ⚠ **Farm boxes need `--force-fp32` too.** A box running ComfyUI without it
> returns buzzing static for every SA3 job — and it returns it *quickly*, so
> dynamic dispatch will happily route most of the batch to the one broken box.

> ⚠ **Don't `pkill -f "…main.py"` over SSH** — the pattern matches the very shell
> running it, so the command kills itself before it can restart anything. Get the
> PID from `ps`/`pgrep` and `kill` that, or just let `bin/soundmon` auto-start.

---

## GUI (`soundmon-gui`)

The same tool in a browser, built like pixelmon-gui (same layout, same look, and both
can run at once):

```bash
soundmon-gui                 # http://127.0.0.1:8191
soundmon-gui --lan           # reachable from other devices on the LAN
soundmon-gui --server rtx    # GPU renders go to rtx (else the Setup tab's pick, else servers.json "_default")
```

- **Every engine on one form.** Eight engine buttons at the top (SFX, Music loop, Song,
  Chiptune, AdLib, Retro FX, Blips, Narrate); only that engine's settings show. Style
  guides, moods, archetypes, voices, keys and hardware formats are read live from
  `sounds.json`, `chip.py`, `chipfx.py`, `opl.py`, `narrate.py` and the node, so the GUI
  never offers something soundmon doesn't have. Hovering a field shows the flag's own
  `--help` text. 🎲 rolls a description, 🕘 brings back ones you rendered; 🎲 next to Seed
  picks one to keep. Form sections fold (and stay folded); sliders step with the mouse wheel
  (shift ×5), reset on middle-click, and take a typed value when you click their number.
- **Every sound is the same player**, wherever it is listed (queue, gallery, sets, find,
  voices, the reference you dropped): a waveform with a playhead you click to play from that
  spot, play/pause, and what the file really is, read from its own header (length, sample
  rate, bit depth or bitrate, channels, codec, size, peak, clipped samples) plus a small
  spectrum showing how high its content reaches. Space plays/pauses, Esc stops.
- **Browse… and Open on every file and folder.** Every path field and drop zone has
  **Browse…** (a picker over your home folder, with shortcuts, a filter and a preview of
  sounds) and every sound has **↗** (also on right-click): open it in the system's default
  app, in a program from Setup (found automatically: Audacity, VLC, OpenMPT…), or show it in
  its folder. With `--lan`, browsing and opening only work from the machine running the GUI.
- **Folders outside home.** The GUI only touches files in your home folder, plus up to 20
  folders you list in Setup → **Folders outside home** (a sample library on another disk):
  Browse… shows each as a ⇱ shortcut, and Find, Sets and references can use them. Not a whole
  drive or a system folder (/etc, /usr…); the list is only changed from the machine running
  the GUI. (A symlink in home to another disk doesn't count — its real place is checked.)
- **The command is always visible.** Under RENDER is the exact `soundmon …` command the
  server will run (the server builds it, so the preview can't drift). COPY CLI takes it
  to a terminal.
  Under it, a **setup check** warns about settings that will render but probably not as
  meant (SAO past 47 s, SA3 steps/CFG, a reference the engine ignores, a server missing the
  model…), each with **Fix**.
- **Queue** shows each job's progress (`✅ [i/n]` lines), log and errors. When ComfyUI
  fails a job, the job fails in seconds with the node's own message. A set render, a Voices
  run or a sweep is one card with one bar and **Cancel all**. **Cancel** really stops the
  GPU: soundmon takes its own prompts off ComfyUI's queue (and interrupts the one rendering
  if it is its own), so a ComfyUI shared with pixelmon keeps pixelmon's work.
  **⛔ CANCEL AND ABORT** in the status bar stops everything this GUI started.
- **SWEEP…** renders one take per value of one setting, same seed (reference strength,
  steps, style, format, mood, archetype, voice, BPM, key…), so you hear what a setting does.
  **🧬** on a take renders four variations of it (the take becomes the reference).
- **⚖ Compare** lines up to nine takes: press `1`–`9` to switch between them *at the same
  moment*, see their loudness and an overlaid spectrum against the ★ reference. Click a
  gallery take's name for a close-up: big waveform, measurements, sonogram, the full
  recipe and every action, `←`/`→` through the gallery.
- **Notifications** (status bar): a desktop notification and/or a chime when jobs finish
  or fail, a coin per take, and when the render server goes down, comes back, gets stuck or
  is restarted. A banner says when the server is down while a GPU engine is selected.
- **Gallery** keeps every take with its waveform, seed, rating and the files beside it
  (`.mod`, `.rad`, `.mid`). **Load** puts a take's settings back in the form; **Again**
  renders that exact take again (same seed). Each job lives in
  `~/soundmon-gallery/gui/<job>/` with a `job.json` recording the settings, the command
  and every seed, so nothing depends on ComfyUI's history.
- **Sets** are for a game's whole sound list, the job neon-caster's Sound Picker and
  qb64-dungeon's `generate-from-manifest.sh` each did by hand:
  1. Paste or drop the list: `key | text`, `key | seconds | prompt`, a tracks.txt line
     (`name | secs | bpm | key | mood`) or plain words, with `[Group]` lines to group them.
  2. The form's current settings become the set's **theme**: every key renders with
     them, N takes per key (narration: one take per voice you pick).
  3. Each key gets a strip of takes beside **▶ NOW**, the file the game plays today.
     Hear them (keys `1`–`0`, `N`), **PICK** one or **Keep old**, flag bad ones, and
     re-roll or edit a key's prompt.
  4. **Export picks** writes `<key>.<ext>` into the game folder (converting to WAV /
     FLAC / OGG, a sample rate or mono if asked). Anything it replaces is copied to
     `.soundmon-backup/<date>/` first. It warns when a key also exists in another
     format, because the game may play that one instead.

  "Skip keys the game has" leaves out keys the game folder already has a file for; each
  key can carry a note; every row is checked against the theme before anything is queued.

  Takes are just gallery jobs tagged with the set and key, so a set survives
  restarts and every take keeps its seed and command.
- **Reference** (SFX, Music, Song): drop a sound to steer the render (`--ref`), with a
  slider from "close variation" to "loosely inspired", plus "use its tempo, key and
  length" (`--ref-match`) and "match its tone" (`--match-tone`). The job keeps a copy
  of the reference, and its gallery card has **ref ▶** to hear it. A set's theme can
  carry a reference too, and a set can export **mastered like the reference**.
- **🔎 Find** is your sound library: ↻ **Index** the folders you choose (remembered; a
  progress bar with the rate and time left, ■ Stop any time), then filter by tag chips (with
  counts, by family), loop or one-shot, length, tempo, key, words in the name, who tagged it
  and folder — or drop a sound for the ones most like it (or 🔎 on any gallery take, set
  take or result), under the same filters. 👂 **Listen** lets the audio model hear them
  all: tags by ear, a **describe** box ("a dark metallic hit"), and **Character** in the
  **Aspects** dial, whose sliders set how much timbre, brightness, envelope, length and
  character count. A result's tag (dashed: by ear) is a menu: set your own, and the next
  Listen learns from it. Each result can be played, **Use**d as the form's reference,
  opened in its folder, compared, or have its path copied.
- **Voices** speaks one of your lines in every Kokoro voice, so you can compare them;
  **Use** puts one in the Narrate form.
- **Presets** (top of the form) save and load the form's settings by name, in folders and
  with a note; 💾 on any queue card saves that job's settings. **⬇** downloads presets as a
  `.zip` with the files they use (reference, MIDI, bank), **⬆** imports one (never
  overwriting).
- **Advanced** exposes the rest (sampler, reprompt, ACE-Step's LLM, trim/normalize/fade,
  OGG quality, MIDI start). Blank means soundmon's own default.
- **Setup** lists every `servers.json` box with its state and whether it has the models
  each engine needs (SA3 medium, ACE-Step, T5Gemma, Qwen), and picks where GPU renders go
  (it says so when `--server` overrides the pick). **Backups** zips the whole
  `~/soundmon-gallery`, restores what's missing from one (never overwriting), and can clear
  the gallery (after a backup).

The GUI only answers its own page: a request from another site, or to a name that isn't this
machine, is refused.

Drop a `.mid` or a recording on the Chiptune / AdLib form to have it followed
(`--from-midi`, `--from-audio`), or a `.txt` of `key | text` lines on the Blips / Narrate
text box to make one file per line.

---

## How it works

```
description
   └─► CLIPLoader (Qwen 3.5) ─► TextGenerate  ← the rewrite stage
         └─► rewritten prompt
               └─► ComfyUI API
                    CheckpointLoaderSimple (Stable Audio 3 medium) ─┐
                    CLIPLoader (T5Gemma) ─► CLIPTextEncode ×2       │
                      └─ ConditioningStableAudio ─► KSampler ───────┘
                           (lcm / simple / 8 steps / cfg 1.0, fp32)
                           └─► VAEDecodeAudio ─► RetroSFX ─► SaveSFX (WAV)
```

Every engine lands its AUDIO on the **same graph node (`"10"`)**, so the tail —
`RetroSFX` then `SaveSFX` — is shared verbatim by SFX, music and songs. Adding a
fourth engine means writing a node-builder that ends at `"10"`; nothing
downstream changes.

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

The lock decimates without an anti-aliasing filter and quantizes without
dither, on purpose: the aliasing and the grit are the sound of the hardware.
Both choices are explained in `_format_lock()` in `custom_nodes/retro_sfx/nodes.py`.

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
7. **fp16 does not merely degrade audio diffusion — it destroys it.** ComfyUI
   picks fp16 by default and `StableAudio3` declares no
   `supported_inference_dtypes` to say otherwise, so the sane-looking default is
   the broken one. Output is buzzing static, not slightly-worse audio. **This is
   the single highest-value line in this file:** `--force-fp32`.
8. **A checkpoint with no published workflow is a research project.** The
   `small_music` / `small_sfx` specialists look like the obvious pick — smaller,
   purpose-trained — and both produced scrambled output at every setting tried.
   Only `medium` has an official ComfyUI workflow, and only `medium` worked.
9. **More steps is not more quality.** SA3 medium wants **8** steps at **cfg 1.0**
   with the **lcm** sampler. At 50/cfg7 it produces noise; at 50 steps on the
   correct sampler it produces audible clacking and popping. Diffusion intuition
   carried over from images ("50 steps is the safe default") is actively wrong
   here.
10. **Do not drop stages of a published pipeline and expect the published
    result.** The Qwen rewrite looked like optional prompt-polish, so it got
    deferred. With it, output was described as "stunning"; without it, "audio
    quality identical, musical quality lower". The pipeline is the product.
11. **Every automated check can pass while the audio is garbage.** Duration,
    peak, LUFS, true-peak and readability were all green across **1,122 files**
    that sounded, per the person who could actually hear them, like "trash".
    Automated audio QA is a smoke test for *plumbing*, not a judgement of sound.
    `tools/spectral-check.py` narrows the gap — it catches missing high end —
    but it cannot tell you a track is musically incoherent. **Listen to the
    output, or have someone who can.**
12. **One-shot defaults silently break loops.** `trim_silence` plus a de-click
    fade is exactly right for a door slam and exactly wrong for a track that
    plays for an hour; it leaves a 45–50 dB hole at the seam. Loops need
    `--no-trim --fade-ms 0`. If your asset list distinguishes loops from
    one-shots, key off *that* rather than a global default.
13. **A fast-failing farm box eats the entire queue.** With dynamic dispatch,
    work goes to whichever box is free — and a box that fails in two seconds
    becomes free far more often than one that succeeds in ninety. One broken
    machine took **30 of 46 jobs** before anything noticed. Dispatch needs a
    circuit-breaker: drop a box after N consecutive failures and requeue its
    work.

---

## Performance

Audio is dramatically cheaper than SDXL — a 4-second latent is far smaller than
a 1024×1024 image, and SA3 needs only 8 steps. Generation is cheap enough that
the intended workflow is *make 8 and keep one*, which is exactly what the cloud
tools charge you per-generation for.

**What actually costs time is model swapping, not sampling.** On an 8 GB card,
SA3 medium in fp32 (~8.6 GB) and the Qwen rewriter (~4.5 GB) cannot co-reside,
so a naive run pays a full model reload per asset:

| Approach | per asset | 336 assets |
|---|---|---|
| inline rewrite, 8 GB card | ~75.6 s | ~7 hours |
| pre-computed rewrites, then `--no-reprompt` | ~4 s | ~1 hour |

The fix is phase separation: rewrite every prompt in one pass while Qwen stays
resident, cache the results, then generate audio with the LLM out of the picture.
On a card that fits both models at once this doesn't arise.

For reference, the legacy `--engine sao` path on an **AMD RX 6600** (RDNA2, 8 GB,
ROCm, `--lowvram`) ran 4-second clips at ~6.4 s (50 steps) or ~4 s (`--fast`).

---

## Repo layout

```
soundmon/
├── README.md
├── AGENTS.md                  orientation for coding agents working on this repo
├── install.sh                 links files; reuses pixelmon's ComfyUI if present
├── download-models.sh         fetch models: --sa3 (default engine), --song, --sao, --opl, --all, --list
├── launch-comfyui.sh          starts ComfyUI (GPU autodetect) — pixelmon's script; linked only if ~/ has none
├── soundmon.py                the CLI brains (talks to ComfyUI's API)
├── narrate.py                 --narrate: Kokoro TTS, in-process on CPU, no ComfyUI
├── blip.py                    --blip: JRPG text-box blips (synth needs no model)
├── chip.py                    --chip: NES 2A03 synthesis + the shared composer
├── chipfx.py                  --chipfx/--oplfx: sfxr-style effect recipes
├── opl.py                     --opl: Nuked OPL3 FM (core fetched, not vendored)
├── record.py                  --record: the terminal recording booth (mic → pack)
├── master.py                  --master: per-sound mastering chain (from NEON-CASTER), + analyse(), match_tone()
├── similar.py                 --index / --similar / --find: the library index (SQLite), DSP fingerprints, filters
├── tags.py                    what a sound is from its own words: tags, loop/one-shot, tempo, key
├── listen.py                  --listen / --describe: CLAP hears the library, learns its tags (optional)
├── sounds.json                --style guide snippets (edit / add your own)
├── sa3_reprompt.json          the four Qwen system prompts, verbatim from ComfyUI
├── servers.example.json       template for --server aliases (copy to servers.json)
├── bin/soundmon               wrapper: ensures the server is up, then runs soundmon.py
├── bin/soundmon-gui           the web front end (gui/server.py on port 8191)
├── gui/
│   ├── server.py              stdlib server: job queue, gallery, setup — runs bin/soundmon
│   ├── sets.py                Sets: manifest → takes per key → picks → export into a game
│   └── index.html             the page (no build step, no CDN)
├── .claude/skills/create-release/   the release checklist (selfcheck, fp32 + smoke render, tag, notes)
├── tools/
│   ├── selfcheck.py           offline invariants — run after touching an engine or the job lifecycle
│   ├── spectral-check.py      measure spectral cutoff — catches hollow output
│   ├── stems.py               a finished track → stems, loops, one-shots, FX (INFINITE-SAMPLES.md)
│   ├── instruments.py         the one-shots → sampler instruments: Bitwig .multisample, Renoise .xrni,
│   │                          Bitwig .bwpreset (a Drum Machine kit, Samplers)
│   └── bwpreset.py            reads and writes Bitwig's binary .bwpreset (used by instruments.py)
└── custom_nodes/
    └── retro_sfx/             the finishing node (trim→format→fade→normalize) + WAV saver
        └── nodes.py           FORMATS registry — add your own hardware formats here
```

### Stems, loops and sampler instruments (`tools/stems.py`, `tools/instruments.py`)

A finished track (a `--music` take, or any WAV) taken apart into a sample pack: stems
(Demucs, DrumSep, de-reverb, synth layers), bar loops, one-shots (drum hits, notes, chords,
held bars) and FX - then the one-shots built into instruments for both DAWs:

```bash
python tools/stems.py track.wav "out/track - TITLE" --name track --title TITLE --bpm 128
python tools/instruments.py "out/track - TITLE"     # stems.py runs it at the end too
```

`5 Instruments/` gets each instrument three ways: `.multisample` (Bitwig's open format),
`Renoise/*.xrni` (drag into Renoise's instrument list; keys as MIDI plays them) and, where
Bitwig is installed, `Bitwig/*.bwpreset` - its own Drum Machine (a Sampler a pad) and Samplers,
written by `tools/bwpreset.py`, which reads and writes Bitwig's binary preset format.
`instruments.py` and `bwpreset.py` need only Python's standard library. With `--bpm`, stems.py
takes the tempo as given and fits only where the beats fall; without it, it fits a line through
the beat tracker's beats (the tracker's own tempo is coarse: 129.5 for a 128 bpm track). The bar's
first beat is where the kick lands hardest, and where the harmony changes when the kick doesn't
decide (a kick on 1 and 3, or no drums). The whole walk-through,
from the prompt to the sampler, is [INFINITE-SAMPLES.md](INFINITE-SAMPLES.md).

### Checking output (`tools/spectral-check.py`)

soundmon's built-in checks verify duration, level and readability — none of which
notice a spectrally hollow file. This measures where the spectrum falls 40 dB
below the 500–2000 Hz reference band:

```bash
tools/spectral-check.py --min 20 render.wav              # WAV: expect Nyquist
tools/spectral-check.py --min 17 assets/music/*.ogg      # OGG: Vorbis rolls off
```

The reference band falls back to the file's loudest octave when 500–2000 Hz holds nothing
but leakage (a pure tone under 500 Hz), and the cutoff is searched above the loudest content,
so a pure tone reads just above its pitch. Files with NaN/inf samples fail; files too short
or silent to measure are listed (and fail a `--min` check); M4A and anything else soundfile
can't open is decoded with ffmpeg.

> **Compare like with like.** Vorbis imposes its own rolloff, so an OGG of
> perfect source still reads ~17 kHz — that is the codec, not a defect. And a
> low reading on a genuinely dark, bass-heavy track is *correct*; real DAW
> masters measured here ranged 9.5–22.1 kHz. This tool tells you whether a
> pipeline lost content, not whether a track is good.

---

## Credits & licenses

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — the engine (GPL-3.0)
- [Stable Audio 3](https://huggingface.co/Comfy-Org/stable-audio-3) — Stability AI, the default engine for SFX and music
- [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) — Stability AI (Stability AI Community License). Trained on Freesound + Free Music Archive.
- [ACE-Step 1.5](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files) — full songs with sung vocals (`--song`)
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) — 82 M TTS, Apache-2.0 (`--narrate`)
- [Qwen 3.5](https://huggingface.co/Comfy-Org/Qwen3.5) — the SA3 prompt rewriter
- [T5](https://huggingface.co/google-t5/t5-base) / T5Gemma — the text encoders

The code in this repo (the CLI, wrapper, installer, and custom node) is released
under the MIT License.

> **Licensing note.** Stable Audio Open ships under the *Stability AI Community
> License*, not a plain OSS license — free for research and for commercial use
> under a revenue threshold, with terms above it. Read it before shipping
> generated audio in a commercial game.
