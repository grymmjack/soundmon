# AGENTS.md

Guidance for AI coding agents working in this repository. Human-facing docs live
in [README.md](README.md); this file is the stuff that is expensive to rediscover.

## What this is

A CLI that turns a text description into audio, running entirely on local
hardware. **Six engines behind one command**, chosen by flag:

| Mode | Engine | Runs on | Output |
|---|---|---|---|
| default | **Stable Audio 3 medium** | ComfyUI / GPU | sound effects |
| `--music` | **Stable Audio 3 medium** | ComfyUI / GPU | loops, beds, stings |
| `--engine sao` | Stable Audio Open 1.0 | ComfyUI / GPU | legacy fallback |
| `--song` | ACE-Step 1.5 | ComfyUI / GPU | full songs, sung vocals |
| `--narrate` | Kokoro | **CPU, in-process** | spoken narration |
| `--record` | **a human + a microphone** | CPU, in-process | spoken narration |
| `--blip` | **an oscillator** (synth) / Kokoro (voice) | CPU, in-process | JRPG text-box blips |

> **Before you change anything about the SA3 path, read gotchas 11–14.** Four
> independent settings each turn its output into unusable noise, and none of them
> fail loudly — you get a valid WAV of the correct length containing static.

Sibling project to **pixelmon** (`~/pixelmon`, github.com/grymmjack/pixelmon),
which does the same thing for pixel-art sprites. soundmon deliberately mirrors
its architecture and reuses its ComfyUI install.

## Architecture — read this before adding an engine

`soundmon.py` never touches the GPU. It builds a ComfyUI graph as JSON, POSTs it
to `/prompt`, polls `/history/<id>`, and fetches results from `/view`. That is
the whole reason the render farm is ~150 lines and vendor-agnostic.

**The engines share a tail.** Every GPU pipeline lands its finished `AUDIO` on
graph node `"10"`; `build_graph()` then attaches `RetroSFX` as `"11"` and a save
node as `"12"`. Adding a fourth GPU engine means writing a `_yourengine_nodes()`
that also ends at `"10"` — do not rewrite the tail.

```
_reprompt_nodes()  ──► rewritten text ──┐   (nodes "20"/"21", SA3 only)
                                        ▼
_sfx_nodes()  ─┐
               ├─► node "10" (AUDIO) ─► "11" RetroSFX ─► "12" Save*
_song_nodes() ─┘
```

This is the same trick pixelmon's `--animate` uses: swap the pipeline, keep the
CLI. Follow it.

**`_reprompt_nodes()` is a stage, not a decoration.** SA3's official workflow
runs Qwen 3.5 over the description first, under one of four system prompts
(`sa3_reprompt.json`, verbatim from ComfyUI). It is on by default; `--no-reprompt`
exists for batch runs that pre-compute rewrites to avoid VRAM thrash, not as a
"skip the slow part" convenience.

> **`TextGenerate`'s nested sampling params are dot-namespaced.** They go in flat
> as `"sampling_mode.temperature"`, `"sampling_mode.top_k"`, … alongside
> `"sampling_mode": "on"` — *not* as a nested dict. A nested dict is accepted by
> the API and then ignored, so you get default sampling and no error.

**`--narrate` is the deliberate exception** — it never touches ComfyUI. SFX and
songs are diffusion problems where ComfyUI earns its keep (model loading, VRAM,
the farm). Kokoro is an 82 M feed-forward model that speaks a line in ~1 s;
wrapping it in a node graph and fanning it across four GPUs would *add* latency.
It lives in `narrate.py` and is handed off early in `main()`, mirroring how
pixelmon hands `--animate` to `animate.py`. **Do not "unify" this into the graph
path.** One interface, the right mechanism per engine.

**`--record` is the same exception again, one step further** — no model at all,
just a terminal booth around a microphone. It lives in `record.py`, hands off
next to `--narrate`, and deliberately **shares two things with it**:

- `narrate.parse_lines()`, verbatim. The `key | text` manifest was never
  TTS-specific; the key becomes the filename either way. This works without
  numpy/soundfile/kokoro installed because `narrate.py` imports those *inside*
  `run()` — keep it that way, or `--record` grows a TTS dependency it does not
  need.
- The output tail. Unlike `--narrate`, `--record` runs its own
  trim → fade → peak-normalize before handing off to `loudness_normalize()` and
  `to_ogg()`, because a mic take arrives with room tone and a reaction-time gap
  that Kokoro output does not have. **A pack whose files were mastered
  differently is a pack you can hear switching**, so the chain must stay
  equivalent.

The payoff is that a hand-voiced pack and a generated pack are interchangeable
*per line* — record the ten lines you care about, generate the other sixty.

## --blip is the far end of the same spectrum

`--narrate` skips ComfyUI because Kokoro is too small to be worth a node graph.
`--record` skips the model entirely. **`--blip-style synth` skips even the audio
*library* stack** — it is `numpy` plus an oscillator, so it runs anywhere,
including a box that has never downloaded a checkpoint.

That is deliberate and worth preserving. It means a game's whole text-box voice
can be regenerated in CI, on a laptop, or by a contributor with no GPU. If you
extend it, do not introduce a torch/kokoro import into the `synth` path — the
Kokoro import in `blip.py` sits *inside* the `voice` branch for exactly this
reason, mirroring how `narrate.py` defers its imports so `--record` stays free
of a TTS dependency.

Two things future-you will be tempted to "fix" and should not:

1. **Per-character pitch is deterministic, not random.** `_char_semitones()`
   hashes the codepoint. Randomizing it sounds like a malfunction; the same word
   producing the same little melody is what makes it read as speech.
2. **`voice` style resamples, it does not pitch-shift.** `narrate._pitch_shift`
   goes to trouble to move pitch *without* changing duration. Animalese is
   speech played fast — the duration change is the effect, so a plain resample
   is correct and calling `_pitch_shift` here would defeat the point.

## Recording — cross-platform gotchas

There is no portable microphone. `_capture_cmd()` in `record.py` picks a backend
per OS, and each one has a different way to be told to *stop*:

| OS | Backend | Stop | Why |
|---|---|---|---|
| Windows | `ffmpeg -f dshow` | `q` on stdin | no SIGINT to send a child process |
| macOS | `ffmpeg -f avfoundation` | `q` on stdin | `-i ":0"` — the leading colon is required, or it opens a **camera** |
| Linux | `pw-record` | SIGTERM | speaks PipeWire natively |
| Linux (fallback) | `ffmpeg -f alsa` / `arecord` | `q` / SIGTERM | for non-PipeWire systems |

1. **The dev box's ffmpeg has no `pulse` demuxer.** `ffmpeg -devices` lists only
   `alsa/oss/fbdev/v4l2/x11grab`, so the obvious `-f pulse -i default` route does
   not exist here. That is why Linux prefers `pw-record`, which is also the
   backend that was actually measured.
2. **Stopping the recorder the wrong way truncates the header.** A WAV's
   RIFF/data size fields are backfilled at close; kill the process without
   letting it finalize and you get a file some players read as zero-length.
   Verified `pw-record` *does* finalize correctly on a signal (size fields
   correct after a 5.7 s signal-terminated capture) — do not "improve" this to
   `kill -9`.
3. **The live VU meter polls the growing file**, it does not tap the audio
   stream — every backend writes a plain WAV as it goes, so the same code works
   on all three OSes. Measured flush granularity is **~0.25 s** (24576 bytes =
   0.256 s of 48 kHz mono s16), which is why `METER_HZ` is 8 and not 30; polling
   faster re-reads the same tail and the bar just freezes.
4. **Envelopes and meters are dB-scaled, not linear.** A linear ramp collapses
   everything under about −18 dBFS into the bottom block, so a perfectly good
   quiet take renders as a flat line — indistinguishable from a muted mic, which
   is the one thing the display exists to tell apart. Both use −48..0 dB so the
   bar you watch while recording agrees with the envelope you see after.
5. **Takes go in `<dest>/takes/`, not beside the pack files.** The output dir
   *is* the game's narration pack — anything in it ships.
6. **Take numbering comes from the highest number on disk, not the count.**
   Deleting take 2 of 3 leaves `{01, 03}`; a count-based number hands back `03`
   and silently overwrites.
7. **Windows consoles need `VIRTUAL_TERMINAL_PROCESSING` enabled** before they
   honor ANSI escapes (`_enable_vt()`). Windows Terminal does it already;
   `conhost.exe` does not, and the whole TUI renders as literal escape garbage.

## Hard-won gotchas — do not re-learn these

1. **Stable Audio's checkpoint has NO text encoder.** It carries the DiT (374
   tensors), VAE (365) and two seconds-embedders, and **zero T5 tensors**.
   `CheckpointLoaderSimple`'s CLIP output (slot 1) is unusable; T5 loads
   separately via `CLIPLoader` with `type=stable_audio`. Wiring slot 1 is the #1
   confusing failure. ACE-Step's all-in-one checkpoint is the opposite — it
   *does* bundle VAE + Qwen encoder, so one loader supplies all three.
2. **Normalize LAST.** Stable Audio frequently starts at full amplitude — the
   loudest sample of a render is often *sample 0* — so the de-click fade lands
   right on the peak. Normalizing before fading silently undoes it (measured
   −5.94 dBFS on a −1.0 dBFS request). Any gain-changing stage after normalize
   invalidates it.
3. **Trimming and capping are different tools.** `trim_silence` removes silence;
   `max_seconds` removes *length*. A UI blip that plays once per glyph came out
   at 0.97 s and no amount of silence-trimming would fix it.
4. **Trimming is wrong for songs.** 30 s at 120 BPM in 4/4 is exactly 60 bars;
   shaving the tail leaves a clip that no longer lines up to a bar. `--song`
   therefore defaults to *keeping* silence. Don't "fix" that.
5. **`--seconds` is conditioning, not duration.** Both models take it as an
   input, so changing it changes *what* is generated, not just how much.
6. **Negative prompts decide the category here.** "A door creak" and "a song
   about a door" are both valid readings of the same text, so `SFX_NEGATIVE`
   pushes music/speech away — and that same negative sabotages `--music`. Hence
   three separate negatives. If you add a mode, add its negative.
7. **The turbo ACE checkpoint wants ~16 steps at cfg 1.0.** Inheriting Stable
   Audio's 50 steps / cfg 5 wastes minutes and oversaturates.
8. **8-bit WAV is unsigned** (offset 128) — the one PCM width that breaks the
   signed pattern. Getting it wrong yields loud static, not quiet audio.
9. **Kokoro must run on CPU here.** On the RX 6600 (gfx1032 masquerading as
   gfx1030 via `HSA_OVERRIDE_GFX_VERSION`) it dies with `HIP error: invalid
   device function`. At 82 M params there is nothing to win on the GPU.
10. **The engine hand-offs `return` before the diffusion-side setup in `main()`,
    so anything they need must be parsed ABOVE them.** `a.lufs_target` was
    derived from `--lufs` *below* the `--narrate` hand-off, so it did not exist
    yet when `narrate.run()` looked for it — and because the lookup is a
    defensive `getattr(a, "lufs_target", None)`, `--narrate --lufs -16` silently
    did nothing instead of raising. The parse now sits above both hand-offs.
    Any future per-engine option has the same trap: a `getattr` default will
    hide it.
11. **fp16 destroys audio diffusion — it does not merely degrade it.** ComfyUI
    defaults audio models to fp16 and `StableAudio3` declares no
    `supported_inference_dtypes` to override that, so the *default* path is the
    broken one. Output is buzzing static. The server must run `--force-fp32`.
    `bin/soundmon` passes it when it starts the server itself, but a ComfyUI
    already up for pixelmon will not have it — **check before blaming the graph.**
12. **Only `stable_audio_3_medium` works.** The `small_music` / `small_sfx`
    specialists look like the obvious choice and produce scrambled output at
    every setting tried. They have no published ComfyUI workflow; `medium` does.
    Treat "no official workflow" as "unsupported", not "undocumented".
13. **SA3 medium wants lcm / simple / 8 steps / cfg 1.0.** Not euler, not 50
    steps, not cfg 5–7. At 50/cfg7 it emits noise; at 50 steps on the right
    sampler it emits audible clacking. Image-diffusion intuition ("more steps is
    safer") is actively wrong here. `soundmon.py` rewrites these defaults when
    `--engine sa3` is active — do not "restore" them for consistency.
14. **Do not drop stages from a published pipeline.** The Qwen rewrite was
    deferred as optional prompt-polish. Result: "audio quality identical,
    musical quality lower." With it: "stunning." If a reference workflow has a
    stage you do not understand, that is a reason to keep it, not to cut it.
15. **Loops are broken by the MODEL, not by post-processing — and the obvious
    fix makes it worse.** A track that plays continuously has a hole at the
    seam because the model *composes an ending*: a 60 s request returns a 60 s
    piece of music with a decay. The instinct is to stop touching the endpoints
    (`--no-trim --fade-ms 0`). Measured, same pack, same models, same prompts:

        trim on,  5 ms fade  ->  tail -30.4 dB vs body
        trim off, no fade    ->  tail -66.6 dB vs body

    Trimming was *helping* — near-silence trimming eats most of the composed
    fade. It cannot finish the job, because a decay is only "silence" at the
    very end. **No endpoint policy fixes this.** Use `--loop`, which crossfades
    the tail over the head so the seam is contiguous by construction (verified:
    tail -39.4 dB -> +0.7 dB). `--loop` forces `trim` on and `fade_ms` to 0
    itself, so the combination cannot be got wrong from outside.

    The meta-lesson is the expensive one: **the first plausible cause was
    asserted and acted on without measuring.** The measurement took two minutes
    and reversed the conclusion. Measure first — especially when you cannot hear
    the result.


16. **A fast-failing farm box eats the queue.** Dynamic dispatch gives work to
    whoever is free, and a box that fails in 2 s becomes free far more often than
    one that succeeds in 90 s. One broken machine took **30 of 46 jobs**. Any
    dispatcher needs a circuit-breaker: drop a box after N consecutive failures
    and requeue its in-flight work.

## Environment gotchas

- **`ls` is aliased to `eza`** in this shell — `ls -t` fails. Use
  `find -printf '%T@ %p\n' | sort -rn` instead.
- **`pkill -f "…main.py"` matches its own shell** and kills the command before it
  can restart anything. Get the PID from `pgrep`/`ps` and `kill` that. Bit us
  twice, once locally and once over SSH.
- **Backgrounding needs `setsid`** for anything that must outlive the invoking
  tool call; a plain `nohup … &` got reaped and truncated a 9-track run.
- **macOS ships bash 3.2**, where expanding an *empty* array under `set -u` is a
  fatal error. Use `${arr[@]+"${arr[@]}"}`, not `"${arr[@]}"`.
- **`df` inside WSL2 reports the sparse vhdx capacity, not the Windows host's
  free space.** It cheerfully said 931 G free while `C:` had 43 G, and a
  download died at 218 MB. Check `df /mnt/c` on WSL boxes.
- **WSL2 shuts the distro down when idle**, taking sshd *and* ComfyUI with it.
  The box still pings (Windows is up) while every WSL port reads `filtered`; a
  lingering Windows port-proxy makes ssh fail at `kex_exchange_identification`
  rather than refusing, which reads like a broken sshd instead of a stopped VM.

## The render farm

`--server a,b,c` fans jobs across boxes with **dynamic dispatch** — each GPU is
handed its next job the moment it frees up. This matters more than it sounds:
measured spread on the dev fleet is **28 s (RTX 3070) vs 468 s (RX 6600)** for
the same 60 s track, ~16×. Any batching layer you write on top must pull work
from a shared queue, not pre-deal it by count; pre-dealing strands the whole run
behind the slowest card.

`run_farm()` also **checks capability before dispatching** — a box with the SFX
models but not the 9.3 GB song model is skipped with a message rather than
failing jobs mid-batch. Extend `server_has_ckpt()` if you add a model.

**Capability is not the only way a box is wrong.** A box with every model present
but ComfyUI running *without* `--force-fp32` passes `server_has_ckpt()` and then
returns static for every SA3 job — quickly, so dispatch sends it more work. There
is no API surface that reports the server's dtype, so this cannot currently be
checked; treat "one box's output sounds different" as a launch-flag question
before a model question.

Aliases live in `servers.json` (gitignored — real LAN IPs stay out of the repo);
`servers.example.json` is the template.

## Working on this repo

- **Models are not in git** (~29 GB across all engines). `./download-models.sh
  --sa3` fetches the default engine + the Qwen rewriter, `--song` adds ACE-Step,
  bare fetches the legacy Stable Audio Open set. It downloads to `.part` and
  renames on success, so an interrupted transfer can never masquerade as a
  complete file.
- **`install.sh` symlinks** this repo into `~/ComfyUI` — the repo stays the
  source of truth. Editing `custom_nodes/retro_sfx/nodes.py` requires a
  **ComfyUI restart** to take effect, and on every farm box, or they will reject
  graphs using a new node input.
- **No test suite.** Verify by generating and measuring: duration, peak dBFS,
  sample rate, LUFS, and an amplitude envelope. An envelope catches "generated
  silence" fast, and some semantic sanity is visible — footsteps show as discrete
  spaced impacts, a drone as a flat bar.

- **⚠ Every one of those checks can pass on unusable audio.** This is the most
  expensive lesson in the repo. **1,122 files** shipped green on duration, peak,
  LUFS, true-peak and readability while sounding, to the person who could
  actually hear them, like "trash" — buzzing, radio-scramble, garbled. The
  envelope did *not* catch it: fp16 noise has a perfectly plausible envelope.

  If you are an agent working here, **you cannot hear the output.** Every metric
  in this repo is a proxy, and the four SA3 bugs above were each found by a human
  listening, not by any check that existed. Practical consequences:

  1. **Do not report audio as working because the checks passed.** Say what was
     measured and that it has not been heard.
  2. **Get output auditioned early and in small batches.** A 6-pack, 1,100-file
     run that turns out to be noise costs hours; three files cost minutes.
  3. **`tools/spectral-check.py` closes part of the gap** — it catches a missing
     top end, which is what "spectrally processed, frequencies just gone" reads
     as. It cannot tell you a track is musically incoherent. Compare like with
     like: OGG always reads ~17 kHz because of Vorbis's rolloff, and a genuinely
     dark track legitimately reads low.
  4. **When a cause is uncertain, run the decisive test, not the cheap one.**
     The four SA3 bugs were untangled by an A/B matrix (dtype × checkpoint ×
     sampler × prompt) that should have been run first — asserting a cause and
     testing it cheaply just produced several confident wrong answers in a row.
- **Style guides live in `sounds.json`**, formats in `custom_nodes/retro_sfx/nodes.py`
  (`FORMATS`), voices in `narrate.py` (`VOICES`). The CLI reads all three at
  runtime, so `--list-styles` / `--list-formats` / `--list-voices` never drift
  from what actually exists. Keep it that way.

## Deliberately unimplemented

`_format_lock()` in `custom_nodes/retro_sfx/nodes.py` raises `NotImplementedError`
**on purpose**. It is the audio twin of pixelmon's palette lock — forcing a
waveform through a sample-rate/bit-depth/channel ceiling (`amiga`, `sb`,
`gameboy`, …) — and how it resamples and quantizes is what decides the tool's
character:

- naive stride decimation *aliases*, which is exactly what an Amiga sounds like;
  a clean anti-aliased resample throws that character away
- plain quantize is harsh and authentic; dithering first trades grit for a
  smoother floor
- staying at 8363 Hz is honest; coming back up to 44.1 kHz is practical

The repo owner is writing this one. **Do not fill it in unless asked.** Default
`--format none` works fully, so only the retro formats are gated on it.

`take_warning()` in `record.py` is the same deal, for the same reason. It decides
whether a recorded take looks bad enough to warn about before you accept it. The
clipping check is implemented because clipping is objectively wrong; everything
below it is **calibration against a specific mic, room and delivery**, which is
not something to guess at from outside:

- a "too quiet" floor depends on the mic's output and how close it is worked —
  and matters because `loudness_normalize()` only ever *attenuates*, so a quiet
  take stays quiet
- duration-vs-word-count catches stopping early and leaving it running, but the
  words/second band that means "wrong" for a grave dungeon-master read is very
  different from conversational narration
- the real trade-off is that a chatty warning you learn to ignore is worse than
  no warning at all

It returns `None` (accept silently) for anything it does not flag, so the booth
is fully functional without it. **Do not fill it in unless asked.**
