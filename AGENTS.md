# AGENTS.md

Guidance for AI coding agents working in this repository. Human-facing docs live
in [README.md](README.md); this file is the stuff that is expensive to rediscover.

## What this is

A CLI that turns a text description into audio, running entirely on local
hardware. **Three engines behind one command**, chosen by flag:

| Mode | Engine | Runs on | Output |
|---|---|---|---|
| default | Stable Audio Open 1.0 | ComfyUI / GPU | sound effects |
| `--music` | Stable Audio Open 1.0 | ComfyUI / GPU | loops, beds, stings |
| `--song` | ACE-Step 1.5 | ComfyUI / GPU | full songs, sung vocals |
| `--narrate` | Kokoro | **CPU, in-process** | spoken narration |

Sibling project to **pixelmon** (`~/pixelmon`, github.com/grymmjack/pixelmon),
which does the same thing for pixel-art sprites. soundmon deliberately mirrors
its architecture and reuses its ComfyUI install.

## Architecture — read this before adding an engine

`soundmon.py` never touches the GPU. It builds a ComfyUI graph as JSON, POSTs it
to `/prompt`, polls `/history/<id>`, and fetches results from `/view`. That is
the whole reason the render farm is ~150 lines and vendor-agnostic.

**The engines share a tail.** Both GPU pipelines land their finished `AUDIO` on
graph node `"10"`; `build_graph()` then attaches `RetroSFX` as `"11"` and a save
node as `"12"`. Adding a fourth GPU engine means writing a `_yourengine_nodes()`
that also ends at `"10"` — do not rewrite the tail.

```
_sfx_nodes()  ─┐
               ├─► node "10" (AUDIO) ─► "11" RetroSFX ─► "12" Save*
_song_nodes() ─┘
```

This is the same trick pixelmon's `--animate` uses: swap the pipeline, keep the
CLI. Follow it.

**`--narrate` is the deliberate exception** — it never touches ComfyUI. SFX and
songs are diffusion problems where ComfyUI earns its keep (model loading, VRAM,
the farm). Kokoro is an 82 M feed-forward model that speaks a line in ~1 s;
wrapping it in a node graph and fanning it across four GPUs would *add* latency.
It lives in `narrate.py` and is handed off early in `main()`, mirroring how
pixelmon hands `--animate` to `animate.py`. **Do not "unify" this into the graph
path.** One interface, the right mechanism per engine.

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

Aliases live in `servers.json` (gitignored — real LAN IPs stay out of the repo);
`servers.example.json` is the template.

## Working on this repo

- **Models are not in git** (~15 GB). `./download-models.sh` fetches SFX models,
  `--song` adds ACE-Step. It downloads to `.part` and renames on success, so an
  interrupted transfer can never masquerade as a complete file.
- **`install.sh` symlinks** this repo into `~/ComfyUI` — the repo stays the
  source of truth. Editing `custom_nodes/retro_sfx/nodes.py` requires a
  **ComfyUI restart** to take effect, and on every farm box, or they will reject
  graphs using a new node input.
- **No test suite.** Verify by generating and measuring: check duration, peak
  dBFS, sample rate, and print an amplitude envelope. An envelope catches
  "generated silence" and "generated noise instead of the thing" faster than
  listening does. Semantic sanity is visible there too — footsteps show as
  discrete spaced impacts, a drone as a flat bar.
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
