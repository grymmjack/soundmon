# AGENTS.md

Guidance for AI coding agents working in this repository. Human-facing docs live
in [README.md](README.md); this file is the stuff that is expensive to rediscover.

## What this is

A CLI that turns a text description into audio, running entirely on local
hardware. **Nine engines behind one command**, chosen by flag:

| Mode | Engine | Runs on | Output |
|---|---|---|---|
| default | **Stable Audio 3 medium** | ComfyUI / GPU | sound effects |
| `--music` | **Stable Audio 3 medium** | ComfyUI / GPU | loops, beds, stings |
| `--engine sao` | Stable Audio Open 1.0 | ComfyUI / GPU | legacy fallback |
| `--song` | ACE-Step 1.5 | ComfyUI / GPU | full songs, sung vocals |
| `--narrate` | Kokoro | **CPU, in-process** | spoken narration |
| `--record` | **a human + a microphone** | CPU, in-process | spoken narration |
| `--blip` | **an oscillator** (synth) / Kokoro (voice) | CPU, in-process | JRPG text-box blips |
| `--chip` | **2A03 synthesis** (pure Python) | CPU, in-process | authentic chiptune music |
| `--opl` | **Nuked OPL3** via ctypes | CPU, in-process | authentic AdLib/OPL3 music |
| `--chipfx` / `--oplfx` | PSG / OPL3 recipes | CPU, in-process | authentic retro SFX |

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

## Diffusion is the wrong tool for chip music — know when to stop prompting

The most valuable judgement call in this repo's history: **when a format is
defined by a constraint, no prompt reaches it.**

Stable Audio 3 asked for "chiptune" returns modern chiptune-influenced music —
DAW-made, reverb, sampled drums, unlimited polyphony. That is a real genre and
the model renders it well. It is not a 2A03. Real chip music is not a style the
model imitates; it is a hardware limit:

    2A03    2 pulse + 1 triangle + 1 noise, no reverb, nothing sampled
    OPL3    FM operator ratios, envelopes, feedback

A model trained on recordings has no representation of an operator ratio and no
way to honour a channel count. Prompting harder produced "closer, still wrong"
for a full round-trip before this was accepted. `--chip` and `--opl` synthesize
directly instead, which makes them authentic *by construction*: there is no code
path that emits reverb, so it cannot.

**Both share `chip.compose()` and nothing else.** Composition returns note
events with no notion of voicing; each renderer consumes them. Keep that seam —
the musical decisions are the hard part and are worth writing once, while voicing
is exactly what differs per chip. A SID or YM2612 is a new renderer, not a new
composer. Do not "unify" the renderers; they have nothing in common but their
input.

**Both loop with no crossfade.** Whole bars, so the last sample runs into the
first. `loop_wrap()` exists to hide a composed ending and these have none, so
`--loop` is explicitly ignored for them. Do not "fix" that by wrapping anyway —
it would shorten the track to solve a problem that isn't there.

**Archetype selection is a TABLE, not inference.** `chipfx.NAME_MAP` names all
46 of the caller's effects explicitly; keyword matching is only the fallback.
Substring matching on short words is unsafe in a namespace you do not control —
"dice" contains "ice", so `diceroll` classified as a frost shimmer. When the asset
list is finite and known, a table is faster, inspectable, and lets each effect be
hand-tuned rather than approximated.

**`--opl` is fetched, not vendored.** Nuked OPL3 is LGPL-2.1; this repo is MIT.
`./download-models.sh --opl` fetches and compiles it into `vendor/` alongside its
licence. Do not commit the C source to make setup shorter — that quietly
relicenses someone else's work, and the models set the precedent for
fetch-at-setup already.

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


17. **A tracker file's TEMPO must come from the composition, never from `--bpm`.**
    Both `write_mod` and `write_rad` took `bpm=int(a.bpm)`, which for `--from-midi`
    is an unrelated CLI default: a 75 bpm source shipped as 120 and played **1.6x
    too fast**. Row duration is `ticks * 2.5 / bpm` in BOTH formats (RAD's "50 Hz
    timer" is exactly BPM 125 at that divisor), so derive both fields from
    seconds-per-bar and never accept a tempo argument.

    Corollary: one BPM byte cannot express a fine row grid. 96 rows/bar at 75 bpm
    needs BPM 450 at the default 6 ticks. Solve for ticks AND bpm together.


18. **Tracker output is where "the bytes look right" fails hardest.** Every defect
    in the MOD/RAD writers was a correct-looking file with a wrong interpretation,
    and none was visible by decoding the pattern data back:

    - the period table spanned three octaves, so `_period` **folded** 12 of 328
      notes an octave — melody peaks became different notes
    - a 64-byte looping cycle at period 428 is **17.7 cents flat** on every note;
      fix with cycle length + the finetune field, not a longer cycle
    - `3xx` tone portamento's parameter is **period units per tick**, not
      semitones — and period distance depends on register, so it must be computed
      from the actual endpoints or the slide never reaches its target
    - `3xx` deliberately does **not** retrigger, so a glide after a `C00` release
      plays at volume zero. Only slide into a note the previous one ran into
    - a release placed at onset+duration lands **inside the next note** on a
      monophonic channel and cuts it off; clamp it to the next onset
    - RAD numbers C as **12**, which is a note-FIELD quirk only. Treating it as an
      octave boundary put every C an octave below its neighbours — wrong for
      exactly one of twelve notes, which is the hardest kind to notice
    - RAD **has** a key-off (note value 15). Without it an OPL voice never
      releases and sustains through the rest of the tune

    `tools/render-mod.py` renders through **libxmp — the same replayer
    pixel-viewer uses**, so it settles these. `tools/render-rad.py` drives Nuked
    OPL3 and catches silence/octave/tempo faults, but it shares this repo's
    reading of the RAD spec, so it cannot prove that reading is right.


19. **Tick 0 of a MOD row does not slide.** It reads the cell and sets the
    target; only ticks 1..speed-1 advance the pitch. A row therefore contributes
    **speed-1** sliding ticks, so at speed 3 a two-row slide has a budget of 4
    ticks, not 6. Dividing by the wrong budget makes every rate too slow — and
    because tone portamento CLAMPS at its target, a slide that runs out of rows
    does not overshoot or retry, it stops and stays there. Every affected note is
    left permanently out of tune. At speed 1 there are no sliding ticks at all,
    so do not emit a portamento.

    Corollary for the whole file: **an effect that needs N rows must be written to
    N rows.** Portamento and vibrato continue with a zero parameter; arpeggio has
    no continue form, because parameter 0 IS no-effect, so its parameter must be
    repeated in full.


20. **A verifier that shares an assumption with the writer is not a verifier.**
    This one cost three failed fixes. `mod.py` computed portamento rates with the
    wrong tick model, and the check that was supposed to catch it SIMULATED
    ProTracker with the same wrong tick model — so it reported "83 slides, 0 land
    short" while the rendered audio sat 2.7 semitones below target and held. The
    belief was stated twice and confirmed zero times.

    What settled it was tracking pitch in libxmp's actual output and working
    backwards from the frequency: 47 period units had moved, not 71, and 47 is
    4 ticks at that rate, not 6. **Measure the artefact, not your model of it.**

    Two traps in doing that, both of which pointed the wrong way first:

    - **libxmp applies Amiga hard panning** — channels 0/3 left, 1/2 right. A
      single-channel comparison made the arp and bass look 50 dB down and the
      arpeggio look inaudible. Use the mono sum for level comparisons.
    - **The right channel carries arp AND bass mixed**, so a pitch tracker there
      locks onto whichever voice is louder and reports errors of tens of
      semitones. Only a voice alone on a channel is cleanly measurable; render
      one voice at a time when it matters.

    And when a musical judgement already exists in one renderer, do not
    reimplement it in another. `chip.py` had working slide rules — leap 3..19,
    notes within 120 ms, fixed 45/60/80 ms, melodic line only — and `mod.py`
    invented its own (leap 1..7, strict legato, "up to 3 rows"), getting every
    parameter wrong including the spurious-glide case `chip.py` explicitly guards
    against.

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

**A job ComfyUI fails is seen at once.** `/history` only grows `outputs` on
success; a node that raises leaves `status.status_str == "error"` and nothing
else, so waiting for outputs alone sat out the full 1800 s timeout. `classify()`
turns a `/history` entry into ok / error / poisoned / transient / **cached** /
pending, and both `wait()` (single server) and `poll()` (farm) act on it. On the
farm a failure feeds the circuit breaker (gotcha 16): a box that fails twice in a
row — or most of its jobs — is dropped, a failed job is retried once on
**another** box, and idle boxes pick up requeued work. selfcheck asserts the
classifier.

**The job lifecycle (ported from pixelmon's `wait()`, keep the two in step):**

- **"Completed, no outputs" is not a failure.** ComfyUI answers a prompt it has
  already run, unchanged, from its cache with success and nothing saved
  (`_cached_empty`). Re-running a seed — the help's own advice — hit it. Answer it
  with `fresh_copy()` (same graph, save node under `rerun-<id>/`: only the save
  re-runs, the file name and seed stay), and on the farm **never count it toward
  the breaker** — it dropped healthy boxes.
- **A restart forgets the whole queue.** `wait()` rides out an unreachable server
  for `SERVER_GRACE` (240 s), notices a prompt the server no longer knows
  (`_known`, three checks, or one right after an outage) and resends it, max
  twice; `main()` then resends every other queued job the server forgot
  (`resend_lost`) instead of letting each sit out the check. The single-server
  loop keeps each job's graph for exactly this.
- **POISONED** (`object has no attribute 'model_size'`/`'is_dynamic'`): the box
  fails every prompt until restarted. `wait()` waits (120 s) for someone — the
  GUI's `_restart` — to restart it, then resends. **Transient** (a loader
  IndexError, out of memory): one retry.
- **Time queued behind other jobs doesn't count** toward the timeout.
- **Farm polls get `FARM_GRACE` (30 s).** Requeueing on the first failed poll
  sent the job elsewhere while it was still rendering — one flaky box issued 75
  prompts for one job. After the grace the box is out of the run.
- **A fetch or post-processing error fails one job**, never the farm; downloads go
  to `.part` and are checked against Content-Length.
- **Ctrl-C / SIGTERM / an early exit** remove this run's prompts from every
  server's queue and interrupt only a running prompt that is ours (`SENT`,
  `cancel_sent`).

**Capability is not the only way a box is wrong.** A box with every model present
but ComfyUI running *without* `--force-fp32` passes `server_has_ckpt()` and then
returns static for every SA3 job — quickly, so dispatch sends it more work.
ComfyUI's `/system_stats` reports its launch arguments as `system.argv`, so
`fp32_warning()` checks it (once per box per run) and prints a `⚠` for SA3 runs.
It warns, it doesn't refuse: an older ComfyUI may not report argv (then nothing
is said), and a box can be fp32 by other means. Still treat "one box's output
sounds different" as a launch-flag question before a model question.

**Target resolution is one rule, in three places:** `--server` > `$SOUNDMON_SERVER`
> servers.json `"_default"` > the `local` alias (`pick_target()`; `bin/soundmon`
resolves the same way to decide whether to start a local ComfyUI; the GUI reads
`_default` too). Keep them agreeing — `install.sh` copies pixelmon's servers.json,
so a `_default` the CLI ignored meant renders on the wrong box.

**Everything after the server works on WAV.** The graph always ends in `SaveSFX`;
`--flac/--mp3/--opus/--ogg` are encoded client-side, last, in `encode()`. Saving
FLAC/MP3 on the server and then running the loudness ceiling, `--loop` or
`--match-tone` wrote WAV data back under the `.flac`/`.mp3` name.
`finish_files()` is the one post-server tail for both paths: tone match → loop →
loudness/true-peak ceiling → encode → `--batch-file` name → recipe tag.

**The recipe goes in the file** (`recipe()`, `tag_file()`): a RIFF LIST/INFO
ICMT chunk appended to WAVs by byte surgery (ffmpeg remuxing would rewrite 24-bit
headers to WAVE_FORMAT_EXTENSIBLE), a comment tag elsewhere. The CPU engines are
tagged by `run_engine()`, which watches `soundfile.write` while the engine runs.
selfcheck asserts the samples are untouched.

Aliases live in `servers.json` (gitignored — real LAN IPs stay out of the repo);
`servers.example.json` is the template.

## Steering with a reference (`--ref`) and finding similar sounds (`similar.py`)

- **`--ref` changes the start of the graph and nothing else.** `_apply_ref()` deletes
  node "9" (the empty latent) and adds `"30": LoadAudio` → `"31": VAEEncodeAudio`
  (VAE from the checkpoint at `["4", 2]`, the same in the SA and song graphs) into
  KSampler, with `denoise = --ref-strength`. The decode/RetroSFX/save tail is unchanged.
  selfcheck asserts the wiring for both graphs.
- **The upload goes to the top of `input/`**, not a subfolder: LoadAudio only lists
  files directly in `input/`. `prepare_ref()` names the file
  `soundmon_ref_<sha1 of content + seconds>.wav`, so a batch or a farm uploads the same
  name once per box. `run_farm()` uploads to every live box before dispatching, and
  drops a box that refuses, like an unreachable one.
- **Only the wiring is proven offline.** Whether SA3's VAE encodes well, and how its
  8-step lcm behaves at denoise < 1, needs a GPU run. If it doesn't work, `--engine sao`
  is the fallback; don't add a silent fallback.
- **`--match-tone` / `--master-like` use `master.match_tone()`**: the 1/3-octave curve
  difference (each curve normalised by its own 125 Hz–8 kHz mean), smoothed, ±9 dB,
  eased at the extremes, through ffmpeg `firequalizer`, then the gain step. As with
  `master()`, that gain step is the only −1 dBTP guard.
- **`similar.py` imports nothing heavy at module level** (the GUI imports it). Its
  fingerprint is DSP only, from the audible part of the file (trimmed at −50 dB):
  game sounds are often padded with silence, and before the trim the padding decided
  the match. **Bump `METHOD` whenever `features()` changes**: an index made with
  another method rebuilds itself rather than mixing two kinds of vector. The GUI's
  `/api/similar/file` serves only paths that are in the index, never arbitrary paths.

## Run tools/selfcheck.py after touching an engine

Assertions over the defects that broke SILENTLY in this repo — every one of
which was found by ear, not by inspection:

    the -16 LUFS ceiling applying to music, and then a fix for it that sat AFTER
    the engine hand-offs and so never executed
    format-locked 8-bit audio promoted back to 16-bit by ffmpeg's default
    -21 dBFS of DC from thin duty cycles
    arpeggios reaching into the bass register
    every pack composing the identical piece

Writing a lesson down does not install it. AGENTS.md gotcha 10 (hand-offs return
early) was written earlier in the same session that it then caught me out again —
an assertion re-checks it at the moment it matters, and costs nothing.

It is NOT a substitute for listening. It catches values that are silently wrong;
it cannot tell you whether the music is any good. Audio quality still needs ears,
and this session's entire quality arc came from someone listening.

One trap worth knowing when writing new checks: **--output-to feeds pack
identity**, so rendering the same seed into two directories deliberately produces
two different compositions. My first FLAC check did that and "failed" on a
7436-frame length difference that was the feature working correctly.

## Working on this repo

- **Models are not in git** (~29 GB across all engines). `./download-models.sh
  --sa3` fetches the default engine + the Qwen rewriter, `--song` adds ACE-Step,
  `--sao` the legacy Stable Audio Open set (also what the bare command fetches,
  for old scripts), `--opl` builds the OPL core; flags combine, `--list` shows
  what's missing. It downloads to `.part` and renames on success, so an
  interrupted transfer can never masquerade as a complete file, and exits
  non-zero naming whatever is still missing.
- **Check arguments in `check_args()`, never next to the diffusion setup.** The
  CPU engines return from `main()` early (gotcha 10), so validation placed below
  them never ran for them (`--mood bogus`, `--bpm 0`, a 270 s `--blip-rate 0`).
  And decide "did the user choose this?" with `given()`, never by comparing to the
  default: a typed `--seconds 10` was read as "no length given" (→ 120 s songs),
  and a typed `--cfg 5` on SA3 was silently replaced. The parser has
  `allow_abbrev=False` so `given()` can't be fooled by `--sec 3`.
- **"No loudness ceiling" is `+inf`, not `None`.** The engines skip
  `loudness_normalize()` on `None`, and it is also the only place `--true-peak`
  is enforced — so chip music and `--lufs off` shipped above -1 dBTP. Files
  under 0.4 s skip only the LUFS part.
- **Keep the GUI's last-line contract:** a failed run's LAST line is what the GUI
  stores as the error, so put the essential facts (which box, which model, the
  `download-models.sh` fix) on one line. Every result line keeps `[i/n]` (even
  `[1/1]`) and `seed=`.
- **Releases**: `.claude/skills/create-release/SKILL.md`. `__version__` is in
  `soundmon.py`.
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

## The GUI (`gui/`, `bin/soundmon-gui`)

A sibling of pixelmon-gui and built the same way: `gui/server.py` is a stdlib
`ThreadingHTTPServer` (port 8191) that runs **`bin/soundmon` as a subprocess**,
one job at a time, always with `--no-open --output-to <job folder>`; `gui/index.html`
is one no-build page. Rules that keep it honest:

- **The server builds the command.** `build_argv()` validates the form and the page
  shows its output (`/api/command`) under RENDER. There is no JS copy of the argv
  logic to drift. A new CLI flag needs a line there (or in `ADV` for the Advanced tab).
- **Lists come from the engines.** Moods, archetypes, OPL instruments and voices are
  imported fresh from `chip.py` / `chipfx.py` / `opl.py` / `narrate.py`; keys and
  negatives are evaluated out of `soundmon.py`'s source (`source_constant`), formats out
  of the node's (importing either pulls in torch). Flag help comes from the
  `add_argument` calls (`cli_spec`). Keep engine modules stdlib-only at import time.
- **Progress is the `✅ [i/n]` line** every engine prints; at exit the job folder is
  trusted over the log (`scan_outputs`). If you change an engine's result line, keep
  the `[i/n]` and `seed=` parts.
- **Every job keeps its recipe** in `~/soundmon-gallery/gui/<job>/job.json` (params,
  argv, command, each take's seed), plus copies of any input MIDI / audio / bank.
- **Sets (`gui/sets.py`) store no takes.** A set is a frozen manifest snapshot +
  theme + picks/flags in `~/soundmon-gallery/sets/<id>/set.json`; a render job carries
  `params.set = {id, key}`, and a key's takes are found by scanning the gallery. Keep
  it that way: a second copy of the takes would drift. Export only ever writes inside
  the set's game folder, and copies whatever it replaces to `.soundmon-backup/` first.
  Blips and narration are deterministic (same text, same sound), so their sets make one
  take per key, or one per voice, rather than seeds.
- **Mastering (`master.py`) is shared by the CLI (`--master`) and the GUI** (measures in
  Sets, ▶ Mastered, master on export). Mastered previews go to `<job>/_mastered/<kind>/`
  and measurements to `_analysis/`; `scan_outputs` skips `_` folders so they never show
  up as takes. The chain has **no limiter on purpose**: the final gain step is the only
  thing keeping a sound under −1 dBTP, and selfcheck asserts it. A lossy export (OGG)
  can overshoot again between samples, so `sets.convert()` measures the OGG it wrote
  and re-encodes quieter if needed.

## The format lock — the owner's call

`_format_lock()` in `custom_nodes/retro_sfx/nodes.py` is the audio twin of
pixelmon's palette lock — forcing a waveform through a sample-rate/bit-depth/
channel ceiling (`amiga`, `sb`, `gameboy`, …). The repo owner wrote it, and its
two choices decide the tool's character:

- naive stride decimation, **no anti-aliasing**: the aliasing is what an Amiga
  sounds like; a clean resample throws that away
- plain quantize, **no dither**: harsh and authentic; dither would smooth the
  floor a hardware ceiling exists to keep rough

**Do not "improve" either into the polite version unless asked.**

## Deliberately unimplemented

`take_warning()` in `record.py` is left to the repo owner on purpose. **Do not
fill it in unless asked.** It decides
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
