# INFINITE SAMPLES — from a prompt to a sampler

soundmon writes music from a description. `tools/stems.py` takes a finished track apart: the
**stems** (drums, kick, snare, hats, bass, synth layers, the reverb), **loops** of 1, 2 and 4 bars
of every part, **one-shots** (drum hits, bass and synth notes named by pitch, chord stabs, held
pads) and **FX** (risers, falls, the reverb tail). Every file is a 44.1 kHz 16-bit stereo WAV with
its loop points and tempo written inside, so a sampler or DAW loops and syncs it as it loads.

Describe ten tracks, get ten sample packs. Describe a hundred... that's the point.

```
"busy neon night market, glitchy synth stabs, ..."      a prompt
        │  soundmon --music                              (Stable Audio 3, on the farm)
        ▼
new3-03_47s_none_s737245421_00001_.wav                   a 47 second track
        │  stems.py                                      (Demucs, BS-Roformer, DrumSep, de-reverb...)
        ▼
new3 - NEON MARKET/
  1 Stems/        15 parts, each the whole track long
  2 Loops/        143 loops, cut on the bar lines, tempo in the file
  3 One-shots/    59 hits and notes, root notes set
  4 FX/           risers, falls, the ending's tail
```

The first pack (NEON-CASTER's `new3`, 2026-10-23) took 47 seconds of music to 224 samples in
under two minutes on an RTX 3070.

---

## 1. Writing a music prompt

`--music` is Stable Audio 3's music mode: loops, beds and stings up to 47 seconds, no vocals.
Before the music model sees your words, soundmon has a small language model (Qwen, its
`TextGenerate` step) rewrite them into a full music prompt — genre, instruments, layers, rhythm,
mood, a BPM. So write *what it's for and how it feels*, and let the rewrite fill in the
musicianship.

A prompt that worked, for every NEON-CASTER sector:

```
THEME, STYLE, dark electronic techno, instrumental video game level music, seamless loop
```

- **THEME** — the place or the moment, in a few concrete images:
  `busy neon night market, glitchy synth stabs, street bustle energy, pulsing bassline`
- **STYLE** — a genre with a reference or two. The ten used for the game's music (one a take):

  | # | style |
  |---|---|
  | 1 | dark cyberpunk techno, hard driving kick |
  | 2 | industrial EBM, aggressive distorted synths like Front 242 and Nine Inch Nails |
  | 3 | dark Berlin warehouse techno, hypnotic rumbling kick |
  | 4 | hard industrial techno, distorted kick, metallic percussion |
  | 5 | dark electro, menacing 808 electro like Drexciya |
  | 6 | dark acid techno, squelching 303 lines |
  | 7 | darksynth techno, heavy distorted synth bass like Perturbator |
  | 8 | dark minimal techno, hypnotic and tense |
  | 9 | dark techno with breakbeats, big beat edge like The Prodigy |
  | 10 | dark driving electro techno, building arpeggios like Carpenter Brut |

- **The tail** — what it is and how it's used: `instrumental`, `video game level music`,
  `seamless loop`. "Instrumental" keeps voices out; "seamless loop" keeps the end from fading.

Tips:

- **For samples, ask for separable music.** Few, distinct instruments separate cleanly; a wall of
  sound doesn't. `sparse`, `minimal`, `punchy kick`, `clean synth bass`, `dry` all help the stems.
  Reverb and swooshes come out as their own layer (`reverb`, `4 FX`), which is fine — but the
  drier the track, the cleaner the notes.
- **Name the instruments you want to sample**: `808 bass`, `303 acid line`, `FM bell`,
  `supersaw pad`, `plucked arpeggio`. They're more likely to be there, and loud.
- **Make many takes.** Generation is cheap: `-n 10` makes ten, each its own seed. Keep the good
  ones; the seed is in every file name.
- **Make it on rtx**, not the Mac: the Mac's renders of music came out as noise.

## 2. Making the takes

```bash
soundmon "busy neon night market, glitchy synth stabs, street bustle energy, pulsing bassline, \
dark Berlin warehouse techno, hypnotic rumbling kick, dark electronic techno, \
instrumental video game level music, seamless loop" \
  --music --seconds 47 -n 10 --server rtx --name new3 --output-to music/new3 --create-dirs --no-open
```

- Keep the **WAV** (the default) for stems; `--ogg` is for shipping, not for taking apart.
- `--seconds 47` is the model's longest. Longer tracks give more bars to choose loops from.
- A whole set: loop over a list of `key|name|theme` lines and the ten styles (take N = style N).
  NEON-CASTER's own was `gen-rtx.sh` (kept in its generation records).

**Is it music?** Some takes come out as noise. Two numbers sort them: spectral flatness under 0.3
(not noise) and a beat — the onset envelope's autocorrelation over 0.4 somewhere in 0.3-1.0 s.
NEON-CASTER's `musicqc.py` does exactly that, and kept re-rolling a take until it passed.

### Making one again, exactly

soundmon's prompt rewrite has a fixed seed, so the same command gives the same rewrite — but the
full recipe (the rewrite step, the model files, steps, sampler, the seed) lives only in ComfyUI's
history on the server, which is **lost when ComfyUI restarts**. Save it:

```bash
curl -s 'http://192.168.1.77:8188/history?max_items=1000000' > history.json
```

Each job's `prompt` is the API graph. POST it back to `/prompt` (with a new `filename_prefix` in
its save node) and the same take comes out — checked byte for byte on a NEON-CASTER track.
NEON-CASTER keeps this automatically (`~/Documents/NEON-CASTER/generation/backup.py`, every 4
hours); `--seed N` with the same prompt also re-makes a take, as long as the models haven't changed.

## 3. Taking it apart: stems, loops, one-shots

`tools/stems.py` runs on the farm box with the GPU (daw: an RTX 3070 under WSL2), in its own
Python environment so ComfyUI's is never touched.

### Once: set it up

```bash
ssh daw
mkdir -p ~/stems && cd ~/stems
python3 -m venv .venv
.venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install "audio-separator[gpu]" librosa soundfile imageio-ffmpeg
ln -sf "$(.venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" .venv/bin/ffmpeg
```

(The separator needs `ffmpeg` on the PATH; `imageio-ffmpeg` brings one without sudo.) Copy the
script over: `scp ~/git/soundmon/tools/stems.py daw:stems/`. The models (~2.3 GB) download into
`~/stems/models` on the first run.

### Each track

```bash
scp new3.wav daw:stems/
ssh daw 'cd ~/stems && PATH=$HOME/stems/.venv/bin:$PATH \
  .venv/bin/python stems.py new3.wav "out/new3 - NEON MARKET" --name new3 --title "NEON MARKET"'
rsync -a "daw:stems/out/new3 - NEON MARKET" ~/Music/NEON-CASTER/samples/
```

`--bpm N` hints the tempo if the beat tracker picks half or double time.

### What it does

| step | how |
|---|---|
| 1. Split | **Demucs htdemucs_ft** (drums, bass, other, vocals) · **BS-Roformer SW** (guitar, piano - synth parts land there too) · **MDX23C DrumSep** on the drums (kick, snare, toms, hats, ride, crash) · **MelBand Roformer de-reverb** on "other" (its dry sound and its reverb) |
| 2. The synths | no model knows one synth from another, so the dry "other" is split two more ways: **by brightness** (NMF: its 18 sound components sorted by spectral centroid into three layers of a third of the sound each - synth low, mid, high) and **sustained vs struck** (HPSS - synth held, synth hits) |
| 3. Tempo, bars | the beat tracker's beats, a straight line fitted through them (AI tracks keep steady time), the bar's first beat where the kick lands hardest; the end, where the music fades, is never looped into |
| 4. Loops | 1, 2 and 4 bars from every bar line, scored by how much what follows the loop's end sounds like its start (so the loop runs on as the track did); the three best that don't overlap, each part. The first 10 ms is blended from the audio that really followed the end - the seam is the track's own |
| 5. One-shots | onsets, each cut to the next (its tail trimmed 50 dB down). Drums: the four cleanest unlike hits (least bleed from the other drums). Bass and synths: pitch by pYIN, one sample per note (the steadiest, loudest, least drum-bled), the rest as chords and stabs; and the steadiest bar of each sustained part, looped - a pad |
| 6. FX | stretches of half a second or more where the brightness climbs or falls steadily (risers, falls, filter sweeps - from "other", the bright layer, the hits and the reverb), and the ending's tail |

Parts that come out near-silent (a guitar in a techno track, say) are left out and listed in
`stems.json` (`left_out_quiet`).

## 4. What you get

```
new3 - NEON MARKET/
├── 1 Stems/          new3 - stem - bass.wav, ... - kick.wav, ... - synth low.wav, ... - reverb.wav
├── 2 Loops/
│   ├── Bass/         new3 - bass loop - 2 bars - 135.9 bpm - bar 11.wav
│   ├── Kick/ Snare/ Hats/ Toms/ Drums/ Full Mix/
│   └── Synth Low/ Synth Mid/ Synth High/ Synth Held/ Synth Hits/ Synth Dry/ Synths + FX/ Reverb/
├── 3 One-shots/
│   ├── Kick/         new3 - kick 01.wav ... 04
│   ├── Bass/         new3 - bass - D1.wav, new3 - bass - E1.wav, new3 - bass - chord 01.wav
│   ├── Synth Mid/    new3 - synth mid - C4.wav, ... - held bar 19.wav (a looped pad)
│   └── ...
├── 4 FX/             new3 - riser - from reverb 03.wav, new3 - ending tail - full mix.wav
└── stems.json        tempo, bar grid, every file and where in the track it came from
```

- The names say the track, the part, then the note or the bars, tempo and the bar it starts on.
  Notes use `s` for sharp (`Cs4`), so file systems and samplers never trip on `#`.
- **44.1 kHz, 16-bit, stereo WAV**, levels as they came out of the split - nothing normalized.
- **Loop points are in the file** (`smpl` chunk: loop start and end, the root note) for every
  loop and every held pad - the standard WAV loop most samplers read (Renoise, Kontakt, the MPC
  and most hardware ones; a DAW that ignores it just plays the file, which is the loop anyway).
- **Tempo and length are in the file** (`acid` chunk: BPM, beats, stretch on) for every loop, so
  a DAW that reads ACID WAVs warps it to the project tempo as it lands. One-shots carry the
  one-shot flag and their root note.

## 5. Using them

- **Drum kit**: drop `3 One-shots/Kick`, `Snare`, `Hats`, `Toms` onto pads.
- **Playable bass or synth**: load a folder's notes into a multi-sample instrument - the root
  notes are set, so each lands on its key and the sampler stretches between them.
- **Pads**: the `held bar` files loop as they are - hold a key.
- **Loops**: the bar number in the name is where in the track it came from; loops of different
  parts from the same bar play together as the track did.
- **Rebuild and remix**: the stems line up sample for sample with the original and each other -
  mute, re-order, re-effect.

## 6. What to expect

Separation isn't magic. Expect:

- **bleed** - a little hi-hat in the synths, a little bass in the kick;
- **reverb and FX coming along** - the de-reverb takes most of it to its own layer, not all;
- **synth layers that are layers, not instruments** - two synths of the same brightness stay
  together;
- **the odd wrong note name** - pitch tracking on a separated part can catch a harmonic;
- **"vocals" in an instrumental** - a voice-like lead or a vocoder-ish synth.

It's raw material: food for the muse, not studio stems.

## 7. Next

- `soundmon --stems FILE...` (and `--stems` straight after `--music`): the same, from the CLI,
  run on the farm like everything else.
- A whole set in one go, with a report page to audition each pack.
- More stems where models exist (a dedicated synth/keys model when one appears).
