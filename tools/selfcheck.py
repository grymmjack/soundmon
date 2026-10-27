#!/usr/bin/env python3
"""Assert the things that broke silently. Run after touching an engine.

WHY THIS EXISTS

Every defect in this repo's recent history was invisible to inspection and only
surfaced by ear:

    a -16 LUFS ceiling that should not apply to music, and then a fix for it that
    sat AFTER the engine hand-offs and so never ran at all
    8-bit format-locked audio silently promoted back to 16-bit by ffmpeg
    -21 dBFS of DC from thin duty cycles
    note durations clamped at bar lines
    note_off matched by pitch, so doubled notes killed each other
    two renderers drifting apart because a fix landed in only one

Documenting a lesson does not install it — I wrote the hand-off gotcha in AGENTS.md
and then walked into it the same day. An assertion re-checks it every time, at the
moment it matters, for free.

These are cheap invariants, not a substitute for listening. Audio quality still
needs ears; this only catches the class of bug where a value is silently wrong.

    tools/selfcheck.py            # all checks
    tools/selfcheck.py -v         # show each result
"""
import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
PY = os.path.expanduser("~/ComfyUI/.venv/bin/python")
SM = os.path.join(HERE, "soundmon.py")

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return ok


def _render(tmp, *args):
    """Run soundmon and return the single file it produced."""
    before = set(os.listdir(tmp))
    r = subprocess.run([PY, SM, "test tone", "--output-to", tmp, "--no-open",
                        "--seconds", "4", *args],
                       capture_output=True, text=True)
    made = [f for f in os.listdir(tmp) if f not in before]
    if not made:
        return None, r.stderr[-300:]
    return os.path.join(tmp, sorted(made)[0]), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import chip
    import theory

    # --- pure-logic invariants ----------------------------------------------
    # Arpeggios must never reach the bass. A fast arp below ~A3 is mud, not a
    # chord, because the ear stops fusing the tones.
    notes = [{"id": i, "t": 0.0, "dur": 1.0, "pitch": p, "prog": 0, "vel": 100}
             for i, p in enumerate((36, 40, 43, 62, 65, 69))]
    out = chip.chippify(notes, "max")
    arped = [n["pitch"] for n in out if isinstance(n.get("id"), tuple)]
    check("chippify never arpeggiates below ARP_MIN_PITCH",
          all(p >= chip.ARP_MIN_PITCH for p in arped),
          f"lowest arped = {min(arped) if arped else 'none'} "
          f"(floor {chip.ARP_MIN_PITCH})")

    # Pack identity must change the plan, or every pack composes one piece.
    mood = chip.MOODS["solemn"]
    p1 = theory.plan_track("packA/level9", mood, "solemn", "minor", 12)
    p2 = theory.plan_track("packB/level9", mood, "solemn", "minor", 12)
    check("pack identity changes the composition",
          (p1["progs"], p1["form"]) != (p2["progs"], p2["form"]),
          "same track name in two packs must not produce one plan")

    # DC blocking must actually remove DC.
    x = np.ones(4096) * 0.5
    check("_dc_block removes a constant offset",
          abs(float(chip._dc_block(x, np, 44100)[2048:].mean())) < 0.02,
          "a 12.5% duty pulse sits at a mean of -0.75")

    # A ComfyUI job that raises gets status "error" and NO outputs. Waiting for
    # outputs alone sat out the full 30-minute timeout on every dead job.
    sys.path.insert(0, HERE)
    import soundmon as _smj
    err = {"outputs": {}, "status": {"status_str": "error", "completed": False,
           "messages": [["execution_error", {"node_type": "CLIPLoader",
                                             "exception_message": "missing model"}]]}}
    running = {"outputs": {}, "status": {"status_str": "success", "completed": False}}
    empty = {"outputs": {}, "status": {"status_str": "success", "completed": True}}
    check("a failed ComfyUI job is seen as failed, not still running",
          bool(_smj._job_error(err)) and "CLIPLoader" in _smj._job_error(err)
          and _smj._job_error(running) is None and bool(_smj._job_error(empty)),
          f"error -> {_smj._job_error(err)!r}")

    # --ref: both AI graphs must start from the encoded reference instead of silence:
    # node "9" (the empty latent) gone, LoadAudio -> VAEEncodeAudio (the checkpoint's own
    # VAE) into KSampler, and denoise == --ref-strength. Wiring checked offline; whether
    # each model's VAE encodes well is only provable on a GPU box.
    from types import SimpleNamespace
    ga = SimpleNamespace(
        prompt="door", style_add="", music=False, negative="x", style_neg="", batch=False, name=None,
        song=False, bpm=120, format="16bit", seconds=2.0, base="b", text_encoder="t", steps=8, cfg=1,
        sampler="lcm", scheduler="simple", no_trim=False, max_seconds=0, threshold_db=-50,
        normalize_db=-1, fade_ms=5, engine="sao", no_reprompt=True, flac=False, mp3=False, opus=False,
        bits=16, ref_name="soundmon_ref_x.wav", ref_strength=0.4, song_base="s", lyrics="",
        no_audio_codes=True, timesig=4, lang="en", key="C major", llm_cfg=2, temperature=1, top_p=1,
        top_k=0, min_p=0)
    okg = []
    for song in (False, True):
        ga.song = song
        g = _smj.build_graph(ga, 1)
        okg.append("9" not in g and g["30"]["class_type"] == "LoadAudio"
                   and g["31"]["inputs"] == {"audio": ["30", 0], "vae": ["4", 2]}
                   and g["3"]["inputs"]["latent_image"] == ["31", 0] and g["3"]["inputs"]["denoise"] == 0.4
                   and "_ref_" in g["12"]["inputs"]["filename_prefix"])
    check("--ref wires LoadAudio -> VAEEncodeAudio -> KSampler at --ref-strength (SFX + song)",
          all(okg), f"sfx/song ok: {okg}")

    # Mastering must calm a bright sound without ever pushing it past -1 dBTP:
    # the chain has no limiter on purpose, so the gain step is the only guard.
    import shutil as _shm
    if _shm.which("ffmpeg"):
        import master as _ms
        with tempfile.TemporaryDirectory() as mt:
            t = np.arange(int(44100 * 1.2)) / 44100
            x = (0.3 * np.sin(2 * np.pi * 3600 * t) + 0.25 * np.random.default_rng(1).standard_normal(len(t))
                 * np.exp(-t * 2) + 0.2 * np.sin(2 * np.pi * 140 * t))
            src, dst = os.path.join(mt, "bright.wav"), os.path.join(mt, "out.wav")
            sf.write(src, np.clip(x, -1, 1) * 0.95, 44100)
            b = _ms.analyse(src)
            _ms.master(src, dst, "sfx", b)
            a2 = _ms.analyse(dst)
            check("--master calms 2.5 kHz+ and stays under -1 dBTP",
                  a2["harsh"] < b["harsh"] - 0.1 and a2["true_peak"] <= -0.9,
                  f"above 2.5k {b['harsh']:.0%} -> {a2['harsh']:.0%}, true peak {a2['true_peak']} dBTP")

            # --match-tone: a bright sound EQ'd toward a dark reference must get darker
            # (its band curve closer to the reference's) and still never pass -1 dBTP.
            dark = os.path.join(mt, "dark.wav")
            lp = np.convolve(np.random.default_rng(2).standard_normal(len(t)), np.ones(24) / 24, "same")
            sf.write(dark, (0.6 * np.sin(2 * np.pi * 110 * t) + 2.0 * lp) * np.exp(-t) * 0.4, 44100)
            r_ = _ms.analyse(dark)
            like = os.path.join(mt, "like.wav")
            _ms.match_tone(src, like, dark)
            a3 = _ms.analyse(like)

            def gap(x):
                mid = lambda y: sum(y["bands"][8:20]) / 12     # noqa: E731
                return sum(abs((p - mid(x)) - (q - mid(r_))) for p, q in zip(x["bands"], r_["bands"]))
            check("--match-tone moves the band curve toward the reference, <= -1 dBTP",
                  gap(a3) < gap(b) and a3["true_peak"] <= -0.9,
                  f"curve gap {gap(b):.0f} -> {gap(a3):.0f} dB, true peak {a3['true_peak']} dBTP")

            # --ref: the reference is trimmed or padded to exactly the render length.
            _smj.REF_CACHE = os.path.join(mt, "refs")
            okl = True
            for want in (0.5, 3.0):
                got = sf.info(_smj.prepare_ref(src, want))
                okl &= abs(got.frames / got.samplerate - want) < 0.01 and got.channels == 2 \
                    and got.samplerate == 44100
            check("prepare_ref trims and pads the reference to --seconds", okl, "1.2 s -> 0.5 s and 3.0 s")

            # similar.py: a sound's near-twins (same kind, other seeds) must outrank other kinds.
            import similar as _sim
            lib = os.path.join(mt, "lib")
            os.makedirs(lib)
            rng = np.random.default_rng(3)
            t2 = np.arange(int(44100 * 0.6)) / 44100
            for i in range(3):
                f0 = 1400 + 120 * i
                sf.write(os.path.join(lib, f"zap{i}.wav"),
                         0.5 * np.sign(np.sin(2 * np.pi * f0 * np.exp(-t2 * 4) * t2)) * np.exp(-t2 * 5), 44100)
                sf.write(os.path.join(lib, f"thud{i}.wav"),
                         np.convolve(rng.standard_normal(len(t2)), np.ones(40) / 40, "same") * 3
                         * np.exp(-t2 * (18 + 3 * i)), 44100)
                sf.write(os.path.join(lib, f"pad{i}.wav"),
                         0.3 * np.sin(2 * np.pi * (220 + 7 * i) * np.arange(44100 * 2) / 44100), 44100)
            ix = _sim.Index(os.path.join(mt, "index.json"))
            ix.update([lib])
            ranks = []
            for kind in ("zap", "thud", "pad"):
                top = [os.path.basename(r["path"]) for r in ix.search(os.path.join(lib, f"{kind}0.wav"), k=2)]
                ranks.append(all(n.startswith(kind) for n in top))
            check("--similar ranks a sound's own kind first", all(ranks),
                  f"zap/thud/pad top-2 own kind: {ranks}")

    # --- tracker output -----------------------------------------------------
    # These four were ALL shipped broken and all were invisible to inspection:
    # correct-looking bytes, wrong interpretation. Only a replayer or arithmetic
    # catches them, so check the arithmetic here every time.
    import math
    import mod as modw
    import rad as radw

    # Tempo must come from the composition. It used to come from --bpm, so a
    # 75 bpm MIDI was written as 120 and played 1.6x too fast.
    worst, detail = 0.0, ""
    for spb in (3.2, 2.0, 1.714, 4.5):
        for rows in (16, 24, 32, 48, 96):
            for name, fn in (("mod", modw.timing_for), ("rad", radw.timing_for)):
                t, b = fn(spb, rows)
                err = abs(t * 2.5 / b - spb / rows) / (spb / rows)
                if err > worst:
                    worst, detail = err, f"{name} spb={spb} rows={rows} -> {t}/{b}"
    check("tracker row duration matches the source tempo", worst < 0.01,
          f"worst {worst*100:.2f}% off ({detail})")

    # The MOD period table must span real game-music range. Three octaves forced
    # octave FOLDING, which turned melody peaks into different notes.
    lo, hi = 41, 82
    folded = [p for p in range(lo, hi + 1)
              if not (modw.MIDI_BASE <= p < modw.MIDI_BASE + len(modw.PERIODS))]
    check("MOD periods cover MIDI 41-82 without folding", not folded,
          f"{len(folded)} of {hi-lo+1} notes outside the table")

    # A looping single cycle is only in tune if the length and finetune agree.
    # A 64-byte cycle at period 428 is 17.7 cents flat, on every note.
    f = 3546894.6 / 428 / modw.CYCLE_LEN * 2 ** (modw.FINETUNE / 96.0)
    cents = 1200 * math.log2(f / (440 * 2 ** ((48 - 69) / 12)))
    check("MOD sample tuning is within 3 cents", abs(cents) < 3.0,
          f"{cents:+.2f} cents (len {modw.CYCLE_LEN}, finetune {modw.FINETUNE})")

    # RAD numbers C as 12, which is a NOTE-field quirk only. Treating it as an
    # octave boundary put every C an octave below its neighbours.
    worst, detail = 0.0, ""
    for p in range(36, 84):
        note, octv = radw._note_octave(p)
        fnum = [0x157, 0x16B, 0x181, 0x198, 0x1B0, 0x1CA,
                0x1E5, 0x202, 0x220, 0x241, 0x263, 0x287][note % 12]
        got = fnum * 49716.0 / (1 << (20 - octv))
        c = abs(1200 * math.log2(got / (440 * 2 ** ((p - 69) / 12))))
        if c > worst:
            worst, detail = c, f"MIDI {p} -> note {note} oct {octv} = {got:.1f} Hz"
    check("RAD note/octave reconstructs the right pitch", worst < 25.0,
          f"worst {worst:.1f} cents ({detail})")

    # A release must never land inside the NEXT note. MOD and RAD channels are
    # both monophonic, so a release at onset+duration silences its successor.
    ev = {"lead": [(0, 0, 8, 60, 0.5, 100), (0, 4, 8, 64, 0.5, 100),
                   (0, 12, 2, 67, 0.5, 100)],
          "arp": [], "bass": [], "drum": []}
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, "t.mod")
        modw.write_mod(mp, ev, 1, 2.0, 16, np, rows_per_bar=16)
        d = open(mp, "rb").read()
        onsets, offs = set(), set()
        for r in range(64):
            b0, b1, b2, b3 = d[1084 + r * 16:1084 + r * 16 + 4]
            if ((b0 & 0x0F) << 8) | b1:
                onsets.add(r)
            elif (b2 & 0x0F) == 0xC and b3 == 0:
                offs.add(r)
        check("MOD release never lands on a following onset",
              not (offs & onsets) and offs,
              f"onsets {sorted(onsets)} releases {sorted(offs)}")

    # Vibrato depth is in PERIOD units, so one fixed depth is a swing that widens
    # without limit going up. A hard-coded 3 measured +/-162 cents at MIDI 80.
    worst, detail = 0.0, ""
    for p in range(36, 84):
        d = modw.vib_depth(p)
        if not d:
            continue
        per = modw._period(p)
        c = 1200 * math.log2(per / max(1.0, per - d * 2.0))
        if c > worst:
            worst, detail = c, f"MIDI {p} depth {d}"
    check("MOD vibrato swing stays under 45 cents", worst < 45.0,
          f"worst {worst:.0f} cents ({detail})")

    # A slide must ARRIVE. 3xx only acts on rows it is written to, so a rate sized
    # for N rows but written on one stops 1/N of the way: 83 of 83 slides landed
    # short, up to 500 cents from the written note.
    notes = [(0, i * 6, 6, 60 + (i % 5) * 2, 0.5, 100) for i in range(24)]
    with tempfile.TemporaryDirectory() as td:
        mp = os.path.join(td, "p.mod")
        modw.write_mod(mp, {"lead": notes, "arp": [], "bass": [], "drum": []},
                       3, 3.2, 96, np, chippy="max", rows_per_bar=96)
        d = open(mp, "rb").read()
        npat = d[950]
        cells = [d[1084 + pat * 1024 + r * 16:1084 + pat * 1024 + r * 16 + 4]
                 for pat in range(npat) for r in range(64)]
        ticks, _bpm = modw.timing_for(3.2, 96)
        cur = None
        i = 0
        tot = bad = 0
        while i < len(cells):
            b0, b1, b2, b3 = cells[i]
            per = ((b0 & 0x0F) << 8) | b1
            if (b2 & 0x0F) == 3 and cur and per and per != cur:
                tot += 1
                tgt, rate, pp, up, j = per, b3, cur, per > cur, i
                while j < len(cells) and (cells[j][2] & 0x0F) == 3:
                    rate = cells[j][3] or rate
                    # ticks - 1: tick 0 of a row sets the target and does not
                    # slide. Simulating `ticks` here is what made this check agree
                    # with a writer that was stopping every slide short.
                    for _ in range(max(1, ticks - 1)):
                        pp = pp + rate if up else pp - rate
                        if (up and pp >= tgt) or (not up and pp <= tgt):
                            pp = tgt
                            break
                    j += 1
                    if pp == tgt:
                        break
                bad += pp != tgt
                cur = tgt
                i = j
                continue
            if per:
                cur = per
            i += 1
        check("MOD tone portamento always reaches its target",
              tot > 0 and bad == 0, f"{bad} of {tot} slides land short")

    # --- end-to-end invariants ----------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        # THE ONE THAT KEEPS BREAKING: the loudness ceiling must not apply to
        # music, and the override must be set ABOVE the engine hand-offs, which
        # return early (AGENTS.md gotcha 10).
        p, err = _render(tmp, "--chip", "--bpm", "120")
        if check("--chip renders", p, err):
            au, sr = sf.read(p, always_2d=True)
            m = au[:, 0]
            peak = 20 * np.log10(max(abs(m).max(), 1e-9))
            check("--chip is NOT held down by the LUFS ceiling", peak > -2.0,
                  f"peak {peak:.2f} dBFS (a -16 LUFS ceiling lands near -3)")
            check("--chip output carries no DC", abs(float(m.mean())) < 0.02,
                  f"dc {float(m.mean()):+.4f}")

        # A format lock must actually change rate AND bit depth, and survive
        # loudness normalisation, which used to promote everything to 16-bit.
        p, err = _render(tmp, "--chip", "--format", "mod6")
        if check("--format renders", p, err):
            i = sf.info(p)
            check("--format sets the sample rate", i.samplerate == 11025,
                  f"{i.samplerate} Hz (expected 11025)")
            check("--format sets the stored bit depth", i.subtype == "PCM_U8",
                  f"{i.subtype} (expected PCM_U8; ffmpeg used to force PCM_16)")
            au, _ = sf.read(p, always_2d=True)
            lv = len(np.unique(np.round(au[:, 0], 6)))
            check("--format quantizes amplitude", lv <= 70,
                  f"{lv} distinct levels (6-bit allows 64)")

        # encode() MUST be idempotent. On the diffusion path --flac attaches a
        # ComfyUI SaveAudio node, so the server already returns a .flac —
        # re-encoding resolved the output to the same path, ffmpeg refused, and the
        # cleanup deleted the file. Every asset downloaded fine and was then
        # destroyed, reported as "nothing produced".
        import soundmon as _smm
        import types as _t
        for fmt, flags in (("flac", dict(flac=True, ogg=False)),
                           ("ogg", dict(flac=False, ogg=True))):
            p = os.path.join(tmp, f"idem.{fmt}")
            sf.write(p, np.zeros(2205, dtype="float32"), 44100,
                     format=fmt.upper())
            n0 = os.path.getsize(p)
            aa = _t.SimpleNamespace(keep_wav=False, ogg_quality=5, **flags)
            out = _smm.encode(p, aa)
            check(f"encode() leaves an existing .{fmt} alone",
                  out == p and os.path.exists(p) and os.path.getsize(p) == n0,
                  f"-> {os.path.basename(out)}, exists={os.path.exists(p)}")
            os.path.exists(p) and os.remove(p)

        # EVERY engine must honour --flac. It was wired into three of seven, and
        # blip.py/narrate.py accepted a to_flac callable and never called it — so
        # the flag was accepted, plumbed, and silently dropped. Only the engines
        # that need no model or GPU are checked here; that is enough to catch a
        # new engine copying the old if-ogg-only shape.
        for eng, extra in (("--chip", []), ("--opl", []),
                           ("--chipfx", []), ("--blip", [])):
            before = set(os.listdir(tmp))
            r = subprocess.run([PY, SM, "beep", "--output-to", tmp, "--no-open",
                                "--seconds", "2", eng, "--flac", *extra],
                               capture_output=True, text=True)
            made = [f for f in os.listdir(tmp) if f not in before]
            got = [f for f in made if f.endswith(".flac")]
            check(f"{eng} honours --flac", bool(got),
                  f"produced {made or r.stderr[-160:]}")
            for f in made:
                os.remove(os.path.join(tmp, f))

        # FLAC must be lossless. Test the CODEC, not two renders: since the
        # pack-identity fix, --output-to feeds the composition, so rendering into
        # two directories deliberately produces two different pieces. My first
        # attempt did exactly that and "failed" on a 7436-frame length difference
        # that was the feature working. Convert one file instead.
        p, err = _render(tmp, "--chip", "--format", "mod6", "--seed", "42")
        if check("--flac source renders", p, err):
            sys.path.insert(0, HERE)
            import soundmon as _sm
            import shutil as _sh
            src = os.path.join(tmp, "flacsrc.wav")
            _sh.copy(p, src)
            out = _sm.to_flac(src, keep=True)
            ok = out.endswith(".flac") and os.path.exists(out)
            if check("to_flac produces a .flac", ok, out):
                x1, _ = sf.read(src, always_2d=True)
                x2, _ = sf.read(out, always_2d=True)
                same = (len(x1) == len(x2)
                        and np.allclose(x1[:, 0], x2[:, 0], atol=1e-6))
                check("FLAC is bit-identical to its WAV", same,
                      f"{len(x1)} vs {len(x2)} frames, "
                      f"max diff {float(np.abs(x1[:min(len(x1),len(x2)),0] - x2[:min(len(x1),len(x2)),0]).max()):.2e}")

    width = max(len(n) for n, _, _ in RESULTS)
    bad = 0
    for name, ok, detail in RESULTS:
        bad += not ok
        if a.verbose or not ok:
            mark = "ok  " if ok else "FAIL"
            print(f"  {mark} {name:<{width}}  {detail}")
    print(f"\n  {len(RESULTS) - bad}/{len(RESULTS)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
