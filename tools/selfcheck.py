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
    def entry(status, outputs=None, msgs=()):
        return {"outputs": outputs or {}, "status": dict(status, messages=list(msgs))}

    def xerr(node, msg, etype=""):
        return entry({"status_str": "error", "completed": False},
                     msgs=[["execution_error", {"node_type": node, "exception_message": msg, "exception_type": etype}]])
    saved = {"12": {"audio": [{"filename": "x_00001_.wav", "subfolder": "soundmon", "type": "output"}]}}
    cases = {
        "error": xerr("CLIPLoader", "missing model"),
        "pending": entry({"status_str": "success", "completed": False}),
        "ok": entry({"status_str": "success", "completed": True}, saved),
        # an identical re-run ComfyUI answers from its cache: success, nothing saved. NOT a failure
        "cached": entry({"status_str": "success", "completed": True}),
        "poisoned": xerr("KSampler", "'NoneType' object has no attribute 'model_size'", "AttributeError"),
        "transient": xerr("CheckpointLoaderSimple", "list index out of range", "IndexError"),
    }
    got = {want: _smj.classify(e)[0] for want, e in cases.items()}
    got["transient(oom)"] = _smj.classify(xerr("KSampler", "CUDA out of memory. Tried to allocate"))[0]
    got["pending(None)"] = _smj.classify(None)[0]
    check("the job classifier: failed, cached-empty, poisoned, transient, pending, done",
          all(k.split("(")[0] == v for k, v in got.items())
          and "CLIPLoader" in _smj._job_error(cases["error"]) and _smj._job_error(cases["cached"]) is None,
          f"{got}")
    fg = {"12": {"class_type": "SaveSFX", "inputs": {"filename_prefix": "soundmon/door_s7", "bit_depth": "16"}},
          "3": {"class_type": "KSampler", "inputs": {"seed": 7}}}
    fc = _smj.fresh_copy(fg)
    pre = fc["12"]["inputs"]["filename_prefix"]
    check("fresh_copy re-saves under a new folder, same file name, nothing else changed",
          pre.startswith("soundmon/rerun-") and pre.endswith("/door_s7") and fc["3"] == fg["3"]
          and fg["12"]["inputs"]["filename_prefix"] == "soundmon/door_s7", pre)

    # Target resolution: --server > $SOUNDMON_SERVER > servers.json "_default" > the `local` alias,
    # and the `local` alias is repointable (not a literal 127.0.0.1).
    pt = _smj.pick_target
    old_servers = dict(_smj.SERVERS)
    _smj.SERVERS.update(local="http://10.0.0.9:8188", rtx="http://192.168.1.77:8188")
    order = [pt("a", "b", "c"), pt(None, "b", "c"), pt("", "", "c"), pt(None, "", "")]
    res = [_smj.resolve_server(x) for x in ("rtx", "local", "box:9000", "http://h:7/")]
    _smj.SERVERS.clear()
    _smj.SERVERS.update(old_servers)
    check("server choice: --server > $SOUNDMON_SERVER > _default > local; aliases resolve",
          order == ["a", "b", "c", ""] and res == ["http://192.168.1.77:8188", "http://10.0.0.9:8188",
                                                   "http://box:9000", "http://h:7"], f"{order} {res}")
    check("given() tells a typed default from no flag at all",
          _smj.given("--seconds", argv=["x", "--seconds", "10"]) and _smj.given("--lufs", argv=["--lufs=-23"])
          and _smj.given("-n", argv=["-n5"]) and not _smj.given("--seconds", argv=["x", "--seconds-x"])
          and not _smj.given("--seconds", argv=["x"]), "")

    # --help must mention every flag (28 were missing once).
    import contextlib
    import io
    import re as _re
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _smj.print_help()
    hidden = [o for act in _smj.build_parser()._actions for o in act.option_strings
              if not _re.search(_re.escape(o) + r"(?![\w-])", buf.getvalue())]
    check("--help mentions every flag", not hidden, f"missing: {hidden}")

    # The wrapper must never start ComfyUI for a command that doesn't render on it: every --list-*,
    # --version, --dry-run and the CPU engines (it matches --flag and --flag=VALUE alike).
    import re as _re2
    wrap = open(os.path.join(HERE, "bin", "soundmon"), encoding="utf-8").read()
    block = wrap[wrap.index("needs_server=1"):wrap.index("needs_server=0 ;;")]
    skip = set(_re2.findall(r"--[a-z][a-z0-9-]*|-h\b", block))
    want = {o for act in _smj.build_parser()._actions for o in act.option_strings if o.startswith("--list-")}
    want |= {"--version", "--dry-run", "--help", "--chip", "--opl", "--chipfx", "--oplfx", "--blip", "--blip-file",
             "--narrate", "--narrate-file", "--record", "--record-file", "--master", "--index", "--similar",
             "--from-midi", "--from-audio"}
    check("bin/soundmon never starts ComfyUI for listings, --version, --dry-run or a CPU engine",
          want <= skip and '"${arg%%=*}"' in wrap, f"missing from its skip-list: {sorted(want - skip)}")

    # --batch-file rows: key | prompt, key | seconds | prompt, plain lines, commas inside prompts.
    with tempfile.TemporaryDirectory() as bt:
        mf = os.path.join(bt, "m.txt")
        with open(mf, "w") as f:
            f.write("# pack\ndoor_open | a heavy door, creaking open\nboom | 3.5 | an explosion, far away\n"
                    "glass breaking\n../evil | x\n")
        rows = _smj.parse_batch_file(mf)
    check("--batch-file parses keys, per-row seconds and comma prompts; keys can't escape",
          rows == [("door_open", None, "a heavy door, creaking open"), ("boom", 3.5, "an explosion, far away"),
                   ("glass_breaking", None, "glass breaking"), ("evil", None, "x")], f"{rows}")
    check("--name can't leave --output-to", _smj.safe_name("../../etc/x") == "x"
          and "/" not in _smj.safe_name("a/b/c") and len(_smj.safe_name("y" * 300)) <= 80,
          _smj.safe_name("../../etc/x"))

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

    # Everything after the server works on a WAV: the graph always ends in SaveSFX, whatever the
    # output format (a server-side FLAC/MP3 got WAV data written back under its name).
    ga.song = False
    saves = []
    for fl in ("flac", "mp3", "opus"):
        setattr(ga, fl, True)
        saves.append(_smj.build_graph(ga, 1)["12"]["class_type"])
        setattr(ga, fl, False)
    check("the server always saves WAV (SaveSFX); --flac/--mp3/--opus are encoded last, client-side",
          saves == ["SaveSFX"] * 3, f"{saves}")

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

            # match_tone is sample-exact: firequalizer's 10 ms linear-phase delay (and its 10 ms
            # tail) are trimmed, so a click stays on its sample and a --loop stays seamless. In
            # place, twice: it used to grow 20 ms a run. And it keeps the source's sample format.
            n_ = 44100
            xc = 0.05 * np.sin(2 * np.pi * 300 * np.arange(n_) / 44100)
            xc[5000] = 0.8
            clk = os.path.join(mt, "click.wav")
            sf.write(clk, xc, 44100, subtype="PCM_24")
            _ms.match_tone(clk, clk, dark)
            _ms.match_tone(clk, clk, dark)
            y_, _ = sf.read(clk)
            check("--match-tone keeps length, alignment and format (twice in place)",
                  len(y_) == n_ and abs(int(np.argmax(np.abs(y_))) - 5000) <= 1 and sf.info(clk).subtype == "PCM_24",
                  f"{len(y_) - n_:+d} samples, click at {int(np.argmax(np.abs(y_))) - 5000:+d}, {sf.info(clk).subtype}")

            # --master never writes onto a source, and same-name sources don't collide.
            import argparse as _ap
            src_d = os.path.join(mt, "src")
            os.makedirs(os.path.join(src_d, "sub"))
            sf.write(os.path.join(src_d, "x.wav"), x * 0.5, 44100)
            sf.write(os.path.join(src_d, "sub", "x.wav"), xc, 44100)
            sf.write(os.path.join(src_d, "hit.wav"), xc, 44100)
            sf.write(os.path.join(src_d, "hit.flac"), x * 0.5, 44100)
            keep = open(os.path.join(src_d, "sub", "x.wav"), "rb").read()
            ma = _ap.Namespace(master=src_d, master_like=None, master_kind="auto", output_to=os.path.join(src_d, "sub"),
                               flac=False, ogg=False, keep_wav=False, ogg_quality=5)
            refused = False
            try:
                _ms.run(ma, "x")
            except SystemExit:
                refused = True
            ma.output_to = None
            plan = sorted(os.path.relpath(d, src_d) for _, _, d in _ms._plan(list(_ms._walk(src_d)), src_d,
                                                                               os.path.join(src_d, "mastered")))
            check("--master never overwrites a source; hit.wav + hit.flac don't collide",
                  refused and open(os.path.join(src_d, "sub", "x.wav"), "rb").read() == keep
                  and len(set(plan)) == len(plan) == 4 and "mastered/hit.flac.wav" in plan,
                  f"output inside the source refused: {refused}; planned {plan}")

            # analyse() on silence (the GUI measures whatever is dropped on it) returns sanely.
            sil = os.path.join(mt, "silence.wav")
            sf.write(sil, np.zeros(4410), 44100)
            a_ = _ms.analyse(sil, sonogram=True)
            check("analyse() on silence: flagged, finite, no traceback",
                  a_.get("silent") and all(np.isfinite(a_["bands"])) and np.isfinite(a_["tilt"])
                  and a_["sonogram"]["cols"] >= 1 and _ms.chain(a_)[0] == ["anull"],
                  f"silent={a_.get('silent')} tilt={a_['tilt']} lufs={a_['lufs']}")

            # similar.py's fingerprint doesn't depend on volume (near-silent mel bands used to count).
            z0 = os.path.join(lib, "zap0.wav")
            zq = os.path.join(mt, "zap_quiet.wav")
            zz, _ = sf.read(z0)
            sf.write(zq, zz * 0.25, 44100)
            dv = float(np.abs(np.array(_sim.features(z0)) - np.array(_sim.features(zq))).max())
            top = ix.search(zq, k=1)
            check("--similar: fingerprint volume-invariant, a -12 dB copy reads as near-identical",
                  dv < 0.1 and top and os.path.basename(top[0]["path"]) == "zap0.wav" and top[0]["score"] >= 90,
                  f"max feature change {dv:.3f}; top {top[0]['score'] if top else None}%")

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
        p, err = _render(tmp, "--chip", "--bpm", "120", "--seed", "7")
        if check("--chip renders", p, err):
            au, sr = sf.read(p, always_2d=True)
            m = au[:, 0]
            import master as _msl
            # What the engine is HANDED, captured in-process — decisive whatever the piece's own
            # loudness: no -16 LUFS ceiling on music (+inf = none; the override must sit ABOVE the
            # hand-off), and a real render still under the -1 dBTP ceiling ("no LUFS ceiling" once
            # meant no true-peak ceiling either: chip moods shipped at +0.8 dBTP).
            seen = {}
            real_run, argv0 = chip.run, sys.argv
            chip.run = lambda a, *x, **k: seen.update(lufs=a.lufs_target, tp=a.true_peak) or []
            try:
                for extra, key in (((), "default"), (("--lufs=-20",), "asked")):
                    sys.argv = ["soundmon", "x", "--chip", "--no-open", "--no-recipe", *extra]
                    _smj.main()
                    seen[key] = seen.pop("lufs")
            finally:
                chip.run, sys.argv = real_run, argv0
            tp = _msl.loudness(p)[1]
            check("--chip is NOT held down by the LUFS ceiling, but stays <= -1 dBTP",
                  seen.get("default") == float("inf") and seen.get("asked") == -20.0 and tp <= -0.95,
                  f"LUFS ceiling handed to the engine: {seen.get('default')} (asked: {seen.get('asked')}), "
                  f"true peak {tp:.2f} dBTP")
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

    import shutil as _sh2
    if _sh2.which("ffmpeg"):
        import master as _mq
        with tempfile.TemporaryDirectory() as tt:
            # Short files keep the TRUE-PEAK ceiling. Below 0.4 s the LUFS integration is skipped, and
            # that skip used to skip the -1 dBTP cap too: a 0.3 s square peak-normalized to -1 dBFS
            # (a chipfx blip) shipped at +1.1 dBTP.
            sr = 44100
            t = np.arange(int(sr * 0.3)) / sr
            sq = os.path.join(tt, "sq.wav")
            sf.write(sq, np.sign(np.sin(2 * np.pi * 440 * t)) * 10 ** (-1 / 20), sr, subtype="PCM_16")
            before = _mq.loudness(sq)[1]
            _smj.loudness_normalize(sq, float("inf"), -1.0)
            after = _mq.loudness(sq)[1]
            check("a 0.3 s full-scale square is pulled under -1 dBTP (no LUFS, true peak still capped)",
                  after <= -0.95, f"{before:+.1f} -> {after:+.1f} dBTP")

            # The recipe rides in the file and changes no sample: WAV gets a LIST/INFO chunk (8- and
            # 24-bit headers untouched), FLAC/OGG a comment tag.
            rec = {"tool": "soundmon", "prompt": "a door, creaking — slowly", "seed": 7, "command": "soundmon x"}
            okr, why = True, []
            for sub, ext in (("PCM_U8", "wav"), ("PCM_24", "wav"), ("PCM_16", "flac")):
                f = os.path.join(tt, f"r_{sub}.{ext}")
                sf.write(f, 0.5 * np.sin(2 * np.pi * 220 * t), sr, subtype=sub)
                raw0, info0 = sf.read(f, dtype="int32")[0], sf.info(f)
                hdr0 = open(f, "rb").read(44)
                _smj.tag_file(f, rec)
                raw1, info1 = sf.read(f, dtype="int32")[0], sf.info(f)
                back = _smj.read_recipe(f) or {}
                same = (np.array_equal(raw0, raw1) and info0.subtype == info1.subtype
                        and (ext != "wav" or open(f, "rb").read(36)[8:36] == hdr0[8:36]))
                okr &= same and back.get("prompt") == rec["prompt"] and back.get("seed") == 7
                why.append(f"{sub}:{'same' if same else 'CHANGED'}/{'tag' if back else 'no tag'}")
            ogg = _smj.to_ogg(os.path.join(tt, "r_PCM_16.flac"), 5, keep=True)
            _smj.tag_file(ogg, rec)
            okr &= (_smj.read_recipe(ogg) or {}).get("seed") == 7
            check("the recipe is embedded (WAV 8/24-bit, FLAC, OGG) without touching the audio", okr,
                  " ".join(why))

            # OGG: never silent at a low quality (the native encoder writes silence at -q:a 0), and a
            # mono file keeps its level (the -ac 2 upmix cost 3 dB).
            mono = os.path.join(tt, "mono.wav")
            t2 = np.arange(sr * 2) / sr
            sf.write(mono, 0.5 * np.sign(np.sin(2 * np.pi * 330 * t2)), sr, subtype="PCM_16")

            def mean_db(path):
                import re
                r = subprocess.run(["ffmpeg", "-i", path, "-af", "volumedetect", "-f", "null", "-"],
                                   capture_output=True, text=True).stderr
                m = re.findall(r"mean_volume:\s*(-?[\d.]+)", r)
                return float(m[-1]) if m else -99.0
            w = mean_db(mono)
            o = mean_db(_smj.to_ogg(mono, 0, keep=True))
            check("--ogg-quality 0 isn't silent, and mono OGG keeps the WAV's level", abs(o - w) < 1.0,
                  f"WAV {w:.1f} dB, OGG q0 {o:.1f} dB")

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
