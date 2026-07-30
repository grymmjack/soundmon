#!/usr/bin/env python3
"""Learn musical idiom from a folder of MIDI files.

WHY

`theory.py` composes from tables I wrote by hand: nine function templates, five
cadences, eleven rhythms, six contours. They are defensible but they are MY
guesses about game music, and they are the reason procedural output has a ceiling.

A corpus of real game MIDI has the actual answers — which chords really follow
which, how long phrases really are, which time signatures really occur, which
instruments really appear together. Mining it replaces guesses with measurements,
using data already on disk rather than a model download.

This is the same move that made `--from-midi` beat `--from-audio`: prefer the
source that never threw the information away.

USAGE

    tools/mine-corpus.py /path/to/midis            # writes corpus.json
    tools/mine-corpus.py /path/to/midis --sample 4000 --out corpus.json

WHAT IT MEASURES

    time signatures      weighted by occurrence, so 4/4 dominates honestly
    tempo                distribution, per time signature
    chord transitions    7x7 scale-degree Markov matrix, separately for
                         major-ish and minor-ish keys
    phrase rhythms       note-duration sequences per bar, as they actually occur
    melodic intervals    semitone step distribution — how far melodies really move
    instrument sets      which GM programs co-occur in one piece
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import midi as midimod                                            # noqa: E402

KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]


def find_key(pc_weights):
    """Krumhansl-Kessler correlation over all 24 keys."""
    tot = sum(pc_weights) or 1.0
    v = [x / tot for x in pc_weights]
    best = (0, "minor", -2.0)
    for r in range(12):
        rot = v[r:] + v[:r]
        for name, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            ps = sum(prof)
            p = [x / ps for x in prof]
            mv, mp = sum(rot) / 12, sum(p) / 12
            num = sum((a - mv) * (b - mp) for a, b in zip(rot, p))
            den = (sum((a - mv) ** 2 for a in rot) *
                   sum((b - mp) ** 2 for b in p)) ** 0.5
            c = num / den if den else -2.0
            if c > best[2]:
                best = (r, name, c)
    return best


def analyse(path):
    """One file -> a small dict of observations, or None."""
    try:
        notes, drums, tpb, tempo_map, timesig = midimod.parse(path)
    except Exception:
        return None
    if len(notes) < 24:
        return None

    beats, unit = timesig
    quarters = beats * (4.0 / unit)
    ticks_per_bar = tpb * quarters
    if ticks_per_bar <= 0:
        return None

    pc = [0.0] * 12
    for st, en, pitch, vel, chan, prog in notes:
        pc[pitch % 12] += max(1, en - st)
    root, mode, conf = find_key(pc)
    if conf < 0.55:                     # too ambiguous to learn harmony from
        return None
    scale = MAJOR if mode == "major" else MINOR
    pcs = {(root + s) % 12: i for i, s in enumerate(scale)}

    # --- chord per bar, as a scale degree -----------------------------------
    per_bar = collections.defaultdict(lambda: [0.0] * 12)
    for st, en, pitch, vel, chan, prog in notes:
        per_bar[int(st / ticks_per_bar)][pitch % 12] += max(1, en - st)
    degrees = []
    for b in sorted(per_bar):
        w = per_bar[b]
        best, bv = None, -1.0
        for deg in range(len(scale)):
            triad = [(root + scale[(deg + k) % len(scale)]) % 12 for k in (0, 2, 4)]
            v = sum(w[t] for t in triad)
            if v > bv:
                best, bv = deg, v
        degrees.append(best)
    trans = collections.Counter(zip(degrees, degrees[1:]))

    # --- melodic intervals, from the highest line ---------------------------
    top = {}
    for st, en, pitch, vel, chan, prog in notes:
        k = int(st / (ticks_per_bar / 16))
        if pitch > top.get(k, (-1,))[0]:
            top[k] = (pitch, en - st)
    seq = [top[k][0] for k in sorted(top)]
    intervals = collections.Counter(
        max(-12, min(12, b - a)) for a, b in zip(seq, seq[1:]))

    # --- rhythm: duration sequences within a bar, in 16ths -----------------
    step = ticks_per_bar / 16.0
    bars = collections.defaultdict(list)
    for k in sorted(top):
        bars[k // 16].append((k % 16, max(1, int(round(top[k][1] / step)))))
    rhythms = collections.Counter()
    for b, items in bars.items():
        if len(items) < 2:
            continue
        durs = tuple(min(16, d) for _, d in sorted(items))[:8]
        if 2 <= len(durs) <= 8 and sum(durs) <= 32:
            rhythms[durs] += 1

    programs = sorted({n[5] for n in notes})
    return {"timesig": f"{beats}/{unit}", "mode": mode,
            "bpm": 60_000_000.0 / tempo_map[0][1],
            "trans": trans, "intervals": intervals, "rhythms": rhythms,
            "programs": tuple(programs[:8]), "has_drums": bool(drums)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--sample", type=int, default=3000,
                    help="how many files to analyse (0 = all)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    print(f"▶ scanning {a.root} ...")
    files = []
    for dirpath, _dirs, names in os.walk(a.root):
        for n in names:
            if n.lower().endswith((".mid", ".midi")):
                files.append(os.path.join(dirpath, n))
    print(f"  found {len(files)} MIDI files")
    if a.sample and len(files) > a.sample:
        random.Random(a.seed).shuffle(files)
        files = files[:a.sample]
        print(f"  sampling {len(files)}")

    ts = collections.Counter()
    tempo = collections.defaultdict(list)
    trans = {"major": collections.Counter(), "minor": collections.Counter()}
    intervals = {"major": collections.Counter(), "minor": collections.Counter()}
    rhythms = collections.defaultdict(collections.Counter)
    progsets = collections.Counter()
    used = skipped = 0

    for i, p in enumerate(files, 1):
        if i % 250 == 0:
            print(f"  {i}/{len(files)}  used={used} skipped={skipped}")
        r = analyse(p)
        if not r:
            skipped += 1
            continue
        used += 1
        ts[r["timesig"]] += 1
        tempo[r["timesig"]].append(round(r["bpm"]))
        trans[r["mode"]].update(r["trans"])
        intervals[r["mode"]].update(r["intervals"])
        rhythms[r["timesig"]].update(r["rhythms"])
        progsets[r["programs"]] += 1

    out = {
        "source": os.path.abspath(a.root),
        "files_seen": len(files), "files_used": used,
        "timesigs": ts.most_common(12),
        "tempo": {k: {"min": min(v), "max": max(v),
                      "median": sorted(v)[len(v) // 2]}
                  for k, v in tempo.items() if v},
        # Markov matrices keyed "from>to" so JSON can hold them.
        "transitions": {m: {f"{x}>{y}": c for (x, y), c in t.most_common()}
                        for m, t in trans.items()},
        "intervals": {m: {str(k): v for k, v in c.most_common()}
                      for m, c in intervals.items()},
        "rhythms": {k: [[list(d), c] for d, c in v.most_common(40)]
                    for k, v in rhythms.items()},
        "instrument_sets": [[list(k), v] for k, v in progsets.most_common(40)],
    }
    dest = a.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)

    print(f"\n▶ {used} usable / {len(files)} sampled  ({skipped} skipped: too few "
          f"notes or ambiguous key)")
    print(f"  time signatures : {ts.most_common(6)}")
    for m in ("minor", "major"):
        top = list(out['transitions'][m].items())[:6]
        print(f"  {m} chord moves : {top}")
    print(f"  wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
