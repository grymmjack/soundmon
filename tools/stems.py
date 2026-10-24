#!/usr/bin/env python3
"""stems.py - a finished track into parts a sampler can use (soundmon's stems, pilot).

    python stems.py TRACK.wav OUTDIR [--name "new3"] [--title "NEON MARKET"] [--bpm N]

Every file is a 44.1 kHz 16-bit stereo WAV named for what it is. A file with a known loop has it
in its smpl chunk (with its root note), and a loop's tempo and length in beats are in its acid
chunk, so a sampler or DAW loops and syncs it as it loads. Into OUTDIR ("new3 - NEON MARKET"):

  1 Stems/        the whole track split into parts, each as long as the track:
                    drums, bass, other, vocals     - Demucs htdemucs_ft (the best 4-part split)
                    guitar, piano                  - BS-Roformer SW (6 parts; synth parts land here too)
                    synth dry, reverb              - "other" split into its dry sound and its reverb
                                                     (MelBand Roformer de-reverb)
                    kick, snare, toms, hats, ride, crash - the drums split again (MDX23C DrumSep)
                    synth low, synth mid, synth high - the dry "other" split by sound and register (NMF:
                                                     what no model knows - pads, leads, arps)
                    synth held, synth hits         - the dry "other" split into sustained and struck (HPSS)
                  ("synths + fx" is all of "other": everything but drums, bass and voice)
  2 Loops/PART/   1, 2 and 4 bar loops of each part (and the full mix), cut on the bar lines
                  where they loop most smoothly: "new3 - bass loop - 2 bars - 127.9 bpm - bar 05.wav"
  3 One-shots/PART/  drum hits ("new3 - kick 01.wav"), notes by pitch ("new3 - bass - C2.wav"),
                  chords and stabs, and a held bar of each sustained part, looped (pads)
  4 FX/           swoops, risers and falls found in the track, and its ending (the reverb tail)
  5 Instruments/  the one-shots as Bitwig .multisample instruments - a drum kit, each part's notes,
                  chords and pads (instruments.py, run at the end when it's beside this script)
  stems.json      the tempo, the bar grid, every file and its numbers

Levels are left as they come out (nothing normalized). Separation isn't clean - bleed, reverb and
effects come along - so these are raw material, not studio stems.
Needs: audio-separator[gpu], librosa, soundfile (the stems venv on the farm box).
"""
import argparse, json, os, re, shutil, struct
import numpy as np
import soundfile as sf
import librosa

SR = 44100
PRETTY = {'hh': 'hats', 'other-dry': 'synth dry', 'other': 'synths + fx', 'synth-low': 'synth low', 'synth-mid': 'synth mid', 'synth-high': 'synth high',
          'other-sustain': 'synth held', 'other-hits': 'synth hits'}
FOLDER = {'kick': 'Kick', 'snare': 'Snare', 'toms': 'Toms', 'hh': 'Hats', 'ride': 'Ride', 'crash': 'Crash',
          'drums': 'Drums', 'bass': 'Bass', 'other': 'Synths + FX', 'vocals': 'Vocals', 'guitar': 'Guitar',
          'piano': 'Piano', 'synth-low': 'Synth Low', 'synth-mid': 'Synth Mid', 'synth-high': 'Synth High',
          'other-sustain': 'Synth Held', 'other-hits': 'Synth Hits', 'mix': 'Full Mix',
          'other-dry': 'Synth Dry', 'reverb': 'Reverb'}


def pretty(k):
    return PRETTY.get(k, k)


def db(x):
    return 20 * np.log10(max(float(x), 1e-12))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def mono(x):
    return x.mean(axis=1) if x.ndim == 2 else x


def stereo(x):
    return np.stack([x, x], 1) if x.ndim == 1 else x


def note_name(midi):
    return librosa.midi_to_note(int(round(midi)), unicode=False).replace('#', 's')


# --- A WAV: 44.1 kHz 16-bit stereo; a loop (start, end in samples) and root note in smpl, and a
# tempo and beats in acid (one-shot flagged when there's no loop)
def write_wav(path, x, loop=None, note=60, tempo=None, beats=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = np.clip(stereo(x), -1, 1)
    sf.write(path, x, SR, subtype='PCM_16')
    extra = b''
    if loop:
        smpl = struct.pack('<9I', 0, 0, int(1e9 / SR), int(note), 0, 0, 0, 1, 0)
        smpl += struct.pack('<6I', 0, 0, int(loop[0]), int(loop[1]) - 1, 0, 0)
        extra += b'smpl' + struct.pack('<I', len(smpl)) + smpl
    flags = (0x04 if tempo else 0x01) | 0x02  # (stretch for a loop, else one-shot; root note set)
    acid = struct.pack('<IHHfIHHf', flags, int(note), 0x8000, 0.0, int(beats), 4, 4, float(tempo or 0))
    extra += b'acid' + struct.pack('<I', len(acid)) + acid
    with open(path, 'r+b') as f:
        f.seek(0, 2)
        f.write(extra)
        size = f.tell() - 8
        f.seek(4)
        f.write(struct.pack('<I', size))


# --- 1. Split: Demucs (4 and 6 parts), then the drums again
def separate(track, work):
    from audio_separator.separator import Separator
    out = {}
    sep = Separator(output_dir=work, output_format='WAV', sample_rate=SR, log_level=40,
                    model_file_dir=os.path.expanduser('~/stems/models'))  # (kept: /tmp is cleared)

    def run(model, src, names):
        sep.load_model(model_filename=model)
        for f in sep.separate(src):
            f = f if os.path.isabs(f) else os.path.join(work, f)
            m = re.search(r'_\(([^)]+)\)_', os.path.basename(f))
            k = m.group(1).lower() if m else ''
            if k in names:
                dst = os.path.join(work, names[k] + '.wav')
                shutil.move(f, dst)
                out[names[k]] = dst
            elif os.path.exists(f):
                os.remove(f)

    run('htdemucs_ft.yaml', track, {'drums': 'drums', 'bass': 'bass', 'other': 'other', 'vocals': 'vocals'})
    run('BS-Roformer-SW.ckpt', track, {'guitar': 'guitar', 'piano': 'piano'})
    run('MDX23C-DrumSep-aufr33-jarredou.ckpt', out['drums'],
        {'kick': 'kick', 'snare': 'snare', 'toms': 'toms', 'hh': 'hh', 'ride': 'ride', 'crash': 'crash'})
    run('dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt', out['other'],
        {'noreverb': 'other-dry', 'reverb': 'reverb'})
    return out


# --- 2. "other" by sound and register (NMF, soft masks) and into held and struck (HPSS)
def split_other(other):
    D = [librosa.stft(other[:, c], n_fft=4096, hop_length=1024) for c in range(2)]
    S = (np.abs(D[0]) + np.abs(D[1])) / 2
    W, H = librosa.decompose.decompose(S, n_components=18, sort=True, max_iter=400, random_state=0)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=4096)
    cent = (W * freqs[:, None]).sum(0) / (W.sum(0) + 1e-9)
    # three layers by brightness, each about a third of the sound (fixed bands left one empty)
    order = np.argsort(cent)
    share = np.cumsum((W * H.sum(1)).sum(0)[order]); share /= share[-1]
    layer = np.empty(len(cent), int); layer[order] = np.searchsorted([1 / 3, 2 / 3], share, side='right').clip(0, 2)
    groups = {'synth-low': layer == 0, 'synth-mid': layer == 1, 'synth-high': layer == 2}
    full = W @ H + 1e-9
    parts = {}
    for name, sel in groups.items():
        if sel.any():
            M = (W[:, sel] @ H[sel]) / full
            parts[name] = np.stack([librosa.istft(D[c] * M, hop_length=1024, length=len(other)) for c in range(2)], 1)
    parts['other-sustain'] = np.stack([librosa.effects.harmonic(other[:, c], margin=1.5) for c in range(2)], 1)
    parts['other-hits'] = np.stack([librosa.effects.percussive(other[:, c], margin=1.5) for c in range(2)], 1)
    return parts


# --- 3. Tempo and bars: the tracker's beats, a straight line through them, the bar's first beat
# where the kick lands hardest
def grid(mix, kick, bpm_hint=None):
    y = mono(mix)
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=512)
    tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=512,
                                           start_bpm=bpm_hint or 128, units='samples')
    tempo = float(np.atleast_1d(tempo)[0]) or 120.0
    while tempo < 90: tempo *= 2
    while tempo > 180: tempo /= 2
    spb, b0 = SR * 60 / tempo, (beats[0] if len(beats) else 0)
    if len(beats) > 8:
        k = np.round((beats - beats[0]) / spb)
        spb, b0 = np.polyfit(k, beats, 1)
    kenv = librosa.onset.onset_strength(y=mono(kick), sr=SR, hop_length=512)
    best, phase = -1, 0
    for ph in range(4):
        t = b0 + (np.arange(64) * 4 + ph) * spb
        t = t[(t >= 0) & (t < len(y))]
        s = kenv[np.minimum((t / 512).astype(int), len(kenv) - 1)].sum()
        if s > best:
            best, phase = s, ph
    first = b0 + phase * spb
    while first - 4 * spb >= 0:
        first -= 4 * spb
    spbar = 4 * spb
    bars = [int(round(first + i * spbar)) for i in range(int((len(y) - first) / spbar) + 1)]
    return SR * 60 / spb, spbar, bars


# Where the music stops: the last bars that fall under half the track's usual level (a fade,
# silence - never loop into it)
def live_end(mix, bars):
    e = [rms(mono(mix[bars[i]:bars[i + 1]])) for i in range(len(bars) - 1)]
    if not e:
        return len(mix)
    top, end = np.median(e), len(bars) - 1
    while end > 1 and e[end - 1] < top * 0.5:
        end -= 1
    return bars[end]


# --- 4. Loops: n bars from a bar line; how like the start what follows the end is (so the loop
# runs on as the track did), the three best that don't overlap
def loops(name, x, bars, spbar, tempo, stop, out, key):
    res, xm = [], mono(x)
    floor = max(rms(xm) * 0.25, 1e-4)
    for nb in (1, 2, 4):
        n = int(round(nb * spbar))
        cands, w = [], 2048
        for b in bars:
            if b + n + w > stop or b < w:
                continue
            if rms(xm[b:b + n]) < floor:
                continue
            A = np.log1p(np.abs(np.fft.rfft(xm[b:b + w] * np.hanning(w))))
            B = np.log1p(np.abs(np.fft.rfft(xm[b + n:b + n + w] * np.hanning(w))))
            cands.append((float(np.sum((A - B) ** 2) / (np.sum(B ** 2) + 1e-9)), b))
        cands.sort()
        taken = []
        for seam, b in cands:
            if all(abs(b - t) >= n for t in taken):
                taken.append(b)
            if len(taken) == 3:
                break
        seams = {b: s for s, b in cands}
        for b in sorted(taken):
            seg = x[b:b + n].copy()
            f = 441  # (10 ms: its start blended from what followed its end - the seam is the track's own)
            r = np.linspace(0, 1, f)[:, None]
            seg[:f] = seg[:f] * r + x[b + n:b + n + f] * (1 - r)
            bar = bars.index(b) + 1
            fn = os.path.join(out, '2 Loops', FOLDER[name],
                              f'{key} - {pretty(name)} loop - {nb} bar{"s" if nb > 1 else ""} - {tempo:.1f} bpm - bar {bar:02d}.wav')
            write_wav(fn, seg, loop=(0, len(seg)), tempo=tempo, beats=4 * nb)
            res.append({'file': os.path.relpath(fn, out), 'part': name, 'bars': nb, 'from_bar': bar,
                        'seam': round(seams[b], 4), 'level_db': round(db(rms(mono(seg))), 1)})
    return res


# --- 5. One-shots: a part's hits or notes, cut at their onsets; the cleanest of each kind
RANGE = {'bass': (30, 262), 'synth-low': (30, 520)}  # (Hz: where a part's notes can be - else 50 to 2000)


def oneshots(name, x, out, key, pitched, others=None, keep=6, max_len=1.5):
    xm = mono(x)
    if rms(xm) < 1e-4:
        return []
    on = librosa.onset.onset_detect(y=xm, sr=SR, hop_length=256, backtrack=True, units='samples')
    ends = list(on[1:]) + [len(xm)]
    top = np.max(np.abs(xm))
    cands = []
    for s, e in zip(on, ends):
        e = min(e, s + int(max_len * SR))
        if e - s < int(0.04 * SR):
            continue
        pk = np.max(np.abs(xm[s:e]))
        if pk < top * 0.2:
            continue
        above = np.nonzero(np.abs(xm[s:e]) > pk * 10 ** (-50 / 20))[0]  # (its tail trimmed 50 dB down)
        e = s + int(above[-1]) + 1
        seg = xm[s:e]
        bleed = rms(mono(others[s:e])) / (rms(seg) + 1e-9) if others is not None else 0.0
        label, midi, wobble = '', 60, 0.0
        if pitched:
            lo, hi = RANGE.get(name, (50, 2000))
            f0, vf, _ = librosa.pyin(seg, fmin=lo, fmax=hi, sr=SR, frame_length=4096, hop_length=512)
            v = f0[vf] if vf is not None else np.array([])
            if len(v) >= 3:
                m = librosa.hz_to_midi(v)
                midi, wobble = float(np.median(m)), float(np.std(m))
                label = note_name(midi)
            else:
                label = 'chord'
        mf = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=13).mean(1)
        cands.append({'s': int(s), 'e': int(e), 'label': label, 'midi': midi,
                      'score': db(pk) - 20 * bleed - 3 * wobble, 'mf': mf})
    cands.sort(key=lambda c: -c['score'])
    picked = []
    for c in cands:  # (one per note; drum hits and chords: ones unlike those already kept)
        same = [p for p in picked if p['label'] == c['label']]
        if pitched and c['label'] != 'chord' and same:
            continue
        if any(np.linalg.norm(c['mf'] - p['mf']) < 25 for p in same):
            continue
        picked.append(c)
        if len(picked) >= keep:
            break
    res, count = [], {}
    for c in sorted(picked, key=lambda c: (c['label'] == 'chord', c['midi'], c['s'])):
        seg = x[c['s']:c['e']].copy()
        f = min(int(0.01 * SR), len(seg) // 4)
        seg[-f:] *= np.linspace(1, 0, f)[:, None]
        count[c['label']] = count.get(c['label'], 0) + 1
        what = f"{pretty(name)} - {c['label']}" if c['label'] else pretty(name)
        nth = count[c['label']]
        fn = os.path.join(out, '3 One-shots', FOLDER[name],
                          f'{key} - {what}' + (f' {nth:02d}' if not pitched or c['label'] == 'chord' else '') + '.wav')
        write_wav(fn, seg, note=round(c['midi']))
        res.append({'file': os.path.relpath(fn, out), 'part': name, 'label': c['label'] or name,
                    'at': round(c['s'] / SR, 3), 'seconds': round((c['e'] - c['s']) / SR, 3)})
    return res


# A held bar of a sustained part (pad, drone): its steadiest loud bar, looped as a whole
def held(name, x, bars, spbar, stop, out, key):
    xm, best = mono(x), None
    for b in bars:
        n = int(spbar)
        if b + n > stop:
            break
        r = librosa.feature.rms(y=xm[b:b + n], hop_length=1024)[0]
        if r.mean() < rms(xm) * 0.5:
            continue
        steady = r.std() / (r.mean() + 1e-9)
        if best is None or steady < best[0]:
            best = (steady, b)
    if best is None:
        return []
    b, n = best[1], int(round(spbar))
    seg = x[b:b + n].copy()
    f = 441
    r = np.linspace(0, 1, f)[:, None]
    seg[:f] = seg[:f] * r + x[b + n:b + n + f] * (1 - r)
    fn = os.path.join(out, '3 One-shots', FOLDER[name], f'{key} - {pretty(name)} - held bar {bars.index(b) + 1:02d}.wav')
    write_wav(fn, seg, loop=(0, len(seg)))
    return [{'file': os.path.relpath(fn, out), 'part': name, 'label': 'held', 'at': round(b / SR, 3)}]


# --- 6. FX: swoops - stretches where the sound's brightness climbs or falls steadily for half a
# second or more (risers, falls, filter sweeps) - and the track's ending (its reverb tail)
def fx(parts, mix, stop, out, key):
    res = []
    hop = 512
    for name in ('other', 'synth-high', 'other-hits', 'reverb'):
        if name not in parts:
            continue
        x = parts[name]
        xm = mono(x)
        c = librosa.feature.spectral_centroid(y=xm, sr=SR, hop_length=hop)[0]
        e = librosa.feature.rms(y=xm, hop_length=hop)[0]
        lc = np.log2(np.maximum(c, 20))
        loud = e > np.median(e) * 0.7
        found = []
        for secs in (2.0, 1.0, 0.5):
            n = int(secs * SR / hop)
            for i in range(0, len(lc) - n, n // 4):
                if not loud[i:i + n].mean() > 0.8:
                    continue
                seg = lc[i:i + n]
                t = np.arange(n)
                r = np.corrcoef(t, seg)[0, 1]
                change = seg[-n // 8:].mean() - seg[:n // 8].mean()
                if abs(r) > 0.85 and abs(change) > 0.7:
                    if all(i + n <= a or i >= b for a, b, _ in found):
                        found.append((i, i + n, 'riser' if change > 0 else 'fall'))
        for k, (a, b, kind) in enumerate(sorted(found)[:6]):
            s, e2 = a * hop, b * hop
            seg = x[s:e2].copy()
            f = 441
            seg[:f] *= np.linspace(0, 1, f)[:, None]
            seg[-f:] *= np.linspace(1, 0, f)[:, None]
            fn = os.path.join(out, '4 FX', f'{key} - {kind} - from {pretty(name)} {k + 1:02d}.wav')
            write_wav(fn, seg)
            res.append({'file': os.path.relpath(fn, out), 'part': name, 'label': kind, 'at': round(s / SR, 2),
                        'seconds': round((e2 - s) / SR, 2)})
    if len(mix) - stop > SR // 2:  # the ending: the mix and the reverb-heavy "other" from where the music stops
        for name, x in (('mix', mix), ('other', parts.get('other'))):
            if x is None:
                continue
            fn = os.path.join(out, '4 FX', f'{key} - ending tail - {pretty(name) if name != "mix" else "full mix"}.wav')
            write_wav(fn, x[stop:])
            res.append({'file': os.path.relpath(fn, out), 'part': name, 'label': 'ending', 'at': round(stop / SR, 2)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('track'); ap.add_argument('outdir')
    ap.add_argument('--name'); ap.add_argument('--title', default=''); ap.add_argument('--bpm', type=float)
    a = ap.parse_args()
    key = a.name or os.path.splitext(os.path.basename(a.track))[0]
    out = a.outdir
    os.makedirs(out, exist_ok=True)
    work = os.path.join(out, '.work'); os.makedirs(work, exist_ok=True)
    mix, sr = sf.read(a.track, always_2d=True)
    mix = stereo(mix[:, 0]) if mix.shape[1] == 1 else mix[:, :2]
    if sr != SR:
        mix = librosa.resample(mix.T, orig_sr=sr, target_sr=SR).T
    print('1/6 splitting (Demucs, DrumSep)...', flush=True)
    files = separate(os.path.abspath(a.track), work)
    parts = {}
    for k, f in files.items():
        y, _ = sf.read(f, always_2d=True)
        y = stereo(y[:, 0]) if y.shape[1] == 1 else y[:, :2]
        parts[k] = np.pad(y, ((0, max(0, len(mix) - len(y))), (0, 0)))[:len(mix)]
    print('2/6 splitting the instruments (NMF, HPSS)...', flush=True)
    parts.update(split_other(parts.get('other-dry', parts['other'])))
    level = {k: db(rms(mono(v))) for k, v in parts.items()}
    mixdb = db(rms(mono(mix)))
    quiet = sorted(k for k in parts if level[k] <= mixdb - 30)  # (next to silent: the 6-part model's guitar in a synth track, say)
    parts = {k: v for k, v in parts.items() if k not in quiet}
    for k, v in parts.items():
        write_wav(os.path.join(out, '1 Stems', f'{key} - stem - {pretty(k)}.wav'), v)
    print('3/6 tempo and bars...', flush=True)
    beat = parts.get('kick', parts.get('drums', mix))   # (a track with no drums - near-silent, left out - finds its bars in the mix)
    tempo, spbar, bars = grid(mix, beat, a.bpm)
    stop = live_end(mix, bars)
    print(f'    {tempo:.2f} bpm, {len(bars)} bars, music to {stop / SR:.1f}s', flush=True)
    info = {'track': os.path.basename(a.track), 'key': key, 'title': a.title, 'bpm': round(tempo, 3),
            'samples_a_bar': spbar, 'bars': bars, 'music_to': stop, 'levels_db': {k: round(level[k], 1) for k in level},
            'left_out_quiet': quiet, 'stems': sorted(parts), 'loops': [], 'oneshots': [], 'fx': []}
    print('4/6 loops...', flush=True)
    for k, v in sorted(parts.items()):
        info['loops'] += loops(k, v, bars, spbar, tempo, stop, out, key)
    info['loops'] += loops('mix', mix, bars, spbar, tempo, stop, out, key)
    print('5/6 one-shots...', flush=True)
    drums = [k for k in ('kick', 'snare', 'toms', 'hh', 'ride', 'crash') if k in parts]
    for k in drums:
        rest = sum(parts[d] for d in drums if d != k) if len(drums) > 1 else None
        info['oneshots'] += oneshots(k, parts[k], out, key, pitched=False, others=rest, keep=4, max_len=1.0)
    for k in ('bass', 'synth-low', 'synth-mid', 'synth-high', 'other-hits', 'piano', 'guitar'):
        if k in parts:
            info['oneshots'] += oneshots(k, parts[k], out, key, pitched=True, others=parts.get('drums'), keep=8, max_len=2.0)
    for k in ('other-sustain', 'synth-low', 'synth-mid'):
        if k in parts:
            info['oneshots'] += held(k, parts[k], bars, spbar, stop, out, key)
    print('6/6 FX...', flush=True)
    info['fx'] = fx(parts, mix, stop, out, key)
    with open(os.path.join(out, 'stems.json'), 'w') as f:
        json.dump(info, f, indent=1)
    shutil.rmtree(work, ignore_errors=True)
    print(f"done: {len(parts)} stems, {len(info['loops'])} loops, {len(info['oneshots'])} one-shots, {len(info['fx'])} fx", flush=True)
    try:  # the one-shots as sampler instruments (instruments.py, beside this script: copy it too)
        import instruments
        print(f"instruments: {len(instruments.build(out))}", flush=True)
    except ImportError:
        print('(no instruments.py beside stems.py: run it on OUTDIR for the .multisample instruments)', flush=True)


if __name__ == '__main__':
    main()
