#!/usr/bin/env python3
"""instruments.py - a stems folder's one-shots as sampler instruments (Bitwig's .multisample).

    python instruments.py "OUTDIR" [...]     # a folder stems.py made ("new3 - NEON MARKET")

Reads OUTDIR/stems.json and writes OUTDIR/5 Instruments/:

  NAME - drum kit.multisample      every drum hit, a key each from C1 (MIDI 36) up, part by part
                                   (kick, snare, toms, hats, ride, crash) - Bitwig's Drum Machine
                                   starts its pads there too. A text file beside it lists the keys.
  NAME - PART.multisample          a part's notes ("bass - E1"), each on its own pitch, its zone
                                   reaching halfway to the next note's - a playable instrument
  NAME - PART chords.multisample   a part's chords and stabs, a key each from C3 (MIDI 60) up
  NAME - PART pad.multisample      a sustained part's held bar, looped as it was cut (its smpl
                                   chunk), on every key; C3 plays it as it was

Key names are Bitwig's (C3 = MIDI 60); stems.py's file names count C4 = 60, an octave up.

A .multisample is a zip of the WAVs and a multisample.xml (Bitwig's open format: drag it onto a
track, or open it in Sampler). Only the standard library: runs anywhere.

Each one also comes as

  Renoise/NAME.xrni        Renoise's own instrument (Instrument.xml and the WAVs): drag it into the
                           instrument list. Keys as MIDI plays them, so the same as Bitwig's - the
                           kit from C-2 (MIDI 36, Bitwig's C1), chords from C-4 (MIDI 60); drums
                           one-shots, unpitched; pads looped.
  Bitwig/NAME.bwpreset     Bitwig's own devices (bwpreset.py): the kit a Drum Machine, a Sampler a
                           pad, the rest Samplers - only where Bitwig is installed (its Library).
"""
import json, os, re, struct, sys, zipfile
from xml.sax.saxutils import escape, quoteattr

DRUMS = ['kick', 'snare', 'toms', 'hh', 'ride', 'crash']
PRETTY = {'hh': 'hats', 'other-dry': 'synth dry', 'other': 'synths + fx', 'synth-low': 'synth low',
          'synth-mid': 'synth mid', 'synth-high': 'synth high', 'other-sustain': 'synth held',
          'other-hits': 'synth hits'}
NOTES = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def pretty(k):
    return PRETTY.get(k, k)


def midi_of(label):
    """'Cs3' (stems.py's note names: s for sharp, C4 = 60) to a MIDI note, else None"""
    m = re.fullmatch(r'([A-G])(s?)(-?\d)', label)
    if not m: return None
    return 12 * (int(m.group(3)) + 1) + NOTES[m.group(1)] + (1 if m.group(2) else 0)


def key_name(n):
    """MIDI n as Bitwig names it (C3 = 60; stems.py's file names count C4 = 60)"""
    return '%s%d' % (KEY_NAMES[n % 12], n // 12 - 2)


def wav_info(path):
    """(frames, loop start, loop end or None) from a WAV's fmt, data and smpl chunks"""
    with open(path, 'rb') as f:
        data = f.read()
    pos, frames, align, loop = 12, 0, 4, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack('<I', data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b'fmt ': align = struct.unpack('<H', body[12:14])[0]
        elif cid == b'data': frames = size // max(align, 1)
        elif cid == b'smpl' and len(body) >= 60 and struct.unpack('<I', body[28:32])[0] > 0:
            s, e = struct.unpack('<II', body[44:52])
            loop = (s, e + 1)
        pos += 8 + size + (size & 1)
    return frames, loop


def sample_xml(fname, frames, lo, hi, root, track=1.0, loop=None):
    lp = ('<loop mode="loop" start="%d.000" stop="%d.000"/>' % loop if loop
          else '<loop mode="off" start="0.000" stop="%d.000"/>' % frames)
    return ('  <sample file=%s gain="0.000" group="0" sample-start="0.000" sample-stop="%d.000" reverse="false" zone-logic="always-play">\n'
            '    <key low="%d" high="%d" root="%d" track="%.4f" tune="0.00"/>\n'
            '    <velocity low="0" high="127"/>\n'
            '    %s\n'
            '  </sample>\n' % (quoteattr(fname), frames, lo, hi, root, track, lp))


def write_multisample(path, name, category, zones, folder, description):
    """zones: (wav path relative to folder, low key, high key, root, track, use its loop)"""
    parts, files = [], []
    for rel, lo, hi, root, track, looped in zones:
        src = os.path.join(folder, rel)
        frames, loop = wav_info(src)
        fname = os.path.basename(rel)
        parts.append(sample_xml(fname, frames, lo, hi, root, track, loop if looped else None))
        files.append((src, fname))
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<multisample name=%s>\n'
           '  <generator>soundmon instruments.py</generator>\n  <category>%s</category>\n'
           '  <creator>soundmon</creator>\n  <description>%s</description>\n'
           '  <group name="%s"/>\n%s</multisample>\n'
           % (quoteattr(name), escape(category), escape(description), escape(category), ''.join(parts)))
    tmp = path + '.part'
    with zipfile.ZipFile(tmp, 'w') as z:
        z.writestr('multisample.xml', xml, compress_type=zipfile.ZIP_DEFLATED)
        for src, fname in files:
            z.write(src, fname, compress_type=zipfile.ZIP_STORED)
    os.replace(tmp, path)


# --- RENOISE (.xrni: a zip of Instrument.xml and SampleData/SampleNN (NAME).wav) -----------------
RN_KEYS = ['C-', 'C#', 'D-', 'D#', 'E-', 'F-', 'F#', 'G-', 'G#', 'A-', 'A#', 'B-']


def rn_note(midi):
    """MIDI note -> Renoise's note (0-119, C-0 = 0): Renoise plays MIDI 60 as C-4, its note 48 - so a
    MIDI keyboard plays the same sounds on the same keys as in Bitwig"""
    return max(0, min(119, midi - 12))


def rn_name(midi):
    n = rn_note(midi)
    return '%s%d' % (RN_KEYS[n % 12], n // 12)


def rn_sample_xml(name, frames, lo, hi, root, pitched, loop):
    lp = ('<LoopMode>Forward</LoopMode><LoopRelease>false</LoopRelease><LoopStart>%d</LoopStart><LoopEnd>%d</LoopEnd>'
          % (loop[0], loop[1] - 1) if loop else
          '<LoopMode>Off</LoopMode><LoopRelease>false</LoopRelease><LoopStart>0</LoopStart><LoopEnd>%d</LoopEnd>' % max(frames - 1, 0))
    return ('<Sample><Name>%s</Name><Volume>1.0</Volume><Panning>0.5</Panning><Transpose>0</Transpose><Finetune>0</Finetune>'
            '<BeatSyncIsActive>false</BeatSyncIsActive><OneShotTrigger>%s</OneShotTrigger><NewNoteAction>%s</NewNoteAction>'
            '<InterpolationMode>Cubic</InterpolationMode><AutoSeek>false</AutoSeek><AutoFade>true</AutoFade>%s'
            '<IsAlias>false</IsAlias><MuteGroupIndex>-1</MuteGroupIndex><ModulationSetIndex>-1</ModulationSetIndex><DeviceChainIndex>-1</DeviceChainIndex>'
            '<Mapping><Layer>Note-On Layer</Layer><BaseNote>%d</BaseNote><NoteStart>%d</NoteStart><NoteEnd>%d</NoteEnd>'
            '<MapKeyToPitch>%s</MapKeyToPitch><VelocityStart>0</VelocityStart><VelocityEnd>127</VelocityEnd>'
            '<MapVelocityToVolume>true</MapVelocityToVolume></Mapping>'
            '<DisplayStart>0</DisplayStart><DisplayLength>%d</DisplayLength></Sample>'
            % (escape(name), 'false' if pitched else 'true', 'NoteOff' if pitched else 'Cut', lp,
               rn_note(root), rn_note(lo), 119 if hi >= 127 else rn_note(hi), 'true' if pitched else 'false', frames))


def write_xrni(path, name, zones, folder, comment=''):
    """Renoise's instrument: zones as in write_multisample (wav, low, high, root MIDI notes, track:
    0 a drum - played as it is, a one-shot - else pitched, use its loop). Drag it into Renoise's
    instrument list, or load it there. The WAVs go in as they are (Renoise reads WAV)."""
    smps, files = [], []
    for k, (rel, lo, hi, root, track, looped) in enumerate(zones):
        src = os.path.join(folder, rel)
        frames, loop = wav_info(src)
        sname = os.path.basename(rel)[:-4]
        smps.append(rn_sample_xml(sname, frames, lo, hi, root, track > 0, loop if looped else None))
        files.append((src, 'SampleData/Sample%02d (%s).wav' % (k, re.sub(r'[\\/:*?"<>|]', '_', sname))))
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<RenoiseInstrument doc_version="33"><Name>%s</Name>'
           '<GlobalProperties><Volume>1.0</Volume><Transpose>0</Transpose><Scale>None</Scale><ScaleKey>C</ScaleKey>'
           '<Quantize>None</Quantize><Monophonic>false</Monophonic><MonophonicGlide>0</MonophonicGlide>'
           '<ShowCommentsAfterLoading>false</ShowCommentsAfterLoading>%s</GlobalProperties>'
           '<SampleGenerator><Samples>%s</Samples><SelectedSampleIndex>0</SelectedSampleIndex></SampleGenerator>'
           '</RenoiseInstrument>\n'
           % (escape(name), '<Comments><Comment>%s</Comment></Comments>' % escape(comment) if comment else '', ''.join(smps)))
    tmp = path + '.part'
    with zipfile.ZipFile(tmp, 'w') as z:
        z.writestr('Instrument.xml', xml, compress_type=zipfile.ZIP_DEFLATED)
        for src, arc in files:
            z.write(src, arc, compress_type=zipfile.ZIP_DEFLATED)
    os.replace(tmp, path)


def bitwig_ok():
    """Bitwig presets need Bitwig's own empty devices (its Library): only where Bitwig is installed"""
    global bwpreset
    try:
        import bwpreset
    except ImportError:
        return False
    return bwpreset.library() is not None


def loop_of(outdir, rel):
    return wav_info(os.path.join(outdir, rel))[1]


def build(outdir):
    d = json.load(open(os.path.join(outdir, 'stems.json')))
    key, title = d['key'], d.get('title', '')
    inst = os.path.join(outdir, '5 Instruments')
    os.makedirs(inst, exist_ok=True)
    shots = d.get('oneshots', [])
    made = []
    bw = bitwig_ok()
    bwdir = os.path.join(inst, 'Bitwig')
    if bw: os.makedirs(bwdir, exist_ok=True)
    rndir = os.path.join(inst, 'Renoise')
    os.makedirs(rndir, exist_ok=True)

    def renoise(stem, zones, description):
        p = os.path.join(rndir, stem + '.xrni')
        write_xrni(p, stem, zones, outdir, description)
        made.append(p)

    def bitwig(stem, zones, category, description):
        if not bw: return
        p = os.path.join(bwdir, stem + '.bwpreset')
        bwpreset.sampler_preset(p, [(os.path.join(outdir, f), lo, hi, root, track, loop_of(outdir, f) if looped else None)
                                    for f, lo, hi, root, track, looped in zones],
                                category=category, comment=description, tags='soundmon %s' % key)
        made.append(p)

    # the drum kit: a key each from C1, part by part
    zones, legend, k = [], [], 36
    for part in DRUMS:
        for o in [o for o in shots if o['part'] == part]:
            zones.append((o['file'], k, k, k, 0.0, False))
            legend.append('%-4s %-4s (MIDI %3d)  %s' % (key_name(k), rn_name(k), k, os.path.basename(o['file'])[:-4]))
            k += 1
    if zones:
        p = os.path.join(inst, '%s - drum kit.multisample' % key)
        write_multisample(p, '%s %s - drum kit' % (key, title), 'Drums', zones, outdir,
                          'Every drum hit of %s, a key each from C1' % key)
        with open(os.path.join(inst, '%s - drum kit keys.txt' % key), 'w') as f:
            f.write('%s - drum kit: its keys (Bitwig\'s names, C3 = MIDI 60, then Renoise\'s, C-4 = MIDI 60)\n\n' % key + '\n'.join(legend) + '\n')
        made.append(p)
        renoise('%s - drum kit' % key, zones, 'Every drum hit of %s, a key each from C-2 (MIDI 36, Bitwig\'s C1)' % key)
        if bw:   # Bitwig's Drum Machine: a pad a hit, each a Sampler
            p = os.path.join(bwdir, '%s - drum kit.bwpreset' % key)
            bwpreset.drum_preset(p, [(z[1], os.path.basename(z[0])[:-4].replace(key + ' - ', ''), os.path.join(outdir, z[0]))
                                     for z in zones], comment='Every drum hit of %s, a pad each from C1' % key,
                                 tags='soundmon %s drums' % key)
            made.append(p)

    for part in sorted({o['part'] for o in shots} - set(DRUMS)):
        mine = [o for o in shots if o['part'] == part]
        # notes: each on its pitch, its zone halfway to its neighbours'
        notes = sorted((midi_of(o['label']), o['file']) for o in mine if midi_of(o['label']) is not None)
        notes = [n for i, n in enumerate(notes) if i == 0 or n[0] != notes[i - 1][0]]
        if notes:
            zones = []
            for i, (m, f) in enumerate(notes):
                lo = 0 if i == 0 else (notes[i - 1][0] + m) // 2 + 1
                hi = 127 if i == len(notes) - 1 else (m + notes[i + 1][0]) // 2
                zones.append((f, lo, hi, m, 1.0, False))
            p = os.path.join(inst, '%s - %s.multisample' % (key, pretty(part)))
            desc = 'The notes of %s\'s %s: %s' % (key, pretty(part), ' '.join(key_name(m) for m, _ in notes))
            write_multisample(p, '%s %s - %s' % (key, title, pretty(part)), pretty(part).title(), zones, outdir, desc)
            made.append(p)
            bitwig('%s - %s' % (key, pretty(part)), zones, pretty(part).title(), desc)
            renoise('%s - %s' % (key, pretty(part)), zones, desc)
        # chords and stabs: a key each from C3
        chords = [o for o in mine if o['label'].split()[0] in ('chord', 'stab', 'hit')]
        if chords:
            zones = [(o['file'], 60 + i, 60 + i, 60 + i, 1.0, False) for i, o in enumerate(chords)]
            p = os.path.join(inst, '%s - %s chords.multisample' % (key, pretty(part)))
            desc = 'The chords and stabs of %s\'s %s, a key each from C3' % (key, pretty(part))
            write_multisample(p, '%s %s - %s chords' % (key, title, pretty(part)), pretty(part).title(), zones, outdir, desc)
            made.append(p)
            bitwig('%s - %s chords' % (key, pretty(part)), zones, pretty(part).title(), desc)
            renoise('%s - %s chords' % (key, pretty(part)), zones, desc.replace('from C3', 'from C3 (Renoise C-4)'))
        # a held bar: looped, on every key (C3 as it was)
        held = [o for o in mine if o['label'].split()[0] == 'held']
        if held:
            zones = [(held[0]['file'], 0, 127, 60, 1.0, True)]
            p = os.path.join(inst, '%s - %s pad.multisample' % (key, pretty(part)))
            desc = 'A held bar of %s\'s %s, looped; C3 plays it as it was' % (key, pretty(part))
            write_multisample(p, '%s %s - %s pad' % (key, title, pretty(part)), 'Pad', zones, outdir, desc)
            made.append(p)
            bitwig('%s - %s pad' % (key, pretty(part)), zones, 'Pad', desc)
            renoise('%s - %s pad' % (key, pretty(part)), zones, desc.replace('C3 plays', 'C3 (Renoise C-4) plays'))
    d['instruments'] = [os.path.relpath(p, outdir) for p in made]
    with open(os.path.join(outdir, 'stems.json'), 'w') as f:
        json.dump(d, f, indent=1)
    return made


def main(argv):
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        return 0
    for outdir in argv:
        made = build(outdir)
        print('%s: %d instruments' % (outdir, len(made)))
        for p in made: print('   ', os.path.basename(p))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
