#!/usr/bin/env python3
"""--opl: real AdLib / Sound Blaster FM, through the Nuked OPL3 core.

WHY AN EMULATOR AND NOT A PROMPT

Same wall `--chip` hit, one step steeper. FM synthesis gets its character from
operator frequency ratios, envelope rates and feedback — parameters a diffusion
model trained on recordings has no representation of. `--style opl3` produces
music that *evokes* DOS; it cannot produce a YMF262, because nothing in the model
is a YMF262.

So drive the actual chip. Nuked OPL3 (nukeykt) is the cycle-accurate core DOSBox
uses, it is a single opl3.c/opl3.h pair, and it needs exactly four entry points:

    OPL3_Reset(chip, samplerate)
    OPL3_WriteReg(chip, reg, value)
    OPL3_GenerateStream(chip, buf, frames)

Everything below is register pokes. The output is not "FM-like"; it is what an
OPL3 does when you write those registers, sample-for-sample.

WHAT IS SHARED WITH --chip, AND WHY

`chip.compose()` produces note events — bars, chords, a motif, drum hits — with
no idea how they will be voiced. This module consumes the same events and writes
OPL registers instead of summing oscillators. **Composition and synthesis are
deliberately separate**: the musical decisions are hard and worth writing once,
while "what does a voice sound like" is exactly what differs between a 2A03 and
a YMF262. Adding a third chip (SID, AY-3-8910, YM2612) means writing a renderer,
not another composer.

THE PATCH BANK IS THE SOUND

An OPL instrument is ~11 bytes of operator config, and the entire difference
between "authentic AdLib" and "awful" lives in those bytes. INSTRUMENTS below is
hand-authored so this works out of the box with nothing to fetch and no
third-party data in the repo. `--opl-bank FILE` loads a standard bank if you
have one — .sbi (single), .op2 (GENMIDI), .wopl (OPL3BankEditor) — which is how
you get a specific game's exact voices without soundmon shipping them.

LICENSING NOTE. Nuked OPL3 is LGPL-2.1 and soundmon is MIT, so the core is
FETCHED AND BUILT at setup (`./download-models.sh --opl`) rather than vendored.
The two licences stay separate and attributed, same pattern as the models.
"""
import ctypes
import os
import struct
import sys

SAMPLE_RATE = 49716          # the OPL3's native rate; resampling it is a choice,
                             # and not resampling is the honest default

# Operator register offsets for melodic channels 0-8 (OPL bank 0).
_OP = [0, 1, 2, 8, 9, 10, 16, 17, 18]

# --- registers ---------------------------------------------------------------
# 0x20+op  AM | VIB | EG-type | KSR | MULT
# 0x40+op  KSL | TL (total level, 0 = loudest)
# 0x60+op  AR | DR
# 0x80+op  SL | RR
# 0xE0+op  waveform select (OPL3: 0-7)
# 0xA0+ch  F-number low
# 0xB0+ch  key-on | block | F-number high
# 0xC0+ch  output L/R | feedback | connection
# 0xBD     rhythm-mode enable + BD/SD/TT/CY/HH triggers

def _op_pair(ch):
    o = _OP[ch % 9]
    return o, o + 3


class Instrument:
    """One 2-operator FM patch. Field order mirrors the register layout so the
    numbers can be read against a chip datasheet or an .sbi dump directly."""

    __slots__ = ("m", "c", "fb", "con", "name")

    def __init__(self, name, mod, car, fb=0, con=0):
        self.name = name
        self.m = mod          # (mult, ksl, tl, ar, dr, sl, rr, wave, am, vib, eg, ksr)
        self.c = car
        self.fb = fb
        self.con = con


#            name          mult ksl  tl  ar  dr  sl  rr  wv am vib eg ksr
INSTRUMENTS = {
    # The signature AdLib sound: bright, nasal, slightly buzzy brass stab.
    "brass":  Instrument("brass",
                         (1, 0, 14, 12, 6, 3, 6, 0, 0, 1, 0, 0),
                         (1, 0,  8, 13, 5, 4, 7, 0, 0, 1, 0, 0), fb=4),
    # Metallic bell/pluck — mult ratio 1:3 is what makes it read as a bell.
    "bell":   Instrument("bell",
                         (3, 0, 18, 15, 9, 0, 8, 0, 0, 0, 1, 0),
                         (1, 0,  6, 15, 7, 0, 7, 0, 0, 0, 1, 0), fb=5),
    # Short percussive bass. eg=0 so it decays rather than sustaining.
    "bass":   Instrument("bass",
                         (1, 0, 16, 14, 7, 2, 8, 0, 0, 0, 1, 0),
                         (1, 0,  4, 14, 6, 3, 8, 0, 0, 0, 1, 0), fb=6),
    # Sustained organ, con=1 (both operators to output) = the hollow OPL organ.
    "organ":  Instrument("organ",
                         (1, 0, 12,  14, 0, 15, 6, 0, 0, 0, 0, 0),
                         (1, 0,  8,  14, 0, 15, 6, 0, 0, 0, 0, 0), fb=0, con=1),
    # Thin square-ish lead. wave=1 (half-sine) is the cheapest route to "chippy"
    # without leaving FM.
    "lead":   Instrument("lead",
                         (1, 0, 20, 15, 4, 2, 7, 1, 0, 1, 0, 0),
                         (1, 0,  7, 15, 3, 3, 7, 1, 0, 1, 0, 0), fb=7),
    "string": Instrument("string",
                         (1, 1, 16, 8, 4, 6, 5, 0, 1, 1, 0, 0),
                         (1, 0, 10, 9, 3, 8, 5, 0, 1, 1, 0, 0), fb=2, con=1),
    # Short plucked attack, long-ish tail — harp/lute territory.
    "pluck":  Instrument("pluck",
                         (2, 0, 16, 15, 8, 1, 6, 0, 0, 0, 1, 0),
                         (1, 0,  6, 15, 6, 2, 7, 0, 0, 0, 1, 0), fb=3),
    # Nasal double-reed. mult 1:2 with heavy feedback is the classic OPL "oboe".
    "reed":   Instrument("reed",
                         (2, 0, 14, 13, 3, 8, 5, 0, 0, 1, 0, 0),
                         (1, 0,  9, 13, 2, 9, 5, 0, 0, 1, 0, 0), fb=6),
    # Soft, breathy, almost pure sine — the quiet end of the bank.
    "flute":  Instrument("flute",
                         (1, 0, 26, 11, 2, 10, 4, 0, 0, 1, 0, 0),
                         (1, 0, 11, 12, 1, 12, 4, 0, 0, 1, 0, 0), fb=0, con=1),
    # Wide, slow, sustained pad for solemn/eerie material.
    "choir":  Instrument("choir",
                         (1, 1, 20, 6, 2, 12, 4, 0, 1, 1, 0, 0),
                         (1, 0, 12, 7, 1, 13, 4, 0, 1, 1, 0, 0), fb=1, con=1),
    # Inharmonic and unpleasant on purpose — dread, not melody.
    "growl":  Instrument("growl",
                         (7, 0, 12, 12, 5, 6, 5, 0, 0, 1, 0, 0),
                         (1, 0,  8, 13, 4, 7, 6, 0, 0, 1, 0, 0), fb=7),
    # Bright metallic mallet — treasure, magic, discovery.
    "mallet": Instrument("mallet",
                         (4, 0, 18, 15, 10, 0, 9, 1, 0, 0, 1, 0),
                         (1, 0,  7, 15, 8, 0, 8, 0, 0, 0, 1, 0), fb=5),
}

# --- mood -> voices ----------------------------------------------------------
# THIS is why mood did not land on AdLib before: every track was voiced
# brass/organ/bass regardless of mood, so 24 tracks shared one timbre and the
# only variation was pitch. chip.py at least varied its duty cycle, which IS
# timbre on a 2A03 — the OPL equivalent is choosing different operator patches.
#
# Each entry offers SEVERAL options per role; track identity picks one, so two
# `solemn` tracks in the same pack get different instrumentation while staying
# recognisably solemn.
MOOD_VOICES = {
    "heroic":     (("brass", "reed"),        ("organ", "string"),  ("bass",)),
    "triumphant": (("brass",),               ("organ", "mallet"),  ("bass",)),
    "ominous":    (("growl", "reed"),        ("choir", "organ"),   ("bass",)),
    "eerie":      (("flute", "bell"),        ("choir", "string"),  ("bass", "growl")),
    "melancholy": (("reed", "flute"),        ("string", "choir"),  ("bass",)),
    "solemn":     (("choir", "organ"),       ("organ", "string"),  ("bass",)),
    "mysterious": (("pluck", "flute", "bell"), ("string", "choir"), ("bass",)),
    "tense":      (("reed", "growl"),        ("string", "organ"),  ("bass",)),
    "frantic":    (("lead", "brass"),        ("organ", "lead"),    ("bass",)),
    "driving":    (("brass", "lead"),        ("organ", "string"),  ("bass",)),
    "playful":    (("pluck", "mallet"),      ("mallet", "organ"),  ("bass",)),
    "serene":     (("flute", "pluck"),       ("choir", "string"),  ("bass",)),
    "grand":      (("brass", "choir"),       ("organ", "choir"),   ("bass",)),
    "wondrous":   (("mallet", "bell"),       ("string", "choir"),  ("bass", "pluck")),
}


def voices_for(mood_name, track, bank):
    """Pick (lead, arp, bass) instruments for this mood + track."""
    import theory
    lead_o, arp_o, bass_o = MOOD_VOICES.get(mood_name, MOOD_VOICES["mysterious"])
    ident = theory.Ident(track, "voice:" + mood_name)
    pick = lambda opts, dflt: bank.get(ident.pick(opts), bank[dflt])
    return pick(lead_o, "brass"), pick(arp_o, "organ"), pick(bass_o, "bass")


class OPL3:
    """ctypes wrapper over the Nuked core. Writes are immediate (OPL3_WriteReg)
    rather than buffered, so the sequencer below controls timing exactly."""

    def __init__(self, lib_path=None):
        self.lib = ctypes.CDLL(lib_path or find_lib())
        self.lib.OPL3_Reset.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.lib.OPL3_WriteReg.argtypes = [ctypes.c_void_p, ctypes.c_uint16,
                                           ctypes.c_uint8]
        self.lib.OPL3_GenerateStream.argtypes = [ctypes.c_void_p,
                                                 ctypes.POINTER(ctypes.c_int16),
                                                 ctypes.c_uint32]
        # opl3_chip is ~12 KB; over-allocate rather than depend on a struct
        # layout we do not parse. We only ever hand the core a pointer.
        self._chip = ctypes.create_string_buffer(131072)
        self.lib.OPL3_Reset(self._chip, SAMPLE_RATE)
        self.write(0x105, 0x01)      # OPL3 mode ("NEW" bit) — 4-op & stereo
        self.write(0x104, 0x00)      # all channels 2-operator
        self.write(0x01, 0x20)       # enable waveform select

    def write(self, reg, val):
        self.lib.OPL3_WriteReg(self._chip, int(reg) & 0x1FF, int(val) & 0xFF)

    def render(self, frames, np):
        """Generate `frames` stereo frames, return float mono-summed array."""
        if frames <= 0:
            return np.zeros(0)
        buf = (ctypes.c_int16 * (frames * 2))()
        self.lib.OPL3_GenerateStream(self._chip, buf, frames)
        a = np.frombuffer(buf, dtype="<i2").astype(np.float64) / 32768.0
        return a.reshape(-1, 2).mean(axis=1)

    # --- higher level -------------------------------------------------------
    def program(self, ch, ins):
        m, c = _op_pair(ch)
        for off, p in ((m, ins.m), (c, ins.c)):
            mult, ksl, tl, ar, dr, sl, rr, wv, am, vib, eg, ksr = p
            self.write(0x20 + off, (am << 7) | (vib << 6) | (eg << 5) |
                                   (ksr << 4) | (mult & 0x0F))
            self.write(0x40 + off, ((ksl & 3) << 6) | (tl & 0x3F))
            self.write(0x60 + off, ((ar & 0x0F) << 4) | (dr & 0x0F))
            self.write(0x80 + off, ((sl & 0x0F) << 4) | (rr & 0x0F))
            self.write(0xE0 + off, wv & 0x07)
        # 0x30 = both speakers on. Mono-summing later, but a channel with neither
        # bit set is silent, which is a very confusing way to hear nothing.
        self.write(0xC0 + ch, 0x30 | ((ins.fb & 7) << 1) | (ins.con & 1))

    def key_on(self, ch, freq):
        fnum, block = fnum_block(freq)
        self.write(0xA0 + ch, fnum & 0xFF)
        self.write(0xB0 + ch, 0x20 | ((block & 7) << 2) | ((fnum >> 8) & 3))

    def key_off(self, ch):
        self.write(0xB0 + ch, 0x00)


def fnum_block(freq):
    """Hz -> (F-Number, Block).

    F-Number = freq * 2^(20-Block) / 49716. Block is picked so F-Number lands in
    the upper half of its 10-bit range, where the chip's frequency resolution is
    best — the same note voiced in a low block is measurably more out of tune.
    """
    if freq <= 0:
        return 0, 0
    for block in range(8):
        fnum = int(round(freq * (2 ** (20 - block)) / 49716.0))
        if fnum < 1024:
            if fnum >= 512 or block == 7:
                return max(1, fnum), block
            # too low in this block; try the next one down only if it fits
            for b2 in range(block, -1, -1):
                f2 = int(round(freq * (2 ** (20 - b2)) / 49716.0))
                if 512 <= f2 < 1024:
                    return f2, b2
            return max(1, fnum), block
    return 1023, 7


def find_lib():
    """Locate libopl3. Built by ./download-models.sh --opl."""
    here = os.path.dirname(os.path.abspath(__file__))
    names = ("libopl3.so", "libopl3.dylib", "opl3.dll")
    roots = (os.path.join(here, "vendor"), here,
             os.path.expanduser("~/.cache/soundmon"))
    for r in roots:
        for n in names:
            p = os.path.join(r, n)
            if os.path.exists(p):
                return p
    sys.exit("--opl needs the Nuked OPL3 core.\n"
             "  build it once:  ./download-models.sh --opl\n"
             "  (fetches nukeykt/Nuked-OPL3 and compiles a small shared library;\n"
             "   it is LGPL-2.1, so it is built locally rather than shipped here)")


# --- external bank loading ---------------------------------------------------
def load_sbi(path):
    """A single-instrument .sbi dump (Sound Blaster Instrument)."""
    with open(path, "rb") as fh:
        d = fh.read()
    if d[:4] not in (b"SBI\x1a", b"2OP\x1a"):
        raise ValueError("not an .sbi file")
    b = d[0x24:0x24 + 11]
    if len(b) < 11:
        raise ValueError("truncated .sbi")
    def unpack(o):
        av, ksltl, ardr, slrr, wv = b[o], b[o + 2], b[o + 4], b[o + 6], b[o + 8]
        return (av & 0x0F, ksltl >> 6, ksltl & 0x3F, ardr >> 4, ardr & 0x0F,
                slrr >> 4, slrr & 0x0F, wv & 7,
                (av >> 7) & 1, (av >> 6) & 1, (av >> 5) & 1, (av >> 4) & 1)
    fbc = b[10]
    return Instrument(os.path.basename(path), unpack(0), unpack(1),
                      fb=(fbc >> 1) & 7, con=fbc & 1)


def load_bank(path):
    """Load a bank file and return {name: Instrument}, merged over the built-ins."""
    ext = os.path.splitext(path)[1].lower()
    bank = dict(INSTRUMENTS)
    if ext == ".sbi":
        ins = load_sbi(path)
        # A single patch replaces the lead — that is almost always why you point
        # soundmon at one .sbi.
        bank["lead"] = ins
        return bank
    if ext in (".op2", ".wopl"):
        sys.exit(f"{ext} bank loading is not implemented yet — use .sbi, or omit "
                 f"--opl-bank for the built-in bank")
    sys.exit(f"unknown bank format {ext!r} (expected .sbi)")


# --- rendering ---------------------------------------------------------------
# Channel assignment. Melodic voices sit low so rhythm mode can own 6/7/8.
CH_LEAD, CH_ARP1, CH_ARP2, CH_BASS = 0, 1, 2, 3
RHYTHM = {"k": 0x10, "s": 0x08, "h": 0x01}      # BD, SD, HH in register 0xBD


def render(a, ev, bars, spb, np, bank, voices, steps=16):
    sr = SAMPLE_RATE
    step_s = spb / float(steps)
    total_steps = bars * steps

    chip = OPL3(getattr(a, "opl_lib", None))
    v_lead, v_arp, v_bass = voices
    chip.program(CH_LEAD, v_lead)
    chip.program(CH_ARP1, v_arp)
    chip.program(CH_ARP2, v_arp)
    chip.program(CH_BASS, v_bass)

    # Rhythm mode: the chip's own BD/SD/HH, which is what AdLib games actually
    # used for drums. Channels 6-8 become percussion and stop being melodic.
    chip.write(0xBD, 0x20)
    for ch, f in ((6, 90.0), (7, 240.0), (8, 640.0)):
        fn, bl = fnum_block(f)
        chip.write(0xA0 + ch, fn & 0xFF)
        chip.write(0xB0 + ch, ((bl & 7) << 2) | ((fn >> 8) & 3))
    for op in (12, 15, 16, 13, 14, 17):
        chip.write(0x20 + op, 0x01)
        chip.write(0x40 + op, 0x00)
        chip.write(0x60 + op, 0xF8)
        chip.write(0x80 + op, 0xF8)

    # Index events by absolute 16th step so the sequencer is one pass.
    lead, arp, bass, drum = {}, {}, {}, {}
    for bar, pos, dur, note, _duty in ev["lead"]:
        lead.setdefault(bar * steps + pos, []).append((dur, note))
    for bar, pos, dur, note, _duty in ev["arp"]:
        arp.setdefault(bar * steps + pos, []).append((dur, note))
    for bar, pos, dur, note in ev["bass"]:
        bass.setdefault(bar * steps + pos, []).append((dur, note))
    for bar, pos, kind in ev["drum"]:
        drum.setdefault(bar * steps + pos, []).append(kind)

    def hz(midi):
        return 440.0 * (2.0 ** ((midi - 69) / 12.0))

    chunks = []
    arp_flip = 0
    for s in range(total_steps):
        for st, ch in ((lead, CH_LEAD), (bass, CH_BASS)):
            if s in st:
                chip.key_off(ch)
                chip.key_on(ch, hz(st[s][0][1]))
        if s in arp:
            # Alternate two channels so consecutive arp notes overlap slightly,
            # which is what gives the fast arpeggio its chord-like shimmer
            # instead of a stuttering single voice.
            ch = CH_ARP1 if arp_flip else CH_ARP2
            arp_flip ^= 1
            chip.key_off(ch)
            chip.key_on(ch, hz(arp[s][0][1]))
        if s in drum:
            bits = 0
            for k in drum[s]:
                bits |= RHYTHM.get(k, 0)
            chip.write(0xBD, 0x20)              # clear triggers
            chip.write(0xBD, 0x20 | bits)       # then strike
        n = int(round((s + 1) * step_s * sr)) - int(round(s * step_s * sr))
        chunks.append(chip.render(n, np))

    out = np.concatenate(chunks) if chunks else np.zeros(1)
    peak = float(np.abs(out).max())
    if peak > 1e-9:
        out = out * (10.0 ** (a.normalize_db / 20.0) / peak)
    return out


def run(a, slug, to_ogg=None, loudness_normalize=None):
    """Generate OPL3 FM music. Same contract as chip.run()."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        sys.exit(f"--opl needs numpy + soundfile: pip install soundfile numpy   ({e})")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import chip as chipmod

    bank = load_bank(a.opl_bank) if getattr(a, "opl_bank", None) else dict(INSTRUMENTS)

    dest = os.path.abspath(os.path.expanduser(a.output_to)) if a.output_to else os.getcwd()
    if not os.path.isdir(dest):
        if a.create_dirs or a.output_to:
            os.makedirs(dest, exist_ok=True)
        else:
            sys.exit(f"output dir does not exist: {dest} (add --create-dirs)")

    base = a.name or slug(a.prompt or "opl")
    n_out = max(1, a.number)
    made = []
    for i in range(n_out):
        # --seed -1 means "random" in soundmon, not None. See the note in chip.py.
        if a.seed is not None and a.seed >= 0:
            seed = (a.seed + i) % (2 ** 32)
        else:
            seed = int.from_bytes(os.urandom(4), "big")
        rng = np.random.default_rng(seed)

        mfile = getattr(a, "from_midi", None)
        if mfile:
            # MIDI: nothing is inferred. Pitches, durations, tempo and metre all
            # come from the file, so this is strictly better than --from-audio
            # when a MIDI exists. Mood still picks the voicing.
            import midi as midimod
            mname = getattr(a, "mood", None)
            if not mname or mname == "auto":
                mname = chipmod.infer_mood(getattr(a, "prompt", "") or "")
            got = midimod.to_events(mfile, np, seconds=a.seconds,
                                    transpose=getattr(a, "transpose", 0),
                                    start_frac=getattr(a, "midi_start", 0.15))
            if not got:
                sys.exit(f"--from-midi: no playable notes in {mfile}")
            ev, bars, spb, info, scale_name = got
            steps = info["steps"]; meter_s = info["timesig"]
            print(f"   \u266a {info['title'][:40]}: {midimod.describe(info)}")
        else:
            src = getattr(a, "from_audio", None)
        if mfile:
            pass
        elif src:
            import transcribe
            mname = getattr(a, "mood", None)
            if not mname or mname == "auto":
                mname = chipmod.infer_mood(getattr(a, "prompt", "") or "")
            got = transcribe.to_events_hi(src, np, sf, seconds=a.seconds,
                                          beats_per_bar=4, div=4)
            if not got:
                sys.exit(f"--from-audio: could not analyze {src}")
            ev, bars, spb, ana, scale_name = got
            steps = 16; meter_s = '4/4'
            print(f"   \u266a {os.path.basename(src)}: "
                  f"{transcribe.NOTE_NAMES[ana['root']]} {ana['mode']}  "
                  f"{ana['bpm']:.0f}bpm  {ana['notes']} notes  "
                  f"{ana['drum_hits']} hits")
        else:
            ev, bars, spb, scale_name, _prog, mname, _mood, _plan = chipmod.compose(a, np, rng)
            steps = _plan["_steps"]; meter_s = _plan["meter"]
        # Voices come from the MOOD unless the caller named one explicitly.
        # Without this, mood was inaudible on AdLib: every track was voiced
        # brass/organ/bass and only the pitches changed.
        track = getattr(a, "name", None) or base
        voices = list(voices_for(mname, track, bank))
        for i_v, (flag, dflt) in enumerate((("opl_lead", "brass"),
                                            ("opl_arp", "organ"),
                                            ("opl_bass", "bass"))):
            chosen = getattr(a, flag, None)
            if chosen and chosen != dflt and chosen in bank:
                voices[i_v] = bank[chosen]
        v_names = "/".join(v.name for v in voices)
        audio = render(a, ev, bars, spb, np, bank, voices, steps)

        # Seed in every filename — see the note in chip.py.
        name = f"{base}_s{seed}"
        path = os.path.join(dest, f"{name}.wav")
        sf.write(path, audio, SAMPLE_RATE,
                 subtype=f"PCM_{a.bits}" if a.bits != 8 else "PCM_U8")
        if getattr(a, "lufs_target", None) is not None and loudness_normalize:
            loudness_normalize(path, a.lufs_target, a.true_peak)
        if getattr(a, "ogg", False) and to_ogg:
            path = to_ogg(path, a.ogg_quality, a.keep_wav)
        made.append(path)
        print(f"   ✅ [{i+1}/{n_out}] {os.path.basename(path):<30} "
              f"{len(audio)/SAMPLE_RATE:5.1f}s  {bars}bar {meter_s:<5}{mname:<11}"
              f"{scale_name:<11}{v_names:<20}seed={seed}")

    print(f"   all done  |  {len(made)} file(s) in {dest}  ({SAMPLE_RATE} Hz, OPL3 native)")
    print("   ↻ loops seamlessly by construction — whole bars, no crossfade needed")
    return made
