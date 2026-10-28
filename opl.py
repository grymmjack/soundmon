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
    """(modulator, carrier) REGISTER ADDRESSES for a 2-op channel.

    OPL3 has 18 channels in two banks of 9. Bank 1 (channels 9-17) lives at the
    SAME register offsets plus 0x100 — which is why the array-of-9 table below
    still works, and why every write needs the bank bit added. Only using bank 0
    is what limited this to 9 channels (and in practice 4), and is most of why
    the result sounded thin.
    """
    add = 0x000 if ch < 9 else 0x100
    o = _OP[ch % 9]
    return add + o, add + o + 3


def _ch_reg(ch, base):
    """Address of a per-channel register (0xA0/0xB0/0xC0) for any channel."""
    return (0x000 if ch < 9 else 0x100) + base + (ch % 9)


# 4-operator channel pairs. Setting a bit in 0x104 fuses a pair into one 4-op
# voice: the primary keeps the note, the secondary contributes operators 3 and 4.
# Six pairs exist, giving at most 6 four-op voices plus 6 remaining 2-op channels.
FOUR_OP_PAIRS = ((0, 3), (1, 4), (2, 5), (9, 12), (10, 13), (11, 14))
TWO_OP_ONLY = (6, 7, 8, 15, 16, 17)


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

    # Samples rendered between a key-off and a key-on of the same channel. The
    # chip only restarts an envelope that has REACHED its release phase, and it
    # only gets there by running at least one sample with the key off. Off-then-
    # on with zero samples between is therefore no note at all: --oplfx coin
    # played one note of two, and a stolen voice kept sounding the old note.
    RETRIGGER_GAP = 64

    def __init__(self, lib_path=None):
        path = lib_path or find_lib()
        if lib_path:
            path = os.path.abspath(os.path.expanduser(lib_path))
            if not os.path.isfile(path):
                sys.exit(f"--opl-lib {lib_path}: no such file")
        try:
            self.lib = ctypes.CDLL(path)
            self.lib.OPL3_Reset
        except (OSError, AttributeError) as e:
            sys.exit(f"--opl-lib {path}: can't load the Nuked OPL3 core ({e}). "
                     f"Rebuild it with ./download-models.sh --opl")
        self.lib.OPL3_Reset.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.lib.OPL3_WriteReg.argtypes = [ctypes.c_void_p, ctypes.c_uint16,
                                           ctypes.c_uint8]
        self.lib.OPL3_GenerateStream.argtypes = [ctypes.c_void_p,
                                                 ctypes.POINTER(ctypes.c_int16),
                                                 ctypes.c_uint32]
        # opl3_chip is ~12 KB; over-allocate rather than depend on a struct
        # layout we do not parse. We only ever hand the core a pointer.
        self._chip = ctypes.create_string_buffer(131072)
        self._off = set()            # channels keyed off since the last sample
        self._rhy = 0                # last 0xBD drum-trigger bits
        self._rhy_off = 0            # drum bits cleared since the last sample
        self._pend = []              # gap samples owed to the next render()
        self.lib.OPL3_Reset(self._chip, SAMPLE_RATE)
        self.write(0x105, 0x01)      # OPL3 mode ("NEW" bit) — 4-op & stereo
        self.write(0x104, 0x00)      # all channels 2-operator
        self.write(0x01, 0x20)       # enable waveform select

    def write(self, reg, val):
        reg, val = int(reg) & 0x1FF, int(val) & 0xFF
        if reg == 0xBD:
            bits = val & 0x1F
            if bits & ~self._rhy & self._rhy_off:
                self._gap()          # re-striking a drum it just released
            self._rhy_off |= self._rhy & ~bits
            self._rhy = bits
        self.lib.OPL3_WriteReg(self._chip, reg, val)

    def _raw(self, frames):
        buf = (ctypes.c_int16 * (frames * 2))()
        self.lib.OPL3_GenerateStream(self._chip, buf, frames)
        return buf

    def _gap(self):
        """Run the chip briefly so a released envelope can restart. The samples
        are not lost or added: they are owed to the next render() and taken out
        of it, so every note still starts within 1.3 ms of its time and the
        total length is exact."""
        self._pend.append(bytes(self._raw(self.RETRIGGER_GAP)))
        self._off.clear()
        self._rhy_off = 0

    def render(self, frames, np):
        """Generate `frames` stereo frames, return float mono-summed array."""
        if frames <= 0:
            return np.zeros(0)
        self._off.clear()
        self._rhy_off = 0
        owed = b"".join(self._pend)
        self._pend = []
        have = len(owed) // 4
        if have >= frames:
            if have > frames:
                self._pend = [owed[frames * 4:]]
            data = owed[:frames * 4]
        else:
            data = owed + bytes(self._raw(frames - have))
        a = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
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
        self.write(_ch_reg(ch, 0xC0), 0x30 | ((ins.fb & 7) << 1) | (ins.con & 1))

    def key_on(self, ch, freq):
        if ch in self._off:
            self._gap()
        fnum, block = fnum_block(freq)
        self.write(_ch_reg(ch, 0xA0), fnum & 0xFF)
        self.write(_ch_reg(ch, 0xB0), 0x20 | ((block & 7) << 2) | ((fnum >> 8) & 3))

    def key_off(self, ch):
        self.write(_ch_reg(ch, 0xB0), 0x00)
        self._off.add(ch)

    def set_four_op(self, mask):
        """Enable 4-operator mode per pair. mask bit i -> FOUR_OP_PAIRS[i]."""
        self.write(0x104, mask & 0x3F)


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


# --- General MIDI bank (Schism Tracker / Creative PLAY.EXE) ------------------
# 128 melodic patches, 11 bytes each, in Impulse Tracker's `adlib_bytes` order:
#
#   0,1  modulator / carrier   AM|VIB|EG|KSR|MULT   -> reg 0x20, 0x23
#   2,3  modulator / carrier   KSL|TL               -> reg 0x40, 0x43
#   4,5  modulator / carrier   AR|DR                -> reg 0x60, 0x63
#   6,7  modulator / carrier   SL|RR                -> reg 0x80, 0x83
#   8,9  modulator / carrier   waveform             -> reg 0xE0, 0xE3
#   10   feedback | connection                      -> reg 0xC0
#
# Writing raw register bytes rather than converting to Instrument() keeps this
# bit-exact: these values were tuned against real hardware, and re-deriving them
# through our own field packing would only introduce rounding.
GM_BANK = None


def load_gm_bank(path=None):
    """Parse fmpatches.c into 128 x 11 raw register bytes. Cached."""
    global GM_BANK
    if GM_BANK is not None:
        return GM_BANK
    here = os.path.dirname(os.path.abspath(__file__))
    src = path or os.path.join(here, "vendor", "fmpatches.c")
    if not os.path.exists(src):
        GM_BANK = []
        return GM_BANK
    rows = []
    with open(src, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{0x") and not line.startswith("{ 0x"):
                continue
            body = line[line.index("{") + 1:line.index("}")]
            vals = [int(v.strip(), 16) for v in body.split(",") if v.strip()]
            if len(vals) == 11:
                rows.append(tuple(vals))
    GM_BANK = rows
    return GM_BANK


def program_gm(chip, ch, raw):
    """Apply an 11-byte GM patch to a channel, register for register."""
    m, c = _op_pair(ch)
    chip.write(0x20 + m, raw[0]);  chip.write(0x20 + c, raw[1])
    chip.write(0x40 + m, raw[2]);  chip.write(0x40 + c, raw[3])
    chip.write(0x60 + m, raw[4]);  chip.write(0x60 + c, raw[5])
    chip.write(0x80 + m, raw[6]);  chip.write(0x80 + c, raw[7])
    chip.write(0xE0 + m, raw[8]);  chip.write(0xE0 + c, raw[9])
    # Force both speakers on; a patch byte with neither set is silent, which is a
    # very confusing way to hear nothing.
    chip.write(_ch_reg(ch, 0xC0), 0x30 | (raw[10] & 0x0F))


# --- external bank loading ---------------------------------------------------
def load_sbi(path):
    """A single-instrument .sbi dump (Sound Blaster Instrument)."""
    with open(path, "rb") as fh:
        d = fh.read()
    if d[:4] not in (b"SBI\x1a", b"2OP\x1a"):
        raise ValueError("not an .sbi file (no SBI header)")
    b = d[0x24:0x24 + 11]
    if len(b) < 11:
        raise ValueError(f"truncated .sbi ({len(d)} bytes; an instrument needs {0x24 + 11})")
    def unpack(o):
        av, ksltl, ardr, slrr, wv = b[o], b[o + 2], b[o + 4], b[o + 6], b[o + 8]
        return (av & 0x0F, ksltl >> 6, ksltl & 0x3F, ardr >> 4, ardr & 0x0F,
                slrr >> 4, slrr & 0x0F, wv & 7,
                (av >> 7) & 1, (av >> 6) & 1, (av >> 5) & 1, (av >> 4) & 1)
    fbc = b[10]
    return Instrument(os.path.splitext(os.path.basename(path))[0], unpack(0), unpack(1),
                      fb=(fbc >> 1) & 7, con=fbc & 1)


def load_bank(path):
    """Load --opl-bank and return {name: Instrument}, merged over the built-ins.

    .sbi   one patch; it becomes the LEAD (that is almost always why you point
           soundmon at one .sbi) and is also selectable by its file name
    .wopl  a whole GM bank (OPL3BankEditor / libADLMIDI); it replaces DMXOPL
           for the GM programs every part plays
    Errors are one line: a missing, truncated or foreign file used to end in a
    traceback.
    """
    p = os.path.abspath(os.path.expanduser(path))
    ext = os.path.splitext(p)[1].lower()
    if not os.path.exists(p):
        sys.exit(f"--opl-bank {path}: no such file")
    if os.path.isdir(p):
        sys.exit(f"--opl-bank {path}: is a folder, not a bank file")
    bank = dict(INSTRUMENTS)
    if ext == ".sbi":
        try:
            ins = load_sbi(p)
        except (OSError, ValueError) as e:
            sys.exit(f"--opl-bank {path}: {e}")
        bank["lead"] = ins
        bank[ins.name] = ins
        return bank
    if ext == ".wopl":
        global WOPL_BANK
        try:
            w = load_wopl(p)
        except (OSError, ValueError, IndexError) as e:
            sys.exit(f"--opl-bank {path}: {e}")
        if not w.get("melodic") or len(w["melodic"]) < 128:
            sys.exit(f"--opl-bank {path}: truncated WOPL bank "
                     f"({len(w.get('melodic') or [])} of 128 melodic instruments)")
        WOPL_BANK = w
        return bank
    if ext == ".op2":
        sys.exit(f"--opl-bank {path}: .op2 bank loading is not implemented yet — use .sbi or "
                 f".wopl, or omit --opl-bank for the built-in bank")
    sys.exit(f"--opl-bank {path}: unknown bank format {ext or '(no extension)'!r} "
             f"(expected .sbi or .wopl)")


# Hand-written stand-ins for the 16 GM program families, used when NO GM bank is
# installed (neither vendor/GENMIDI.wopl nor vendor/fmpatches.c). Without this a
# MIDI file or the procedural parts played on unprogrammed channels: a valid,
# normalised WAV of near-silence. The docstring's promise — music out of the box
# with nothing to fetch — is kept by voicing each family on INSTRUMENTS.
GM_FAMILY_VOICE = ("pluck", "mallet", "organ", "pluck", "bass", "string",
                   "choir", "brass", "reed", "flute", "lead", "choir",
                   "bell", "pluck", "mallet", "growl")


def have_gm_bank():
    return bool(load_wopl_bank() or load_gm_bank())


def fallback_voice(prog, bank=None):
    bank = bank or INSTRUMENTS
    return bank.get(GM_FAMILY_VOICE[(int(prog) // 8) % 16], INSTRUMENTS["organ"])


def program_instrument(chip, ch, ins, vel=None):
    """A hand-written Instrument on a channel, with velocity on the CARRIER only
    (attenuating the modulator changes timbre, not level)."""
    chip.program(ch, ins)
    if vel is not None:
        add = min(24, int(round((127 - max(1, min(127, vel))) * 0.28)))
        _m, _c = _op_pair(ch)
        chip.write(0x40 + _c, ((ins.c[1] & 3) << 6) | min(63, (ins.c[2] & 0x3F) + add))


def _apply_program(chip, ch, prog):
    """Select GM program `prog` on channel `ch`, best bank available.

    DMXOPL (WOPL) is preferred over the Creative PLAY.EXE set because it was
    voiced for the OPL3 rather than carried over from OPL2. Falling back rather
    than failing matters: a user who has not run `download-models.sh --opl` still
    gets music, just on the hand-authored bank.
    """
    w = load_wopl_bank()
    if w and 0 <= prog < len(w["melodic"]):
        program_wopl(chip, ch, w["melodic"][prog])
        return
    if load_gm_bank() and 0 <= prog < len(GM_BANK):
        program_gm(chip, ch, GM_BANK[prog])


# --- rendering ---------------------------------------------------------------
# Channel assignment. Melodic voices sit low so rhythm mode can own 6/7/8.
CH_LEAD, CH_ARP1, CH_ARP2, CH_BASS = 0, 1, 2, 3
RHYTHM = {"k": 0x10, "s": 0x08, "h": 0x01}      # BD, SD, HH in register 0xBD


def render(a, ev, bars, spb, np, bank, voices, steps=16, gmprog=None):
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
    _rhythm_setup(chip)

    # Index events by absolute 16th step so the sequencer is one pass.
    lead, arp, bass, drum = {}, {}, {}, {}
    for _it in ev["lead"]:
        bar, pos, dur, note = _it[0], _it[1], _it[2], _it[3]
        vel = _it[5] if len(_it) > 5 else None
        lead.setdefault(bar * steps + pos, []).append((dur, note, vel))
    for _it in ev["arp"]:
        bar, pos, dur, note = _it[0], _it[1], _it[2], _it[3]
        vel = _it[5] if len(_it) > 5 else None
        arp.setdefault(bar * steps + pos, []).append((dur, note, vel))
    for _it in ev["bass"]:
        bar, pos, dur, note = _it[0], _it[1], _it[2], _it[3]
        vel = _it[4] if len(_it) > 4 else None
        bass.setdefault(bar * steps + pos, []).append((dur, note, vel))
    for bar, pos, kind in ev["drum"]:
        drum.setdefault(bar * steps + pos, []).append(kind)

    def hz(midi):
        return 440.0 * (2.0 ** ((midi - 69) / 12.0))

    chunks = []
    arp_flip = 0
    for s in range(total_steps):
        for st, ch, role in ((lead, CH_LEAD, "lead"), (bass, CH_BASS, "bass")):
            if s in st:
                # Accent from the composer, so procedural output has dynamics
                # too. Without this the 4-voice path renders every note at one
                # level — measured 0.62 dB of spread across a whole track.
                _v = st[s][0][2] if len(st[s][0]) > 2 else None
                # Real GM patch for this note, when a MIDI supplied one. Real OPL
                # drivers reprogrammed the channel per note exactly like this —
                # without it every instrument in the file plays as one timbre.
                if gmprog:
                    p = gmprog.get(role, {}).get(s)
                    if p is not None:
                        _apply_program(chip, ch, p)
                chip.key_off(ch)
                _ins = (v_lead if role == "lead" else v_bass)
                if _v is not None:
                    _base = [_ins.c[2]] if hasattr(_ins, "c") else [0]
                    add = int(round((127 - max(1, min(127, _v))) * 0.28))
                    _m, _c = _op_pair(ch)
                    chip.write(0x40 + _c, min(63, (_base[0] & 0x3F) + add))
                chip.key_on(ch, hz(st[s][0][1]))
        if s in arp:
            # Alternate two channels so consecutive arp notes overlap slightly,
            # which is what gives the fast arpeggio its chord-like shimmer
            # instead of a stuttering single voice.
            ch = CH_ARP1 if arp_flip else CH_ARP2
            arp_flip ^= 1
            if gmprog:
                p = gmprog.get("arp", {}).get(s)
                if p is not None:
                    _apply_program(chip, ch, p)
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


VOICE_FLAGS = (("lead", "opl_lead", "--opl-lead", "brass"),
               ("arp", "opl_arp", "--opl-arp", "organ"),
               ("bass", "opl_bass", "--opl-bass", "bass"))


def pick_voices(a, bank, mname, track):
    """{part: Instrument} for the parts that play a hand-written patch, plus a
    label for the summary line.

    Explicit choices always win: --opl-lead/--opl-arp/--opl-bass, and an .sbi
    bank (its patch is the lead). They used to be built into a list nothing
    read — the summary named them while the audio came from the GM bank. With no
    GM bank installed at all, every part takes the mood's hand-written voice.
    Otherwise parts play the mood's GM programs from the bank, as before.
    """
    import chip as _cm
    mood_v = dict(zip(("lead", "arp", "bass"), voices_for(mname, track, bank)))
    sbi = getattr(a, "opl_bank", None) and str(a.opl_bank).lower().endswith(".sbi")
    out = {}
    for part, attr, flag, dflt in VOICE_FLAGS:
        chosen = getattr(a, attr, None)
        if chosen and (_cm._flag_given(flag) or chosen != dflt):
            out[part] = bank[chosen]
        elif part == "lead" and sbi:
            out[part] = bank["lead"]
    if not have_gm_bank():
        for part in ("lead", "arp", "bass"):
            out.setdefault(part, mood_v[part])
        return out, "/".join(out[p].name for p in ("lead", "arp", "bass"))
    if not out:
        return out, "GM " + _bank_label().split(" ")[0]
    return out, "/".join(out[p].name if p in out else "GM" for p in ("lead", "arp", "bass"))


def check_voice_names(a, bank):
    """--opl-lead nonexistent used to be ignored without a word."""
    for part, attr, flag, dflt in VOICE_FLAGS:
        v = getattr(a, attr, None)
        if v and v not in bank:
            sys.exit(f"unknown {flag} {v!r}. One of: {', '.join(sorted(bank))}")


def run(a, slug, to_ogg=None, loudness_normalize=None, to_flac=None):
    """Generate OPL3 FM music. Same contract as chip.run()."""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        sys.exit(f"--opl needs numpy + soundfile: pip install soundfile numpy   ({e})")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import chip as chipmod

    chipmod.check_seconds(a, "--opl")
    try:
        chipmod.parse_key(a.key)
    except ValueError as e:
        sys.exit(f"--opl: {e}")
    chipmod.note_unused_writers(a, "opl")
    bank = load_bank(a.opl_bank) if getattr(a, "opl_bank", None) else dict(INSTRUMENTS)
    check_voice_names(a, bank)
    OPL3(getattr(a, "opl_lib", None))          # a bad --opl-lib fails here, in one line
    dest = chipmod.out_dir(a)
    base = chipmod.base_name(a, slug, "opl")
    if not have_gm_bank():
        print("   \u2139 no GM bank in vendor/ (./download-models.sh --opl fetches DMXOPL); "
              "using the built-in hand-written patches")

    mfile = getattr(a, "from_midi", None)
    src = getattr(a, "from_audio", None)
    mname = chipmod.resolve_mood(a)
    if mfile:
        tnotes, tdrums, info = chipmod.load_midi_source(a, "--opl")
        if any(chipmod._flag_given(f) for _, _, f, _ in VOICE_FLAGS):
            print("   \u2139 --opl-lead/--opl-arp/--opl-bass voice composed parts; "
                  "a MIDI file plays its own GM programs")
        if getattr(a, "write_midi", False):
            print("   \u2139 --write-midi writes composed music; with --from-midi you already have the MIDI")
    elif src:
        import transcribe
        got = chipmod.load_audio_source(a, np, sf, "--opl")

    n_out = chipmod.takes_for(a)
    made = []
    for i in range(n_out):
        # --seed -1 means "random" in soundmon, not None. See the note in chip.py.
        if a.seed is not None and a.seed >= 0:
            seed = (a.seed + i) % (2 ** 32)
        else:
            seed = int.from_bytes(os.urandom(4), "big")
        rng = np.random.default_rng(seed)

        loop = True
        key_name = ""
        track = chipmod.track_identity(a)
        if mfile:
            # MIDI: nothing is inferred. Pitches, durations, tempo and metre all
            # come from the file, so this is strictly better than --from-audio
            # when a MIDI exists.
            ev, bars, spb, scale_name = None, 0, info["spb"], "(from MIDI)"
            steps = 16; meter_s = info["timesig"]; bpm = info["bpm"]
            loop = False
            if i == 0:
                n_gm = len(info.get("gm_used", []))
                print(f"   \u266a {info['title'][:38]}: {info['timesig']} "
                      f"{info['bpm']:.0f}bpm  {info['duration']:.1f}s  "
                      f"{info['notes']} notes  {info['drum_hits']} hits"
                      f"  {n_gm} instr / {len(info.get('gm_drums_used', []))} perc"
                      f" via {_bank_label()}")
            voices, v_names = {}, _bank_label().split(" (")[0]
            # EXACT time, no grid: 91% of notes in a real file sit off a 16th.
            audio = render_timed(a, tnotes, tdrums, info["duration"], np,
                                 load_wopl_bank(), with_drums=bool(tdrums))
        else:
            if src:
                ev, bars, spb, ana, scale_name = got
                steps = 16; meter_s = '4/4'; bpm = ana["bpm"]
                key_name = f"{transcribe.NOTE_NAMES[ana['root']]} {ana['mode']}"
                if i == 0:
                    print(f"   \u266a {os.path.basename(src)}: {key_name}  "
                          f"{ana['bpm']:.0f}bpm  {ana['notes']} notes  "
                          f"{ana['drum_hits']} hits")
            else:
                ev, bars, spb, scale_name, _prog, mname, _mood, _plan = chipmod.compose(a, np, rng)
                steps = _plan["_steps"]; meter_s = _plan["meter"]; bpm = _plan["bpm"]
                key_name = _plan.get("key_name", "")
            # Voices: explicit flags/.sbi, else the mood's GM programs from the
            # bank, else (no bank) the mood's hand-written patches. The take's
            # identity picks among the mood's options.
            voices, v_names = pick_voices(a, bank, mname,
                                          chipmod.take_identity(track, np.random.default_rng(seed)))
            # PROCEDURAL through the same renderer, so it inherits the DMXOPL
            # bank, level calibration, 18-channel polyphony, the GM percussion kit
            # and --chippy. Exactly bars*spb long: it loops on the bar.
            pnotes, pdrums = chipmod.events_to_timed(ev, bars, spb, steps, mname)
            audio = render_timed(a, pnotes, pdrums, bars * spb, np, load_wopl_bank(),
                                 with_drums=bool(pdrums), voices=voices, loop=True)

        # Seed in every filename — see the note in chip.py.
        name = f"{base}_s{seed}"
        audio, out_sr, out_bits = chipmod.apply_format(audio, SAMPLE_RATE, a, np)
        path = os.path.join(dest, f"{name}.wav")
        sf.write(path, audio, out_sr, subtype=chipmod.wav_subtype(out_bits))
        if getattr(a, "lufs_target", None) is not None and loudness_normalize:
            loudness_normalize(path, a.lufs_target, a.true_peak)
        # FLAC before OGG: it is lossless, so for format-locked audio it is the
        # only compression that preserves the crush. Asking for both means FLAC.
        if getattr(a, "flac", False) and to_flac:
            path = to_flac(path, a.keep_wav)
        elif getattr(a, "ogg", False) and to_ogg:
            path = to_ogg(path, a.ogg_quality, a.keep_wav)
        if getattr(a, "write_rad", False):
            import rad as _radw
            import midi as _mm
            if ev is not None:
                # The plan's own steps-per-bar, which is not always 16 \u2014 a 6/8
                # mood uses 12, and writing those as 16 lines per bar walked every
                # bar further out of place.
                gev, gbars, glines = ev, bars, steps
            else:
                # Let the source pick the line grid. RAD allows 100 patterns of 64
                # lines, so the cap is 6400 lines rather than the MOD's 8192.
                glines = chipmod.best_grid(tnotes, spb,
                                           cap_rows=_radw.MAX_PATTERNS * _radw.LINES)
                gev, gbars = chipmod.timed_to_grid(tnotes, tdrums, spb, glines)
            lead_gm, arp_gm, bass_gm = _mm.MOOD_GM.get(mname,
                                                       _mm.MOOD_GM["mysterious"])
            rp = os.path.splitext(path)[0] + ".rad"
            try:
                _radw.write_rad(rp, gev, gbars, spb, glines, np,
                                title=base[:20], wopl=load_wopl_bank(),
                                lines_per_bar=glines,
                                progs={"lead": lead_gm, "arp": arp_gm,
                                       "bass": bass_gm})
                _s, _b = _radw.timing_for(spb, glines)
                print(f"   \u266b also wrote {os.path.basename(rp)}"
                      f"  ({glines} lines/bar, speed {_s}, {_b} bpm)")
            except Exception as e:
                print(f"   \u26a0 rad write failed: {e}")
        if getattr(a, "write_midi", False) and ev is not None:
            # Additional output, not a substitute: the chip render is the point.
            import midi as _midiw
            mp = os.path.splitext(path)[0] + ".mid"
            try:
                _midiw.write_smf(mp, ev, bars, spb, steps, mood=mname,
                                 timesig=meter_s, title=base)
                print(f"   \u266b also wrote {os.path.basename(mp)}")
            except Exception as e:
                print(f"   \u26a0 midi write failed: {e}")
        made.append(path)
        real_s = len(audio) / float(out_sr)
        print(f"   \u2705 [{i+1}/{n_out}] {os.path.basename(path):<30} "
              f"{real_s:5.1f}s  {(str(bars) + 'bar') if bars else '  -  '} {meter_s:<5}{mname:<11}"
              f"{(key_name or scale_name):<14}{bpm:.0f}bpm  {v_names:<20}seed={seed}"
              f"{chipmod.length_note(a, real_s) if loop else ''}")

    print(f"   all done  |  {len(made)} file(s) in {dest}  ({out_sr} Hz"
          f"{', OPL3 native' if out_sr == SAMPLE_RATE else ', format-locked'})")
    if not mfile:
        print("   \u21bb loops seamlessly by construction — whole bars, no crossfade needed")
    return made


# --- WOPL banks (DMXOPL, libADLMIDI format) ----------------------------------
# A far better bank than the Creative PLAY.EXE set: DMXOPL is the Doom OPL3 bank,
# voiced specifically for a YMF262 rather than adapted from OPL2. MIT licensed,
# same as this repo.
#
# Format per Wohlstand's WOPL-and-OPLI specification. The one non-obvious detail,
# and the one that will silently produce wrong timbres if you get it backwards:
#
#     Operator 1 in the file is the CARRIER, operator 2 is the MODULATOR
#
# i.e. the reverse of the register order, where the modulator sits at the base
# offset and the carrier at +3. Swapping these does not error — it just makes
# every instrument sound wrong, which is exactly the kind of bug that survives.
WOPL_MAGIC = b"WOPL3-BANK\0"


def load_wopl(path):
    """Parse a .wopl bank -> {"melodic": [...], "percussion": [...]}.

    Each entry is a dict with the raw operator bytes plus the metadata soundmon
    can use (key offset, 4-op flag, rhythm type).
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:11] != WOPL_MAGIC:
        raise ValueError("not a WOPL bank")
    version = int.from_bytes(data[11:13], "little")
    mbanks = int.from_bytes(data[13:15], "big")
    pbanks = int.from_bytes(data[15:17], "big")
    pos = 19                                    # header is 19 bytes
    if version >= 2:                            # bank meta-data arrays
        pos += 34 * (mbanks + pbanks)
    ins_size = 66 if version >= 3 else 62

    def one(off):
        name = data[off:off + 32].split(b"\0")[0].decode("utf-8", "replace")
        key_off = int.from_bytes(data[off + 32:off + 34], "big", signed=True)
        key_off2 = int.from_bytes(data[off + 34:off + 36], "big", signed=True)
        vel_off = int.from_bytes(data[off + 36:off + 37], "big", signed=True)
        detune = int.from_bytes(data[off + 37:off + 38], "big", signed=True)
        perc_key = data[off + 38]
        flags = data[off + 39]
        fbc1, fbc2 = data[off + 40], data[off + 41]
        ops = []
        for k in range(4):
            b = off + 42 + k * 5
            ops.append(tuple(data[b:b + 5]))    # AVEKM, KSL|TL, AR|DR, SR|RR, WS
        return {"name": name, "key_off": key_off, "key_off2": key_off2,
                "detune": detune, "vel_off": vel_off,
                "perc_key": perc_key, "four_op": bool(flags & 0x01),
                "pseudo4": bool(flags & 0x02), "blank": bool(flags & 0x04),
                "rhythm": (flags & 0x38) >> 3, "fixed_note": bool(flags & 0x40),
                "fbc1": fbc1, "fbc2": fbc2, "ops": ops}

    need = pos + (mbanks + pbanks) * 128 * ins_size
    if len(data) < need:
        raise ValueError(f"truncated WOPL bank ({len(data)} of {need} bytes)")
    out = {"melodic": [], "percussion": [], "version": version,
           "name": os.path.basename(path)}
    for _ in range(mbanks):
        for i in range(128):
            out["melodic"].append(one(pos)); pos += ins_size
    for _ in range(pbanks):
        for i in range(128):
            out["percussion"].append(one(pos)); pos += ins_size
    return out


WOPL_BANK = None


def _bank_label():
    w = load_wopl_bank()
    if w:
        return f"DMXOPL ({len(w['melodic'])} instr)"
    return "Creative PLAY.EXE" if load_gm_bank() else "built-in (no GM bank)"


def load_wopl_bank(path=None):
    """Load the WOPL bank if present, else None. Cached."""
    global WOPL_BANK
    if WOPL_BANK is not None:
        return WOPL_BANK or None
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in ([path] if path else []) + [
            os.path.join(here, "vendor", "GENMIDI.wopl")]:
        if cand and os.path.exists(cand):
            try:
                WOPL_BANK = load_wopl(cand)
                return WOPL_BANK
            except Exception:
                pass
    WOPL_BANK = {}
    return None


# Carrier Total Level compression.
#
# DMXOPL spreads 24 dB across its patches: a church organ sits at TL=0 (absolute
# maximum) while SynthStrings 1 sits at TL=32. That is deliberate voicing, and it
# is meant to be evened out by MIDI channel volume — but a great many game MIDIs
# have no dynamics at all. Zelda's Boss Battle is velocity 110 on every note of
# every channel, 0 dB of spread, so nothing in the file corrects for it and one
# instrument blares while another is inaudible.
#
# Compressing toward the bank's median narrows the spread while preserving the
# ORDER (an organ still reads louder than strings). Only the CARRIER is touched:
# the modulator's TL sets FM index, so changing that would alter timbre rather
# than level — the classic mistake when adding level control to FM.
TL_MEDIAN = 5           # measured median carrier TL across DMXOPL's 128 patches
TL_SQUEEZE = 0.5        # 1.0 = bank as-is, 0.0 = every patch identical


def _balance_tl(tl):
    out = TL_MEDIAN + (tl - TL_MEDIAN) * TL_SQUEEZE
    # Never fully open: TL=0 is the chip's absolute maximum and stacking 15 voices
    # there is what clips the mix.
    return max(3, min(63, int(round(out))))


def program_wopl(chip, ch, ins):
    """Apply a WOPL instrument to a 2-operator channel.

    Only the first voice is used: a 4-op or pseudo-4-op instrument needs two
    channels, and with four voices total we cannot spend two on one note. Taking
    voice 1 is the documented degradation path and is what 2-op hardware did.
    """
    m, c = _op_pair(ch)
    car, mod = ins["ops"][0], ins["ops"][1]      # op1=CARRIER, op2=MODULATOR
    for off, o in ((m, mod), (c, car)):
        chip.write(0x20 + off, o[0])
        lvl = o[1]
        if off == c:
            lvl = (lvl & 0xC0) | _balance_tl(lvl & 0x3F)
        chip.write(0x40 + off, lvl)
        chip.write(0x60 + off, o[2])
        chip.write(0x80 + off, o[3])
        chip.write(0xE0 + off, o[4])
    chip.write(_ch_reg(ch, 0xC0), 0x30 | (ins["fbc1"] & 0x0F))


# =============================================================================
# POLYPHONIC MIDI RENDERING — "make it sing"
#
# The 4-voice renderer above throws away most of both the chip and the music:
#
#   the chip   OPL3 has 18 two-op channels, or 6 four-op plus 6 two-op. Four were
#              being used, all in bank 0.
#   the bank   102 of DMXOPL's 128 instruments are 4-operator, and only voice 1
#              was applied — literally half of each patch.
#   the music  a MIDI file's polyphony was collapsed to highest/lowest/middle,
#              so chords became three notes and inner parts vanished.
#
# All three compound into "thin". This path fixes all three: real 4-op voices for
# melodic instruments, every available channel, and a voice allocator so the
# arrangement plays as written.
# =============================================================================

def program_wopl_4op(chip, ch1, ch2, ins):
    """Program a fused 4-operator voice across a channel pair.

    Operator roles per the WOPL spec, in FILE order:
        ops[0] = Carrier1   -> in 4-op terms, operator 2
        ops[1] = Modulator1 -> operator 1
        ops[2] = Carrier2   -> operator 4
        ops[3] = Modulator2 -> operator 3
    So the primary channel carries (Modulator1, Carrier1) and the secondary
    carries (Modulator2, Carrier2) — the same mod/car pairing as 2-op, applied
    twice. fbc1 goes to the primary and fbc2 to the secondary, and the pair's
    connection bits decide which of the four FM algorithms results.
    """
    m1, c1 = _op_pair(ch1)
    m2, c2 = _op_pair(ch2)
    for off, o in ((m1, ins["ops"][1]), (c1, ins["ops"][0]),
                   (m2, ins["ops"][3]), (c2, ins["ops"][2])):
        chip.write(0x20 + off, o[0])
        lvl = o[1]
        if off in (c1, c2):
            lvl = (lvl & 0xC0) | _balance_tl(lvl & 0x3F)
        chip.write(0x40 + off, lvl)
        chip.write(0x60 + off, o[2])
        chip.write(0x80 + off, o[3])
        chip.write(0xE0 + off, o[4])
    chip.write(_ch_reg(ch1, 0xC0), 0x30 | (ins["fbc1"] & 0x0F))
    chip.write(_ch_reg(ch2, 0xC0), 0x30 | (ins["fbc2"] & 0x0F))


def _carrier_regs(chans, ins):
    """Register addresses of the operators that reach the output.

    Only the CARRIERS should be attenuated for velocity. Turning down a modulator
    changes the timbre — less FM index, so a duller sound — rather than the
    volume, which is the classic mistake when adding dynamics to FM.
    """
    if len(chans) == 2 and ins and (ins.get("four_op") or ins.get("pseudo4")):
        # 4-op: ops[0] and ops[2] are Carrier1 and Carrier2.
        return [_op_pair(chans[0])[1], _op_pair(chans[1])[1]]
    return [_op_pair(chans[0])[1]]


def _apply_velocity(chip, chans, ins, base_tl, vel, trim=0):
    """Attenuate the carriers for MIDI velocity.

    Total Level is 6 bits where 0 is loudest and each step is 0.75 dB, so
    velocity maps to an ADDITIVE offset on the patch's own TL — preserving the
    instrument's designed balance instead of overwriting it.
    """
    # Cap the attenuation. A patch already 24 dB down in the bank, hit by a low
    # velocity on a channel with low CC7, would otherwise disappear entirely —
    # which is worse than being slightly too loud, because a missing inner voice
    # reads as a broken arrangement rather than a quiet one.
    add = min(24, int(round((127 - max(1, min(127, vel or 127))) * 0.28)))
    add += trim                              # per-patch level calibration
    for i, reg in enumerate(_carrier_regs(chans, ins)):
        tl = base_tl[i] if i < len(base_tl) else 0
        ksl = tl & 0xC0
        chip.write(0x40 + reg, ksl | min(63, (tl & 0x3F) + add))


def program_wopl_voice2(chip, ch, ins):
    """Program the SECOND voice of a pseudo-4-operator patch.

    Pseudo-4-op is not a fused 4-operator voice — it is two independent 2-op
    voices played together, slightly detuned, and the beating between them is the
    effect. ops[2]/ops[3] are Carrier2/Modulator2, and fbc2 is their
    feedback/connection.
    """
    m, c = _op_pair(ch)
    car, mod = ins["ops"][2], ins["ops"][3]
    for off, o in ((m, mod), (c, car)):
        chip.write(0x20 + off, o[0])
        lvl = o[1]
        if off == c:
            lvl = (lvl & 0xC0) | _balance_tl(lvl & 0x3F)
        chip.write(0x40 + off, lvl)
        chip.write(0x60 + off, o[2])
        chip.write(0x80 + off, o[3])
        chip.write(0xE0 + off, o[4])
    chip.write(_ch_reg(ch, 0xC0), 0x30 | (ins["fbc2"] & 0x0F))


class Allocator:
    """Assigns notes to OPL channels. All 18 channels, all in 2-operator mode.

    WHY NO 4-OPERATOR FUSION. Measured against DMXOPL: it contains ZERO real
    4-operator patches. All 102 of its "4-op" instruments are pseudo-4-op — two
    independent detuned 2-op voices — and 26 are plain 2-op.

    Fusing channel pairs was therefore wrong twice over: it broke the 26 plain
    2-op patches outright (a 2-op patch on a fused channel is silent or nearly
    so — Tenor Sax measured -93 dB), and it mangled the 102 pseudo-4-op ones by
    routing two separate voices through one fused algorithm (-52 dB against
    -28 dB done properly). That is the whole reason one instrument blared while
    another was inaudible.

    Unfused, all 18 channels are usable, which is also more polyphony than
    6 fused + 6 spare.

    Voice stealing takes the OLDEST sounding note: the newest is the one the
    listener is waiting for, the oldest is usually already decaying.
    """

    def __init__(self, chip, wopl, use_four_op=False, reserve_drums=False):
        self.chip = chip
        self.wopl = wopl
        self.chip.set_four_op(0x00)          # never fuse; see the class docstring
        pool = list(range(18))
        if reserve_drums:                     # rhythm mode owns 6,7,8
            pool = [c for c in pool if c not in (6, 7, 8)]
        self.free = pool
        self.busy = {}                        # key -> (age, pitch, channels)
        self.progs = {}
        self.age = 0

    def _take(self, n):
        """Grab n channels, stealing the oldest notes if necessary."""
        got = []
        while len(got) < n:
            if self.free:
                got.append(self.free.pop(0))
                continue
            if not self.busy:
                return None
            key = min(self.busy, key=lambda k: self.busy[k][0])
            _age, _p, chans = self.busy.pop(key)
            for ch in chans:
                self.chip.key_off(ch)
                self.free.append(ch)
        return got

    def note_on(self, pitch, prog, hz, vel=None, nid=None, voice=None):
        """`voice`: a hand-written Instrument that overrides the GM program —
        --opl-lead/--opl-arp/--opl-bass, an .sbi, or the no-bank fallback."""
        if voice is None and not have_gm_bank():
            voice = fallback_voice(prog)
        if voice is not None:
            chans = self._take(1)
            if not chans:
                return None
            program_instrument(self.chip, chans[0], voice, vel)
            self.chip.key_on(chans[0], hz)
            self.age += 1
            key = nid if nid is not None else ("p", pitch, self.age)
            self.busy[key] = (self.age, pitch, chans)
            return chans
        ins = None
        if self.wopl and 0 <= prog < len(self.wopl["melodic"]):
            ins = self.wopl["melodic"][prog]
        two_voice = bool(ins and ins.get("pseudo4"))
        chans = self._take(2 if two_voice else 1)
        if not chans:
            return None

        trim = calibration_trim(prog)
        if ins:
            program_wopl(self.chip, chans[0], ins)
            _apply_velocity(self.chip, (chans[0],), ins,
                            [ins["ops"][0][1]], vel, trim)
            self.chip.key_on(chans[0], hz * (2.0 ** (ins["key_off"] / 12.0)))
            if two_voice:
                program_wopl_voice2(self.chip, chans[1], ins)
                _apply_velocity(self.chip, (chans[1],), ins,
                                [ins["ops"][2][1]], vel, trim)
                # Detune the second voice slightly: the beating between the pair
                # IS the pseudo-4-op effect. `detune` is a fine offset, key_off2
                # a coarse one in semitones.
                cents = (ins.get("detune") or 0) / 64.0
                hz2 = hz * (2.0 ** ((ins.get("key_off2", 0) + cents / 100.0) / 12.0))
                self.chip.key_on(chans[1], hz2)
        else:
            _apply_program(self.chip, chans[0], prog)
            self.chip.key_on(chans[0], hz)

        self.age += 1
        key = nid if nid is not None else ("p", pitch, self.age)
        self.busy[key] = (self.age, pitch, chans)
        return chans

    def perc_on(self, gm_note):
        """Strike a GM percussion instrument. One-shot: keyed on and left to
        decay, because a drum's envelope IS its length."""
        if not (self.wopl and self.wopl.get("percussion")):
            return
        if not (0 <= gm_note < len(self.wopl["percussion"])):
            return
        ins = self.wopl["percussion"][gm_note]
        if ins.get("blank"):
            return
        chans = self._take(1)
        if not chans:
            return
        program_wopl(self.chip, chans[0], ins)
        key = ins["perc_key"] if ins["perc_key"] else gm_note
        self.age += 1
        self.busy[("perc", gm_note, self.age)] = (self.age, -gm_note, chans)
        self.chip.key_on(chans[0], 440.0 * (2.0 ** ((key - 69) / 12.0)))

    def note_off(self, key):
        """End a specific note by its id, not by pitch."""
        ent = self.busy.pop(key, None)
        if ent is None:
            return
        for ch in ent[2]:
            self.chip.key_off(ch)
            self.free.append(ch)

    def all_off(self):
        for key in list(self.busy):
            for ch in self.busy[key][2]:
                self.chip.key_off(ch)
        self.busy.clear()


def render_poly(a, poly, drums, bars, spb, np, steps, wopl, with_drums=True,
                drums_gm=None):
    """Render a polyphonic note list. `poly` is (bar, step, dur, pitch, prog).

    When the bank has a percussion set and `drums_gm` is supplied, drums are
    played as REAL GM kit instruments on ordinary channels rather than through
    OPL3 rhythm mode. Rhythm mode gives five fixed sounds; the bank has 128
    indexed by GM note, so toms, congas, rides and crashes survive instead of
    collapsing into kick/snare/hat. It also frees channels 6/7/8 back into the
    melodic pool, since rhythm mode is no longer occupying them.
    """
    sr = SAMPLE_RATE
    step_s = spb / float(steps)
    total = bars * steps
    chip = OPL3(getattr(a, "opl_lib", None))
    use_kit = bool(with_drums and drums_gm and wopl and wopl.get("percussion"))
    alloc = Allocator(chip, wopl, use_four_op=True,
                      reserve_drums=bool(with_drums and not use_kit))

    if with_drums and not use_kit:
        _rhythm_setup(chip)

    # Every note gets a unique id. Matching note_off by PITCH was the last
    # source of holes: 47% of consecutive same-pitch pairs overlap after
    # quantization (usually two tracks doubling a line), and the first note's
    # off-event then silenced the SECOND note. With ids, doubled notes simply
    # occupy two voices and each ends when it should — no clamping, so the source
    # keeps its own articulation instead of being forced legato.
    ons, offs = {}, {}
    for nid, item in enumerate(poly):
        bar, st, dur, pitch, prog = item[:5]
        vel = item[5] if len(item) > 5 else None
        s = bar * steps + st
        if s >= total:
            continue
        ons.setdefault(s, []).append((nid, pitch, prog, vel))
        offs.setdefault(min(total, s + max(1, dur)), []).append(nid)
    dmap = {}
    for bar, st, kind in (drums or []):
        dmap.setdefault(bar * steps + st, set()).add(kind)
    kmap = {}
    if use_kit:
        for bar, st, note in drums_gm:
            kmap.setdefault(bar * steps + st, []).append(note)

    hz = lambda n: 440.0 * (2.0 ** ((n - 69) / 12.0))
    RHY = {"k": 0x10, "s": 0x08, "h": 0x01}
    out = []
    for s in range(total):
        for nid in offs.get(s, ()):
            alloc.note_off(nid)
        for nid, pitch, prog, vel in ons.get(s, ()):
            alloc.note_on(pitch, prog, hz(pitch), vel, nid)
        if use_kit and s in kmap:
            for note in kmap[s][:3]:            # cap: a crash + a kick, not 9 toms
                alloc.perc_on(note)
        elif with_drums and s in dmap:
            bits = 0
            for k in dmap[s]:
                bits |= RHY.get(k, 0)
            chip.write(0xBD, 0x20)
            chip.write(0xBD, 0x20 | bits)
        n = int(round((s + 1) * step_s * sr)) - int(round(s * step_s * sr))
        if n > 0:
            out.append(chip.render(n, np))
    alloc.all_off()
    audio = np.concatenate(out) if out else np.zeros(1)
    peak = float(np.abs(audio).max())
    if peak > 1e-9:
        audio = audio * (10.0 ** (a.normalize_db / 20.0) / peak)
    return audio


# =============================================================================
# PATCH LEVEL CALIBRATION
#
# The complaint that drove this: in a rendered MIDI one instrument blared while
# another was nearly inaudible.
#
# Three candidate causes were measured before fixing anything:
#   patch TL spread   the bank spans 24 dB, but real MELODIC patches sit within
#                     ~5 dB of each other — not the cause
#   4-op vs 2-op      mean difference +0.1 dB — not the cause in general, BUT
#                     individual patches differ wildly (Trumpet is 11.3 dB
#                     quieter in a 4-op slot than a 2-op one)
#   voice stealing    this file needed 6 voices of 12 — not the cause here
#
# So the spread is per-patch and irregular, which no single formula fixes. The
# reliable answer is to MEASURE it: render every patch through the actual chip,
# note its output level, and apply a compensating attenuation so they all land at
# a common reference. Empirical, cached, and verifiable without listening.
#
# Only attenuation is applied, never boost: TL 0 is the chip's maximum and there
# is no headroom above it.
# =============================================================================
CAL_REF_DB = -30.0          # target level; near the bank's own median
CAL_MAX_TRIM = 18           # TL steps, ~13 dB. Beyond this a patch is broken,
                            # not quiet, and dragging it up only adds noise.
_CAL = None


def calibrate_bank(np, force=False, path=None):
    """Measure each melodic patch's output level; return a list of TL offsets."""
    global _CAL
    if _CAL is not None and not force:
        return _CAL
    import json
    here = os.path.dirname(os.path.abspath(__file__))
    cache = path or os.path.join(here, "vendor", "opl-calibration.json")
    w = load_wopl_bank()
    if not w:
        _CAL = []
        return _CAL
    if os.path.exists(cache) and not force:
        try:
            with open(cache) as fh:
                data = json.load(fh)
            if len(data.get("trim", [])) == len(w["melodic"]):
                _CAL = data["trim"]
                return _CAL
        except Exception:
            pass

    trim = []
    for ins in w["melodic"]:
        chip = OPL3()
        # Measure exactly what the Allocator will actually play, including the
        # second voice of a pseudo-4-op patch. Calibrating a rendering path that
        # differs from the playback path is worse than not calibrating: it
        # confidently applies the wrong correction. That is what happened first
        # time round — it measured fused 4-op and reported Tenor Sax at -93 dB.
        chip.set_four_op(0x00)
        program_wopl(chip, 0, ins)
        chip.key_on(0, 261.6)                       # middle C, a fair reference
        if ins.get("pseudo4"):
            program_wopl_voice2(chip, 1, ins)
            chip.key_on(1, 261.6)
        a = chip.render(int(0.4 * SAMPLE_RATE), np)
        rms = float(np.sqrt(np.mean(a ** 2))) if len(a) else 0.0
        db = 20.0 * (np.log10(rms) if rms > 1e-9 else -9)
        # Positive trim = attenuate. Louder than reference -> pull down.
        steps = int(round((db - CAL_REF_DB) / 0.75)) if rms > 1e-9 else 0
        trim.append(max(0, min(CAL_MAX_TRIM, steps)))
    _CAL = trim
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as fh:
            json.dump({"ref_db": CAL_REF_DB, "trim": trim}, fh)
    except Exception:
        pass
    return _CAL


def calibration_trim(prog):
    return _CAL[prog] if (_CAL and 0 <= prog < len(_CAL)) else 0


def _rhythm_setup(chip):
    """OPL rhythm mode: the chip's own BD/SD/HH on channels 6-8, which is what
    AdLib games used for drums when they had no percussion bank."""
    chip.write(0xBD, 0x20)
    for ch, fq in ((6, 90.0), (7, 240.0), (8, 640.0)):
        fn, bl = fnum_block(fq)
        chip.write(_ch_reg(ch, 0xA0), fn & 0xFF)
        chip.write(_ch_reg(ch, 0xB0), ((bl & 7) << 2) | ((fn >> 8) & 3))
    # The six drum operators are REGISTER offsets 0x10-0x15 (channels 6-8's
    # modulators, then carriers: BD mod, HH, TT, BD car, SD, CY). They were
    # written as 12-17 decimal — 0x0C-0x11 — which programmed two channel-3/4
    # operators and left kick, snare, tom and cymbal on reset values: only the
    # hi-hat (0x11) ever sounded, kick and snare measured -72 dB.
    for op in (0x10, 0x11, 0x12, 0x13, 0x14, 0x15):
        chip.write(0x20 + op, 0x01); chip.write(0x40 + op, 0x00)
        chip.write(0x60 + op, 0xF8); chip.write(0x80 + op, 0xF8)
    # Both speakers on for the three drum channels. In OPL3 mode a channel with
    # neither output bit set is silent — kick and snare were, measured -72 dB.
    for ch in (6, 7, 8):
        chip.write(_ch_reg(ch, 0xC0), 0x30)


_RHY_BIT = {"k": 0x10, "s": 0x08, "h": 0x01}


def render_timed(a, notes, drums, duration, np, wopl, with_drums=True,
                 voices=None, loop=False):
    """Event-driven rendering at EXACT times — no grid, no quantization.

    render_poly() steps a fixed 16th grid, which forces every onset and length to
    the nearest step. Here the schedule is built from real note times and the chip
    is advanced only as far as the next event, so timing is sample-accurate and
    expressive gate times survive.

    `voices`: {part: Instrument} for notes that carry a "role" (composed music) —
    what --opl-lead/--opl-arp/--opl-bass and an .sbi bank choose. Parts without
    an entry play their GM program from the installed bank; with NO bank
    installed every note falls back to the hand-written INSTRUMENTS and the
    drums to OPL rhythm mode, so --opl makes music without the download.

    `loop`: whole bars that repeat — the file is exactly `duration` long and the
    final release is folded onto the head (chip.fold_tail) instead of appended.
    """
    sr = SAMPLE_RATE
    voices = voices or {}
    chip = OPL3(getattr(a, "opl_lib", None))
    use_kit = bool(with_drums and drums and wopl and wopl.get("percussion"))
    use_rhythm = bool(with_drums and drums and not use_kit)
    alloc = Allocator(chip, wopl, reserve_drums=use_rhythm)
    if use_rhythm:
        _rhythm_setup(chip)

    # Chip idiom, if asked: chords become fast arpeggios and leaps get slides.
    # Period-correct for AdLib too — DOS drivers arpeggiated constantly even
    # though an OPL3 can hold a chord.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import chip as _chipmod
    level = getattr(a, "chippy", "off") or "off"
    if level != "off":
        notes = _chipmod.chippify(notes, level)

    ev = []
    for n in notes:
        ev.append((n["t"], 0, n))                       # 0 = note on
        ev.append((n["t"] + n["dur"], 1, n))            # 1 = note off
    if use_kit:
        for t, gm in drums:
            ev.append((t, 2, gm))
    elif use_rhythm:
        for t, gm in drums:
            ev.append((t, 3, _RHY_BIT[_chipmod.GM_DRUM_KIND.get(gm, "h")]))
            # Release the trigger shortly after, so the next hit is a new edge.
            ev.append((t + 0.005, -1, None))
    ev.sort(key=lambda e: (e[0], e[1]))

    hz = lambda p: 440.0 * (2.0 ** ((p - 69) / 12.0))
    out = []
    now = 0.0
    done = 0                                            # samples rendered so far
    rhy = 0
    end_t = duration if loop else None
    for t, kind, payload in ev:
        if end_t is not None and t >= end_t and kind in (0, 2, 3):
            continue                                    # a loop has no notes past its end
        if t > now:
            nsamp = int(round(t * sr)) - done
            if nsamp > 0:
                out.append(chip.render(nsamp, np))
                done += nsamp
                now = t
        if kind == 0:
            alloc.note_on(payload["pitch"], payload["prog"],
                          hz(payload["pitch"]), payload["vel"], payload["id"],
                          voice=voices.get(payload.get("role")))
        elif kind == 1:
            alloc.note_off(payload["id"])
        elif kind == 2:
            alloc.perc_on(payload)
        elif kind == 3:
            rhy |= payload
            chip.write(0xBD, 0x20 | rhy)
        else:
            rhy = 0
            chip.write(0xBD, 0x20)
    n_body = int(round(duration * sr))
    if n_body > done:
        out.append(chip.render(n_body - done, np))
        done = n_body
    alloc.all_off()
    chip.write(0xBD, 0x20)
    # Let the final release ring out rather than truncating it: appended for an
    # excerpt, folded onto the head for a loop.
    out.append(chip.render(int(0.35 * sr), np))

    audio = np.concatenate(out) if out else np.zeros(1)
    if loop and n_body > 0:
        audio = _chipmod.fold_tail(audio, n_body)
        if level != "off":
            cfg = _chipmod.CHIPPY.get(level, _chipmod.CHIPPY["off"])
            audio = _chipmod._chip_verb(audio, sr, cfg["verb"], np, circular=True)
        audio = _chipmod._dc_block(audio, np, sr, prime=audio[-min(len(audio), sr):])
    else:
        audio = _chipmod._dc_block(audio, np, sr)
        if level != "off":
            cfg = _chipmod.CHIPPY.get(level, _chipmod.CHIPPY["off"])
            audio = _chipmod._chip_verb(audio, sr, cfg["verb"], np)
        # An excerpt ends wherever --seconds cut it, often mid-release: land on
        # zero instead of clicking.
        f = min(len(audio), int(0.01 * sr))
        if f > 1:
            audio[-f:] *= np.linspace(1.0, 0.0, f)
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    if peak > 1e-9:
        audio = audio * (10.0 ** (a.normalize_db / 20.0) / peak)
    return audio


def opl_slide(chip, ch, hz0, hz1, ms, np, sr, render_cb, steps=8):
    """Glide a channel's pitch by rewriting its F-number in small steps.

    An OPL has no portamento; the era's drivers faked it exactly this way, by
    rewriting the frequency registers a few dozen times a second. Eight steps is
    enough to read as a glide rather than a chromatic run.
    """
    per = max(1, int(sr * (ms / 1000.0) / steps))
    for i in range(steps):
        f = hz0 + (hz1 - hz0) * ((i + 1) / steps)
        fn, bl = fnum_block(f)
        chip.write(_ch_reg(ch, 0xA0), fn & 0xFF)
        chip.write(_ch_reg(ch, 0xB0), 0x20 | ((bl & 7) << 2) | ((fn >> 8) & 3))
        render_cb(per)
