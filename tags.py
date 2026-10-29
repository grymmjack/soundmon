"""What a sound is, from its own words: the file name, the folders it sits in, and what the file says
about itself (WAV/AIFF chunks: acid, smpl, LIST/INFO, bext, iXML, Apple Loops; tags in other formats).

Nothing here knows any one library. It is the vocabulary sample packs use everywhere ("kick", "BD",
"hihat", "riser"...), including the ways they write it: glued together ("DryKick", "noisesnare"),
without vowels ("kcks", "snrs"), abbreviated ("SD", "OH").

    tag      one of TAGS (kick, snare, pad, riser...), with how sure the words are (0-1)
    kind     loop or one-shot: an acid/Apple Loops flag, then the words near it ("loop", "one shot",
             "120 bpm"), then a sampler's loop points (over the whole file of 2 s or more: a loop; part way
             in: a sustained note, one-shot), then a length that fits whole bars at the tempo, then a short
             length (under 1.2 s: a one-shot, whatever the ear says)
    bpm, key from the words or the file's own chunks ("Fmin", "C# minor", acid tempo)

listen.py tags the rest by ear (CLAP) and uses these tags to learn each library's own sounds.
Pure Python; no numpy.
"""
import json
import os
import re
import struct
import subprocess

VERSION = 4             # bump when the rules change: the index re-tags from what it stored (no file reads)
EMB_VERSION = 2         # bump when embedded() reads more: the index reads those files' headers again

# (id, label, family, whole words, words also found inside a glued token, phrases the ear model hears)
TAGS = [
    ("kick", "Kick", "drums",
     "kick kicks kik kck kcks bd bdrum bassdrum bassdrums kickdrum kickdrums",
     "kick bassdrum",
     ["a kick drum hit", "a bass drum one-shot", "an electronic kick drum"]),
    ("snare", "Snare", "drums",
     "snare snares snr snrs sd rim rims rimshot rimshots sidestick crossstick",
     "snare rimshot",
     ["a snare drum hit", "a snare drum one-shot", "a rimshot"]),
    ("clap", "Clap & snap", "drums",
     "clap claps clp cp handclap handclaps snap snaps fingersnap fingersnaps",
     "clap fingersnap",
     ["a hand clap", "an electronic clap", "a finger snap"]),
    ("hat", "Hi-hat", "drums",
     "hat hats hihat hihats hh hhs hhat oh ch ohh chh hho hhc hhp hhr openhat closedhat pedalhat",
     "hihat hatclos hatopen openhat closedhat",
     ["a hi-hat hit", "a closed hi-hat", "an open hi-hat"]),
    ("cymbal", "Cymbal", "drums",
     "cymbal cymbals cym cyms crash crashes ride rides splash splashes china cy rd",
     "cymbal",
     ["a crash cymbal", "a ride cymbal", "a cymbal hit ringing out"]),
    ("tom", "Tom", "drums",
     "tom toms floortom floortoms lowtom midtom hitom hightom tomtom tomtoms lt mt ht",
     "floortom",
     ["a tom drum hit", "a floor tom", "a tuned tom-tom"]),
    ("handdrum", "Hand drum", "drums",
     "conga congas bongo bongos djembe djembes timbale timbales tabla tablas cajon darbuka dumbek doumbek "
     "quinto tumba udu taiko surdo frame",
     "conga bongo djembe tabla darbuka timbale",
     ["a conga hit", "a bongo drum", "a hand drum"]),
    ("shaker", "Shaker & tambourine", "drums",
     "shaker shakers shk shake tamb tambourine tambourines tambo cabasa maraca maracas guiro vibraslap ratchet",
     "shaker tambourine maraca",
     ["a shaker", "a tambourine", "a rattle"]),
    ("hit", "Wood & metal hit", "drums",
     "woodblock woodblocks block blocks clave claves cowbell cowbells cow cb agogo anvil gong gongs stick sticks "
     "knock knocks clank",
     "woodblock cowbell",
     ["a wood block hit", "a cowbell", "a metallic hit"]),
    ("perc", "Percussion", "drums",
     "perc percs percussion percussive",
     "perc",
     ["a percussion hit", "a short percussive sound"]),
    ("drums", "Drums", "drums",
     "drum drums drumloop drumloops beat beats break breaks breakbeat breakbeats groove grooves drumkit fill fills "
     "top tops",
     "drumloop breakbeat",
     ["a drum beat", "a drum loop with kick snare and hi-hats", "a breakbeat"]),

    ("bass", "Bass", "tonal",
     "bass basses bassline basslines sub subbass reese wobble",
     "bass",
     ["a bass synth note", "a deep sub bass", "an electric bass guitar note"]),
    ("synth", "Synth & lead", "tonal",
     "lead leads synth synths synthesizer saw sine square osc waveform waveforms",
     "synth",
     ["a synth lead", "a synthesizer melody", "a single synth note"]),
    ("keys", "Keys", "tonal",
     "keys piano pianos epiano epianos rhodes wurli wurlitzer organ organs hammond clav clavinet harpsichord "
     "electricpiano electricpianos",
     "piano organ",
     ["a piano note", "an electric piano", "an organ chord"]),
    ("pad", "Pad & drone", "tonal",
     "pad pads texture textures drone drones",
     "drone",
     ["a soft synth pad", "a sustained ambient drone", "an evolving texture"]),
    ("pluck", "Pluck & arp", "tonal",
     "pluck plucks plucked arp arps arpeggio arpeggios arpeggiator arpeggiators",
     "pluck arpegg",
     ["a plucked synth note", "an arpeggio", "a short plucked string"]),
    ("stab", "Stab & chord", "tonal",
     "stab stabs chord chords orchhit",
     "stab chord",
     ["a synth chord stab", "an orchestra hit", "a short chord"]),
    ("orch", "Strings, brass & winds", "tonal",
     "string strings violin violins viola violas cello cellos contrabass orchestra orchestral ensemble pizz "
     "pizzicato harp harps brass horn horns trumpet trumpets trombone trombones tuba sax saxes saxophone "
     "flute flutes clarinet oboe bassoon woodwind woodwinds",
     "violin cello orchestr trumpet trombone saxophone flute",
     ["strings playing", "a brass section", "a flute"]),
    ("guitar", "Guitar", "tonal",
     "guitar guitars gtr gtrs banjo ukulele uke mandolin sitar strum strums strummed",
     "guitar",
     ["an electric guitar", "an acoustic guitar strum", "a plucked guitar note"]),
    ("bells", "Bells & mallets", "tonal",
     "bell bells chime chimes glock glockenspiel celesta vibes vibraphone mallet mallets marimba xylophone "
     "kalimba musicbox triangle triangles",
     "glock marimba kalimba xylophone",
     ["a bell", "a glockenspiel note", "a marimba"]),
    ("vocal", "Vocal", "tonal",
     "vox vocal vocals voice voices acapella acapellas choir choirs chant chants shout shouts spoken speech "
     "phrase phrases adlib adlibs sing singing breath breaths words speak hum",
     "vocal acapella choir",
     ["a human voice", "a sung vocal phrase", "a spoken word"]),

    ("riser", "Riser & sweep", "fx",
     "riser risers rise rising uplifter uplifters downlifter downlifters sweep sweeps swell swells buildup "
     "buildups",
     "riser sweep uplift downlift",
     ["a rising sweep", "a riser building up", "a filter sweep"]),
    ("whoosh", "Whoosh & transition", "fx",
     "whoosh whooshes swoosh swooshes woosh swish swishes whip whips passby flyby transition transitions",
     "whoosh swoosh",
     ["a whoosh", "something swishing past", "a fast transition swoosh"]),
    ("impact", "Impact & boom", "fx",
     "impact impacts boom booms slam slams thud thuds braam stomp stomps subdrop",
     "impact",
     ["a cinematic impact", "a deep boom", "a heavy slam"]),
    ("explosion", "Explosion & gun", "fx",
     "explosion explosions explode blast blasts bomb bombs gun guns gunshot gunshots weapon weapons rifle "
     "pistol missile ricochet",
     "explosion gunshot",
     ["an explosion", "a gunshot", "a blast"]),
    ("zap", "Zap, blip & glitch", "fx",
     "zap zaps blip blips bleep bleeps beep beeps laser lasers lazer lazers glitch glitches glitched computer "
     "robot robotic scifi ui button",
     "glitch laser lazer blip",
     ["a laser zap", "a computer blip", "a digital glitch"]),
    ("noise", "Noise", "fx",
     "noise noises static hiss distortion feedback",
     "noise",
     ["white noise", "static hiss", "harsh distorted noise"]),
    ("ambience", "Ambience & nature", "fx",
     "ambience ambiences ambient ambiance atmos atmo atmosphere atmospheres atmospheric nature rain wind water "
     "ocean sea river birds bird forest thunder storm fieldrecording city traffic crowd",
     "ambien atmos",
     ["an ambient background", "rain and wind outside", "birds in a forest"]),
    ("foley", "Foley", "fx",
     "foley footstep footsteps door doors paper cloth glass smash shatter debris rustle creak squeak drip drips "
     "vehicle vehicles car cars engine animal animals dog",
     "foley footstep",
     ["footsteps", "a door closing", "glass breaking"]),
    ("scratch", "Scratch", "fx",
     "scratch scratches scratching turntable",
     "scratch",
     ["a vinyl record scratch", "a DJ scratching a record"]),
]
IDS = [t[0] for t in TAGS]
LABEL = {t[0]: t[1] for t in TAGS}
FAMILY = {t[0]: t[2] for t in TAGS}
PROMPTS = {t[0]: t[5] for t in TAGS}
KIND_PROMPTS = {"loop": ["a music loop", "a drum loop", "a repeating rhythmic pattern"],
                "oneshot": ["a single one-shot sound", "a single drum hit", "one short sound"]}

WORD = {}
for _id, _l, _f, _words, _glued, _p in TAGS:
    for _w in _words.split():
        WORD.setdefault(_w, _id)
GLUED = sorted(((w, t[0]) for t in TAGS for w in t[4].split()), key=lambda x: -len(x[0]))
# two words in a row that say more than either alone
PAIRS = {("bass", "drum"): "kick", ("kick", "drum"): "kick", ("hi", "hat"): "hat", ("high", "hat"): "hat",
         ("open", "hat"): "hat", ("closed", "hat"): "hat", ("pedal", "hat"): "hat", ("ride", "bell"): "cymbal",
         ("finger", "snap"): "clap", ("side", "stick"): "snare", ("cross", "stick"): "snare",
         ("rim", "shot"): "snare", ("sub", "drop"): "impact", ("orch", "hit"): "stab", ("orchestra", "hit"): "stab",
         ("music", "box"): "bells", ("wood", "block"): "hit", ("field", "recording"): "ambience",
         ("white", "noise"): "noise", ("pink", "noise"): "noise", ("vinyl", "scratch"): "scratch",
         ("electric", "piano"): "keys", ("drum", "loop"): "drums", ("drum", "kit"): "drums",
         ("synth", "bass"): "bass", ("bass", "synth"): "bass", ("snare", "roll"): "snare"}
# abbreviations that mean something else often enough to count for less
WEAK = {"snap", "snaps", "oh", "ch", "cp", "sd", "bd", "lt", "mt", "ht", "rd", "cb", "cy", "hh", "rim", "rims", "top", "tops",
        "cow", "block", "blocks", "stick", "sticks", "frame", "sub", "saw", "sine", "square", "osc", "words",
        "hum", "break", "breaks", "fill", "fills", "rise", "water", "glass", "car", "cars", "dog", "ui",
        "button", "computer", "robot", "sing", "speak", "shake"}
# words that say a tag only weakly (the family more than the sound): beaten by any specific word
BROAD = {"perc", "drums"}
LOOPY = set("loop loops looped bpm beat beats groove grooves break breaks breakbeat breakbeats drumloop drumloops "
            "phrase phrases construction".split())
ONESHOTY = set("oneshot oneshots single hit hits shot shots".split())
NOTE = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
NOTES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# a token that is only noise in a path (format, take numbers, version marks)
NOISE = set("wav wave wavs aif aiff flac ogg mp3 the and of a in by for with to from at on samples sample "
            "sounds sound pack packs vol volume version ver v final new old copy part disk disc cd".split())


def tokens(s):
    """Lowercase words of s: split on anything not a letter or digit, camelCase, and letters next to digits."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])(?=\d)|(\d)(?=[A-Za-z])", lambda m: (m.group(1) or m.group(2)) + " ", s)
    return [t for t in re.split(r"[^a-z0-9#]+", s.lower()) if t]


def path_parts(path, root=None, depth=6):
    """[(text, weight)]: the file's name weighs most, then its folder, then each folder above for less.
    Folders above `root` (the indexed folder) are left out: they're where the user keeps things."""
    rel = os.path.relpath(path, root) if root else path
    parts = rel.replace("\\", "/").split("/")
    name = os.path.splitext(parts[-1])[0]
    out = [(name, 3.0)]
    w = 2.0
    for d in reversed(parts[:-1][-depth:]):
        if d in ("", ".", ".."):
            continue
        out.append((d, w))
        w = max(0.4, w * 0.5)
    return out


def _glued(tok):
    """the tag of a known word found inside a glued token, or None. The last word wins, as in English:
    "drykick" is a kick, "synthperc" a perc."""
    if len(tok) < 5 or tok in WORD:
        return None
    best = None
    for w, t in GLUED:
        i = tok.rfind(w)
        if i >= 0 and (best is None or i + len(w) > best[0]):
            best = (i + len(w), t)
    return best[1] if best else None


def _one(w):
    """a plural's singular, for two-word names: drums -> drum, hats -> hat (not "bass", "glass")"""
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def tag_of(parts):
    """(tag, sureness 0-1) from [(text, weight)], or (None, 0)."""
    score, broad = {}, {}
    for text, w in parts:
        t = [x for x in tokens(text) if x not in NOISE and not x.isdigit()]
        used = set()
        for i, (a, b) in enumerate(zip(t, t[1:])):
            hit = PAIRS.get((a, b)) or PAIRS.get((a, _one(b)))   # "Bass Drums", "Hi Hats", "Snare Rolls"
            if hit:
                dst = broad if hit in BROAD else score
                dst[hit] = dst.get(hit, 0) + 1.5 * w
                used.update((i, i + 1))
        for i, x in enumerate(t):
            if i in used:
                continue
            hit, k = WORD.get(x), 1.0
            if hit is None:
                hit, k = _glued(x), 0.7
            if hit is None:
                continue
            if x in WEAK:
                k = 0.5
            dst = broad if hit in BROAD else score
            dst[hit] = dst.get(hit, 0) + k * w
    for t, v in broad.items():                      # "Drums", "Percussion": they count, for less
        score[t] = score.get(t, 0) + 0.6 * v
    if not score:
        return None, 0.0
    best = sorted(score.items(), key=lambda kv: -kv[1])
    top, second = best[0][1], (best[1][1] if len(best) > 1 else 0.0)
    if top < 0.9 or top == second:                  # one weak word in a far folder, or a tie: can't say
        return None, 0.0
    sure = (top - second) / top * min(1.0, top / 3.0)
    if best[0][0] in BROAD:
        sure *= 0.6
    return best[0][0], round(sure, 3)


_BPM = re.compile(r"(?<![\d.])(\d{2,3}(?:[.,]\d+)?)\s*(?:bpm|_bpm)|bpm\s*[_ -]?(\d{2,3})(?!\d)", re.I)
_KEY = re.compile(r"(?<![A-Za-z0-9])([A-G])(#|b|sharp|flat)?[ _-]?"
                  r"(maj(?:or)?|min(?:or)?|m(?![a-z]))(?![a-z])", re.I)


def bpm_of(text):
    for m in _BPM.finditer(text):
        v = float((m.group(1) or m.group(2)).replace(",", "."))
        if 50 <= v <= 250:
            return v
    return None


def key_of(text):
    """'F#m' / 'C' style, from 'Fmin', 'F# minor', 'Gbmaj', 'C#m'. A bare 'Am' or 'Dm' doesn't count: too often
    it's something else (DM is a drum machine)."""
    for m in _KEY.finditer(text):
        root, acc, mode = m.group(1).upper(), (m.group(2) or "").lower(), m.group(3).lower()
        if mode == "m" and not acc:
            continue                                    # "Am", "Em", "Dm": too often not a key
        n = NOTE[root.lower()] + (1 if acc in ("#", "sharp") else -1 if acc in ("b", "flat") else 0)
        return NOTES[n % 12] + ("m" if mode.startswith("m") and not mode.startswith("maj") else "")
    return None


def kind_of(parts, emb, seconds):
    """('loop' | 'oneshot' | None, how it was decided: flag, words, bars, short)."""
    if emb and emb.get("loop") is not None:
        return ("loop" if emb["loop"] else "oneshot"), "flag"
    near = [x for text, w in parts[:2] for x in tokens(text)]
    joined = " ".join(text for text, _ in parts)
    bpm = bpm_of(joined)
    near_set = set(near)
    if near_set & LOOPY or bpm_of(" ".join(text for text, _ in parts[:2])):
        return "loop", "words"
    if near_set & ONESHOTY or "one shot" in " ".join(near) or ("one" in near_set and "shot" in near_set):
        return "oneshot", "words"
    smp = (emb or {}).get("sampler")
    if smp == "sustain":                               # a held note: a sampler plays it as one
        return "oneshot", "sampler"
    if smp == "whole" and (seconds is None or seconds >= 2.0):
        return "loop", "sampler"                       # looped end to end (drum hits often are too: not under 2 s)
    tempo = (emb or {}).get("tempo") or bpm
    if tempo and seconds:
        beats = seconds * tempo / 60
        if beats >= 3.8 and abs(beats / 4 - round(beats / 4)) < 0.06:
            return "loop", "bars"
    if seconds is not None and seconds < 1.2:          # under a bar even at 200 bpm
        return "oneshot", "short"
    return None, ""


def describe(path, root=None, emb=None, seconds=None):
    """Everything the words say: {tag, sure, kind, kind_by, bpm, key}."""
    parts = path_parts(path, root)
    texts = [t for t in (emb or {}).get("text", []) if t]
    tag, sure = tag_of(parts + [(t[:200], 1.0) for t in texts])
    kind, kind_by = kind_of(parts, emb, seconds)
    near = " ".join(t for t, _ in parts[:2])
    bpm = (emb or {}).get("tempo") if kind == "loop" else None
    bpm = bpm if bpm and 50 <= bpm <= 250 else bpm_of(near) or bpm_of(" ".join(texts))
    key = key_of(near) or key_of(" ".join(texts))
    if key is None and kind == "loop" and (emb or {}).get("root") is not None:
        key = NOTES[int(emb["root"]) % 12]
    return {"tag": tag, "sure": sure, "kind": kind, "kind_by": kind_by,
            "bpm": round(bpm, 2) if bpm else None, "key": key}


# ----------------------------------------------------------------------------------------------
# what a file says about itself
# ----------------------------------------------------------------------------------------------
def _txt(b):
    return re.sub(r"\s+", " ", b.split(b"\0", 1)[0].decode("latin-1", "replace")).strip()


_SPAM = ("www.", "http", ".com", "copyright", "(c)", "all rights")


def _keep(t):
    t = t.strip()
    return t if t and len(t) > 1 and not any(s in t.lower() for s in _SPAM) else None


def embedded(path):
    """{text: [...], loop: True/False/None, sampler, tempo, beats, root (MIDI note), v} from the file's own
    chunks (WAV/RF64/AIFF), else its tags (ffprobe). Reads headers only, never the audio.

    loop is a loop flag (acid, Apple Loops). sampler is a sampler's loop points (smpl, AIFF INST): "whole"
    (looped end to end: a loop), "sustain" (part way in: a note held while the key is down) or None."""
    out = {"text": [], "loop": None, "sampler": None, "tempo": None, "beats": None, "root": None,
           "v": EMB_VERSION}
    align = frames = sloop = None
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if len(head) < 12:
                return out
            riff = head[:4] in (b"RIFF", b"RF64") and head[8:12] == b"WAVE"
            aiff = head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC")
            if not (riff or aiff):
                return _probe(path, out)
            end = os.fstat(f.fileno()).st_size
            pos = 12
            for _ in range(64):
                if pos + 8 > end:
                    break
                f.seek(pos)
                h = f.read(8)
                if len(h) < 8:
                    break
                cid, n = h[:4], struct.unpack("<I" if riff else ">I", h[4:8])[0]
                big = cid in (b"data", b"SSND")
                body = b"" if big or n > 1 << 20 else f.read(n)
                if riff and cid == b"fmt " and len(body) >= 14:
                    align = struct.unpack("<H", body[12:14])[0] or None
                elif riff and cid == b"data" and n != 0xFFFFFFFF and align:
                    frames = n // align
                elif riff and cid == b"acid" and len(body) >= 24:
                    flags, root, _, _, beats, _, _, tempo = struct.unpack("<IHHfIHHf", body[:24])
                    out["loop"] = not (flags & 1)
                    out["root"] = root if flags & 2 and 0 <= root < 128 else out["root"]
                    out["beats"] = beats or None
                    out["tempo"] = round(tempo, 2) if tempo and 20 < tempo < 400 else None
                elif riff and cid == b"smpl" and len(body) >= 36:
                    unity, loops = struct.unpack("<I", body[12:16])[0], struct.unpack("<I", body[28:32])[0]
                    if loops and len(body) >= 60:
                        sloop = struct.unpack("<II", body[44:52])     # the first loop's start and end
                    if out["root"] is None and 0 < unity < 128:
                        out["root"] = unity
                elif riff and cid == b"LIST" and body[:4] == b"INFO":
                    i = 4
                    while i + 8 <= len(body):
                        k, m = body[i:i + 4], struct.unpack("<I", body[i + 4:i + 8])[0]
                        if k in (b"INAM", b"ICMT", b"IGNR", b"IKEY", b"ISBJ"):
                            t = _keep(_txt(body[i + 8:i + 8 + m]))
                            if t:
                                out["text"].append(t[:200])
                        i += 8 + m + (m & 1)
                elif riff and cid == b"bext" and body:
                    t = _keep(_txt(body[:256]))
                    if t:
                        out["text"].append(t)
                elif cid in (b"iXML", b"NAME", b"ANNO") and body:
                    t = re.sub(r"<[^>]+>", " ", body.decode("utf-8", "replace"))
                    t = _keep(re.sub(r"\s+", " ", t))
                    if t:
                        out["text"].append(t[:300])
                elif aiff and cid == b"INST" and len(body) >= 20:
                    if 0 < body[0] < 128:
                        out["root"] = body[0]
                    if struct.unpack(">H", body[8:10])[0]:
                        out["sampler"] = "sustain"
                elif aiff and cid == b"basc" and len(body) >= 24:  # Apple Loops: beats, loop or one-shot
                    out["beats"] = struct.unpack(">I", body[4:8])[0] or out["beats"]
                    if out["loop"] is None:
                        out["loop"] = struct.unpack(">I", body[16:20])[0] != 2
                elif aiff and cid == b"cate" and body:
                    t = _keep(" ".join(_txt(body[i:i + 50]) for i in range(4, min(len(body), 204), 50)))
                    if t:
                        out["text"].append(t)
                pos += 8 + n + (n & 1)
    except (OSError, struct.error):
        pass
    if sloop:                                          # data may come before or after smpl: decided here
        a, b = sloop
        whole = a <= max(64, (frames or 0) // 100) and (frames is None or b >= frames * 0.98 - 2)
        if not whole:
            out["sampler"] = "sustain"
        elif frames is not None:
            out["sampler"] = "whole"
    return out


def _probe(path, out):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json", path],
                           capture_output=True, text=True, timeout=30)
        tags = json.loads(r.stdout or "{}").get("format", {}).get("tags", {}) or {}
    except (ValueError, OSError, subprocess.SubprocessError):
        tags = {}
    for k, v in tags.items():
        k = k.lower()
        if k in ("title", "genre", "comment", "description", "keywords", "grouping"):
            t = _keep(str(v))
            if t:
                out["text"].append(t[:200])
        elif k in ("bpm", "tbpm"):
            try:
                out["tempo"] = float(v) if 20 < float(v) < 400 else None
            except ValueError:
                pass
    return out
