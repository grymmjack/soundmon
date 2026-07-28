"""soundmon --record — record the narration yourself, one line at a time.

The fourth engine, and the only one whose audio source is a person. It reads
the same `key | text` manifest --narrate does, shows you a line, records you
reading it, and lands the result in the same place under the same filename a
generated line would have used. The game cannot tell the two apart, which is
the whole point: a hand-voiced pack and a Kokoro pack are interchangeable, and
you can mix them line by line.

Like --narrate, this never touches ComfyUI — there is no diffusion here at all.
But unlike --narrate it does NOT skip the output tail: takes go through the
same trim -> fade -> peak-normalize -> loudness-cap -> ogg chain the generated
packs go through, because a pack whose files were mastered differently is a
pack you can hear switching.

CROSS-PLATFORM, and that is the hard part — see _capture_cmd() for the
per-OS backends. Recording is the one thing in soundmon that cannot be done
with a portable library-free subprocess call, because every OS exposes its
microphones through a different demuxer.
"""
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import time
import wave

IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"

RATE = 48000          # capture rate; 48k is the native rate of essentially every
                      # USB mic and webcam, so this resamples nothing on the way in
MAX_TAKE = 180.0      # hard stop, in seconds, so walking away from a hot mic
                      # cannot fill the disk with a recording of an empty room

# Poll interval for the live meter. pw-record flushes in ~0.25 s chunks (measured:
# 24576 bytes = 0.256 s of 48 kHz mono s16 per flush), so polling faster than this
# just re-reads the same tail and shows a frozen bar.
METER_HZ = 8.0


def _colors():
    """ANSI codes — same convention as soundmon.py, redefined rather than imported
    because soundmon.py runs as __main__ and importing it back re-executes it."""
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return {k: "" for k in ("b", "dim", "cyan", "grn", "yel", "mag", "red", "rst")}
    return {"b": "\033[1m", "dim": "\033[2m", "cyan": "\033[36m", "grn": "\033[32m",
            "yel": "\033[33m", "mag": "\033[35m", "red": "\033[31m", "rst": "\033[0m"}


C = _colors()


def _enable_vt():
    """Windows consoles need VIRTUAL_TERMINAL_PROCESSING switched on before they
    honor ANSI escapes. Windows Terminal does it already; conhost.exe does not,
    and without this the whole TUI renders as literal '\\033[2J' garbage."""
    if not IS_WIN:
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        # -11 = STD_OUTPUT_HANDLE, 0x0004 = ENABLE_VIRTUAL_TERMINAL_PROCESSING
        k.SetConsoleMode(k.GetStdHandle(-11), 7)
    except Exception:
        pass


# Block-drawing characters for the meter and envelope. Windows consoles still
# default to a legacy codepage in plenty of setups, and a UnicodeEncodeError
# mid-render would kill a recording session, so fall back to ASCII if the
# terminal cannot actually print these.
_BLOCKS = "▁▂▃▄▅▆▇█"
_ASCII = ".:-=+*#@"
try:
    (_BLOCKS + "●░▒▓").encode(sys.stdout.encoding or "utf-8")
    RAMP, DOT, EMPTY, FULL = _BLOCKS, "●", "░", "▓"
except (UnicodeEncodeError, LookupError):
    RAMP, DOT, EMPTY, FULL = _ASCII, "*", ".", "#"


# ---------------------------------------------------------------------------
# Keyboard: one keypress, no Enter, cross-platform
# ---------------------------------------------------------------------------

class Keyboard:
    """Single-keypress input with a timeout, as a context manager.

    The timeout is what makes the live meter possible: the record loop needs to
    redraw the VU bar several times a second AND notice the stop key the instant
    it is pressed, so it cannot block on input.

    POSIX uses cbreak rather than raw mode on purpose — raw mode swallows Ctrl-C,
    and a recording tool that cannot be interrupted is a recording tool that
    eventually has to be killed from another terminal.
    """

    def __enter__(self):
        self.fd = None
        if not IS_WIN and sys.stdin.isatty():
            import termios
            import tty
            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def get(self, timeout=None):
        """Return a normalized key name, or None if `timeout` elapsed first."""
        if IS_WIN:
            import msvcrt
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("\x00", "\xe0"):    # arrow/function key: eat the second byte
                        msvcrt.getwch()
                        continue
                    return self._norm(ch)
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                time.sleep(0.01)
        import select
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        return self._norm(sys.stdin.read(1)) if r else None

    @staticmethod
    def _norm(ch):
        if ch in ("\r", "\n"):
            return "ENTER"
        if ch == " ":
            return "SPACE"
        if ch == "\x1b":
            return "ESC"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch.lower()


# ---------------------------------------------------------------------------
# Capture backends — one per OS, because there is no portable microphone
# ---------------------------------------------------------------------------

def _have(cmd):
    return shutil.which(cmd) is not None


def _capture_cmd(device, rate, path):
    """Build the capture command for this platform. Returns (argv, stop_mode).

    `stop_mode` is how the process must be told to finish, and it is not
    cosmetic — stopping a recorder the wrong way leaves a WAV whose RIFF/data
    size fields were never backfilled, which some players read as a zero-length
    file:

      "stdin-q"  ffmpeg. Writing 'q' to its stdin is the documented clean
                 shutdown and is the ONLY one that works the same on Windows,
                 where there is no SIGINT to send to a child process.
      "signal"   pw-record / arecord. Both install a handler that finalizes the
                 header; verified on pw-record, whose size fields came back
                 correct after a signal-terminated 5.7 s capture.

    Platform notes:
      Windows  DirectShow. The official gyan.dev/BtbN builds ship it; the device
               string is the friendly name from --list-devices, verbatim, spaces
               and parentheses included (no shell quoting — argv form handles it).
      macOS    AVFoundation. Its -i syntax is "<video>:<audio>", so an
               audio-only capture is ":0" — the leading colon is required and
               omitting it silently opens a camera instead.
      Linux    pw-record first: it speaks to PipeWire natively, and this
               machine's ffmpeg build has NO pulse demuxer at all (its -devices
               list is alsa/oss/fbdev/v4l2/x11grab only), so the obvious
               `-f pulse` route is not available. ALSA and arecord are the
               fallbacks for non-PipeWire systems.
    """
    if IS_WIN:
        if not _have("ffmpeg"):
            sys.exit("--record on Windows needs ffmpeg on PATH (gyan.dev or BtbN build).")
        src = f"audio={device}" if device else "audio=default"
        return (["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "dshow",
                 "-i", src, "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le",
                 "-y", path], "stdin-q")

    if IS_MAC:
        if not _have("ffmpeg"):
            sys.exit("--record on macOS needs ffmpeg on PATH (brew install ffmpeg).")
        return (["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "avfoundation",
                 "-i", f":{device or '0'}", "-ac", "1", "-ar", str(rate),
                 "-c:a", "pcm_s16le", "-y", path], "stdin-q")

    if _have("pw-record"):
        cmd = ["pw-record", "--rate", str(rate), "--channels", "1"]
        if device:
            cmd += ["--target", device]
        return (cmd + [path], "signal")
    if _have("ffmpeg"):
        return (["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "alsa",
                 "-i", device or "default", "-ac", "1", "-ar", str(rate),
                 "-c:a", "pcm_s16le", "-y", path], "stdin-q")
    if _have("arecord"):
        return (["arecord", "-q", "-f", "S16_LE", "-r", str(rate), "-c", "1",
                 "-D", device or "default", path], "signal")
    sys.exit("--record needs pw-record, ffmpeg or arecord on PATH.")


def list_devices():
    """Print the microphones this platform can see, in the form --device wants."""
    c = C
    print(f"{c['b']}{c['cyan']}Recording devices{c['rst']}  "
          f"{c['dim']}(use with --device, --record){c['rst']}\n")
    found = []

    if IS_WIN:
        # dshow enumerates to stderr and always "fails" — the dummy input is
        # never opened. Non-zero exit here is expected, not an error.
        out = subprocess.run(["ffmpeg", "-hide_banner", "-list_devices", "true",
                              "-f", "dshow", "-i", "dummy"],
                             capture_output=True, text=True).stderr
        audio = False
        for line in out.splitlines():
            if "DirectShow audio devices" in line:
                audio = True
                continue
            if "DirectShow video devices" in line:
                audio = False
                continue
            m = re.search(r'"([^"]+)"', line)
            if audio and m and "Alternative name" not in line:
                found.append((m.group(1), ""))
    elif IS_MAC:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation",
                              "-list_devices", "true", "-i", ""],
                             capture_output=True, text=True).stderr
        audio = False
        for line in out.splitlines():
            if "AVFoundation audio devices" in line:
                audio = True
                continue
            if "AVFoundation video devices" in line:
                audio = False
                continue
            m = re.search(r"\[(\d+)\]\s+(.+)$", line)
            if audio and m:
                found.append((m.group(1), m.group(2).strip()))
    else:
        if _have("pactl"):
            out = subprocess.run(["pactl", "list", "short", "sources"],
                                 capture_output=True, text=True).stdout
            for line in out.splitlines():
                cols = line.split("\t")
                # .monitor sources are loopbacks of an OUTPUT — recording one
                # captures what the speakers are playing, not what you say.
                if len(cols) >= 2 and not cols[1].endswith(".monitor"):
                    found.append((cols[1], "input"))
        if not found and _have("arecord"):
            out = subprocess.run(["arecord", "-L"], capture_output=True, text=True).stdout
            found = [(ln, "") for ln in out.splitlines() if ln and not ln[0].isspace()]

    if not found:
        print(f"  {c['dim']}none found — is a microphone connected?{c['rst']}")
        return
    for name, note in found:
        print(f"  {c['grn']}{name}{c['rst']}  {c['dim']}{note}{c['rst']}")
    print(f"\n  {c['dim']}omit --device to use the system default input.{c['rst']}")


def _play(path):
    """Play a file back. Blocks until it finishes."""
    if _have("ffplay"):
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path])
        return
    for cmd in (["afplay"] if IS_MAC else ["pw-play", "paplay", "aplay"]):
        if _have(cmd):
            subprocess.run([cmd, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
    if IS_WIN:
        try:
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
            return
        except Exception:
            pass
    print("   ⚠ no player found (install ffmpeg for ffplay)")


# ---------------------------------------------------------------------------
# Measurement — stdlib only, so recording works without the Kokoro deps
# ---------------------------------------------------------------------------

def _samples(path):
    """Read a WAV as a list of floats in -1..1, mixed to mono.

    Uses the stdlib `wave` module rather than soundfile/numpy on purpose:
    --record should work on a machine that never installed the TTS stack. The
    takes are always 16-bit mono because we ask every backend for exactly that,
    but the other widths are handled so a hand-dropped file does not crash the
    envelope.
    """
    try:
        with wave.open(path, "rb") as w:
            ch, sw, n = w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = w.readframes(n)
    except (wave.Error, EOFError, FileNotFoundError):
        return [], 0
    if not raw:
        return [], 0
    import array
    if sw == 2:
        a, scale, off = array.array("h"), 32768.0, 0.0
    elif sw == 4:
        a, scale, off = array.array("i"), 2147483648.0, 0.0
    elif sw == 1:
        a, scale, off = array.array("B"), 128.0, -128.0   # 8-bit WAV is UNSIGNED
    else:
        return [], 0
    a.frombytes(raw[:len(raw) - len(raw) % a.itemsize])
    if sys.byteorder == "big":
        a.byteswap()
    vals = [(v + off) / scale for v in a]
    if ch > 1:
        vals = [sum(vals[i:i + ch]) / ch for i in range(0, len(vals) - ch + 1, ch)]
    return vals, ch


def _db(x):
    """Amplitude (0..1) to dBFS, floored so log10(0) never appears in the UI."""
    return 20.0 * math.log10(max(abs(x), 1e-6))


def _peak_db(path):
    vals, _ = _samples(path)
    return _db(max((abs(v) for v in vals), default=0.0)) if vals else -99.0


def _duration(path):
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate() or RATE)
    except (wave.Error, EOFError, FileNotFoundError):
        return 0.0


def _envelope(path, width=56):
    """An ASCII amplitude envelope of the whole take.

    AGENTS.md makes the case for this on the generated side: an envelope catches
    'generated silence' and 'generated noise instead of the thing' faster than
    listening does. The same is true of a voice take — a flat bar means the mic
    was muted, a wall of full blocks means it clipped, and you can see the word
    rhythm well enough to spot a fluffed line without playing it back.
    """
    vals, _ = _samples(path)
    if not vals:
        return ""
    step = max(1, len(vals) // width)
    out = []
    for i in range(0, len(vals), step):
        pk = max((abs(v) for v in vals[i:i + step]), default=0.0)
        # dB, not raw amplitude. Linear scaling makes anything below about
        # -18 dBFS collapse into the bottom block, so a perfectly good quiet take
        # renders as a flat line — indistinguishable from a muted mic, which is
        # the one thing this display exists to tell you apart. -48..0 dB across
        # the ramp is the same mapping _meter uses, so the bar you watched while
        # recording and the envelope you see afterwards agree.
        lvl = (_db(pk) + 48.0) / 48.0
        out.append(RAMP[min(len(RAMP) - 1, max(0, int(lvl * len(RAMP))))])
    return "".join(out[:width])


def _meter(peak, width=22):
    """A VU bar for a single peak amplitude, scaled over the useful -48..0 dB."""
    db = _db(peak)
    filled = int(max(0.0, min(1.0, (db + 48.0) / 48.0)) * width)
    c = C
    col = c["red"] if db > -1.0 else (c["grn"] if db > -24.0 else c["yel"])
    return f"{col}{FULL * filled}{c['dim']}{EMPTY * (width - filled)}{c['rst']} {db:6.1f} dB"


# ---------------------------------------------------------------------------
# THE ONE JUDGMENT CALL — see the TODO
# ---------------------------------------------------------------------------

def take_warning(peak_db, seconds, text):
    """Decide whether a take looks bad enough to warn about before accepting it.

    Return a short warning string to show next to the take, or None to accept it
    silently. This is deliberately a policy function and not a hard gate — it
    never blocks you, it only puts a note on screen.

    The clipping case below is objective: digital clipping is wrong no matter
    who is reading or how. Everything else is calibration that depends on your
    mic, your room and your delivery, which is why the rest is left to you.

    TODO(grymmjack): add the subjective checks. Things worth considering:

      * TOO QUIET — a take peaking at, say, under -30 dBFS has little signal to
        work with; the loudness stage only ever attenuates (see
        loudness_normalize in soundmon.py), so a quiet take stays quiet.
      * DURATION vs WORD COUNT — `text` is right here, so len(text.split())
        against `seconds` catches the two classic mistakes: stopping the
        recording before you finished the line, and leaving it running after.
        Conversational narration runs roughly 2-3 words/second, but a grave
        dungeon-master read is much slower, so pick the band that matches how
        YOU are actually reading these.
      * The trade-off to weigh: a chatty warning you learn to ignore is worse
        than no warning at all. Prefer flagging few things confidently.
    """
    if peak_db > -0.5:
        return "CLIPPED — back off the mic or lower the input gain"
    # TODO: your checks here.
    return None


# ---------------------------------------------------------------------------
# The post-take chain — the same tail the generated packs go through
# ---------------------------------------------------------------------------

def _ff(args):
    """Run ffmpeg quietly; return stderr (some filters report through it)."""
    return subprocess.run(["ffmpeg", "-y", "-hide_banner", "-nostats"] + args,
                          capture_output=True, text=True).stderr


def _master(src, dst, threshold_db, fade_ms, normalize_db):
    """trim -> fade -> peak-normalize, landing at `dst`.

    ORDER MATTERS, and it is the order AGENTS.md gotcha #2 spells out: normalize
    LAST. A de-click fade is a gain change, so normalizing before it silently
    undoes the normalization. Here it bites harder than on the generated side —
    trimming pulls the leading silence off right up to the first syllable, so a
    plosive onset can genuinely be the loudest sample in the file and land
    directly under the fade-in.

    Falls back to a plain copy if ffmpeg is missing, so a machine with a mic but
    no ffmpeg still records usable (if unmastered) takes.
    """
    if not _have("ffmpeg"):
        shutil.copyfile(src, dst)
        return
    tmp = dst + ".trim.wav"
    # Trailing silence is removed by reversing, trimming the (now leading)
    # silence, and reversing back — silenceremove only ever works on the head.
    # The small start_silence pads keep a breath of room rather than butting the
    # first sample against the onset: 50 ms in front, 100 ms behind so a word is
    # allowed to decay instead of being chopped mid-tail.
    sr = (f"silenceremove=start_periods=1:start_duration=0:"
          f"start_threshold={threshold_db}dB:start_silence=0.05:detection=peak")
    sr_end = sr.replace("start_silence=0.05", "start_silence=0.10")
    if _ff(["-i", src, "-af", f"{sr},areverse,{sr_end},areverse", tmp]) and not os.path.exists(tmp):
        shutil.copyfile(src, dst)
        return
    if not os.path.exists(tmp) or os.path.getsize(tmp) < 128:
        # Trimming ate the whole file — the take was silence. Keep the original
        # so the envelope shows you a flat line instead of an empty file.
        if os.path.exists(tmp):
            os.remove(tmp)
        shutil.copyfile(src, dst)
        return

    dur = _duration(tmp)
    f = max(0.0, fade_ms / 1000.0)
    af = []
    if f and dur > 2 * f:
        af.append(f"afade=t=in:st=0:d={f}")
        af.append(f"afade=t=out:st={dur - f:.4f}:d={f}")
    # Peak-normalize by measuring POST-fade and applying one fixed gain, which is
    # what keeps the fade from invalidating the result.
    faded = dst + ".fade.wav"
    err = _ff(["-i", tmp, "-af", ",".join(af + ["volumedetect"]) if af else "volumedetect", faded])
    m = re.search(r"max_volume:\s*(-?[\d.]+) dB", err)
    gain = (normalize_db - float(m.group(1))) if m else 0.0
    if abs(gain) > 0.1 and os.path.exists(faded):
        _ff(["-i", faded, "-af", f"volume={gain:.2f}dB", dst])
    elif os.path.exists(faded):
        os.replace(faded, dst)
    for p in (tmp, faded):
        if os.path.exists(p):
            os.remove(p)
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)


# ---------------------------------------------------------------------------
# The booth
# ---------------------------------------------------------------------------

def _takes_for(takedir, key):
    """Every take already on disk for this key, in recording order."""
    if not os.path.isdir(takedir):
        return []
    pre = f"{key}.take"
    return sorted(os.path.join(takedir, f) for f in os.listdir(takedir)
                  if f.startswith(pre) and f.endswith(".wav"))


def _next_take_path(takedir, key):
    """Path for the next take of `key`.

    Numbered from the HIGHEST take on disk, not from how many there are:
    deleting take 2 of 3 leaves {01, 03}, and a count-based number would hand
    back 03 and silently overwrite a take you meant to keep.
    """
    nums = [int(m.group(1)) for p in _takes_for(takedir, key)
            if (m := re.search(r"\.take(\d+)\.wav$", os.path.basename(p)))]
    return os.path.join(takedir, f"{key}.take{(max(nums, default=0) + 1):02d}.wav")


def _finished(dest, key):
    """The pack file for this key, if one exists (either extension)."""
    for ext in (".ogg", ".wav"):
        p = os.path.join(dest, key + ext)
        if os.path.exists(p):
            return p
    return None


def _clear():
    sys.stdout.write("\033[2J\033[H")


def _draw(dest, idx, items, takes, sel, status, done):
    key, text = items[idx]
    c = C
    w = min(shutil.get_terminal_size((80, 24)).columns, 92)
    _clear()
    bar = f"{c['dim']}{'─' * w}{c['rst']}"
    print(f"{c['b']}{c['cyan']}soundmon record{c['rst']}  {c['dim']}·{c['rst']}  "
          f"line {c['b']}{idx + 1}{c['rst']}/{len(items)}  {c['dim']}·{c['rst']}  "
          f"{c['grn']}{done}{c['rst']} done  {c['dim']}·  {dest}{c['rst']}")
    print(bar)
    mark = f"  {c['grn']}✓ recorded{c['rst']}" if _finished(dest, key) else ""
    print(f"\n  {c['dim']}key{c['rst']}  {c['b']}{c['yel']}{key}{c['rst']}{mark}\n")
    for ln in textwrap.wrap(text, width=w - 6) or [""]:
        print(f"  {c['b']}{ln}{c['rst']}")
    print()
    if takes:
        chips = "  ".join(
            (f"{c['b']}{c['grn']}▸{i + 1}{c['rst']}" if i == sel else f"{c['dim']}{i + 1}{c['rst']}")
            for i in range(len(takes)))
        print(f"  {c['dim']}takes{c['rst']}  {chips}")
        pk, dur = _peak_db(takes[sel]), _duration(takes[sel])
        print(f"  {c['dim']}take {sel + 1}{c['rst']}  {dur:5.2f}s  peak {pk:6.1f} dBFS")
        print(f"  {c['dim']}{_envelope(takes[sel], w - 4)}{c['rst']}")
        warn = take_warning(pk, dur, text)
        if warn:
            print(f"  {c['red']}⚠ {warn}{c['rst']}")
    else:
        print(f"  {c['dim']}no takes yet — press R to record{c['rst']}")
    print(f"\n{bar}")
    if status:
        print(f"  {status}")
    print(f"  {c['grn']}R{c['rst']} record   {c['grn']}P{c['rst']} play   "
          f"{c['grn']}A{c['rst']} cycle take   {c['grn']}ENTER{c['rst']} accept + next   "
          f"{c['grn']}S{c['rst']} skip")
    print(f"  {c['grn']}B{c['rst']} back     {c['grn']}D{c['rst']} delete take   "
          f"{c['grn']}Q{c['rst']} quit {c['dim']}(progress is on disk — just rerun to resume){c['rst']}")


def _record_take(kb, path, device, rate):
    """Record until a key is pressed. Returns True unless the take was aborted.

    The live meter works by polling the growing file rather than tapping the
    audio stream: every backend writes a plain WAV as it goes, so reading the
    bytes that appeared since the last poll gives a real level with no extra
    capture path to keep in sync. Measured flush granularity is ~0.25 s, which
    is why METER_HZ is not set higher — a faster poll re-reads the same tail.
    """
    cmd, stop_mode = _capture_cmd(device, rate, path)
    c = C
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except FileNotFoundError:
        return False, f"{c['red']}capture command not found: {cmd[0]}{c['rst']}"

    started, pos, peak = time.monotonic(), 44, 0.0   # 44 = past the WAV header
    aborted = False
    sys.stdout.write("\n")
    while True:
        if proc.poll() is not None:                  # backend died (bad device?)
            err = (proc.stderr.read() or b"").decode(errors="replace").strip()
            return False, f"{c['red']}capture failed{c['rst']} {c['dim']}{err.splitlines()[-1] if err else ''}{c['rst']}"
        k = kb.get(timeout=1.0 / METER_HZ)
        if k in ("SPACE", "ENTER", "r", "s"):
            break
        if k == "ESC":
            aborted = True
            break
        elapsed = time.monotonic() - started
        if elapsed > MAX_TAKE:
            break
        # Peak of everything written since the last poll.
        try:
            size = os.path.getsize(path)
            if size > pos:
                with open(path, "rb") as f:
                    f.seek(pos)
                    chunk = f.read(size - pos)
                pos = size
                import array
                a = array.array("h")
                a.frombytes(chunk[:len(chunk) - len(chunk) % 2])
                if sys.byteorder == "big":
                    a.byteswap()
                if a:
                    peak = max(abs(v) for v in a) / 32768.0
        except OSError:
            pass
        sys.stdout.write(f"\r  {c['red']}{DOT} REC{c['rst']}  {elapsed:4.1f}s  "
                         f"{_meter(peak)}   {c['dim']}[SPACE] stop  [ESC] abort{c['rst']}  ")
        sys.stdout.flush()
        peak *= 0.55        # decay, so the bar falls back between words

    if stop_mode == "stdin-q":
        try:
            proc.stdin.write(b"q")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            proc.terminate()
    else:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    if proc.stdin:
        try:
            proc.stdin.close()
        except OSError:
            pass

    if aborted:
        if os.path.exists(path):
            os.remove(path)
        return False, f"{c['dim']}take discarded{c['rst']}"
    if not os.path.exists(path) or os.path.getsize(path) <= 44:
        if os.path.exists(path):
            os.remove(path)
        return False, f"{c['red']}nothing was captured — check --device / --list-devices{c['rst']}"
    return True, ""


def run(a, slug, to_ogg, loudness_normalize):
    """Record narration for a manifest. Called from soundmon.py.

    `to_ogg` and `loudness_normalize` are injected for the same reason
    narrate.run takes them: soundmon.py is __main__, so importing it back from
    here would re-execute the whole module.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import narrate                      # reuse the manifest parser, verbatim

    items = (narrate.parse_lines(a.record_file) if a.record_file
             else [(a.name or slug(a.prompt), a.prompt)] if a.prompt else [])
    if not items:
        sys.exit(f"--record: nothing to read in {a.record_file or '(no prompt given)'}")

    dest = os.path.abspath(os.path.expanduser(a.output_to)) if a.output_to else os.getcwd()
    if not os.path.isdir(dest):
        if a.create_dirs or a.output_to:
            os.makedirs(dest, exist_ok=True)
        else:
            sys.exit(f"output dir does not exist: {dest} (add --create-dirs)")
    # Takes live in a subdirectory, not beside the pack files. The output dir IS
    # the game's narration pack — anything dropped in it ships — and a pack
    # holding four rejected reads of every line is a pack nobody wants to copy
    # around. Sweeping them afterwards is one `rm -r takes/`.
    takedir = os.path.join(dest, "takes")
    os.makedirs(takedir, exist_ok=True)

    _enable_vt()
    c = C
    if not sys.stdin.isatty():
        sys.exit("--record needs an interactive terminal.")

    # Resume: start at the first line with no pack file. B still walks backwards,
    # so a finished line is revisitable — it is a starting point, not a filter.
    start = next((i for i, (k, _) in enumerate(items) if not _finished(dest, k)), 0)
    idx, status = start, ""
    done = sum(1 for k, _ in items if _finished(dest, k))
    if done:
        status = f"{c['dim']}resuming — {done}/{len(items)} already recorded{c['rst']}"

    sel_by_key = {}
    with Keyboard() as kb:
        while True:
            key, text = items[idx]
            takes = _takes_for(takedir, key)
            sel = min(sel_by_key.get(key, len(takes) - 1), len(takes) - 1)
            if sel < 0:
                sel = 0
            _draw(dest, idx, items, takes, sel, status, done)
            status = ""
            k = kb.get()

            if k == "q":
                _clear()
                print(f"{c['grn']}✔{c['rst']} {done}/{len(items)} recorded in {dest}")
                print(f"  {c['dim']}rerun the same command to pick up where you left off.{c['rst']}")
                return
            if k == "r":
                path = _next_take_path(takedir, key)
                ok, msg = _record_take(kb, path, a.device, a.record_rate)
                if ok:
                    # Select by looking the new file up in the re-listed takes
                    # rather than assuming it landed last: after a deletion the
                    # take numbers are sparse and its position is not the end.
                    fresh = _takes_for(takedir, key)
                    sel_by_key[key] = fresh.index(path) if path in fresh else len(fresh) - 1
                    status = msg or f"{c['grn']}take captured{c['rst']}"
                else:
                    status = msg
            elif k == "p" and takes:
                _play(takes[sel])
            elif k == "a" and takes:
                sel_by_key[key] = (sel + 1) % len(takes)
            elif k == "d" and takes:
                os.remove(takes[sel])
                sel_by_key[key] = max(0, sel - 1)
                status = f"{c['dim']}take {sel + 1} deleted{c['rst']}"
            elif k == "b":
                idx = (idx - 1) % len(items)
            elif k == "s":
                idx = (idx + 1) % len(items)
            elif k == "ENTER":
                if not takes:
                    status = f"{c['yel']}nothing to accept — press R first{c['rst']}"
                    continue
                out = os.path.join(dest, f"{key}.wav")
                _master(takes[sel], out, a.threshold_db, a.fade_ms, a.normalize_db)
                if getattr(a, "lufs_target", None) is not None:
                    loudness_normalize(out, a.lufs_target, a.true_peak)
                # A previous pass may have left a .ogg here; masters must not
                # coexist with a stale compressed twin the game would load first.
                stale = os.path.join(dest, f"{key}.ogg")
                if getattr(a, "ogg", False):
                    out = to_ogg(out, a.ogg_quality, a.keep_wav)
                elif os.path.exists(stale):
                    os.remove(stale)
                done = sum(1 for kk, _ in items if _finished(dest, kk))
                status = f"{c['grn']}✓ {os.path.basename(out)}{c['rst']}"
                if idx + 1 >= len(items):
                    _clear()
                    print(f"{c['grn']}✔ all {len(items)} lines recorded{c['rst']} in {dest}")
                    return
                idx += 1
