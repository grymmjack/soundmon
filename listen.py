"""--listen: what an audio model hears in every indexed sound (optional; needs torch and transformers).

The model is LAION's CLAP (laion/larger_clap_general: open weights, downloaded once from Hugging Face,
~600 MB). It turns the first 10 s of a sound into 512 numbers that put sounds that are alike near
each other - and near words that describe them. soundmon uses that three ways:

    tags     Your library teaches it. The sounds whose own words say what they are (tags.py) are the
             lessons; a small classifier learns them from what the model hears, blended with the model's
             own guess from each tag's description, and tags the sounds whose words say nothing. Each
             library teaches itself: nothing is shipped but the words. Your own tags (the GUI) teach it
             too, and always win.
    similar  "character": what it hears, as one more aspect of --similar (its default share CHARACTER)
    describe search by words: "a dark metallic hit"

It runs on a GPU when torch has one that works (CUDA, or ROCm: an RDNA2 laptop chip may need
HSA_OVERRIDE_GFX_VERSION=10.3.0), else on the CPU (~8 sounds a second). Incremental: only sounds not yet
heard are listened to; stop it any time.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tags  # noqa: E402

MODEL = "laion/larger_clap_general"
REVISION = "refs/pr/2"                  # the same weights as safetensors (no pickle to load)
SR = 48000
CLIP = 10.0                             # seconds the model hears: the start of the audible part
BATCH = 32
TRAIN_SHARE = 0.7                       # the learned classifier's share; the rest is the model's own guess
MIN_LESSONS = 15                        # a tag needs this many taught sounds to be learned (else: own guess)
PRECISION = 0.75                        # the ear tags a sound only as sure as it was right this often on held-back ones
_EAR = {}


def available():
    """(True, "") or (False, what to install)."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False, "listening needs torch and transformers — run soundmon with ComfyUI's venv, or " \
                      "pip install torch transformers"
    return True, ""


def _device(want=None):
    import torch
    want = want or os.environ.get("SOUNDMON_LISTEN_DEVICE") or "auto"
    if want != "auto":
        return want
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class Ear:
    def __init__(self, device=None):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch
        self.proc = ClapProcessor.from_pretrained(MODEL, revision=REVISION)
        model = ClapModel.from_pretrained(MODEL, revision=REVISION, use_safetensors=True).eval()
        self.model, self.device = model, _device(device)
        if self.device != "cpu":
            try:                                         # a GPU torch can't drive fails here, not mid-run
                self.model = model.to(self.device)
                self.audio([[0.01] * SR])
            except Exception:
                self.model, self.device = model.to("cpu"), "cpu"
        _speed_up_spectrogram(self.proc.feature_extractor)
        # on a GPU the CPU only feeds it: torch's own thread pool stays small (it otherwise takes every core)
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 2) if self.device == "cpu" else 2)

    def _unit(self, t):
        t = t if isinstance(t, self.torch.Tensor) else t.pooler_output
        return self.torch.nn.functional.normalize(t.float(), dim=-1).cpu().numpy()

    def features(self, clip):
        """the model's input for one clip (a mel spectrogram, float32): thread-safe, so the decode threads
        make them while the model works. One clip at a time, and "longer" from its own length: in batches
        the feature extractor marks a random clip as longer when none is, so a sound's embedding would
        depend on its batch."""
        import numpy as np
        fe = self.proc.feature_extractor
        x = fe(np.asarray(clip, dtype=np.float32), sampling_rate=SR, return_tensors="np")
        return x["input_features"][0].astype(np.float32), len(clip) > fe.nb_max_samples

    def embed(self, feats):
        """unit embeddings (n, 512) of features() results"""
        import numpy as np
        t = self.torch
        with t.no_grad():
            x = {"input_features": t.from_numpy(np.stack([f for f, _ in feats])).to(self.device),
                 "is_longer": t.tensor([[lg] for _, lg in feats], dtype=t.bool).to(self.device)}
            return self._unit(self.model.get_audio_features(**x))

    def audio(self, clips):
        """unit embeddings (n, 512) of mono 48 kHz clips"""
        return self.embed([self.features(c) for c in clips])

    def text(self, words):
        with self.torch.no_grad():
            x = self.proc(text=list(words), return_tensors="pt", padding=True)
            x = {k: v.to(self.device) for k, v in x.items()}
            return self._unit(self.model.get_text_features(**x))


def _fast_spectrogram(orig):
    """transformers.audio_utils.spectrogram with its per-frame Python loop done in one numpy call. The loop
    (a thousand frames a clip) holds Python's lock, so the decode threads took turns: 27 ms a clip alone,
    860 ms with 8 threads at once. The same maths for what CLAP asks of it; anything else goes to orig."""
    import numpy as np
    import transformers.audio_utils as au

    def f(waveform, window, frame_length, hop_length, fft_length=None, power=1.0, center=True, pad_mode="reflect",
          onesided=True, dither=0.0, preemphasis=None, mel_filters=None, mel_floor=1e-10, log_mel=None,
          reference=1.0, min_value=1e-10, db_range=None, remove_dc_offset=False, dtype=np.float32):
        fft_length = fft_length or frame_length
        if (dither or preemphasis is not None or remove_dc_offset or not onesided or power not in (1.0, 2.0)
                or waveform.ndim != 1 or len(window) != frame_length or frame_length > fft_length
                or log_mel not in (None, "dB") or hop_length <= 0):
            return orig(waveform, window, frame_length, hop_length, fft_length=fft_length, power=power,
                        center=center, pad_mode=pad_mode, onesided=onesided, dither=dither, preemphasis=preemphasis,
                        mel_filters=mel_filters, mel_floor=mel_floor, log_mel=log_mel, reference=reference,
                        min_value=min_value, db_range=db_range, remove_dc_offset=remove_dc_offset, dtype=dtype)
        if center:
            waveform = np.pad(waveform, [(frame_length // 2, frame_length // 2)], mode=pad_mode)
        w = waveform.astype(np.float64)
        n = int(1 + np.floor((w.size - frame_length) / hop_length))
        frames = np.lib.stride_tricks.sliding_window_view(w, frame_length)[::hop_length][:n]
        spec = np.fft.rfft(frames * window.astype(np.float64), n=fft_length, axis=1).astype(np.complex64)
        spec = (np.abs(spec, dtype=np.float64) ** power).T
        if mel_filters is not None:
            spec = np.maximum(mel_floor, np.dot(mel_filters.T, spec))
        if log_mel == "dB":
            spec = (au.amplitude_to_db if power == 1.0 else au.power_to_db)(spec, reference, min_value, db_range)
            spec = np.asarray(spec, dtype)
        return spec
    f.soundmon = True
    return f


def _speed_up_spectrogram(fe):
    """Use _fast_spectrogram in CLAP's feature extractor if it gives the same input on a test clip (else keep
    the library's own: a transformers that computes it differently is left alone)."""
    import numpy as np
    try:
        import transformers.models.clap.feature_extraction_clap as fc
        if getattr(fc.spectrogram, "soundmon", False):
            return
        clip = (np.random.default_rng(1).standard_normal(int(SR * 3.3)) * 0.2).astype(np.float32)
        want = fe(clip, sampling_rate=SR, return_tensors="np")["input_features"]
        orig = fc.spectrogram
        fc.spectrogram = _fast_spectrogram(orig)
        got = fe(clip, sampling_rate=SR, return_tensors="np")["input_features"]
        if got.shape != want.shape or np.abs(got - want).max() > 1e-3:
            fc.spectrogram = orig
    except Exception:
        pass


def ear(device=None):
    key = device or "auto"
    if key not in _EAR:
        _EAR[key] = Ear(device)
    return _EAR[key]


def clip_of(path):
    """The first CLIP seconds of a file's audible part, mono 48 kHz, scaled to its peak; None when silent."""
    import numpy as np
    import similar
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-t", "40", "-i", similar.here(path), "-ac", "1", "-ar", str(SR),
                        "-f", "f32le", "-"], capture_output=True, timeout=120)
    if r.returncode:
        raise RuntimeError(r.stderr.decode(errors="replace")[-200:])
    x = np.nan_to_num(np.frombuffer(r.stdout, dtype=np.float32))
    if len(x) < 64 or np.abs(x).max() < 1e-4:
        return None
    x = x / np.abs(x).max() * 0.9
    loud = np.where(np.abs(x) > 10 ** (-50 / 20))[0]
    return x[loud[0]:loud[0] + int(CLIP * SR)]


def text_query(device=None):
    """f(words) -> the unit embedding of a description (for Index.browse(describe=...))."""
    e = ear(device)
    return lambda words: e.text([words])[0]


def audio_query(device=None):
    """f(path) -> the unit embedding of a file that isn't in the index (for Index.search's character)."""
    e = ear(device)

    def f(path):
        c = clip_of(path)
        if c is None:
            raise ValueError("silent")
        return e.audio([c])[0]
    return f


def _prompts(e):
    """unit embedding of each tag's descriptions (averaged), and of loop / one-shot."""
    import numpy as np

    def avg(phrases):
        v = e.text(phrases).mean(axis=0)
        return v / (np.linalg.norm(v) + 1e-9)
    T = np.stack([avg(tags.PROMPTS[t]) for t in tags.IDS])
    K = np.stack([avg(tags.KIND_PROMPTS["loop"]), avg(tags.KIND_PROMPTS["oneshot"])])
    return T, K


def idx_cooler():
    import similar
    return similar.Cooler()


def _one_blas_thread():
    """numpy's BLAS held to one thread while listening (process-wide: each decode thread would otherwise
    start one per core - the oversubscription the index's workers had). A no-op without threadpoolctl."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(1)
    except ImportError:
        import contextlib
        return contextlib.nullcontext()


def listen(idx, progress=None, device=None, stop=None, workers=None):
    """Hear every indexed sound not yet heard, then learn() the tags. Returns stats."""
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    e = ear(device)
    workers = workers or max(2, min(8, (os.cpu_count() or 4) // 2))
    db = idx.db()
    rows = db.execute("SELECT id, path FROM files WHERE status = 0 AND clap IS NULL ORDER BY path").fetchall()
    n, heard, failed, last, last_commit = len(rows), 0, 0, 0.0, time.time()
    def load(r):                                    # in a decode thread: ffmpeg, then the model's input
        try:
            c = clip_of(r[1])
            return r, (None if c is None else e.features(c))
        except Exception:
            return r, None
    from collections import deque
    with _one_blas_thread(), ThreadPoolExecutor(workers) as ex:
        todo, ahead = iter(rows), deque()              # only a few batches decoded ahead: memory stays flat
        batch, done, cooler = [], 0, idx_cooler()
        while True:
            while not cooler.ok() and not (stop and stop()):   # too hot: the GPU and the decoding rest
                if progress and time.time() - last > 2:
                    last = time.time()
                    progress("cooling", done, n, cooler.note())
                time.sleep(0.5)
            while len(ahead) < BATCH * 3:
                r = next(todo, None)
                if r is None:
                    break
                ahead.append(ex.submit(load, r))
            if not ahead:
                break
            r, f = ahead.popleft().result()
            done += 1
            if f is None:
                failed += 1
                db.execute("UPDATE files SET clap = ? WHERE id = ?", (b"", r[0]))   # not asked again till it changes
            else:
                batch.append((r, f))
            if len(batch) >= BATCH or (done == n and batch):
                V = e.embed([f_ for _, f_ in batch]).astype(np.float16)
                db.executemany("UPDATE files SET clap = ? WHERE id = ?",
                               [(v.tobytes(), r_[0]) for (r_, _), v in zip(batch, V)])
                heard += len(batch)
                batch = []
            if progress and (time.time() - last > 1 or done == n):
                last = time.time()
                progress("listening", done, n, r[1])
            if time.time() - last_commit > 3:
                idx._commit(db)
                last_commit = time.time()
            if stop and stop():
                for fu in ahead:
                    fu.cancel()
                break
        if batch:                                      # stopped part way through a batch: keep what was heard
            V = e.embed([f_ for _, f_ in batch]).astype(np.float16)
            db.executemany("UPDATE files SET clap = ? WHERE id = ?",
                           [(v.tobytes(), r_[0]) for (r_, _), v in zip(batch, V)])
            heard += len(batch)
    idx._commit(db)
    st = learn(idx, e, progress)
    st.update(heard=heard, failed=failed, device=e.device, stopped=bool(stop and stop()), cooled=round(cooler.waited))
    return st


def _softmax_fit(X, y, k, weight, steps=300, l2=1e-4):
    """multinomial logistic regression (full batch, Adam) -> W (d+1, k)."""
    import torch
    Xt = torch.tensor(X, dtype=torch.float32)
    Xt = torch.cat([Xt, torch.ones(len(Xt), 1)], dim=1)
    yt = torch.tensor(y, dtype=torch.long)
    W = torch.zeros(Xt.shape[1], k, requires_grad=True)
    opt = torch.optim.Adam([W], lr=0.05)
    wt = torch.tensor(weight, dtype=torch.float32)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W * 10, yt, weight=wt) + l2 * (W ** 2).sum()
        loss.backward()
        opt.step()
    return W.detach().numpy()


def _predict(W, X):
    import numpy as np
    z = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], axis=1) @ W * 10
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def learn(idx, e=None, progress=None):
    """Teach a classifier the library's own tagged sounds and tag every heard sound by ear (stag, ssure,
    skind). Returns {"learned_from", "tagged", "agree"}: agree = how often it matched the words on sounds
    it wasn't taught (a tenth held back), so you can see how far to trust it."""
    import numpy as np
    import similar
    e = e or ear()
    db = idx.db()
    rows = db.execute("SELECT id, clap, utag, wtag, wsure, ukind, kind, kind_by, seconds FROM files "
                      "WHERE status = 0 AND length(clap) > 0").fetchall()
    if not rows:
        return {"learned_from": 0, "tagged": 0, "agree": None}
    if progress:
        progress("learning", 0, len(rows), "")
    X = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float16).reshape(len(rows), -1).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    T, K = _prompts(e)
    zs = X @ T.T * 100
    zs = np.exp(zs - zs.max(axis=1, keepdims=True))
    zs /= zs.sum(axis=1, keepdims=True)
    code = {t: i for i, t in enumerate(tags.IDS)}
    lab = np.full(len(rows), -1)
    for i, r in enumerate(rows):
        if r[2] and r[2] != "-":
            lab[i] = code.get(r[2], -1)
        elif r[3] and (r[4] or 0) >= similar.WORDS_SURE:
            lab[i] = code.get(r[3], -1)
    counts = np.bincount(lab[lab >= 0], minlength=len(tags.IDS))
    taught = np.nonzero(counts >= MIN_LESSONS)[0]
    P, agree, bar = zs * 0.5, None, None                # untaught, its own guesses count for half
    if len(taught) >= 2:
        sel = np.nonzero(np.isin(lab, taught))[0]
        remap = {c: j for j, c in enumerate(taught)}
        y = np.array([remap[c] for c in lab[sel]])
        w = 1.0 / np.sqrt(np.bincount(y, minlength=len(taught)).clip(1))
        w = w / w.mean()
        rng = np.random.default_rng(0)
        hold = rng.random(len(sel)) < 0.1
        if hold.sum() >= 20 and (~hold).sum() >= 50:   # a check first, on a tenth held back
            Wc = _softmax_fit(X[sel[~hold]], y[~hold], len(taught), w)
            Pc = np.zeros((hold.sum(), len(tags.IDS)))
            Pc[:, taught] = _predict(Wc, X[sel[hold]])
            Pc = TRAIN_SHARE * Pc + (1 - TRAIN_SHARE) * zs[sel[hold]]
            right = Pc.argmax(axis=1) == lab[sel[hold]]
            agree = float(right.mean())
            # the bar: the least sureness at which the held-back tags were right PRECISION of the time
            sure_c = Pc.max(axis=1)
            for t in np.arange(0.2, 0.96, 0.05):
                m = sure_c >= t
                if m.sum() >= 20 and right[m].mean() >= PRECISION:
                    bar = round(float(t), 2)
                    break
        W = _softmax_fit(X[sel], y, len(taught), w)
        Pt = np.zeros_like(zs)
        Pt[:, taught] = _predict(W, X)
        P = TRAIN_SHARE * Pt + (1 - TRAIN_SHARE) * zs
        untaught = ~np.isin(np.arange(len(tags.IDS)), taught)
        P[:, untaught] *= 0.5                           # a tag nobody taught it: its own guess, half believed
    stag, ssure = P.argmax(axis=1), P.max(axis=1)
    bar = bar if bar is not None else similar.EAR_SURE
    db.execute("INSERT INTO meta VALUES ('ear_sure', ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v", (str(bar),))
    # loop or one-shot: taught by the flags and words, with the length; blended with the model's own guess
    secs = np.log10(np.array([max(r[8] or 0.01, 0.01) for r in rows], dtype=np.float32))[:, None]
    zk = X @ K.T * 100
    zk = np.exp(zk - zk.max(axis=1, keepdims=True))
    zk /= zk.sum(axis=1, keepdims=True)
    klab = np.array([0 if (r[5] or (r[6] if r[7] in ("flag", "words", "sampler") else None)) == "loop" else
                     1 if (r[5] or (r[6] if r[7] in ("flag", "words", "sampler") else None)) == "oneshot" else -1 for r in rows])
    PK = None                                       # untaught, the model's own loop-or-not is a coin toss
    if (klab == 0).sum() >= MIN_LESSONS and (klab == 1).sum() >= MIN_LESSONS:
        ks = np.nonzero(klab >= 0)[0]
        XK = np.concatenate([X, secs], axis=1)
        wk = 1.0 / np.sqrt(np.bincount(klab[ks], minlength=2).clip(1))
        WK = _softmax_fit(XK[ks], klab[ks], 2, wk / wk.mean())
        PK = TRAIN_SHARE * _predict(WK, XK) + (1 - TRAIN_SHARE) * zk
    skind = np.where(PK[:, 0] >= 0.5, "loop", "oneshot") if PK is not None else [None] * len(rows)
    db.executemany("UPDATE files SET stag = ?, ssure = ?, skind = ? WHERE id = ?",
                   [(tags.IDS[int(t)], round(float(s), 3), k and str(k), r[0])
                    for t, s, k, r in zip(stag, ssure, skind, rows)])
    idx._commit(db)
    tagged = sum(1 for r, s in zip(rows, ssure) if not r[2] and not (r[3] and (r[4] or 0) >= similar.WORDS_SURE)
                 and s >= bar)
    return {"learned_from": int((lab >= 0).sum()), "tagged": tagged, "agree": agree, "bar": bar,
            "taught": [tags.IDS[int(t)] for t in taught]}
