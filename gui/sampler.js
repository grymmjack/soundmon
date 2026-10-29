/* soundmon-gui: SAMPLER - 16 pads played from a MIDI controller, the computer's keys or the mouse.
   Drop sounds onto pads (from the clip bin, Gallery, Queue, Find, Assemble's media or the desktop). Each pad
   plays on its own note (Pads), the selected pad plays across the keyboard pitched from its root (Pitched), or
   both at once (Both: the notes with a pad play it, the rest play the selected pad pitched - kaleidotron's way).
   Per pad: volume, pan, pitch, start/end, one-shot / hold / loop, reverse, an ADSR, a filter, velocity,
   choke groups. Any knob can be MIDI-learned. The playing is Web Audio + Web MIDI in the page; the server keeps
   the sounds (Assemble's media store), kits (sampler/kits) and writes recordings and SFZ exports.
   Loaded after index.html's script, whose helpers it uses ($, el, api, store, toast, pickPath, …). */
(() => {
  document.head.append(Object.assign(document.createElement("style"), { textContent: `
#p-sampler { gap: 8px; }
.sm-top { flex-wrap: wrap; gap: 6px 10px; align-items: center; }
.sm-top h2 { margin: 0 6px 0 0; }
.sm-top button { padding: 3px 10px; font-size: 12px; }
.sm-f { display: inline-flex; align-items: center; gap: 5px; font-size: 12px; color: var(--dim); }
.sm-f select, .sm-f input { width: auto; padding: 3px 5px; } .sm-f input[type=number] { width: 58px; }
#smKit { width: 160px; } #smOpen, #smRecFmt, #smMidi { width: auto; padding: 3px 5px; }
.sm-seg { display: inline-flex; } .sm-seg button { border-radius: 0; margin-left: -1px; }
.sm-seg button:first-child { border-radius: var(--radius) 0 0 var(--radius); } .sm-seg button:last-child { border-radius: 0 var(--radius) var(--radius) 0; }
.sm-seg button[aria-pressed="true"] { background: var(--accent); color: var(--on-accent); border-color: var(--accent); font-weight: 700; }
.sm-led { display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: var(--line); transition: background .25s; }
.sm-led.on { background: var(--ok); transition: none; }
.sm-meter { width: 90px; height: 8px; border: 1px solid var(--line); border-radius: 3px; overflow: hidden; background: var(--bg); }
.sm-meter i { display: block; height: 100%; width: 0; background: var(--ok); }
.sm-meter.hot i { background: var(--bad); }
#smRec[aria-pressed="true"] { background: var(--bad); color: #000; border-color: var(--bad); font-weight: 700; }
.sm-body { display: grid; grid-template-columns: minmax(280px, 440px) minmax(0, 1fr); gap: 10px; align-items: start; }
@media (max-width: 900px) { .sm-body { grid-template-columns: minmax(0, 1fr); } }
.sm-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; user-select: none; }
.sm-pad { position: relative; aspect-ratio: 1 / .82; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); overflow: hidden; cursor: pointer; touch-action: none; }
.sm-pad.empty { border-style: dashed; cursor: copy; }
.sm-pad.sel { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
.sm-pad.keys::after { content: "🎹"; position: absolute; right: 4px; bottom: 2px; font-size: 11px; }
.sm-pad.drop { border-color: var(--ok); box-shadow: 0 0 0 2px var(--ok); }
.sm-pad.quiet { opacity: .45; }
.sm-pad canvas { position: absolute; left: 0; right: 0; bottom: 0; width: 100%; height: 55%; pointer-events: none; opacity: .65; }
.sm-pad .hit { position: absolute; inset: 0; background: var(--accent); opacity: 0; pointer-events: none; transition: opacity .3s; }
.sm-pad.lit .hit { opacity: .35; transition: none; }
.sm-pad.on { border-color: var(--ok); }
.sm-pad .n { position: absolute; left: 5px; top: 3px; font: 600 10.5px var(--mono); color: var(--dim); }
.sm-pad .k { position: absolute; right: 5px; top: 3px; font: 10.5px var(--mono); color: var(--accent2); }
.sm-pad .nm { position: absolute; left: 5px; right: 5px; top: 18px; font: 600 11.5px var(--mono); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sm-pad .ch { position: absolute; left: 5px; bottom: 2px; font: 10px var(--mono); color: var(--dim); text-shadow: 0 0 3px #000; }
.sm-ed { display: flex; flex-direction: column; gap: 8px; padding: 8px 10px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--panel); min-width: 0; }
.sm-edh { display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; font-size: 12px; }
.sm-edh input.nm { font: 600 13px var(--mono); width: 200px; padding: 2px 5px; }
.sm-edh button { padding: 2px 8px; font-size: 12px; }
.sm-edh .m[aria-pressed="true"] { background: var(--bad); color: #000; border-color: var(--bad); }
.sm-edh .s[aria-pressed="true"] { background: var(--warn); color: #000; border-color: var(--warn); }
#smLearn[aria-pressed="true"] { background: var(--warn); color: #000; border-color: var(--warn); }
.sm-wave { position: relative; height: 120px; border: 1px solid var(--line); border-radius: 4px; background: var(--bg); cursor: crosshair; touch-action: none; }
.sm-wave canvas { display: block; width: 100%; height: 100%; }
.sm-knobs { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 4px 14px; font-size: 12px; }
.sm-k { display: grid; grid-template-columns: 78px minmax(0, 1fr) 62px; align-items: center; gap: 5px; }
.sm-k > span:first-child { color: var(--dim); white-space: nowrap; }
.sm-k input[type=range] { width: 100%; min-width: 0; padding: 0; }
.sm-k .v { font: 11px var(--mono); text-align: right; white-space: nowrap; }
.sm-k .cc { color: var(--warn); font-size: 10px; margin-left: 3px; }
.sm-k.learn > span:first-child { color: var(--warn); }
.sm-k select { width: 100%; padding: 1px 3px; }
.sm-sect { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--dim); grid-column: 1 / -1; margin-top: 4px; border-bottom: 1px solid var(--line); }
.sm-kb { position: relative; height: 84px; border: 1px solid var(--line); border-radius: 4px; overflow: hidden; user-select: none; touch-action: none; }
.sm-kb canvas { display: block; width: 100%; height: 100%; cursor: pointer; }
.sm-rec { display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; padding: 6px 10px; border: 1px solid var(--bad); border-radius: var(--radius); font-size: 12px; }
.sm-rec button { padding: 2px 8px; font-size: 12px; }
.sm-menu { position: fixed; z-index: 60; }
.sm-hint { font-size: 11.5px; color: var(--dim); }
` }));

  const PADS = 16, NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  const noteName = n => NOTES[((n % 12) + 12) % 12] + (Math.floor(n / 12) - 1);      // 60 = C4, 48 = C3 (kaleidotron's)
  const cutHz = v => 20 * Math.pow(1000, v);
  const dB = x => Math.pow(10, x / 20);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const fmtS = v => v < 1 ? Math.round(v * 1000) + " ms" : v.toFixed(2) + " s";
  const PAD_TYPE = "application/x-soundmon-pad";
  const blankPad = () => ({ media: null, name: "", note: null, root: 60, vol: 0, pan: 0, semis: 0, fine: 0, start: 0, end: 0, play: "oneshot", ls: 0, le: 0,
    rev: false, a: 0.002, d: 0.1, s: 1, r: 0.05, filter: "off", cut: 1, res: 0.7, vel: 0.7, choke: 0, mono: false, mute: false, solo: false, dur: 0 });
  const blankKit = () => ({ v: 1, name: "", mode: "both", base: 48, sel: 0, bend: 2, master: { vol: 0, limit: true }, pads: Array.from({ length: PADS }, blankPad) });
  function fix(k) {
    const b = blankKit();
    if (!k || typeof k !== "object") return b;
    const m = Object.assign({}, b, k);
    m.master = Object.assign({}, b.master, k.master || {});
    m.pads = Array.from({ length: PADS }, (_, i) => Object.assign(blankPad(), (k.pads || [])[i] || {}));
    if (!["pads", "keys", "both"].includes(m.mode)) m.mode = "both";
    m.sel = clamp(m.sel | 0, 0, PADS - 1);
    return m;
  }
  let K = fix(store.get("smp", null)), built = false, shown = false, savedSig = "";
  const pad = () => K.pads[K.sel];

  /* ---------- the knobs: every one MIDI-learnable (right-click it) ---------- */
  const PLAY = [["oneshot", "One-shot"], ["hold", "Hold"], ["loop", "Loop"]];
  const PARAMS = {
    "pad.vol":   { l: "Volume", min: -40, max: 12, step: 0.5, def: 0, f: v => (v > 0 ? "+" : "") + v.toFixed(1) + " dB" },
    "pad.pan":   { l: "Pan", min: -1, max: 1, step: 0.01, def: 0, f: v => Math.abs(v) < 0.005 ? "C" : (v < 0 ? "L" : "R") + Math.round(Math.abs(v) * 100) },
    "pad.semis": { l: "Pitch", min: -24, max: 24, step: 1, def: 0, f: v => (v > 0 ? "+" : "") + v + " st" },
    "pad.fine":  { l: "Fine", min: -100, max: 100, step: 1, def: 0, f: v => (v > 0 ? "+" : "") + v + " ct" },
    "pad.vel":   { l: "Velocity", min: 0, max: 1, step: 0.01, def: 0.7, f: v => Math.round(v * 100) + "%", tip: "how much harder hits are louder (0: every hit the same)" },
    "pad.a":     { l: "Attack", min: 0, max: 2, step: 0.001, def: 0.002, f: fmtS, curve: 3 },
    "pad.d":     { l: "Decay", min: 0, max: 4, step: 0.001, def: 0.1, f: fmtS, curve: 3 },
    "pad.s":     { l: "Sustain", min: 0, max: 1, step: 0.01, def: 1, f: v => Math.round(v * 100) + "%" },
    "pad.r":     { l: "Release", min: 0, max: 6, step: 0.001, def: 0.05, f: fmtS, curve: 3 },
    "pad.cut":   { l: "Cutoff", min: 0, max: 1, step: 0.001, def: 1, f: v => fmtHz(cutHz(v)) },
    "pad.res":   { l: "Resonance", min: 0.5, max: 18, step: 0.1, def: 0.7, f: v => "Q " + v.toFixed(1) },
    "master.vol": { l: "Master", min: -40, max: 12, step: 0.5, def: 0, f: v => (v > 0 ? "+" : "") + v.toFixed(1) + " dB" },
  };
  const getP = key => key.startsWith("master.") ? K.master[key.slice(7)] : pad()[key.slice(4)];
  function setP(key, v) {
    if (key.startsWith("master.")) K.master[key.slice(7)] = v; else pad()[key.slice(4)] = v;
    live(key); dirty();
  }
  const fmtHz = typeof window.fmtHz === "function" ? window.fmtHz : hz => hz >= 1000 ? (hz / 1000).toFixed(hz >= 10000 ? 0 : 1) + " kHz" : Math.round(hz) + " Hz";
  // a slider position (0..1) <-> a value; envelope times are curved so short times get most of the travel
  const toV = (p, x) => p.curve ? p.min + (p.max - p.min) * Math.pow(x, p.curve) : p.min + (p.max - p.min) * x;
  const toX = (p, v) => p.curve ? Math.pow((v - p.min) / (p.max - p.min), 1 / p.curve) : (v - p.min) / (p.max - p.min);
  let CC = store.get("smpCC", {}), learnCC = null;          // {cc number: param key}
  function knob(key) {
    const p = PARAMS[key], r = el("input", { type: "range", min: 0, max: 1000, step: 1 }), v = el("span", { class: "v" });
    const ccOf = () => Object.keys(CC).find(c => CC[c] === key);
    const box = el("label", { class: "sm-k", "data-k": key, title: (p.tip ? p.tip + ". " : "") + "Right-click: learn a knob or slider on your controller. Double-click: reset." },
      el("span", {}, p.l, ...(ccOf() ? [el("span", { class: "cc", text: "CC" + ccOf() })] : [])), r, v);
    const show = () => { const x = getP(key); r.value = Math.round(clamp(toX(p, x), 0, 1) * 1000); v.textContent = p.f(x); };
    r.addEventListener("input", () => { let x = toV(p, r.value / 1000); x = Math.round(x / p.step) * p.step; setP(key, +x.toFixed(4)); v.textContent = p.f(getP(key)); });
    r.addEventListener("dblclick", () => { setP(key, p.def); show(); });
    box.addEventListener("contextmenu", e => { e.preventDefault(); ccMenu(key, e); });
    box._show = show; show();
    return box;
  }
  function ccMenu(key, e) {
    const cc = Object.keys(CC).find(c => CC[c] === key);
    smallMenu(e, [
      [learnCC === key ? "Stop learning" : "Learn: move a knob on your controller", () => { learnCC = learnCC === key ? null : key; markLearn(); if (learnCC) toast(`Move a knob, fader or wheel on your controller for ${PARAMS[key].l}…`); }],
      ...(cc ? [[`Forget CC ${cc}`, () => { delete CC[cc]; store.set("smpCC", CC); drawEditor(); drawTop(); }]] : []),
      ...(Object.keys(CC).length ? [["Forget every learned knob", () => { CC = {}; store.set("smpCC", CC); drawEditor(); drawTop(); }]] : [])]);
  }
  function markLearn() { document.querySelectorAll("#p-sampler .sm-k").forEach(b => b.classList.toggle("learn", b.dataset.k === learnCC)); }
  function smallMenu(e, items) {
    document.querySelectorAll(".omenu").forEach(m => m.remove());
    const m = el("div", { class: "omenu sm-menu", role: "menu" }, ...items.map(([label, fn]) => el("button", { type: "button", role: "menuitem", text: label, onclick: () => { m.remove(); fn(); } })));
    document.body.append(m);
    m.style.left = Math.min(innerWidth - m.offsetWidth - 8, e.clientX) + "px"; m.style.top = Math.min(innerHeight - m.offsetHeight - 8, e.clientY) + "px";
    setTimeout(() => document.addEventListener("click", function off(ev) { if (!m.contains(ev.target)) { m.remove(); document.removeEventListener("click", off); } }), 0);
  }

  /* ---------- sounds ---------- */
  let ctx = null, bus = null, master = null, limiter = null, meter = null, padGain = [];
  const BUF = new Map(), PEAKS = new Map();
  const mediaUrl = m => "/api/assemble/media?f=" + encodeURIComponent(m);
  function ac() {
    if (ctx) return ctx;
    ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: "interactive" });
    master = ctx.createGain(); limiter = ctx.createDynamicsCompressor(); meter = ctx.createAnalyser(); meter.fftSize = 1024;
    limiter.threshold.value = -1; limiter.knee.value = 0; limiter.ratio.value = 20; limiter.attack.value = 0.002; limiter.release.value = 0.12;
    bus = ctx.createGain();
    padGain = K.pads.map(() => { const g = ctx.createGain(); g.connect(bus); return g; });
    bus.connect(master); route(); padLevels();
    return ctx;
  }
  function route() {                                         // master -> (limiter) -> meter -> speakers (and the recorder)
    if (!ctx) return;
    master.disconnect(); limiter.disconnect();
    if (K.master.limit) { master.connect(limiter); limiter.connect(meter); } else master.connect(meter);
    meter.disconnect(); meter.connect(ctx.destination); if (rec.node) meter.connect(rec.node);
    master.gain.value = dB(K.master.vol);
  }
  const audible = i => { const solo = K.pads.some(p => p.solo && p.media); const p = K.pads[i]; return !p.mute && (!solo || p.solo); };
  function padLevels() { if (!ctx) return; K.pads.forEach((p, i) => padGain[i].gain.setTargetAtTime(audible(i) ? dB(p.vol) : 0, ctx.currentTime, 0.01)); }
  function getBuf(media, rev) {
    const key = media + (rev ? "#rev" : "");
    if (BUF.has(key)) return BUF.get(key);
    const pr = rev ? getBuf(media).then(b => {
      const r = ac().createBuffer(b.numberOfChannels, b.length, b.sampleRate);
      for (let ch = 0; ch < b.numberOfChannels; ch++) r.getChannelData(ch).set(b.getChannelData(ch).slice().reverse());
      return r;
    }) : fetch(mediaUrl(media)).then(r => { if (!r.ok) throw new Error(`can't load ${media} (HTTP ${r.status})`); return r.arrayBuffer(); })
      .then(a => ac().decodeAudioData(a));
    pr.then(b => { pr.now = b; }).catch(() => BUF.delete(key));
    BUF.set(key, pr);
    return pr;
  }
  const bufNow = (media, rev) => { const pr = BUF.get(media + (rev ? "#rev" : "")); return pr && pr.now; };
  function warm() { K.pads.forEach(p => { if (p.media) getBuf(p.media, p.rev).then(() => { drawGrid(); if (p === pad()) drawWave(); }).catch(e => toast(e.message, true)); }); }
  function peaks(media) {                                    // a min/max outline for drawing, 600 columns
    if (PEAKS.has(media)) return PEAKS.get(media);
    const b = bufNow(media, false); if (!b) return null;
    const n = 600, d = b.getChannelData(0), d2 = b.numberOfChannels > 1 ? b.getChannelData(1) : d, step = d.length / n, out = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) { let lo = 0, hi = 0; const a = Math.floor(i * step), z = Math.min(d.length, Math.floor((i + 1) * step) + 1);
      for (let j = a; j < z; j++) { const x = (d[j] + d2[j]) / 2; if (x < lo) lo = x; if (x > hi) hi = x; } out[2 * i] = lo; out[2 * i + 1] = hi; }
    PEAKS.set(media, out); return out;
  }

  /* ---------- voices ---------- */
  let voices = [], bendCents = 0, sustain = false;
  const MAX_VOICES = 64;
  function envAt(v, t) {                                     // where the voice's ADSR is at time t (for a smooth release)
    const p = v.p, dt = t - v.t0;
    if (dt <= 0) return 0;
    if (dt < p.a) return v.peak * dt / Math.max(p.a, 1e-4);
    if (dt < p.a + p.d) return v.peak * (1 - (1 - p.s) * (dt - p.a) / Math.max(p.d, 1e-4));
    return v.peak * p.s;
  }
  function fire(i, vel, transpose, key) {
    const c = ac(), p = K.pads[i];
    flash(i);
    if (!p.media) return;
    if (c.state !== "running") c.resume();
    const b = bufNow(p.media, p.rev);
    if (!b) { getBuf(p.media, p.rev).catch(e => toast(e.message, true)); return; }
    // who this hit silences: the same key still held, this pad's own voices when mono, its choke group
    for (const v of voices.slice()) {
      if (v.key === key && v.held && v.pad === i) stopVoice(v, 0.008);
      else if (p.mono && v.pad === i) stopVoice(v, 0.008);
      else if (p.choke && v.pad !== i && K.pads[v.pad].choke === p.choke) stopVoice(v, 0.008);
    }
    while (voices.length >= MAX_VOICES) stopVoice(voices[0], 0.005);
    const now = c.currentTime + 0.003, dur = b.duration;
    let st = clamp(p.start || 0, 0, dur), en = clamp(p.end || dur, st + 0.001, dur);
    const rate = Math.pow(2, ((p.semis || 0) + (p.fine || 0) / 100 + transpose) / 12);
    const s = c.createBufferSource(); s.buffer = b; s.playbackRate.value = rate; if (s.detune) s.detune.value = bendCents;
    const off = p.rev ? dur - en : st, len = en - st;
    let head = s;
    let f = null;
    if (p.filter === "lp" || p.filter === "hp") { f = c.createBiquadFilter(); f.type = p.filter === "lp" ? "lowpass" : "highpass"; f.frequency.value = cutHz(p.cut); f.Q.value = p.res; head.connect(f); head = f; }
    const g = c.createGain(), pan = c.createStereoPanner ? c.createStereoPanner() : null;
    const vv = clamp(vel / 127, 0, 1), peak = (1 - p.vel) + p.vel * vv * vv;
    const v = { pad: i, key, s, g, f, pan, t0: now, rate, tr: transpose, held: true, sus: false, peak, p: { a: p.a, d: p.d, s: p.s, r: p.r }, play: p.play, off, len, st, en, dur, rev: p.rev };
    g.gain.setValueAtTime(0, now);
    g.gain.linearRampToValueAtTime(peak, now + Math.max(p.a, 0.001));
    if (p.d > 0 && p.s < 1) g.gain.linearRampToValueAtTime(peak * p.s, now + Math.max(p.a, 0.001) + p.d);
    else g.gain.setValueAtTime(peak * p.s, now + Math.max(p.a, 0.001) + Math.max(p.d, 0));
    head.connect(g); head = g;
    if (pan) { pan.pan.value = p.pan; head.connect(pan); head = pan; }
    head.connect(padGain[i]);
    if (p.play === "loop") {
      const ls = clamp(p.ls || st, st, en), le = clamp(p.le || en, ls + 0.005, en);
      s.loop = true; s.loopStart = p.rev ? dur - le : ls; s.loopEnd = p.rev ? dur - ls : le;
      s.start(now, off);
    } else {
      s.start(now, off, len);
      const real = len / rate;                               // a one-shot fades out over its release at the end
      if (p.play === "oneshot" && p.r > 0 && real > Math.max(p.a, 0.001) + p.d + p.r) {
        g.gain.setValueAtTime(peak * p.s, now + real - p.r); g.gain.linearRampToValueAtTime(0, now + real);
      }
    }
    s.onended = () => { voices = voices.filter(x => x !== v); try { (v.pan || v.g).disconnect(); } catch (e) { /* gone */ } padsOn(); };
    voices.push(v); padsOn();
  }
  function stopVoice(v, fade) {                              // a release (or a quick fade, when it's cut off)
    const c = ac(), now = c.currentTime, r = fade != null ? fade : Math.max(0.005, v.p.r);
    if (v.dying) { if (fade == null) return; }
    v.dying = true; v.held = false;
    const val = envAt(v, now);
    v.g.gain.cancelScheduledValues(now); v.g.gain.setValueAtTime(val, now); v.g.gain.linearRampToValueAtTime(0, now + r);
    try { v.s.stop(now + r + 0.02); } catch (e) { /* not started */ }
  }
  function release(key) {                                    // a key or pad let go
    for (const v of voices) if (v.key === key && v.held) {
      if (sustain) { v.sus = true; continue; }
      if (v.play === "oneshot") { v.held = false; continue; }
      stopVoice(v);
    }
  }
  function sustainOff() { sustain = false; for (const v of voices.slice()) if (v.sus) { v.sus = false; if (v.play !== "oneshot") stopVoice(v); else v.held = false; } }
  function panic() { for (const v of voices.slice()) stopVoice(v, 0.01); sustain = false; held.clear(); drawKeys(); }
  function live(key) {                                       // a knob turned while voices sound
    if (!ctx) return;
    const t = ctx.currentTime;
    if (key === "master.vol") master.gain.setTargetAtTime(dB(K.master.vol), t, 0.01);
    if (key === "pad.vol") padLevels();
    const p = pad();
    for (const v of voices) if (v.pad === K.sel) {
      if (key === "pad.pan" && v.pan) v.pan.pan.setTargetAtTime(p.pan, t, 0.01);
      if (key === "pad.cut" && v.f) v.f.frequency.setTargetAtTime(cutHz(p.cut), t, 0.01);
      if (key === "pad.res" && v.f) v.f.Q.setTargetAtTime(p.res, t, 0.01);
      if (key === "pad.semis" || key === "pad.fine") { v.rate = Math.pow(2, ((p.semis || 0) + (p.fine || 0) / 100 + v.tr) / 12); v.s.playbackRate.setTargetAtTime(v.rate, t, 0.01); }
    }
  }
  function bend(value14) {
    bendCents = ((value14 - 8192) / 8192) * (K.bend || 2) * 100;
    if (!ctx) return;
    for (const v of voices) if (v.s.detune) v.s.detune.setTargetAtTime(bendCents, ctx.currentTime, 0.004);
  }

  /* ---------- notes: from MIDI, the computer's keys, the on-screen keys and the pads ---------- */
  const held = new Set();
  let learnNote = false;
  const padNote = i => clamp(K.pads[i].note != null ? K.pads[i].note : K.base + i, 0, 127);
  const padForNote = n => { for (let i = 0; i < PADS; i++) if (K.pads[i].media && padNote(i) === n) return i; return -1; };
  function noteOn(n, vel) {
    held.add(n); drawKeys();
    if (learnNote) {
      learnNote = false; const was = K.pads.findIndex((p, i) => i !== K.sel && p.note === n);
      pad().note = n; dirty(); drawGrid(); drawEditor();
      toast(`Pad ${K.sel + 1} plays on ${noteName(n)}` + (was >= 0 ? ` (pad ${was + 1} is on it too: the first one with a sound wins)` : ""));
      return;
    }
    const i = K.mode === "keys" ? -1 : padForNote(n);
    if (i >= 0) { fire(i, vel, 0, n); if (follow && K.sel !== i) select(i); return; }
    if (K.mode !== "pads" && pad().media) fire(K.sel, vel, n - (pad().root ?? 60), n);
  }
  function noteOff(n) { held.delete(n); drawKeys(); release(n); }
  function hitPad(i, vel) { held.add("p" + i); fire(i, vel, 0, "p" + i); }
  function letPad(i) { held.delete("p" + i); release("p" + i); }

  /* ---------- MIDI in ---------- */
  let midi = null, midiErr = "", follow = store.get("smpFollow", false);
  let midiWant = store.get("smpMidiIn", "all"), midiCh = store.get("smpMidiCh", 0);   // "all", "none" or a device's name; channel 0 = any
  async function midiStart(ask) {
    if (midi || !navigator.requestMIDIAccess) { if (!navigator.requestMIDIAccess) midiErr = "this browser has no Web MIDI (Chrome, Edge and Firefox do)"; drawTop(); return; }
    if (!ask && !store.get("smpMidiOk", false)) { drawTop(); return; }   // the first time waits for a click (Firefox asks)
    try {
      midi = await navigator.requestMIDIAccess({ sysex: false });
      store.set("smpMidiOk", true); midiErr = "";
      midi.onstatechange = () => { hookInputs(); drawTop(); };
      hookInputs();
    } catch (e) { midiErr = (e && e.message) || String(e); midi = null; }
    drawTop();
  }
  function hookInputs() {
    if (!midi) return;
    for (const inp of midi.inputs.values()) inp.onmidimessage = midiWant === "all" || inp.name === midiWant ? onMidi : null;
  }
  function onMidi(e) {
    const [st, d1 = 0, d2 = 0] = e.data, kind = st & 0xf0, ch = (st & 0x0f) + 1;
    if (st >= 0xf0) return;                                  // clock, sense, sysex
    if (midiCh && ch !== midiCh) return;
    blink();
    if (kind === 0x90 && d2 > 0) noteOn(d1, d2);
    else if (kind === 0x80 || (kind === 0x90 && d2 === 0)) noteOff(d1);
    else if (kind === 0xe0) bend(d1 | (d2 << 7));
    else if (kind === 0xb0) {
      if (learnCC) { for (const c of Object.keys(CC)) if (CC[c] === learnCC) delete CC[c]; CC[d1] = learnCC; store.set("smpCC", CC); toast(`${PARAMS[learnCC].l} follows CC ${d1}`); learnCC = null; drawEditor(); drawTop(); return; }
      if (d1 === 64) { if (d2 >= 64) sustain = true; else sustainOff(); return; }
      if (d1 === 120 || d1 === 123) { panic(); return; }
      const key = CC[d1];
      if (key) { const p = PARAMS[key]; let x = toV(p, d2 / 127); x = Math.round(x / p.step) * p.step; setP(key, +x.toFixed(4)); document.querySelectorAll(`#p-sampler .sm-k[data-k="${key}"]`).forEach(b => b._show()); }
    }
  }
  let blinkT = 0;
  function blink() { const l = $("smLed"); if (!l) return; l.classList.add("on"); clearTimeout(blinkT); blinkT = setTimeout(() => l.classList.remove("on"), 60); }

  /* ---------- the computer's keys ---------- */
  const PIANO_LO = "zsxdcvgbhnjm,l.;/", PIANO_HI = "q2w3er5t6y7ui9o0p[=]";
  const PADKEYS = "1234qwerasdfzxcv";
  let kbMode = store.get("smpKeys", "piano"), kbOct = store.get("smpOct", 4), kbVel = store.get("smpVel", 100);
  const downKeys = new Map();                                 // e.code -> what it started
  function keyNote(k) {
    let i = PIANO_LO.indexOf(k); if (i >= 0) return 12 * (kbOct) + i;
    i = PIANO_HI.indexOf(k); if (i >= 0) return 12 * (kbOct + 1) + i;
    return -1;
  }
  document.addEventListener("keydown", e => {
    if (typeof tab === "undefined" || tab !== "sampler" || /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName) || document.querySelector("dialog[open]")) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (e.key === "Escape") { panic(); return; }
    if (kbMode === "off") return;
    if (e.repeat) { if (downKeys.has(e.code)) e.preventDefault(); return; }
    if (kbMode === "piano") {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") { e.preventDefault(); kbOct = clamp(kbOct + (e.key === "ArrowLeft" ? -1 : 1), 0, 8); store.set("smpOct", kbOct); drawTop(); drawKeys(); return; }
      const n = keyNote(k); if (n < 0 || n > 127) return;
      e.preventDefault(); e.stopImmediatePropagation(); downKeys.set(e.code, n); noteOn(n, kbVel);
    } else {
      const i = PADKEYS.indexOf(k); if (i < 0) return;
      e.preventDefault(); e.stopImmediatePropagation(); downKeys.set(e.code, "p" + i); hitPad(i, kbVel); if (follow) select(i);
    }
  }, true);
  document.addEventListener("keyup", e => {
    const what = downKeys.get(e.code); if (what == null) return;
    downKeys.delete(e.code);
    if (typeof what === "string") letPad(+what.slice(1)); else noteOff(what);
  }, true);
  window.addEventListener("blur", () => { for (const w of downKeys.values()) (typeof w === "string" ? letPad(+w.slice(1)) : noteOff(w)); downKeys.clear(); });

  /* ---------- recording what's played ---------- */
  const rec = { node: null, on: false, chunks: [], buf: null, sr: 44100 };
  async function recToggle() {
    const c = ac(); if (c.state !== "running") await c.resume();
    if (rec.on) { rec.on = false; recFinish(); drawTop(); return; }
    if (!rec.node) {
      const code = `class R extends AudioWorkletProcessor { process(i) { const c = i[0]; if (c && c.length) this.port.postMessage([c[0].slice(), (c[1] || c[0]).slice()]); return true; } } registerProcessor("sm-rec", R);`;
      const url = URL.createObjectURL(new Blob([code], { type: "text/javascript" }));
      await c.audioWorklet.addModule(url); URL.revokeObjectURL(url);
      rec.node = new AudioWorkletNode(c, "sm-rec", { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [2] });
      const z = c.createGain(); z.gain.value = 0; rec.node.connect(z).connect(c.destination);
      rec.node.port.onmessage = ev => { if (rec.on) rec.chunks.push(ev.data); };
      route();
    }
    rec.chunks = []; rec.sr = c.sampleRate; rec.on = true; rec.buf = null; drawTop(); drawRec();
  }
  function recFinish() {
    const n = rec.chunks.reduce((m, c) => m + c[0].length, 0);
    const L = new Float32Array(n), R = new Float32Array(n); let o = 0;
    for (const [a, b] of rec.chunks) { L.set(a, o); R.set(b, o); o += a.length; }
    rec.chunks = [];
    const th = dB(-60); let a = 0, z = n;                    // leave out the silence before the first sound and after the last
    while (a < n && Math.abs(L[a]) < th && Math.abs(R[a]) < th) a++;
    while (z > a && Math.abs(L[z - 1]) < th && Math.abs(R[z - 1]) < th) z--;
    if (z - a < rec.sr * 0.02) { toast("Nothing was recorded: play something while ⏺ is on", true); rec.buf = null; drawRec(); return; }
    a = Math.max(0, a - Math.round(rec.sr * 0.005)); z = Math.min(n, z + Math.round(rec.sr * 0.05));
    rec.buf = { L: L.slice(a, z), R: R.slice(a, z), sr: rec.sr };
    drawRec();
  }
  function wavBlob(b) {
    const n = b.L.length, buf = new ArrayBuffer(44 + n * 4), v = new DataView(buf);
    const s = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
    s(0, "RIFF"); v.setUint32(4, 36 + n * 4, true); s(8, "WAVE"); s(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 2, true);
    v.setUint32(24, b.sr, true); v.setUint32(28, b.sr * 4, true); v.setUint16(32, 4, true); v.setUint16(34, 16, true); s(36, "data"); v.setUint32(40, n * 4, true);
    let o = 44;
    for (let i = 0; i < n; i++) for (const d of [b.L, b.R]) { const x = clamp(d[i], -1, 1); v.setInt16(o, x < 0 ? x * 32768 : x * 32767, true); o += 2; }
    return new Blob([buf], { type: "audio/wav" });
  }
  async function recSend(to, extra = {}) {
    if (!rec.buf) return null;
    const name = (($("smRecName") || {}).value || "").trim() || (K.name || "sampler") + "-take";
    const q = new URLSearchParams({ to, name, format: to === "ref" ? "wav" : ($("smRecFmt") || {}).value || "wav", area: "sampler", ...extra });
    const r = await fetch("/api/assemble/export?" + q, { method: "POST", body: wavBlob(rec.buf) });
    const j = await r.json().catch(() => ({ error: r.statusText }));
    if (!r.ok || j.error) throw new Error(j.error || r.statusText);
    return j;
  }
  let recPlay = null;
  function recListen() {
    if (recPlay) { try { recPlay.stop(); } catch (e) { /* done */ } recPlay = null; drawRec(); return; }
    const c = ac(), b = c.createBuffer(2, rec.buf.L.length, rec.buf.sr); b.getChannelData(0).set(rec.buf.L); b.getChannelData(1).set(rec.buf.R);
    const s = c.createBufferSource(); s.buffer = b; s.connect(c.destination); s.onended = () => { if (recPlay === s) { recPlay = null; drawRec(); } }; s.start(); recPlay = s; drawRec();
  }

  /* ---------- loading sounds onto pads ---------- */
  async function loadPad(i, media, name) {
    const b = await getBuf(media);
    const p = Object.assign(blankPad(), { note: K.pads[i].note, root: K.pads[i].root, choke: K.pads[i].choke, media, name: (name || media).replace(/\.[^.]+$/, ""), dur: b.duration,
      play: b.duration > 2 ? "hold" : "oneshot" });
    K.pads[i] = p; PEAKS.delete(media);
  }
  async function dropOn(i, targets, files) {
    let at = i, n = 0;
    const next = () => { while (at < PADS && n > 0 && K.pads[at].media) at++; return at; };
    try {
      for (const f of files) {
        if (next() >= PADS) break;
        const r = await fetch("/api/assemble/upload?filename=" + encodeURIComponent(f.name), { method: "POST", body: f });
        const j = await r.json().catch(() => ({ error: r.statusText }));
        if (!r.ok || j.error) throw new Error(j.error || r.statusText);
        await loadPad(at, j.media, f.name); n++; at++;
      }
      for (const tg of targets) {
        if (next() >= PADS) break;
        const j = await api("/api/assemble/import", { target: tg });
        await loadPad(at, j.media, (tg.file || tg.path || tg.input || tg.media || j.media).split("/").pop()); n++; at++;
      }
      if (n) { select(i); dirty(); toast(n > 1 ? `${n} sounds on pads ${i + 1}–${i + n}` : `On pad ${i + 1}: ${K.pads[i].name}`); }
    } catch (e) { toast(e.message, true); }
    drawGrid();
  }
  function movePad(a, b, copy) {
    if (a === b) return;
    const keepA = K.pads[a].note, keepB = K.pads[b].note;
    if (copy) K.pads[b] = Object.assign({}, K.pads[a], { note: keepB, mute: false, solo: false });
    else { const t = K.pads[a]; K.pads[a] = K.pads[b]; K.pads[b] = t; K.pads[a].note = keepA; K.pads[b].note = keepB; }   // notes stay with the slots
    select(b); dirty(); padLevels();
  }

  /* ---------- kits ---------- */
  const snap = () => JSON.stringify(K);
  function dirty() {
    store.set("smp", K); drawSaved();
    clearTimeout(dirty.t);
    if (!K.name) return;
    dirty.t = setTimeout(() => api("/api/sampler/save", { name: K.name, kit: JSON.parse(snap()) }).then(r => { savedSig = snap(); kitList(r.kits); drawSaved(); }).catch(e => toast(e.message, true)), 900);
  }
  function drawSaved() { const n = $("smSaved"); if (n) n.textContent = !K.name ? "(not saved: name it to keep it)" : savedSig === snap() ? "saved" : "saving…"; }
  let folders = null;
  function kitList(list) {
    const s = $("smOpen"); if (!s) return;
    s.replaceChildren(el("option", { value: "", text: list.length ? `open… (${list.length})` : "no saved kits" }), ...list.map(p => el("option", { value: p.name, text: p.name })));
  }
  async function loadKits() { try { const r = await api("/api/sampler/kits"); folders = r; kitList(r.kits); } catch (e) { toast(e.message, true); } }
  function setKit(k) { panic(); K = fix(k); savedSig = snap(); store.set("smp", K); padLevels(); route(); warm(); drawAll(); }
  const run = f => async () => { try { await f(); } catch (e) { toast(e.message, true); } };
  async function sfz(folder) {
    if (!K.pads.some(p => p.media)) { toast("no pads with sounds yet: drop sounds on the pads", true); return; }
    const r = await api("/api/sampler/sfz", { kit: K, name: K.name || "kit", folder: folder || "" });
    toast("Saved " + r.path);
  }

  /* ---------- drawing ---------- */
  function select(i) { K.sel = i; store.set("smp", K); drawGrid(); drawEditor(); drawKeys(); }
  const lit = new Map();
  function flash(i) { lit.set(i, performance.now()); const t = $("smPad-" + i); if (t) { t.classList.add("lit"); setTimeout(() => t.classList.remove("lit"), 70); } }
  function padsOn() {
    const on = new Set(voices.map(v => v.pad));
    for (let i = 0; i < PADS; i++) $("smPad-" + i)?.classList.toggle("on", on.has(i));
    if (!padsOn.raf && voices.length) padsOn.raf = requestAnimationFrame(function tick() { drawWave(true); padsOn.raf = voices.length ? requestAnimationFrame(tick) : 0; if (!voices.length) drawWave(); });
  }
  function miniWave(cv, media) {
    const pk = peaks(media); if (!pk) return;
    const w = cv.clientWidth || 80, h = cv.clientHeight || 30, r = devicePixelRatio || 1;
    cv.width = Math.round(w * r); cv.height = Math.round(h * r);
    const g = cv.getContext("2d"); g.setTransform(r, 0, 0, r, 0, 0); g.fillStyle = getComputedStyle(document.body).getPropertyValue("--accent2") || "#88f";
    const n = pk.length / 2;
    for (let x = 0; x < w; x++) { const i = Math.floor(x / w * n), lo = pk[2 * i], hi = pk[2 * i + 1]; g.fillRect(x, h / 2 - hi * h / 2, 1, Math.max(1, (hi - lo) * h / 2)); }
  }
  function drawGrid() {
    const grid = $("smGrid"); if (!grid) return;
    grid.replaceChildren(...K.pads.map((p, i) => {
      const chips = [p.play === "loop" ? "⟲" : p.play === "hold" ? "hold" : "", p.rev ? "rev" : "", p.choke ? "ch" + p.choke : "", p.mono ? "mono" : "",
        p.mute ? "M" : "", p.solo ? "S" : "", p.semis ? (p.semis > 0 ? "+" : "") + p.semis : ""].filter(Boolean).join(" ");
      const t = el("div", { class: "sm-pad" + (p.media ? "" : " empty") + (i === K.sel ? " sel" : "") + (p.media && !audible(i) ? " quiet" : "") + (i === K.sel && K.mode !== "pads" && p.media ? " keys" : ""),
        id: "smPad-" + i, draggable: !!p.media,
        title: p.media ? `${p.name}\nClick to play (higher up = harder), drag onto another pad to swap (Alt: copy), drag out to the clip bin, Assemble or Start from. Right-click: more.`
          : "Drop a sound here: from the clip bin, Gallery, Queue, Find, or files from your desktop (several fill the next pads)" },
        el("div", { class: "hit" }),
        p.media ? el("canvas") : null,
        el("span", { class: "n", text: String(i + 1) }),
        el("span", { class: "k", text: K.mode === "keys" ? "" : noteName(padNote(i)) + (p.note != null ? "•" : ""), title: p.note != null ? "a note of its own (learned)" : "" }),
        el("span", { class: "nm", text: p.media ? p.name : "" }),
        el("span", { class: "ch", text: chips }));
      t.addEventListener("pointerdown", e => {
        if (e.button !== 0) return;
        if (!p.media) { select(i); return; }
        const r = t.getBoundingClientRect(), vel = Math.round(127 * (0.35 + 0.65 * clamp(1 - (e.clientY - r.top) / r.height, 0, 1)));
        if (K.sel !== i) select(i);
        if (e.shiftKey) return;                              // Shift: just pick it
        t._down = true; hitPad(i, vel);
      });
      const up = () => { if (t._down) { t._down = false; letPad(i); } };
      t.addEventListener("pointerup", up); t.addEventListener("pointerleave", up); t.addEventListener("dragstart", up);
      t.addEventListener("contextmenu", e => { e.preventDefault(); padMenu(i, e); });
      t.addEventListener("dragstart", e => {
        if (!p.media) return e.preventDefault();
        e.dataTransfer.setData(PAD_TYPE, String(i)); e.dataTransfer.setData(CLIP_TYPE, JSON.stringify({ media: p.media })); e.dataTransfer.setData("text/plain", p.name);
        e.dataTransfer.effectAllowed = "copyMove";
      });
      const ok = e => [...e.dataTransfer.types].some(x => x === CLIP_TYPE || x === PAD_TYPE || x === "Files");
      t.addEventListener("dragover", e => { if (!ok(e)) return; e.preventDefault(); e.dataTransfer.dropEffect = e.altKey || ![...e.dataTransfer.types].includes(PAD_TYPE) ? "copy" : "move"; t.classList.add("drop"); });
      t.addEventListener("dragleave", () => t.classList.remove("drop"));
      t.addEventListener("drop", e => {
        e.preventDefault(); t.classList.remove("drop");
        const from = e.dataTransfer.getData(PAD_TYPE);
        if (from !== "") { movePad(+from, i, e.altKey); return; }
        let tg = []; try { const raw = e.dataTransfer.getData(CLIP_TYPE); if (raw) tg = [JSON.parse(raw)]; } catch (err) { /* not ours */ }
        dropOn(i, tg, tg.length ? [] : [...e.dataTransfer.files]);
      });
      return t;
    }));
    requestAnimationFrame(() => K.pads.forEach((p, i) => { const cv = $("smPad-" + i)?.querySelector("canvas"); if (cv && p.media) miniWave(cv, p.media); }));
    padsOn();
  }
  function padMenu(i, e) {
    const p = K.pads[i];
    select(i);
    smallMenu(e, [
      ["Load a sound from disk…", run(async () => { const f = await pickPath({ kind: "file", title: `A sound for pad ${i + 1}` }); if (f) await dropOn(i, [{ path: f }], []); })],
      ...(p.media ? [
        ["Learn its note: play a key or pad on your controller", () => { learnNote = true; drawEditor(); toast(`Play the note pad ${i + 1} should be on…`); }],
        ...(p.note != null ? [["Back to its usual note (" + noteName(K.base + i) + ")", () => { p.note = null; dirty(); drawGrid(); }]] : []),
        ["Open in…", () => openMenu({ media: p.media }, { getBoundingClientRect: () => ({ left: e.clientX, right: e.clientX, top: e.clientY, bottom: e.clientY }) })],
        ["🎞 Into the clip bin", () => binAdd({ media: p.media })],
        ["🔎 Find sounds like it", run(async () => { const j = await api("/api/input/from-target", { target: { media: p.media } }); findFrom({ input: j.file }, p.name); })],
        ["Clear the pad", () => { const n = p.note; K.pads[i] = Object.assign(blankPad(), { note: n }); dirty(); drawGrid(); drawEditor(); }]] : [])]);
  }

  let wv = { t0: 0, t1: 0, media: null };                  // the editor's waveform view (seconds)
  function drawWave(heads) {
    const cv = $("smWaveC"); if (!cv) return;
    const p = pad(), w = cv.clientWidth, h = cv.clientHeight, r = devicePixelRatio || 1;
    if (cv.width !== Math.round(w * r) || cv.height !== Math.round(h * r)) { cv.width = Math.round(w * r); cv.height = Math.round(h * r); }
    const g = cv.getContext("2d"); g.setTransform(r, 0, 0, r, 0, 0); g.clearRect(0, 0, w, h);
    const css = getComputedStyle(document.body), col = n => css.getPropertyValue(n).trim();
    if (!p.media) { g.fillStyle = col("--dim"); g.font = "12px sans-serif"; g.fillText("Drop a sound on this pad (or on the waveform)", 10, h / 2); return; }
    const b = bufNow(p.media, false); if (!b) { g.fillStyle = col("--dim"); g.fillText("loading…", 10, h / 2); return; }
    const dur = b.duration;
    if (wv.media !== p.media || !(wv.t1 > wv.t0)) wv = { t0: 0, t1: dur, media: p.media };
    const X = t => (t - wv.t0) / (wv.t1 - wv.t0) * w;
    const d = b.getChannelData(0), spp = (wv.t1 - wv.t0) * b.sampleRate / w;
    g.fillStyle = col("--accent2");
    for (let x = 0; x < w; x++) {
      const a = Math.floor((wv.t0 * b.sampleRate) + x * spp), z = Math.min(d.length, Math.floor(a + spp) + 1); let lo = 0, hi = 0;
      for (let j = Math.max(0, a); j < z; j += Math.max(1, Math.floor(spp / 64))) { const v = d[j]; if (v < lo) lo = v; if (v > hi) hi = v; }
      g.fillRect(x, h / 2 - hi * h / 2, 1, Math.max(1, (hi - lo) * h / 2));
    }
    const st = p.start || 0, en = p.end || dur;
    g.fillStyle = "rgba(0,0,0,.55)"; g.fillRect(0, 0, Math.max(0, X(st)), h); g.fillRect(X(en), 0, w - X(en), h);
    if (p.play === "loop") { const ls = p.ls || st, le = p.le || en; g.fillStyle = "rgba(80,255,140,.12)"; g.fillRect(X(ls), 0, X(le) - X(ls), h);
      g.strokeStyle = col("--ok"); g.setLineDash([3, 3]); for (const t of [ls, le]) { g.beginPath(); g.moveTo(X(t) + .5, 0); g.lineTo(X(t) + .5, h); g.stroke(); } g.setLineDash([]); }
    g.strokeStyle = col("--text"); g.lineWidth = 2;
    for (const t of [st, en]) { g.beginPath(); g.moveTo(X(t), 0); g.lineTo(X(t), h); g.stroke(); }
    g.fillStyle = col("--text"); g.fillRect(X(st), 0, 8, 8); g.fillRect(X(en) - 8, 0, 8, 8);
    g.lineWidth = 1; g.font = "10.5px monospace";
    g.fillText(`${fmtS(st)} → ${fmtS(en)}  (${fmtS(en - st)} of ${fmtS(dur)})` + (wv.t0 > 0 || wv.t1 < dur ? "   wheel: zoom · double-click: all" : "   wheel: zoom"), 6, h - 5);
    if (heads && ctx) {                                      // where each sounding voice of this pad is
      g.strokeStyle = col("--ok");
      for (const v of voices) if (v.pad === K.sel && !v.dying) {
        let pos = (ctx.currentTime - v.t0) * v.rate;
        if (v.play === "loop") { const ls = clamp(p.ls || v.st, v.st, v.en), le = clamp(p.le || v.en, ls + 0.005, v.en); const intro = ls - v.st; if (pos > intro + (le - ls)) pos = intro + ((pos - intro) % (le - ls)); }
        if (pos < 0 || pos > v.len) continue;
        const t = v.rev ? v.en - pos : v.st + pos; g.beginPath(); g.moveTo(X(t) + .5, 0); g.lineTo(X(t) + .5, h); g.stroke();
      }
    }
  }
  function hookWave() {
    const box = $("smWave"), cv = $("smWaveC");
    let drag = null;
    const T = x => wv.t0 + (x / cv.clientWidth) * (wv.t1 - wv.t0);
    cv.addEventListener("pointerdown", e => {
      const p = pad(); if (!p.media) return; const b = bufNow(p.media, false); if (!b) return;
      const x = e.offsetX, dur = b.duration, Xo = t => (t - wv.t0) / (wv.t1 - wv.t0) * cv.clientWidth;
      const hs = [["start", p.start || 0], ["end", p.end || dur]];
      if (p.play === "loop") hs.push(["ls", p.ls || p.start || 0], ["le", p.le || p.end || dur]);
      let best = null; for (const [k, t] of hs) { const dx = Math.abs(Xo(t) - x); if (dx < 10 && (!best || dx < best[1])) best = [k, dx]; }
      drag = best ? best[0] : (e.shiftKey ? "end" : "start");          // nothing near: a click moves the start (Shift: the end)
      cv.setPointerCapture(e.pointerId); move(e);
    });
    const move = e => {
      if (!drag) return; const p = pad(), b = bufNow(p.media, false); if (!b) return;
      const dur = b.duration, t = clamp(T(e.offsetX), 0, dur), st = p.start || 0, en = p.end || dur;
      if (drag === "start") p.start = +clamp(t, 0, en - 0.002).toFixed(4);
      if (drag === "end") p.end = +clamp(t, st + 0.002, dur).toFixed(4);
      if (drag === "ls") p.ls = +clamp(t, st, (p.le || en) - 0.005).toFixed(4);
      if (drag === "le") p.le = +clamp(t, (p.ls || st) + 0.005, en).toFixed(4);
      drawWave();
    };
    cv.addEventListener("pointermove", move);
    cv.addEventListener("pointerup", () => { if (drag) { drag = null; dirty(); } });
    cv.addEventListener("wheel", e => {
      const p = pad(); const b = p.media && bufNow(p.media, false); if (!b) return; e.preventDefault();
      const dur = b.duration, span = wv.t1 - wv.t0;
      if (e.shiftKey) { const d = Math.sign(e.deltaY || e.deltaX) * span * 0.15; wv.t0 = clamp(wv.t0 + d, 0, dur - span); wv.t1 = wv.t0 + span; }
      else { const t = T(e.offsetX), k = e.deltaY > 0 ? 1.25 : 0.8, ns = clamp(span * k, 0.005, dur); wv.t0 = clamp(t - (t - wv.t0) * ns / span, 0, dur - ns); wv.t1 = wv.t0 + ns; }
      drawWave();
    }, { passive: false });
    cv.addEventListener("dblclick", () => { wv.t1 = 0; drawWave(); });
    box.addEventListener("dragover", e => { if ([...e.dataTransfer.types].some(x => x === CLIP_TYPE || x === "Files")) e.preventDefault(); });
    box.addEventListener("drop", e => {
      e.preventDefault(); const from = e.dataTransfer.getData(PAD_TYPE); if (from !== "") { movePad(+from, K.sel, true); return; }
      let tg = []; try { const raw = e.dataTransfer.getData(CLIP_TYPE); if (raw) tg = [JSON.parse(raw)]; } catch (err) { /* not ours */ }
      dropOn(K.sel, tg, tg.length ? [] : [...e.dataTransfer.files]);
    });
    new ResizeObserver(() => drawWave()).observe(box);
  }
  function drawEditor() {
    const ed = $("smEd"); if (!ed) return;
    const p = pad(), i = K.sel;
    const sel = (opts, val, on, title) => { const s = el("select", { title }, ...opts.map(([v, l]) => el("option", { value: v, text: l }))); s.value = String(val); s.addEventListener("change", () => on(s.value)); return s; };
    const tog = (cls, text, on, title, fn) => el("button", { type: "button", class: cls, text, title, "aria-pressed": String(!!on), onclick: fn });
    const notes = Array.from({ length: 128 }, (_, n) => [n, noteName(n)]);
    const name = el("input", { class: "nm", value: p.name, placeholder: "empty pad", maxlength: 60, onchange: e => { p.name = e.target.value.trim(); dirty(); drawGrid(); } });
    $("smEdHead").replaceChildren(
      el("b", { class: "mono", text: `Pad ${i + 1}` }), name,
      tog("", "▶", false, "play it (as a hit at velocity 100)", () => { hitPad(i, 100); setTimeout(() => letPad(i), 400); }),
      tog("m", "M", p.mute, "mute this pad", () => { p.mute = !p.mute; padLevels(); dirty(); drawGrid(); drawEditor(); }),
      tog("s", "S", p.solo, "solo: only soloed pads sound", () => { p.solo = !p.solo; padLevels(); dirty(); drawGrid(); drawEditor(); }),
      el("label", { class: "sm-f", title: "the note this pad is on (Pads and Both modes)" }, "Note",
        sel(notes, padNote(i), v => { p.note = +v === K.base + i ? null : +v; dirty(); drawGrid(); drawKeys(); }, "")),
      Object.assign(tog("", learnNote ? "…play a note" : "Learn", learnNote, "play a key or pad on your controller to put this pad on it", () => { learnNote = !learnNote; drawEditor(); }), { id: "smLearn" }),
      el("label", { class: "sm-f", title: "Pitched and Both modes: the key that plays this pad at its own pitch" }, "Root",
        sel(notes, p.root ?? 60, v => { p.root = +v; dirty(); drawKeys(); }, "")),
      el("span", { class: "sp" }),
      ...(p.media ? [tog("", "🎞", false, "put this pad's sound in the clip bin", () => binAdd({ media: p.media })),
        tog("", "↗", false, "open in…", e => openMenu({ media: p.media }, e.currentTarget))] : []));
    const knobs = $("smKnobs");
    knobs.replaceChildren(
      el("div", { class: "sm-sect", text: "Sound" }),
      knob("pad.vol"), knob("pad.pan"), knob("pad.semis"), knob("pad.fine"), knob("pad.vel"),
      el("label", { class: "sm-k", title: "One-shot: plays to its end whatever you do. Hold: plays while the key is down, then releases. Loop: loops the green part while the key is down." },
        el("span", { text: "Play" }), sel(PLAY, p.play, v => { p.play = v; if (v === "loop" && !(p.le > p.ls)) { p.ls = p.start || 0; p.le = p.end || p.dur; } dirty(); drawGrid(); drawWave(); }, ""), el("span")),
      el("label", { class: "sm-k", title: "play it backwards" }, el("span", { text: "Reverse" }),
        el("input", { type: "checkbox", checked: p.rev, onchange: e => { p.rev = e.target.checked; if (p.media) getBuf(p.media, true); dirty(); drawGrid(); } }), el("span")),
      el("label", { class: "sm-k", title: "a new hit on this pad stops the last one" }, el("span", { text: "Mono" }),
        el("input", { type: "checkbox", checked: p.mono, onchange: e => { p.mono = e.target.checked; dirty(); drawGrid(); } }), el("span")),
      el("label", { class: "sm-k", title: "hitting a pad stops the other pads in its group (open and closed hi-hats)" }, el("span", { text: "Choke group" }),
        sel([[0, "none"], ...Array.from({ length: 8 }, (_, g) => [g + 1, String(g + 1)])], p.choke, v => { p.choke = +v; dirty(); drawGrid(); }, ""), el("span")),
      el("div", { class: "sm-sect", text: "Envelope" }),
      knob("pad.a"), knob("pad.d"), knob("pad.s"), knob("pad.r"),
      el("div", { class: "sm-sect", text: "Filter" }),
      el("label", { class: "sm-k" }, el("span", { text: "Type" }), sel([["off", "off"], ["lp", "low-pass"], ["hp", "high-pass"]], p.filter, v => { p.filter = v; dirty(); }, ""), el("span")),
      knob("pad.cut"), knob("pad.res"));
    markLearn(); drawWave();
  }
  const KB_LO = () => clamp(store.get("smpKbLo", 36), 0, 67);
  function drawKeys() {
    const cv = $("smKbC"); if (!cv) return;
    const w = cv.clientWidth, h = cv.clientHeight, r = devicePixelRatio || 1;
    if (cv.width !== Math.round(w * r) || cv.height !== Math.round(h * r)) { cv.width = Math.round(w * r); cv.height = Math.round(h * r); }
    const g = cv.getContext("2d"); g.setTransform(r, 0, 0, r, 0, 0); g.clearRect(0, 0, w, h);
    const css = getComputedStyle(document.body), col = n => css.getPropertyValue(n).trim();
    const L = kbLayout(w, h);
    const padNotes = new Set(); if (K.mode !== "keys") K.pads.forEach((p, i) => { if (p.media) padNotes.add(padNote(i)); });
    const root = K.mode !== "pads" && pad().media ? (pad().root ?? 60) : -1;
    for (const blackPass of [false, true]) for (const k of L) {
      if (k.black !== blackPass) continue;
      const on = held.has(k.n);
      g.fillStyle = on ? col("--ok") : padNotes.has(k.n) ? (k.black ? "#5a2a6a" : "#d9b8ef") : k.black ? "#111" : "#e8e8e8";
      g.fillRect(k.x, 0, k.w, k.h); g.strokeStyle = "#555"; g.strokeRect(k.x + .5, .5, k.w - 1, k.h - 1);
      if (k.n === root) { g.fillStyle = col("--accent"); g.fillRect(k.x + 2, k.h - 7, k.w - 4, 5); }
      if (!k.black && k.n % 12 === 0) { g.fillStyle = "#333"; g.font = "9.5px monospace"; g.fillText(noteName(k.n), k.x + 2, h - 10); }
    }
  }
  function kbLayout(w, h) {
    const lo = KB_LO(), hi = lo + 60, whites = []; const isB = n => [1, 3, 6, 8, 10].includes(n % 12);
    for (let n = lo; n <= hi; n++) if (!isB(n)) whites.push(n);
    const ww = w / whites.length, out = [];
    whites.forEach((n, i) => out.push({ n, x: i * ww, w: ww, h, black: false }));
    for (let n = lo; n <= hi; n++) if (isB(n)) { const i = whites.indexOf(n - 1); if (i >= 0) out.push({ n, x: (i + 1) * ww - ww * 0.32, w: ww * 0.64, h: h * 0.6, black: true }); }
    return out;
  }
  function hookKeys() {
    const cv = $("smKbC"); let cur = -1;
    const at = e => { const L = kbLayout(cv.clientWidth, cv.clientHeight); const hit = L.filter(k => k.black).find(k => e.offsetX >= k.x && e.offsetX < k.x + k.w && e.offsetY < k.h) || L.find(k => !k.black && e.offsetX >= k.x && e.offsetX < k.x + k.w); return hit ? hit.n : -1; };
    const vel = e => Math.round(127 * (0.35 + 0.65 * clamp(e.offsetY / cv.clientHeight, 0, 1)));   // lower down the key = harder, as a real one
    cv.addEventListener("pointerdown", e => { cv.setPointerCapture(e.pointerId); cur = at(e); if (cur >= 0) noteOn(cur, vel(e)); });
    cv.addEventListener("pointermove", e => { if (cur < 0 || !(e.buttons & 1)) return; const n = at(e); if (n !== cur) { noteOff(cur); cur = n; if (n >= 0) noteOn(n, vel(e)); } });
    const up = () => { if (cur >= 0) noteOff(cur); cur = -1; };
    cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
    new ResizeObserver(drawKeys).observe(cv);
  }
  function drawTop() {
    const s = $("smMidi"); if (!s) return;
    const opts = [["all", "every input"], ...(midi ? [...midi.inputs.values()].map(x => [x.name, x.name]) : []), ["none", "none"]];
    if (midiWant !== "all" && midiWant !== "none" && !opts.some(o => o[0] === midiWant)) opts.splice(1, 0, [midiWant, midiWant + " (not plugged in)"]);
    s.replaceChildren(...opts.map(([v, l]) => el("option", { value: v, text: l }))); s.value = midiWant;
    s.hidden = !midi; $("smMidiGo").hidden = !!midi || !navigator.requestMIDIAccess;
    $("smMidiNote").textContent = midiErr ? "MIDI: " + midiErr : !midi ? "" : [...midi.inputs.values()].length ? "" : "no MIDI inputs found: plug one in";
    $("smCh").value = String(midiCh);
    document.querySelectorAll("#smMode button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.m === K.mode)));
    $("smBase").value = String(K.base); $("smBend").value = K.bend; $("smLimit").checked = !!K.master.limit;
    $("smKeys").value = kbMode; $("smOct").textContent = kbMode === "piano" ? `octave ${kbOct} (← →)` : kbMode === "pads" ? "1-4 Q-R A-F Z-V" : "";
    $("smFollow").checked = follow; $("smKit").value = K.name || "";
    const rb = $("smRec"); rb.setAttribute("aria-pressed", String(rec.on)); rb.textContent = rec.on ? "⏺ Recording… (stop)" : "⏺ Rec";
    const mv = $("smMasterBox"); if (mv) mv.replaceChildren(knob("master.vol"));
    drawSaved();
  }
  function drawRec() {
    const box = $("smRecBox"); if (!box) return;
    if (!rec.buf) { box.hidden = true; return; }
    box.hidden = false;
    const secs = rec.buf.L.length / rec.buf.sr;
    box.replaceChildren(
      el("b", { text: "Recorded" }), el("span", { class: "mono", text: fmtS(secs) }),
      el("button", { type: "button", text: recPlay ? "■" : "▶", title: "listen", onclick: recListen }),
      el("label", { class: "sm-f" }, "Name", el("input", { id: "smRecName", placeholder: (K.name || "sampler") + "-take", style: "width:150px" })),
      el("select", { id: "smRecFmt", title: "format" }, ...[["wav", "WAV"], ["flac", "FLAC"], ["ogg", "OGG"], ["mp3", "MP3"]].map(([v, l]) => el("option", { value: v, text: l }))),
      el("button", { type: "button", text: "💾 Save", title: "save into ~/soundmon-gallery/sampler/exports", onclick: run(async () => { const j = await recSend("file"); if (j) toast("Saved " + j.path); }) }),
      el("button", { type: "button", text: "🎞 To clip bin", onclick: run(async () => { const j = await recSend("bin", { bin: BIN.name }); if (j) { binSet(j); toast(`In the clip bin “${BIN.name}” as ${j.file}`); } }) }),
      el("button", { type: "button", text: "➕ Onto a pad", title: "resample: put it on the next empty pad", onclick: run(async () => {
        const i = K.pads.findIndex(p => !p.media); if (i < 0) { toast("every pad has a sound: clear one first", true); return; }
        const r = await fetch("/api/assemble/upload?filename=" + encodeURIComponent((($("smRecName").value.trim()) || "resampled") + ".wav"), { method: "POST", body: wavBlob(rec.buf) });
        const j = await r.json(); if (j.error) throw new Error(j.error); await loadPad(i, j.media, $("smRecName").value.trim() || "resampled"); select(i); dirty(); toast(`On pad ${i + 1}`); }) }),
      el("button", { type: "button", text: "✨ Make one like this", title: "make it the form's reference sound (Start from)", onclick: run(async () => {
        const j = await recSend("ref"); if (!j) return; readForm(); if (!["sfx", "music", "song"].includes(eng)) { eng = "sfx"; store.set("engine", eng); }
        cur().ref = j.file; store.set("form", F); showEngine(); toast("It's the reference now: describe what to make from it and press RENDER."); }) }),
      el("button", { type: "button", text: "🔎 Find similar", onclick: run(async () => { const j = await recSend("ref"); if (j) findFrom({ input: j.file }, "sampler take"); }) }),
      el("button", { type: "button", text: "✕", title: "throw it away", onclick: () => { rec.buf = null; drawRec(); } }));
  }
  function tickMeter() {
    if (!shown) { tickMeter.on = false; return; }
    tickMeter.on = true;
    if (meter) {
      const a = tickMeter.a || (tickMeter.a = new Float32Array(meter.fftSize)); meter.getFloatTimeDomainData(a);
      let pk = 0; for (let i = 0; i < a.length; i++) pk = Math.max(pk, Math.abs(a[i]));
      tickMeter.pk = Math.max(pk, (tickMeter.pk || 0) * 0.9);
      const m = $("smMeter"); if (m) { m.firstChild.style.width = Math.min(100, tickMeter.pk * 100) + "%"; m.classList.toggle("hot", tickMeter.pk > 0.98); }
    }
    requestAnimationFrame(tickMeter);
  }
  function drawAll() { drawTop(); drawGrid(); drawEditor(); drawKeys(); drawRec(); }

  /* ---------- the tab ---------- */
  function build() {
    const pane = $("p-sampler");
    const opt = (v, l) => el("option", { value: v, text: l });
    pane.replaceChildren(
      el("div", { class: "paneh sm-top" },
        el("h2", { text: "Sampler" }),
        el("span", { class: "sm-seg", id: "smMode", title: "how notes are played" },
          ...[["pads", "Pads", "each pad plays on its own note, at its own pitch"], ["keys", "Pitched", "every key plays the selected pad, pitched from its root"],
            ["both", "Both", "keys with a pad play it; the others play the selected pad pitched (kaleidotron's way)"]]
            .map(([m, l, t]) => el("button", { type: "button", "data-m": m, text: l, title: t, onclick: () => { K.mode = m; dirty(); drawTop(); drawGrid(); drawKeys(); } }))),
        el("label", { class: "sm-f", title: "the note pad 1 is on; the others follow it up" }, "Pads from",
          el("select", { id: "smBase", onchange: e => { K.base = +e.target.value; dirty(); drawGrid(); drawKeys(); drawEditor(); } }, ...Array.from({ length: 113 }, (_, n) => opt(n, noteName(n))))),
        el("label", { class: "sm-f", title: "how far the pitch wheel bends, in semitones" }, "Bend ±", el("input", { id: "smBend", type: "number", min: 0, max: 24, step: 1, onchange: e => { K.bend = clamp(+e.target.value || 0, 0, 24); dirty(); } })),
        el("span", { id: "smMasterBox", style: "min-width:240px" }),
        el("label", { class: "sm-f c", title: "keep the sum from clipping (a fast limiter on the master)" }, el("input", { type: "checkbox", id: "smLimit", onchange: e => { K.master.limit = e.target.checked; route(); dirty(); } }), "limit"),
        el("span", { class: "sm-meter", id: "smMeter", title: "the master's level" }, el("i")),
        el("button", { type: "button", id: "smRec", text: "⏺ Rec", title: "record what you play, then save it, put it in the clip bin or onto a pad", onclick: run(recToggle) }),
        el("button", { type: "button", text: "■ Panic", title: "stop every sound (Esc)", onclick: panic })),
      el("div", { class: "paneh sm-top" },
        el("span", { class: "sm-f" }, el("span", { class: "sm-led", id: "smLed", title: "lights up when MIDI comes in" }), "MIDI in"),
        el("button", { type: "button", id: "smMidiGo", text: "🎹 Connect MIDI", title: "let the page use your MIDI controllers (the browser asks the first time)", onclick: () => midiStart(true) }),
        el("select", { id: "smMidi", hidden: true, title: "the controller to listen to", onchange: e => { midiWant = e.target.value; store.set("smpMidiIn", midiWant); hookInputs(); } }),
        el("label", { class: "sm-f" }, "Channel", el("select", { id: "smCh", onchange: e => { midiCh = +e.target.value; store.set("smpMidiCh", midiCh); } }, opt(0, "any"), ...Array.from({ length: 16 }, (_, c) => opt(c + 1, String(c + 1))))),
        el("label", { class: "sm-f c", title: "a pad played from the controller becomes the selected one" }, el("input", { type: "checkbox", id: "smFollow", onchange: e => { follow = e.target.checked; store.set("smpFollow", follow); } }), "follow"),
        el("span", { id: "smMidiNote", class: "dim" }),
        el("span", { class: "sp" }),
        el("label", { class: "sm-f", title: "what the computer's keys play while this tab is open" }, "Computer keys",
          el("select", { id: "smKeys", onchange: e => { kbMode = e.target.value; store.set("smpKeys", kbMode); drawTop(); } }, opt("piano", "piano (Z, Q rows)"), opt("pads", "pads (4×4)"), opt("off", "off"))),
        el("span", { id: "smOct", class: "dim mono" })),
      el("div", { class: "paneh sm-top" },
        el("label", { class: "sm-f" }, "Kit", el("input", { id: "smKit", placeholder: "name it to keep it", maxlength: 64, onchange: e => { K.name = e.target.value.trim().replace(/[^\w.-]+/g, "-"); e.target.value = K.name; dirty(); } })),
        el("span", { id: "smSaved", class: "dim" }),
        el("select", { id: "smOpen", title: "open a saved kit", onchange: async e => { const n = e.target.value; e.target.value = ""; if (!n) return;
          try { const r = await api("/api/sampler/kit?name=" + encodeURIComponent(n)); setKit(r.kit); toast("Opened " + n); } catch (err) { toast(err.message, true); } } }),
        el("button", { type: "button", text: "New", title: "an empty kit (the current one stays saved if it has a name)", onclick: () => {
          if (!K.name && K.pads.some(p => p.media) && !confirm("This kit has no name, so it isn't saved. Start a new one anyway?")) return; setKit(blankKit()); } }),
        el("button", { type: "button", text: "Delete", title: "delete this saved kit", onclick: run(async () => { if (!K.name || !confirm(`Delete the saved kit “${K.name}”?`)) return;
          kitList((await api("/api/sampler/delete", { name: K.name })).kits); K.name = ""; dirty(); drawTop(); }) }),
        el("span", { class: "sp" }),
        el("button", { type: "button", text: "Export SFZ", title: "the kit as an SFZ instrument (a .sfz and its samples) for any SFZ sampler or DAW: into ~/soundmon-gallery/sampler/exports", onclick: run(() => sfz("")) }),
        el("button", { type: "button", text: "SFZ as…", title: "pick the folder to export the SFZ into", onclick: run(async () => { const d = await pickPath({ kind: "dir", start: folders?.exports || "", title: "Export the SFZ into…" }); if (d) await sfz(d); }) })),
      el("div", { class: "sm-rec", id: "smRecBox", hidden: true }),
      el("div", { class: "sm-body" },
        el("div", {}, el("div", { class: "sm-grid", id: "smGrid" }),
          el("p", { class: "sm-hint", text: "Drop sounds on the pads. Click a pad to play it (higher up = harder); Shift+click just selects it. Right-click a pad or a knob for more." })),
        el("div", { class: "sm-ed", id: "smEd" },
          el("div", { class: "sm-edh", id: "smEdHead" }),
          el("div", { class: "sm-wave", id: "smWave", title: "drag the white edges for the start and end (green dashes: the loop); a click moves the start, Shift+click the end" }, el("canvas", { id: "smWaveC" })),
          el("div", { class: "sm-knobs", id: "smKnobs" }))),
      el("div", { class: "paneh sm-top" },
        el("span", { class: "sm-f" }, "Keyboard"),
        el("button", { type: "button", text: "◀", title: "an octave down", onclick: () => { store.set("smpKbLo", KB_LO() - 12); drawKeys(); } }),
        el("button", { type: "button", text: "▶", title: "an octave up", onclick: () => { store.set("smpKbLo", KB_LO() + 12); drawKeys(); } }),
        el("span", { class: "dim", text: "lilac keys have a pad; the bar marks the selected pad's root" })),
      el("div", { class: "sm-kb" }, el("canvas", { id: "smKbC" })));
    pane.addEventListener("pointerdown", () => { if (ctx && ctx.state !== "running") ctx.resume(); });
    built = true; hookWave(); hookKeys(); loadKits(); savedSig = K.name ? snap() : "";
    drawAll(); warm();
    midiStart(false);
  }
  window.smpShow = on => {
    shown = !!on;
    if (!on) return;
    if (!built) build(); else { drawGrid(); drawKeys(); drawWave(); }
    if (!tickMeter.on) tickMeter();
  };
  window.smpDebug = { kit: () => K, voices: () => voices.map(v => ({ pad: v.pad, key: v.key, rate: +v.rate.toFixed(4), held: v.held })), noteOn, noteOff, midi: data => onMidi({ data }), rec: () => rec, recToggle, CC: () => CC };
  if (typeof tab !== "undefined" && tab === "sampler") smpShow(true);
})();
