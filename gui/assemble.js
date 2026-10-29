/* soundmon-gui: ASSEMBLE - an 8-lane sketchpad for sounds.
   Drag sounds onto lanes (from the clip bin, Gallery, Queue, Find, Sets or the desktop), move and trim
   them on a musical grid, give each clip volume, pitch (sampler-style: it changes the length too, unless
   "keep length"), stretch and reverse, then shape the sum on the master: pitch, filter cutoff and amp
   envelopes drawn as breakpoints. Everything is mixed in the page (Web Audio): the preview and the export
   are the same offline render, so what you hear is what you save. The server keeps the sounds dropped
   here, makes stretched copies with ffmpeg, keeps projects and writes exports.
   Loaded after index.html's script, whose helpers it uses ($, el, api, store, toast, pickPath, …). */
(() => {
  document.head.append(Object.assign(document.createElement("style"), { textContent: `
#p-assemble { gap: 8px; }
.as-top { flex-wrap: wrap; gap: 6px 10px; align-items: center; }
.as-top h2 { margin: 0 6px 0 0; }
.as-top button { padding: 3px 10px; font-size: 12px; }
.as-play { min-width: 76px; background: var(--accent); color: var(--on-accent); border-color: var(--accent); font-weight: 700; }
#asLoop[aria-pressed="true"] { color: var(--accent); border-color: var(--accent); }
.as-f { display: inline-flex; align-items: center; gap: 5px; font-size: 12px; color: var(--dim); }
.as-f input { width: 62px; padding: 3px 5px; } .as-f select { width: auto; padding: 3px 5px; }
#asName, #asFile { width: 170px; } #asOpen, #asFmt { width: auto; padding: 3px 5px; }
.as-f.c input { width: auto; }
.as-g { display: inline-flex; gap: 2px; }
.as-props { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; padding: 8px 10px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--panel); min-height: 44px; font-size: 12px; }
.as-cname { font: 600 12px var(--mono); max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.as-p { display: inline-flex; align-items: center; gap: 5px; } .as-p > span:first-child { color: var(--dim); }
.as-p input[type=range] { width: 110px; } .as-p input[type=number] { width: 64px; padding: 2px 4px; } .as-p input[type=checkbox] { width: auto; }
.as-p button { padding: 2px 8px; font-size: 12px; }
.as-tl { display: grid; grid-template-columns: ${156}px minmax(0, 1fr); border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; background: var(--bg); user-select: none; }
.as-heads { border-right: 1px solid var(--line); background: var(--panel); }
.as-lh { display: flex; align-items: center; gap: 4px; padding: 0 6px; border-bottom: 1px solid var(--line); box-sizing: border-box; font-size: 12px; }
.as-lh b { min-width: 14px; color: var(--dim); font-family: var(--mono); }
.as-lh button { padding: 0 5px; font-size: 11px; line-height: 18px; } .as-lh input[type=range] { flex: 1; min-width: 0; padding: 0; }
.as-lh .m[aria-pressed="true"] { background: var(--bad); color: #000; border-color: var(--bad); }
.as-lh .s[aria-pressed="true"] { background: var(--warn); color: #000; border-color: var(--warn); }
.as-lh.ruler { color: var(--dim); font-size: 10.5px; } .as-lh.master b { min-width: 0; margin-right: 4px; color: var(--ok); }
.as-lh.env { flex-direction: column; align-items: stretch; justify-content: center; gap: 2px; }
.as-lh.env label { display: flex; gap: 5px; align-items: center; } .as-lh.env label input { width: auto; }
.as-lh.env .row { display: flex; gap: 4px; align-items: center; } .as-lh.env select { width: auto; padding: 0 3px; font-size: 11px; }
.as-lh.env .clr { align-self: flex-start; font-size: 10px; padding: 0 5px; line-height: 15px; }
.as-view { position: relative; overflow: hidden; min-width: 0; }
.as-view canvas { display: block; }
.as-ruler { border-bottom: 1px solid var(--line); cursor: text; }
.as-lanes { position: relative; overflow: hidden; border-bottom: 1px solid var(--line); }
.as-lanes > canvas { position: absolute; left: 0; top: 0; }
.as-clip { position: absolute; border: 1px solid var(--accent2); border-radius: 4px; background: rgba(120, 90, 255, .14); overflow: hidden; cursor: grab; box-sizing: border-box; touch-action: none; }
.as-clip.sel { border-color: var(--accent); background: rgba(85, 255, 255, .12); box-shadow: 0 0 0 1px var(--accent); }
.as-clip.quiet { opacity: .4; } .as-clip.busy { opacity: .5; cursor: progress; }
.as-clip canvas { position: absolute; top: 0; pointer-events: none; }
.as-clip::before, .as-clip::after { content: ""; position: absolute; top: 0; bottom: 0; width: 6px; cursor: ew-resize; z-index: 1; }
.as-clip::before { left: 0; } .as-clip::after { right: 0; }
.as-name { position: absolute; left: 6px; bottom: 2px; font: 10.5px var(--mono); color: var(--text); text-shadow: 0 0 3px #000, 0 0 3px #000; white-space: nowrap; pointer-events: none; }
.as-clip .fh { position: absolute; top: 0; width: 9px; height: 9px; background: var(--text); opacity: 0; cursor: ew-resize; z-index: 2; }
.as-clip:hover .fh, .as-clip.sel .fh { opacity: .75; } .as-clip .fh.l { left: 6px; } .as-clip .fh.r { right: 6px; }
.as-drop { position: absolute; width: 2px; background: var(--accent); pointer-events: none; }
.as-master { border-bottom: 1px solid var(--line); }
.as-env { border-bottom: 1px solid var(--line); cursor: crosshair; touch-action: none; }
.as-cursor, .as-head { position: absolute; top: 0; bottom: 0; width: 0; pointer-events: none; }
.as-cursor { border-left: 1px dashed var(--dim); } .as-head { border-left: 2px solid var(--ok); }
.as-scroll { margin: -4px 0 0 156px; width: calc(100% - 156px); padding: 0; }
.as-out b { color: var(--dim); text-transform: uppercase; letter-spacing: .08em; font-size: 12px; }
.as-tip { position: fixed; z-index: 50; background: var(--panel); border: 1px solid var(--line); border-radius: 4px; padding: 2px 6px; font: 11px var(--mono); pointer-events: none; }
` }));
  const LANES = 8, SR = 44100, LANE_H = 54, ENV_H = 58, RULER_H = 24, MASTER_H = 46, HEAD_W = 156;
  const SNAPS = [["off", "off"], ["ms", "10 ms"], ["1", "1 bar"], ["1/2", "1/2"], ["1/4", "1/4"], ["1/8", "1/8"], ["1/16", "1/16"], ["1/32", "1/32"]];
  const cutHz = v => 20 * Math.pow(1000, v);                 // 0..1 -> 20 Hz .. 20 kHz
  const ENVS = [
    { k: "pitch", name: "Pitch", min: -24, max: 24, def: 0, fmt: v => (v > 0 ? "+" : "") + v.toFixed(1) + " st", tip: "varispeed of the whole mix, like a sampler's pitch envelope (it bends time too)" },
    { k: "cut", name: "Filter cutoff", min: 0, max: 1, def: 1, fmt: v => fmtHz(cutHz(v)), tip: "the master filter's cutoff over time" },
    { k: "amp", name: "Amp", min: 0, max: 1, def: 1, fmt: v => Math.round(v * 100) + "%", tip: "the master volume over time" }];
  const dB = x => Math.pow(10, x / 20);
  const uid = () => Math.random().toString(36).slice(2, 10);
  const blank = () => ({
    v: 1, name: "", bpm: 120, sig: 4, snap: "1/16", loop: false, cursor: 0,
    lanes: Array.from({ length: LANES }, () => ({ vol: 0, mute: false, solo: false })),
    clips: [],
    master: { vol: 0, filter: "off", res: 0.7, safe: true, env: { pitch: { on: true, pts: [] }, cut: { on: true, pts: [] }, amp: { on: true, pts: [] } } },
    out: { fmt: "wav", file: "" }
  });
  let P = fix(store.get("asm", null));
  let view = { t0: 0, pps: 0 }, sel = null, built = false, undo = [], redo = [], savedSig = "";
  let ctx = null, out = null, outSig = "", rendering = null, dirty = true;
  let play = null;                                          // {src, t0, from}
  const BUF = new Map(), MEDIA_SECS = new Map();

  function fix(p) {                                         // an old or odd project: fill in what's missing
    const b = blank();
    if (!p || typeof p !== "object") return b;
    const m = Object.assign({}, b, p);
    m.lanes = Array.from({ length: LANES }, (_, i) => Object.assign({ vol: 0, mute: false, solo: false }, (p.lanes || [])[i] || {}));
    m.master = Object.assign({}, b.master, p.master || {});
    m.master.env = Object.assign({}, b.master.env, (p.master || {}).env || {});
    for (const e of ENVS) m.master.env[e.k] = Object.assign({ on: true, pts: [] }, m.master.env[e.k]);
    m.out = Object.assign({}, b.out, p.out || {});
    m.clips = (p.clips || []).filter(c => c && c.src);
    return m;
  }
  const ac = () => ctx || (ctx = new (window.AudioContext || window.webkitAudioContext)());
  const rate = c => c.keep ? 1 : Math.pow(2, (c.semis || 0) / 12);
  const dur = c => c.len / rate(c);
  const projEnd = () => P.clips.reduce((m, c) => Math.max(m, c.start + dur(c)), 0);
  const beat = () => 60 / Math.max(20, Math.min(300, +P.bpm || 120));
  const bar = () => beat() * Math.max(1, Math.min(16, +P.sig || 4));
  function unit() {
    if (P.snap === "off") return 0;
    if (P.snap === "ms") return 0.01;
    if (P.snap === "1") return bar();
    return 4 * beat() / (+P.snap.split("/")[1] || 16);
  }
  const snapT = (t, force) => { const u = unit(); return u && !force ? Math.max(0, Math.round(t / u) * u) : Math.max(0, t); };
  const mediaUrl = m => "/api/assemble/media?f=" + encodeURIComponent(m);

  /* ---------- sounds ---------- */
  function getBuf(media, rev) {
    const key = media + (rev ? "#rev" : "");
    if (BUF.has(key)) return BUF.get(key);
    const pr = rev ? getBuf(media).then(b => {
      const r = ac().createBuffer(b.numberOfChannels, b.length, b.sampleRate);
      for (let ch = 0; ch < b.numberOfChannels; ch++) r.getChannelData(ch).set(b.getChannelData(ch).slice().reverse());
      return r;
    }) : fetch(mediaUrl(media)).then(r => { if (!r.ok) throw new Error(`can't load ${media} (HTTP ${r.status})`); return r.arrayBuffer(); })
      .then(a => ac().decodeAudioData(a)).then(b => { MEDIA_SECS.set(media, b.duration); return b; });
    pr.catch(() => BUF.delete(key));
    BUF.set(key, pr);
    return pr;
  }
  const bufNow = new Map();                                  // decoded buffers, for drawing
  function want(c) { const k = c.fx + (c.rev ? "#rev" : ""); if (!bufNow.has(k)) getBuf(c.fx, c.rev).then(b => { bufNow.set(k, b); draw(); }).catch(e => toast(e.message, true)); return bufNow.get(k); }

  async function addClip(media, name, lane, t) {
    const b = await getBuf(media);
    const c = { id: uid(), lane, start: t, src: media, fx: media, name: name || media, off: 0, len: b.duration, full: b.duration, gain: 0, semis: 0, keep: false, stretch: 1, rev: false, fin: 0, fout: 0 };
    P.clips.push(c); sel = c.id; return c;
  }
  async function dropTargets(targets, files, lane, t) {
    let at = t;
    try {
      for (const f of files) {
        const r = await fetch("/api/assemble/upload?filename=" + encodeURIComponent(f.name), { method: "POST", body: f });
        const j = await r.json().catch(() => ({ error: r.statusText }));
        if (!r.ok || j.error) throw new Error(j.error || r.statusText);
        const c = await addClip(j.media, f.name, lane, at); at = snapT(at + dur(c), false);
      }
      for (const tg of targets) {
        const j = await api("/api/assemble/import", { target: tg });
        const c = await addClip(j.media, (tg.file || tg.path || tg.input || j.media).split("/").pop(), lane, at); at = snapT(at + dur(c), false);
      }
      commit();
    } catch (e) { toast(e.message, true); draw(); }
  }
  async function reprocess(c, stretch, semisKeep) {        // a stretched / length-kept pitched copy from the server
    const was = c.stretch || 1;
    c.busy = true; draw();
    try {
      const j = await api("/api/assemble/process", { media: c.src, stretch, semis: semisKeep });
      const b = await getBuf(j.media);
      const k = stretch / was;
      c.fx = j.media; c.stretch = stretch; c.full = b.duration;
      c.off = Math.min(c.off * k, c.full); c.len = Math.min(c.len * k, c.full - c.off);
    } catch (e) { toast(e.message, true); }
    c.busy = false; commit();
  }

  /* ---------- the mix: lanes -> sum, then the master (pitch env -> filter -> amp env) ---------- */
  const envOn = k => P.master.env[k].on && P.master.env[k].pts.length;
  function envAt(k, t) {
    const e = ENVS.find(x => x.k === k), pts = P.master.env[k].pts;
    if (!P.master.env[k].on || !pts.length) return e.def;
    if (t <= pts[0][0]) return pts[0][1];
    for (let i = 1; i < pts.length; i++) if (t <= pts[i][0]) { const [a, va] = pts[i - 1], [b, vb] = pts[i]; return b > a ? va + (vb - va) * (t - a) / (b - a) : vb; }
    return pts[pts.length - 1][1];
  }
  function audible(c) { const solo = P.lanes.some(l => l.solo), l = P.lanes[c.lane]; return l && !l.mute && (!solo || l.solo); }
  function outLength(sumDur) {                              // the pitch envelope bends time: walk it
    if (!envOn("pitch")) return sumDur;
    let pos = 0, t = 0; const dt = 0.002;
    while (pos < sumDur && t < 300) { pos += Math.pow(2, envAt("pitch", t) / 12) * dt; t += dt; }
    return t;
  }
  function automate(param, k, map, len, ramp) {
    const pts = P.master.env[k].pts;
    param.setValueAtTime(map(envAt(k, 0)), 0);
    for (const [t, v] of pts) if (t > 0 && t <= len) (ramp === "exp" ? param.exponentialRampToValueAtTime(Math.max(1e-4, map(v)), t) : param.linearRampToValueAtTime(map(v), t));
    if (pts.length && pts[pts.length - 1][0] < len) param.setValueAtTime(map(pts[pts.length - 1][1]), len);
  }
  async function render() {
    const sig = JSON.stringify([P.clips.map(c => [c.lane, c.start, c.fx, c.off, c.len, c.gain, c.semis, c.keep, c.rev, c.fin, c.fout]), P.lanes, P.master]);
    if (out && sig === outSig) return out;
    const clips = P.clips.filter(audible);
    const sumDur = Math.max(0.05, clips.reduce((m, c) => Math.max(m, c.start + dur(c)), 0));
    const bufs = await Promise.all(clips.map(c => getBuf(c.fx, c.rev)));
    const off1 = new OfflineAudioContext(2, Math.ceil(sumDur * SR) + 1, SR);
    clips.forEach((c, i) => {
      const s = off1.createBufferSource(); s.buffer = bufs[i]; s.playbackRate.value = rate(c);
      const g = off1.createGain(), base = dB(c.gain || 0) * dB(P.lanes[c.lane].vol || 0), end = c.start + dur(c);
      const fin = Math.min(c.fin || 0, dur(c) / 2), fout = Math.min(c.fout || 0, dur(c) / 2);
      g.gain.setValueAtTime(fin > 0 ? 0 : base, c.start);
      if (fin > 0) g.gain.linearRampToValueAtTime(base, c.start + fin);
      if (fout > 0) { g.gain.setValueAtTime(base, end - fout); g.gain.linearRampToValueAtTime(0, end); }
      s.connect(g).connect(off1.destination);
      s.start(c.start, Math.min(c.off, bufs[i].duration), c.len);
    });
    const sum = await off1.startRendering();
    const len = Math.max(0.05, outLength(sum.duration));
    const off2 = new OfflineAudioContext(2, Math.ceil(len * SR) + 1, SR);
    const s = off2.createBufferSource(); s.buffer = sum;
    if (envOn("pitch")) automate(s.playbackRate, "pitch", v => Math.pow(2, v / 12), len, "exp");
    let node = s;
    if (P.master.filter !== "off") {
      const f = off2.createBiquadFilter(); f.type = P.master.filter === "hp" ? "highpass" : "lowpass"; f.Q.value = +P.master.res || 0.7;
      if (envOn("cut")) automate(f.frequency, "cut", cutHz, len, "exp"); else f.frequency.value = cutHz(envAt("cut", 0));
      node = node.connect(f);
    }
    const g = off2.createGain(), mv = dB(P.master.vol || 0);
    if (envOn("amp")) automate(g.gain, "amp", v => v * mv, len); else g.gain.value = mv;
    node.connect(g).connect(off2.destination);
    s.start(0);
    const res = await off2.startRendering();
    let peak = 0;
    for (let ch = 0; ch < res.numberOfChannels; ch++) { const d = res.getChannelData(ch); for (let i = 0; i < d.length; i++) { const a = Math.abs(d[i]); if (a > peak) peak = a; } }
    res._peak = peak;
    out = res; outSig = sig; return out;
  }
  function invalidate() { dirty = true; clearTimeout(invalidate.t); invalidate.t = setTimeout(rerender, 120); }
  async function rerender() {
    if (!built) return;
    if (!P.clips.length) { out = null; outSig = ""; drawMaster(); info(); return; }
    try {
      const r = render(); rendering = r; await r;
      if (rendering !== r) return;
      dirty = false; drawMaster(); info();
      if (play) { const at = playPos(); startPlay(at < out.duration ? at : 0); }
    } catch (e) { toast("Assemble: " + e.message, true); }
  }
  function info() {
    const n = $("asInfo"); if (!n) return;
    if (!out || !P.clips.length) { n.textContent = "empty — drag sounds onto a lane"; return; }
    const pk = 20 * Math.log10(out._peak || 1e-9);
    n.textContent = `${out.duration.toFixed(2)} s · peak ${pk.toFixed(1)} dB${pk > -0.1 ? (P.master.safe ? " (turned down on export)" : " ⚠ clips") : ""}`;
    n.style.color = pk > -0.1 && !P.master.safe ? "var(--bad)" : "";
  }

  /* ---------- playing ---------- */
  function playPos() { if (!play) return P.cursor || 0; const t = play.from + (ac().currentTime - play.t0); return P.loop && out ? t % out.duration : t; }
  function startPlay(from) {
    stopPlay_(true);
    if (!out) return;
    if (typeof nowKey !== "undefined" && nowKey) stopPlay();         // the page's own player stops
    const s = ac().createBufferSource(); s.buffer = out; s.loop = !!P.loop; s.connect(ac().destination);
    const t0 = ac().currentTime + 0.03; from = Math.max(0, Math.min(from, out.duration - 0.001));
    s.start(t0, from); play = { src: s, t0, from };
    s.onended = () => { if (play && play.src === s) { play = null; drawHead(); btns(); } };
    btns(); tickHead();
  }
  function stopPlay_(quiet) { if (play) { const s = play.src; play = null; try { s.stop(); } catch (e) {} } if (!quiet) { drawHead(); btns(); } }
  async function togglePlay() {
    if (play) { stopPlay_(); return; }
    if (!P.clips.length) { toast("drag a sound onto a lane first", true); return; }
    await ac().resume();
    try { $("asPlay").textContent = "…"; await render(); dirty = false; drawMaster(); info(); startPlay(P.cursor || 0); } catch (e) { toast(e.message, true); btns(); }
  }
  function tickHead() { drawHead(); if (play) requestAnimationFrame(tickHead); }
  function btns() { const b = $("asPlay"); if (b) b.textContent = play ? "■ Stop" : "▶ Play"; const l = $("asLoop"); if (l) l.setAttribute("aria-pressed", String(!!P.loop)); }
  player.addEventListener("play", () => stopPlay_());          // the page's player wins

  /* ---------- history ---------- */
  const snap = () => JSON.stringify(P, (k, v) => k === "busy" ? undefined : v);
  let last = snap();
  function commit() {
    const s = snap();
    if (s !== last) { undo.push(last); if (undo.length > 120) undo.shift(); redo = []; last = s; }
    store.set("asm", JSON.parse(s)); autosave(); invalidate(); draw();
  }
  function stepHistory(from, to) { if (!from.length) return; to.push(last); last = from.pop(); P = fix(JSON.parse(last)); if (!P.clips.some(c => c.id === sel)) sel = null; store.set("asm", P); autosave(); syncTop(); invalidate(); draw(); }
  function autosave() {
    clearTimeout(autosave.t);
    if (!P.name) { saved(); return; }
    autosave.t = setTimeout(() => api("/api/assemble/save", { name: P.name, project: JSON.parse(snap()) }).then(r => { savedSig = snap(); projList(r.projects); saved(); }).catch(e => toast(e.message, true)), 1200);
  }
  function saved() { const n = $("asSaved"); if (n) n.textContent = !P.name ? "(not saved: name it to keep it)" : savedSig === snap() ? "saved" : "saving…"; }

  /* ---------- drawing ---------- */
  const W = () => Math.max(100, $("asView")?.clientWidth || 600);
  const X = t => (t - view.t0) * view.pps;
  const T = x => view.t0 + x / view.pps;
  function gridLines(g, w, h, labels) {
    const css = getComputedStyle(document.documentElement), line = css.getPropertyValue("--line").trim() || "#333", dim = css.getPropertyValue("--dim").trim() || "#888";
    const u = unit(), musical = P.snap !== "off" && P.snap !== "ms";
    const steps = musical ? [[u, 0.35], [beat(), 0.6], [bar(), 1]] : [[0.01, 0.25], [0.1, 0.5], [1, 1]];
    for (const [step, a] of steps) {
      if (!step || step * view.pps < 6) continue;
      g.globalAlpha = a; g.fillStyle = a === 1 ? dim : line;
      for (let k = Math.floor(view.t0 / step); ; k++) { const x = Math.round(X(k * step)); if (x > w) break; if (x >= 0) g.fillRect(x, 0, 1, h); }
    }
    g.globalAlpha = 1;
    if (labels) {
      g.fillStyle = dim; g.font = "10px " + (css.getPropertyValue("--mono").trim() || "monospace"); g.textBaseline = "top";
      const step = musical ? (bar() * view.pps >= 34 ? bar() : bar() * Math.ceil(34 / (bar() * view.pps))) : [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30].find(s => s * view.pps >= 46) || 60;
      for (let k = Math.floor(view.t0 / step); ; k++) {
        const x = X(k * step); if (x > w) break; if (x < -40) continue;
        g.fillText(musical ? String(Math.round(k * step / bar()) + 1) : (k * step).toFixed(step < 0.1 ? 2 : step < 1 ? 1 : 0) + "s", Math.round(x) + 3, 3);
        if (musical && beat() * view.pps >= 40 && step === bar()) for (let b = 1; b < P.sig; b++) { const xb = X(k * step + b * beat()); if (xb < w) g.fillText(`.${b + 1}`, Math.round(xb) + 3, 3); }
      }
    }
  }
  function canvasFit(cv, w, h) { const r = devicePixelRatio || 1; cv.width = Math.max(1, Math.round(w * r)); cv.height = Math.max(1, Math.round(h * r)); cv.style.width = w + "px"; cv.style.height = h + "px"; const g = cv.getContext("2d"); g.setTransform(r, 0, 0, r, 0, 0); g.clearRect(0, 0, w, h); return g; }
  function waveInto(g, b, from, len, x0, w, h, col) {
    g.fillStyle = col; const d0 = b.getChannelData(0), d1 = b.numberOfChannels > 1 ? b.getChannelData(1) : d0, sr = b.sampleRate, mid = h / 2;
    for (let x = 0; x < w; x++) {
      const a = Math.floor((from + (x / w) * len) * sr), z = Math.max(a + 1, Math.floor((from + ((x + 1) / w) * len) * sr)), st = Math.max(1, Math.floor((z - a) / 48));
      let pk = 0; for (let i = a; i < z && i < d0.length; i += st) { const v = Math.max(Math.abs(d0[i]), Math.abs(d1[i])); if (v > pk) pk = v; }
      const hh = Math.max(0.5, pk * (mid - 1)); g.fillRect(x0 + x, mid - hh, 1, hh * 2);
    }
  }
  function draw() {
    if (!built || $("p-assemble").hidden) return;
    cancelAnimationFrame(draw.raf); draw.raf = requestAnimationFrame(drawNow);
  }
  function drawNow() {
    const w = W();
    if (!view.pps) fit();
    // ruler
    const rg = canvasFit($("asRuler"), w, RULER_H); gridLines(rg, w, RULER_H, true);
    // lanes: grid behind, clips on top
    const lanesH = LANES * LANE_H, lg = canvasFit($("asGrid"), w, lanesH); gridLines(lg, w, lanesH, false);
    const css = getComputedStyle(document.documentElement), line = css.getPropertyValue("--line").trim();
    lg.fillStyle = line; for (let i = 1; i < LANES; i++) lg.fillRect(0, i * LANE_H, w, 1);
    const host = $("asClips"); host.replaceChildren();
    for (const c of P.clips) {
      const x = X(c.start), cw = Math.max(3, dur(c) * view.pps);
      if (x > w || x + cw < 0) continue;
      const lane = P.lanes[c.lane], quiet = !audible(c);
      const d = el("div", { class: "as-clip" + (c.id === sel ? " sel" : "") + (quiet ? " quiet" : "") + (c.busy ? " busy" : ""), "data-id": c.id,
        style: `left:${x}px;top:${c.lane * LANE_H + 2}px;width:${cw}px;height:${LANE_H - 4}px`, title: `${c.name}\ndrag: move · edges: trim · top corners: fades · Alt+drag: copy` });
      const vx0 = Math.max(0, -x), vx1 = Math.min(cw, w - x);
      const b = want(c);
      if (b && vx1 > vx0) {
        const cv = el("canvas", { style: `left:${vx0}px` }); const g = canvasFit(cv, vx1 - vx0, LANE_H - 4);
        waveInto(g, b, c.off + (vx0 / cw) * c.len, ((vx1 - vx0) / cw) * c.len, 0, vx1 - vx0, LANE_H - 4, c.id === sel ? css.getPropertyValue("--accent").trim() : css.getPropertyValue("--accent2").trim());
        // fades drawn as a shade over the wave
        g.fillStyle = "rgba(0,0,0,.45)";
        if (c.fin > 0) { g.beginPath(); g.moveTo(-vx0, 0); g.lineTo(-vx0 + c.fin * view.pps, 0); g.lineTo(-vx0, LANE_H - 4); g.fill(); }
        if (c.fout > 0) { g.beginPath(); g.moveTo(cw - vx0, 0); g.lineTo(cw - vx0 - c.fout * view.pps, 0); g.lineTo(cw - vx0, LANE_H - 4); g.fill(); }
        d.append(cv);
      }
      const tags = [c.gain ? `${c.gain > 0 ? "+" : ""}${(+c.gain).toFixed(1)} dB` : "", c.semis ? `${c.semis > 0 ? "+" : ""}${c.semis} st${c.keep ? " =len" : ""}` : "", c.stretch !== 1 ? `×${(+c.stretch).toFixed(2).replace(/\.?0+$/, "")}` : "", c.rev ? "rev" : ""].filter(Boolean).join(" · ");
      d.append(el("span", { class: "as-name", text: c.name + (tags ? "  " + tags : "") }), el("i", { class: "fh l" }), el("i", { class: "fh r" }));
      host.append(d);
    }
    // envelopes
    for (const e of ENVS) drawEnv(e, w);
    drawMaster(); drawHead(); drawProps(); syncHeads();
    const sb = $("asScroll"), total = Math.max(projEnd() * 1.3, w / view.pps, bar() * 4);
    sb.max = String(Math.max(0, total - w / view.pps)); sb.step = "any"; sb.value = String(view.t0);
  }
  function drawEnv(e, w) {
    const cv = $("asEnv-" + e.k), g = canvasFit(cv, w, ENV_H); gridLines(g, w, ENV_H, false);
    const env = P.master.env[e.k], css = getComputedStyle(document.documentElement), acc = css.getPropertyValue("--accent").trim(), dim = css.getPropertyValue("--dim").trim();
    const Y = v => 4 + (1 - (v - e.min) / (e.max - e.min)) * (ENV_H - 8);
    if (e.k === "pitch") { g.fillStyle = dim; g.globalAlpha = 0.5; g.fillRect(0, Math.round(Y(0)), w, 1); g.globalAlpha = 1; }
    g.strokeStyle = env.on && env.pts.length ? acc : dim; g.lineWidth = 1.5; g.setLineDash(env.pts.length ? [] : [4, 4]); g.globalAlpha = env.on ? 1 : 0.4;
    g.beginPath();
    for (let x = 0; x <= w; x += 2) { const y = Y(env.pts.length ? envAt2(e.k, T(x)) : e.def); x ? g.lineTo(x, y) : g.moveTo(x, y); }
    g.stroke(); g.setLineDash([]);
    g.fillStyle = acc;
    for (const [t, v] of env.pts) { const x = X(t); if (x < -5 || x > w + 5) continue; g.beginPath(); g.arc(x, Y(v), 4, 0, Math.PI * 2); g.fill(); }
    g.globalAlpha = 1;
  }
  const envAt2 = (k, t) => { const on = P.master.env[k].on; P.master.env[k].on = true; const v = envAt(k, t); P.master.env[k].on = on; return v; };
  function drawMaster() {
    if (!built) return;
    const w = W(), g = canvasFit($("asMaster"), w, MASTER_H), css = getComputedStyle(document.documentElement);
    gridLines(g, w, MASTER_H, false);
    if (out) { const a = Math.max(0, view.t0), z = Math.min(out.duration, T(w)); if (z > a) waveInto(g, out, a, z - a, X(a), Math.round((z - a) * view.pps), MASTER_H, css.getPropertyValue("--ok").trim() || "#5f5"); }
  }
  function drawHead() {
    if (!built) return;
    const ph = $("asHead"), cu = $("asCursor"), w = W();
    const xc = X(P.cursor || 0); cu.style.left = xc + "px"; cu.hidden = xc < 0 || xc > w;
    if (play) { const x = X(playPos()); ph.style.left = x + "px"; ph.hidden = x < 0 || x > w; } else ph.hidden = true;
  }
  function syncHeads() {
    P.lanes.forEach((l, i) => {
      const h = $("asLane-" + i); if (!h) return;
      h.querySelector(".m").setAttribute("aria-pressed", String(!!l.mute)); h.querySelector(".s").setAttribute("aria-pressed", String(!!l.solo));
      const r = h.querySelector("input"); if (document.activeElement !== r) r.value = l.vol; r.title = `lane ${i + 1} volume: ${(+l.vol).toFixed(1)} dB (double-click: 0)`;
    });
    for (const e of ENVS) { const c = $("asEnvOn-" + e.k); if (c) c.checked = !!P.master.env[e.k].on; }
    $("asFilt").value = P.master.filter; $("asRes").value = P.master.res; $("asMVol").value = P.master.vol; $("asSafe").checked = !!P.master.safe;
    $("asMVol").title = `master volume ${(+P.master.vol).toFixed(1)} dB (double-click: 0)`;
  }
  function syncTop() { $("asBpm").value = P.bpm; $("asSig").value = P.sig; $("asSnap").value = P.snap; $("asName").value = P.name || ""; $("asFmt").value = P.out.fmt; $("asFile").value = P.out.file || ""; btns(); saved(); }

  /* ---------- the selected clip's properties ---------- */
  function drawProps() {
    const box = $("asProps"), c = P.clips.find(x => x.id === sel);
    if (!c) { box.replaceChildren(el("span", { class: "dim", text: "Click a clip for its volume, pitch, length and fades. Space plays, Delete removes, Ctrl+D duplicates, Ctrl+Z undoes." })); box.dataset.id = ""; return; }
    if (box.dataset.id === c.id && box.contains(document.activeElement)) { upd(); return; }
    box.dataset.id = c.id;
    const num = (label, key, min, max, step, tip, fmt) => {
      const v = el("span", { class: "mono" }), r = el("input", { type: "range", min, max, step, value: c[key] ?? 0, title: tip + " (double-click: reset)" });
      r.addEventListener("input", () => { c[key] = +r.value; v.textContent = fmt(+r.value); if (key !== "semis" || !c.keep) { invalidate(); draw(); } });
      r.addEventListener("change", () => key === "semis" && c.keep ? reprocess(c, c.stretch, c.semis) : commit());
      r.addEventListener("dblclick", () => { r.value = 0; r.dispatchEvent(new Event("input")); r.dispatchEvent(new Event("change")); });
      v.textContent = fmt(+r.value); return el("label", { class: "as-p" }, el("span", { text: label }), r, v);
    };
    const ms = (label, key) => { const i = el("input", { type: "number", min: 0, step: 5, value: Math.round((c[key] || 0) * 1000), title: label + " in milliseconds" });
      i.addEventListener("change", () => { c[key] = Math.max(0, (+i.value || 0) / 1000); commit(); }); return el("label", { class: "as-p n" }, el("span", { text: label }), i, el("span", { class: "dim", text: "ms" })); };
    const keep = el("input", { type: "checkbox", checked: !!c.keep });
    keep.addEventListener("change", () => { c.keep = keep.checked; reprocess(c, c.stretch, c.keep ? c.semis : 0); });
    const rev = el("input", { type: "checkbox", checked: !!c.rev });
    rev.addEventListener("change", () => { c.rev = rev.checked; c.off = Math.max(0, c.full - c.off - c.len); commit(); });
    const sx = el("input", { type: "number", min: 0.25, max: 4, step: 0.05, value: c.stretch, title: "length × this, pitch kept (ffmpeg)" });
    sx.addEventListener("change", () => reprocess(c, Math.max(0.25, Math.min(4, +sx.value || 1)), c.keep ? c.semis : 0));
    const st = k => el("button", { type: "button", text: k > 1 ? "×2" : "×½", title: k > 1 ? "twice as long, same pitch" : "half as long, same pitch",
      onclick: () => reprocess(c, Math.max(0.25, Math.min(4, (c.stretch || 1) * k)), c.keep ? c.semis : 0) });
    box.replaceChildren(
      el("b", { class: "as-cname", text: c.name, title: c.name }),
      num("Volume", "gain", -36, 12, 0.5, "clip volume in dB", v => (v > 0 ? "+" : "") + v.toFixed(1) + " dB"),
      num("Pitch", "semis", -24, 24, 1, "semitones; sampler-style it changes the length too, unless 'keep length'", v => (v > 0 ? "+" : "") + v + " st"),
      el("label", { class: "as-p c", title: "pitch without changing the length (a stretched copy from ffmpeg)" }, keep, el("span", { text: "keep length" })),
      el("span", { class: "as-p" }, el("span", { text: "Stretch" }), st(0.5), sx, st(2)),
      el("label", { class: "as-p c" }, rev, el("span", { text: "reverse" })),
      ms("Fade in", "fin"), ms("Fade out", "fout"),
      el("span", { class: "as-p" },
        el("button", { type: "button", text: "⧉ Duplicate", title: "Ctrl+D", onclick: () => dup(c) }),
        el("button", { type: "button", text: "↔ Whole sound", title: "undo any trim: the whole sound again", onclick: () => { c.off = 0; c.len = c.full; commit(); } }),
        el("button", { type: "button", text: "✕ Remove", title: "Delete", onclick: () => remove(c) })));
    function upd() { }
  }
  function dup(c) { const n = Object.assign({}, c, { id: uid(), start: snapT(c.start + dur(c)) }); delete n.busy; P.clips.push(n); sel = n.id; commit(); }
  function remove(c) { P.clips = P.clips.filter(x => x !== c); if (sel === c.id) sel = null; commit(); }

  /* ---------- interaction ---------- */
  function fit() { const e = Math.max(projEnd(), bar() * 2, 1); view.t0 = 0; view.pps = Math.max(10, (W() - 20) / (e * 1.08)); }
  function zoomAt(x, k) { const t = T(x); view.pps = Math.max(10, Math.min(30000, view.pps * k)); view.t0 = Math.max(0, t - x / view.pps); draw(); }
  function hookTimeline() {
    const v = $("asView");
    v.addEventListener("wheel", e => {
      e.preventDefault();
      const r = v.getBoundingClientRect();
      if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) { view.t0 = Math.max(0, view.t0 + (e.deltaX || e.deltaY) / view.pps); draw(); }
      else zoomAt(e.clientX - r.left, e.deltaY < 0 ? 1.18 : 1 / 1.18);
    }, { passive: false });
    $("asRuler").addEventListener("pointerdown", e => { const r = e.currentTarget.getBoundingClientRect(); P.cursor = snapT(T(e.clientX - r.left)); store.set("asm", P); drawHead(); if (play) startPlay(P.cursor); });
    // clips: move / trim / fade / Alt-copy; empty lane: deselect and set the cursor
    $("asLanes").addEventListener("pointerdown", e => {
      if (e.button !== 0) return;
      const lr = $("asLanes").getBoundingClientRect(), d = e.target.closest(".as-clip");
      if (!d) { sel = null; P.cursor = snapT(T(e.clientX - lr.left)); draw(); return; }
      let c = P.clips.find(x => x.id === d.dataset.id); if (!c || c.busy) return;
      const cr = d.getBoundingClientRect(), lx = e.clientX - cr.left;
      let mode = e.target.classList.contains("fh") ? (e.target.classList.contains("l") ? "fin" : "fout") : lx < 7 ? "left" : lx > cr.width - 7 ? "right" : "move";
      let copy = mode === "move" && e.altKey;                    // Alt: leave a copy where it was
      sel = c.id;
      const o = { start: c.start, lane: c.lane, off: c.off, len: c.len, fin: c.fin, fout: c.fout, x: e.clientX, y: e.clientY }, rt = rate(c);
      let moved = false;
      const lanes = $("asLanes");                                 // (the clip's own div is redrawn while it moves)
      lanes.setPointerCapture(e.pointerId);
      const mv = ev => {
        const dt = (ev.clientX - o.x) / view.pps, free = ev.shiftKey;   // Shift: off the grid
        if (Math.abs(ev.clientX - o.x) + Math.abs(ev.clientY - o.y) > 2) moved = true; if (!moved) return;
        if (copy) { copy = false; const n = Object.assign({}, c, { id: uid() }); delete n.busy; P.clips.push(n); }
        if (mode === "move") { c.start = snapT(o.start + dt, free); c.lane = Math.max(0, Math.min(LANES - 1, o.lane + Math.round((ev.clientY - o.y) / LANE_H))); }
        else if (mode === "left") {
          let ns = snapT(o.start + dt, free); const end = o.start + o.len / rt;
          ns = Math.min(ns, end - 0.005); ns = Math.max(ns, o.start - o.off / rt);
          c.start = ns; c.off = o.off + (ns - o.start) * rt; c.len = o.len - (ns - o.start) * rt;
        } else if (mode === "right") {
          let ne = snapT(o.start + o.len / rt + dt, free); ne = Math.max(ne, o.start + 0.005);
          c.len = Math.min((ne - o.start) * rt, c.full - c.off);
        } else if (mode === "fin") c.fin = Math.max(0, Math.min(dur(c), o.fin + dt));
        else c.fout = Math.max(0, Math.min(dur(c), o.fout - dt));
        draw();
      };
      const up = () => { lanes.removeEventListener("pointermove", mv); lanes.removeEventListener("pointerup", up); lanes.removeEventListener("pointercancel", up); commit(); };
      lanes.addEventListener("pointermove", mv); lanes.addEventListener("pointerup", up); lanes.addEventListener("pointercancel", up);
      draw();
    });
    // drops: sounds from anywhere on the page, or files from the desktop
    const lanes = $("asLanes");
    const ok = e => [...e.dataTransfer.types].some(t => t === CLIP_TYPE || t === "Files");
    lanes.addEventListener("dragover", e => { if (!ok(e)) return; e.preventDefault(); const r = lanes.getBoundingClientRect(), ln = Math.max(0, Math.min(LANES - 1, Math.floor((e.clientY - r.top) / LANE_H)));
      const m = $("asDrop"); m.hidden = false; m.style.top = ln * LANE_H + "px"; m.style.left = X(snapT(T(e.clientX - r.left))) + "px"; });
    lanes.addEventListener("dragleave", e => { if (!lanes.contains(e.relatedTarget)) $("asDrop").hidden = true; });
    lanes.addEventListener("drop", e => {
      e.preventDefault(); $("asDrop").hidden = true;
      const r = lanes.getBoundingClientRect(), ln = Math.max(0, Math.min(LANES - 1, Math.floor((e.clientY - r.top) / LANE_H))), t = snapT(T(e.clientX - r.left));
      let tg = []; try { const raw = e.dataTransfer.getData(CLIP_TYPE); if (raw) tg = [JSON.parse(raw)]; } catch (err) { /* not ours */ }
      dropTargets(tg, tg.length ? [] : [...e.dataTransfer.files], ln, t);
    });
    // envelopes: click to add a point, drag to move, right-click or double-click to delete
    for (const en of ENVS) {
      const cv = $("asEnv-" + en.k);
      const val = y => Math.max(en.min, Math.min(en.max, en.min + (1 - (y - 4) / (ENV_H - 8)) * (en.max - en.min)));
      const Yv = v => 4 + (1 - (v - en.min) / (en.max - en.min)) * (ENV_H - 8);
      const hit = (x, y) => P.master.env[en.k].pts.findIndex(([t, v]) => Math.abs(X(t) - x) < 7 && Math.abs(Yv(v) - y) < 7);
      cv.addEventListener("pointerdown", e => {
        if (e.button !== 0) return;
        const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, env = P.master.env[en.k];
        let i = hit(x, y);
        if (i < 0) { env.pts.push([snapT(T(x), e.shiftKey), +val(y).toFixed(3)]); env.pts.sort((a, b) => a[0] - b[0]); i = env.pts.findIndex(p => p[0] === snapT(T(x), e.shiftKey)); env.on = true; }
        const pt = env.pts[i];
        cv.setPointerCapture(e.pointerId);
        const mv = ev => { pt[0] = snapT(T(ev.clientX - r.left), ev.shiftKey); pt[1] = +val(ev.clientY - r.top).toFixed(3); env.pts.sort((a, b) => a[0] - b[0]); tipEnv(en, pt, ev); drawEnv(en, W()); invalidate(); };
        const up = () => { cv.removeEventListener("pointermove", mv); cv.removeEventListener("pointerup", up); cv.removeEventListener("pointercancel", up); $("asTip").hidden = true; commit(); };
        cv.addEventListener("pointermove", mv); cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up); tipEnv(en, pt, e); drawEnv(en, W());
      });
      const del = e => { e.preventDefault(); const r = cv.getBoundingClientRect(), i = hit(e.clientX - r.left, e.clientY - r.top); if (i >= 0) { P.master.env[en.k].pts.splice(i, 1); commit(); } };
      cv.addEventListener("contextmenu", del); cv.addEventListener("dblclick", del);
      cv.addEventListener("pointermove", e => { if (e.buttons) return; const r = cv.getBoundingClientRect(); cv.style.cursor = hit(e.clientX - r.left, e.clientY - r.top) >= 0 ? "grab" : "crosshair"; });
    }
    $("asScroll").addEventListener("input", e => { view.t0 = +e.target.value; draw(); });
    new ResizeObserver(() => draw()).observe(v);
  }
  function tipEnv(en, pt, e) { const t = $("asTip"); t.hidden = false; t.textContent = `${en.fmt(pt[1])} at ${pt[0].toFixed(3)} s`; t.style.left = (e.clientX + 12) + "px"; t.style.top = (e.clientY - 26) + "px"; }

  /* ---------- exporting ---------- */
  function wavBytes(b) {
    const ch = b.numberOfChannels, n = b.length, peak = b._peak || 0, k = P.master.safe && peak > 0.98 ? 0.98 / peak : 1;   // only ever down
    const buf = new ArrayBuffer(44 + n * ch * 2), v = new DataView(buf);
    const s = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
    s(0, "RIFF"); v.setUint32(4, 36 + n * ch * 2, true); s(8, "WAVE"); s(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, ch, true);
    v.setUint32(24, b.sampleRate, true); v.setUint32(28, b.sampleRate * ch * 2, true); v.setUint16(32, ch * 2, true); v.setUint16(34, 16, true); s(36, "data"); v.setUint32(40, n * ch * 2, true);
    const d = Array.from({ length: ch }, (_, i) => b.getChannelData(i));
    let o = 44;
    for (let i = 0; i < n; i++) for (let c = 0; c < ch; c++) { const x = Math.max(-1, Math.min(1, d[c][i] * k)); v.setInt16(o, x < 0 ? x * 32768 : x * 32767, true); o += 2; }
    return new Blob([buf], { type: "audio/wav" });
  }
  async function exportTo(to, extra = {}) {
    if (!P.clips.length) { toast("nothing to export: drag a sound onto a lane", true); return null; }
    const b = await render(), name = ($("asFile").value.trim() || P.name || "assembled").replace(/\.(wav|flac|ogg|mp3)$/i, "");
    const q = new URLSearchParams({ to, name, format: to === "file" || to === "bin" ? P.out.fmt : "wav", ...extra });
    const r = await fetch("/api/assemble/export?" + q, { method: "POST", body: wavBytes(b) });
    const j = await r.json().catch(() => ({ error: r.statusText }));
    if (!r.ok || j.error) throw new Error(j.error || r.statusText);
    return j;
  }
  const run = f => async () => { try { await f(); } catch (e) { toast(e.message, true); } };

  /* ---------- projects ---------- */
  function projList(list) {
    const s = $("asOpen"); if (!s) return;
    s.replaceChildren(el("option", { value: "", text: list.length ? `open… (${list.length})` : "no saved projects" }), ...list.map(p => el("option", { value: p.name, text: p.name })));
  }
  async function loadProjects() { try { const r = await api("/api/assemble/projects"); projList(r.projects); $("asStretcher").textContent = `stretching by ${r.stretcher}`; asmFolders = r; } catch (e) { toast(e.message, true); } }
  let asmFolders = null;
  function setProject(p) { stopPlay_(); P = fix(p); sel = null; undo = []; redo = []; last = snap(); savedSig = snap(); store.set("asm", P); out = null; outSig = ""; view.pps = 0; syncTop(); invalidate(); draw(); }

  /* ---------- the tab ---------- */
  function build() {
    const pane = $("p-assemble");
    const sel_ = (id, opts, title) => el("select", { id, title }, ...opts.map(([v, l]) => el("option", { value: v, text: l })));
    const laneHead = i => el("div", { class: "as-lh", id: "asLane-" + i, style: `height:${LANE_H}px` },
      el("b", { text: String(i + 1) }),
      el("button", { type: "button", class: "m", text: "M", title: "mute", onclick: () => { P.lanes[i].mute = !P.lanes[i].mute; commit(); } }),
      el("button", { type: "button", class: "s", text: "S", title: "solo", onclick: () => { P.lanes[i].solo = !P.lanes[i].solo; commit(); } }),
      (() => { const r = el("input", { type: "range", min: -30, max: 6, step: 0.5, value: 0 }); r.addEventListener("input", () => { P.lanes[i].vol = +r.value; invalidate(); });
        r.addEventListener("change", commit); r.addEventListener("dblclick", () => { P.lanes[i].vol = 0; commit(); }); return r; })());
    const envHead = e => el("div", { class: "as-lh env", style: `height:${ENV_H}px`, title: e.tip },
      el("label", {}, el("input", { type: "checkbox", id: "asEnvOn-" + e.k, onchange: ev => { P.master.env[e.k].on = ev.target.checked; commit(); } }), el("span", { text: e.name })),
      e.k === "cut" ? el("span", { class: "row" }, sel_("asFilt", [["off", "off"], ["lp", "LP"], ["hp", "HP"]], "the master filter: off, low-pass or high-pass"),
        (() => { const r = el("input", { type: "range", id: "asRes", min: 0.5, max: 18, step: 0.1, title: "resonance" }); r.addEventListener("input", () => { P.master.res = +r.value; invalidate(); }); r.addEventListener("change", commit); return r; })()) : null,
      el("button", { type: "button", class: "clr", text: "clear", title: "remove this envelope's points", onclick: () => { P.master.env[e.k].pts = []; commit(); } }));
    pane.replaceChildren(
      el("div", { class: "paneh as-top" },
        el("h2", { text: "Assemble" }),
        el("button", { type: "button", id: "asPlay", class: "as-play", text: "▶ Play", title: "play from the cursor (Space)", onclick: togglePlay }),
        el("button", { type: "button", id: "asLoop", text: "⟲ Loop", title: "loop the playback", onclick: () => { P.loop = !P.loop; btns(); store.set("asm", P); if (play) startPlay(playPos()); } }),
        el("label", { class: "as-f" }, "BPM", el("input", { id: "asBpm", type: "number", min: 20, max: 300, step: 1, onchange: e => { P.bpm = Math.max(20, Math.min(300, +e.target.value || 120)); commit(); } })),
        el("label", { class: "as-f" }, "Beats", el("input", { id: "asSig", type: "number", min: 1, max: 16, step: 1, title: "beats in a bar", onchange: e => { P.sig = Math.max(1, Math.min(16, Math.round(+e.target.value || 4))); commit(); } })),
        el("label", { class: "as-f" }, "Snap", sel_("asSnap", SNAPS, "what clips and points snap to (hold Shift for off-grid)")),
        el("span", { class: "as-g" },
          el("button", { type: "button", text: "−", title: "zoom out (or the wheel over the lanes)", onclick: () => zoomAt(W() / 2, 1 / 1.5) }),
          el("button", { type: "button", text: "fit", title: "show the whole thing", onclick: () => { fit(); draw(); } }),
          el("button", { type: "button", text: "+", title: "zoom in", onclick: () => zoomAt(W() / 2, 1.5) })),
        el("span", { class: "as-g" },
          el("button", { type: "button", text: "↶", title: "undo (Ctrl+Z)", onclick: () => stepHistory(undo, redo) }),
          el("button", { type: "button", text: "↷", title: "redo (Ctrl+Y)", onclick: () => stepHistory(redo, undo) }))),
      el("div", { class: "paneh as-top" },
        el("label", { class: "as-f" }, "Project", el("input", { id: "asName", placeholder: "name it to keep it", maxlength: 64, onchange: e => { P.name = e.target.value.trim().replace(/[^\w.-]+/g, "-"); e.target.value = P.name; commit(); } })),
        el("span", { id: "asSaved", class: "dim" }),
        el("select", { id: "asOpen", title: "open a saved project", onchange: async e => { const n = e.target.value; e.target.value = ""; if (!n) return;
          try { const r = await api("/api/assemble/project?name=" + encodeURIComponent(n)); setProject(r.project); toast("Opened " + n); } catch (err) { toast(err.message, true); } } }),
        el("button", { type: "button", text: "New", title: "start an empty project (the current one stays saved if it has a name)", onclick: () => {
          if (!P.name && P.clips.length && !confirm("This project has no name, so it isn't saved. Start a new one anyway?")) return; setProject(blank()); } }),
        el("button", { type: "button", text: "Delete", title: "delete this saved project", onclick: run(async () => { if (!P.name || !confirm(`Delete the saved project “${P.name}”?`)) return;
          projList((await api("/api/assemble/delete", { name: P.name })).projects); P.name = ""; commit(); syncTop(); }) }),
        el("span", { class: "sp" }),
        el("span", { id: "asInfo", class: "mono dim" })),
      el("div", { class: "as-props", id: "asProps" }),
      el("div", { class: "as-tl" },
        el("div", { class: "as-heads" },
          el("div", { style: `height:${RULER_H}px`, class: "as-lh ruler", text: "click: cursor" }),
          ...Array.from({ length: LANES }, (_, i) => laneHead(i)),
          el("div", { class: "as-lh master", style: `height:${MASTER_H}px` }, el("b", { text: "Master" }),
            (() => { const r = el("input", { type: "range", id: "asMVol", min: -30, max: 12, step: 0.5 }); r.addEventListener("input", () => { P.master.vol = +r.value; invalidate(); });
              r.addEventListener("change", commit); r.addEventListener("dblclick", () => { P.master.vol = 0; commit(); }); return r; })()),
          ...ENVS.map(envHead)),
        el("div", { class: "as-view", id: "asView" },
          el("canvas", { id: "asRuler", class: "as-ruler" }),
          el("div", { class: "as-lanes", id: "asLanes", style: `height:${LANES * LANE_H}px` }, el("canvas", { id: "asGrid" }), el("div", { id: "asClips" }), el("div", { id: "asDrop", class: "as-drop", hidden: true, style: `height:${LANE_H}px` })),
          el("canvas", { id: "asMaster", class: "as-master", title: "the result: what Play plays and Export saves" }),
          ...ENVS.map(e => el("canvas", { id: "asEnv-" + e.k, class: "as-env", title: `${e.name}: click to add a point, drag to move it (Shift: off the grid), right-click or double-click to delete` })),
          el("div", { id: "asCursor", class: "as-cursor" }), el("div", { id: "asHead", class: "as-head", hidden: true }))),
      el("input", { type: "range", id: "asScroll", class: "as-scroll", min: 0, value: 0, title: "scroll along (or Shift+wheel)" }),
      el("div", { class: "paneh as-top as-out" },
        el("b", { text: "Out" }),
        el("label", { class: "as-f" }, "File name", el("input", { id: "asFile", placeholder: "the project's name", onchange: e => { P.out.file = e.target.value.trim(); store.set("asm", P); autosave(); } })),
        sel_("asFmt", [["wav", "WAV"], ["flac", "FLAC"], ["ogg", "OGG"], ["mp3", "MP3"]], "format"),
        el("label", { class: "as-f c", title: "if the mix would clip, turn it down to just under 0 dB (never up)" }, el("input", { type: "checkbox", id: "asSafe", onchange: e => { P.master.safe = e.target.checked; commit(); } }), "no clipping"),
        el("button", { type: "button", text: "💾 Export", title: "save into ~/soundmon-gallery/assemble/exports", onclick: run(async () => { const j = await exportTo("file"); if (j) toast("Saved " + j.path); }) }),
        el("button", { type: "button", text: "Export as…", title: "pick the folder to save into", onclick: run(async () => {
          const d = await pickPath({ kind: "dir", start: asmFolders?.exports || "", title: "Save the sound into…" }); if (!d) return; const j = await exportTo("file", { folder: d }); if (j) toast("Saved " + j.path); }) }),
        el("button", { type: "button", text: "🎞 To clip bin", title: "put it in the clip bin that's open", onclick: run(async () => { const j = await exportTo("bin", { bin: BIN.name }); if (j) { binSet(j); toast(`In the clip bin “${BIN.name}” as ${j.file}`); } }) }),
        el("button", { type: "button", text: "✨ Make one like this", title: "make it the form's reference sound (Start from), then describe and RENDER", onclick: run(async () => {
          const j = await exportTo("ref"); if (!j) return; readForm(); if (!["sfx", "music", "song"].includes(eng)) { eng = "sfx"; store.set("engine", eng); }
          cur().ref = j.file; store.set("form", F); showEngine(); toast("It's the reference now: describe what to make from it and press RENDER."); }) }),
        el("button", { type: "button", text: "🔎 Find similar", title: "find the sounds in your library most like it", onclick: run(async () => { const j = await exportTo("ref"); if (j) findFrom({ input: j.file }, (P.name || "assembled") + ".wav"); }) }),
        el("span", { class: "sp" }), el("span", { id: "asStretcher", class: "dim" })),
      el("div", { id: "asTip", class: "as-tip", hidden: true }));
    $("asSnap").addEventListener("change", e => { P.snap = e.target.value; commit(); });
    $("asFmt").addEventListener("change", e => { P.out.fmt = e.target.value; store.set("asm", P); autosave(); });
    $("asFilt").addEventListener("change", e => { P.master.filter = e.target.value; commit(); });
    built = true; hookTimeline(); syncTop(); loadProjects(); savedSig = P.name ? snap() : ""; saved();
  }
  document.addEventListener("keydown", e => {
    if (typeof tab === "undefined" || tab !== "assemble" || /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName) || document.querySelector("dialog[open]")) return;
    const c = P.clips.find(x => x.id === sel), k = e.key.toLowerCase(), mod = e.ctrlKey || e.metaKey;
    if (e.key === " ") { e.preventDefault(); e.stopImmediatePropagation(); togglePlay(); }
    else if ((e.key === "Delete" || e.key === "Backspace") && c) { e.preventDefault(); remove(c); }
    else if (mod && k === "d" && c) { e.preventDefault(); dup(c); }
    else if (mod && k === "z" && !e.shiftKey) { e.preventDefault(); stepHistory(undo, redo); }
    else if (mod && (k === "y" || (k === "z" && e.shiftKey))) { e.preventDefault(); stepHistory(redo, undo); }
    else if (e.key === "Escape") { stopPlay_(); sel = null; draw(); }
    else if (e.key === "Home") { P.cursor = 0; drawHead(); }
  }, true);
  window.asmShow = () => { if (!built) build(); view.pps = view.pps || 0; draw(); invalidate(); };
  window.asmFolders = () => asmFolders;
})();
