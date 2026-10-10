import { api, BEV, color, drawBox, esc, fitCanvas, h, pretty, streamJob, VRU, wsUrl } from './util.js';

const STATUS_COL = { visible: '#34d399', occluded: '#ef4444', undetected: '#a78bfa', out_of_view: '#64748b', uncalibrated: '#64748b' };

export default async function guardian(root, args) {
  root.append(h(`<section class="hero" style="padding-top:.5rem">
    <span class="pill cyan">Guardian · cooperative perception</span>
    <h1 style="margin-top:.5rem">See what your camera cannot</h1>
    <p>Each camera detects and tracks road users and projects them onto a shared ground plane, with uncertainty. The network
      associates and fuses the observations, then tells every camera which objects are <b>hidden from it</b> and what is hiding them.
      This is a research demonstrator: advisory only, never connected to vehicle control.</p>
  </section>`));
  const tabs = h(`<div class="tabs">
    <button data-t="demo">Demo intersection (V2X-Sim)</button>
    <button data-t="videos">Your two videos</button>
    <button data-t="live">Live: two phones</button>
  </div>`);
  root.append(tabs);
  const pane = h('<div></div>');
  root.append(pane);
  let cleanup = null;
  const open = t => {
    if (cleanup) { cleanup(); cleanup = null; }
    tabs.querySelectorAll('button').forEach(b => b.classList.toggle('active', b.dataset.t === t));
    pane.innerHTML = '';
    cleanup = ({ demo, videos, live })[t](pane, args.slice(1));
  };
  tabs.onclick = e => e.target.dataset.t && open(e.target.dataset.t);
  open(['demo', 'videos', 'live'].includes(args[0]) ? args[0] : 'demo');
  return () => cleanup && cleanup();
}

// ------------------------------------------------------------------ shared rendering
function drawCamView(canvas, imgEl, view, srcW, opts = {}) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  const ctx = fitCanvas(canvas, Math.round(r.width * dpr), Math.round(r.height * dpr));
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!view) return;
  const s = canvas.width / srcW;
  for (const tr of view.own_tracks || view.tracks || []) drawBox(ctx, tr.bbox, `${pretty(tr.cls)} #${tr.id}`, color(tr.cls), { scale: s, lw: 2 * dpr });
  for (const o of view.objects || []) {
    if (!o.ghost_box || !['occluded', 'undetected'].includes(o.status)) continue;
    const src = (o.sources || []).map(c => opts.camLabel?.(c) || c).join(', ');
    const lbl = o.status === 'occluded'
      ? `HIDDEN ${pretty(o.cls)} · behind ${pretty(o.occluded_by?.cls)} #${o.occluded_by?.id} · ${o.dist_m} m`
      : `MISSED ${pretty(o.cls)} · ${o.dist_m} m`;
    drawBox(ctx, o.ghost_box, lbl, o.status === 'occluded' ? '#ef4444' : '#a78bfa', { scale: s, dashed: true, lw: 2.5 * dpr, fill: true });
    if (src) {
      ctx.font = `${11 * dpr}px system-ui`; ctx.fillStyle = '#e6edf7';
      ctx.fillText(`from ${src} · ${o.age_ms} ms old`, o.ghost_box[0] * s + 3, o.ghost_box[3] * s - 4);
    }
  }
}

function drawTracks(canvas, tracks, srcW) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  const ctx = fitCanvas(canvas, Math.round(r.width * dpr), Math.round(r.height * dpr));
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const s = canvas.width / srcW;
  for (const tr of tracks || []) drawBox(ctx, tr.bbox, '', color(tr.cls), { scale: s, lw: 1.5 * dpr });
}

function drawWorld(bev, { cams = {}, ego, objects = [], gt = [], showGt = true }) {
  bev.begin();
  for (const [c, info] of Object.entries(cams)) {
    if (info.fov) bev.poly(info.fov, c === ego ? '#ffb02088' : '#38bdf855', c === ego ? '#ffb02014' : '#38bdf80d');
  }
  if (showGt) for (const g of gt) bev.box(g.xy, g.yaw, g.l, g.w, '#94a3b8aa');
  for (const o of objects) {
    const col = color(o.cls);
    bev.trail(o.trail, col);
    bev.dot(o.xy, o.status === 'occluded' ? 6 : 4.5, col, o.status === 'occluded' ? `${pretty(o.cls)} (hidden)` : '', o.r2s);
    if (o.status && o.status !== 'visible') {
      const [x, y] = bev.p(o.xy), d = window.devicePixelRatio || 1;
      bev.ctx.beginPath(); bev.ctx.arc(x, y, 9 * d, 0, Math.PI * 2); bev.ctx.strokeStyle = STATUS_COL[o.status]; bev.ctx.lineWidth = 2 * d; bev.ctx.stroke();
    }
  }
  for (const [c, info] of Object.entries(cams)) {
    if (info.pos) bev.dot(info.pos, c === ego ? 7 : 5, c === ego ? '#ffb020' : '#38bdf8', info.label || c);
  }
}

function alertsHtml(alerts, camLabel) {
  if (!alerts?.length) return '<p class="muted">No hidden road users for this camera right now.</p>';
  return alerts.slice(0, 12).map(a => `<div class="alert ${a.vru ? 'vru' : ''} ${a.status}">
    <div class="t">${a.status === 'occluded' ? 'Hidden' : 'Missed by own camera'}: ${pretty(a.cls)} · ${a.distance_m} m
      ${a.occluded_by ? `· behind ${pretty(a.occluded_by.cls)} #${a.occluded_by.id}` : ''}</div>
    <div class="m">from ${a.sources.map(s => camLabel?.(s) || s).join(', ')} · age ${a.age_ms} ms · ±${a.uncertainty_m} m · TTL ${a.ttl_ms} ms · priority ${a.priority}</div>
  </div>`).join('');
}

// ------------------------------------------------------------------ demo (V2X-Sim)
function demo(pane) {
  const ui = h(`<div class="stack">
    <div class="card stack">
      <div class="row between">
        <div class="row">
          <label class="field">Ego camera (receives alerts)
            <select id="ego"><option value="CAM_FRONT_id_1">Vehicle 1 · front (stopped behind a car)</option>
              <option value="CAM_FRONT_id_4">Vehicle 4 · front</option></select></label>
          <label class="field">Sources
            <span class="row" id="srcs"></span></label>
        </div>
        <div class="row">
          <label class="row muted"><input type="checkbox" id="gt" checked> ground truth in map</label>
          <button class="btn primary" id="run">Run live on server</button>
        </div>
      </div>
      <div class="progress"><div id="pbar"></div></div>
      <p class="note">CARLA scene from the public V2X-Sim 2.0 dataset (Apache-2.0, synthetic, so no real people). We bundle only raw
        frames and calibration. Detection, tracking, fusion and occlusion reasoning run on our server when you press Run.</p>
    </div>
    <div class="split">
      <div class="stack">
        <div class="stage" id="ego-stage"><img id="ego-img" alt="ego camera"><canvas class="overlay" id="ego-ov"></canvas>
          <span class="badge" id="ego-badge">ego camera</span><div class="banner hidden" id="ego-banner"></div></div>
        <div class="thumbs" id="thumbs"></div>
        <div class="row"><button class="btn" id="play" disabled>▶ Replay</button><input type="range" class="scrub" id="scrub" min="0" max="0" value="0" disabled></div>
      </div>
      <div class="stack">
        <canvas class="bev" id="bev"></canvas>
        <div class="legend"><span><i style="background:#ffb020"></i>ego camera &amp; FOV</span><span><i style="background:#38bdf8"></i>other cameras</span>
          <span><i style="border:2px solid #ef4444;background:none"></i>hidden from ego</span><span><i style="border:2px solid #a78bfa;background:none"></i>missed by ego</span>
          <span><i style="border:1px dashed #94a3b8;background:none"></i>ground truth</span></div>
        <div class="card flat"><h3>Alerts for the ego camera</h3><div class="alerts" id="alerts"><p class="muted">Press Run.</p></div></div>
        <div class="card flat"><h3>Timing (this server)</h3><div id="timing" class="mono muted">—</div></div>
        <div class="card flat"><h3>This run</h3><div id="kpis" class="grid g2"></div></div>
      </div>
    </div>
  </div>`);
  pane.append(ui);
  const bev = new BEV(ui.querySelector('#bev'));
  let scene = null, job = null, frames = [], cur = 0, timer = 0;
  const camLabel = c => scene?.cameras?.[c] || c;

  const preview = () => { if (scene && !frames.length) ui.querySelector('#ego-img').src = `/demo/v2xsim/frames/${scene.preview[ui.querySelector('#ego').value]}`; };
  ui.querySelector('#ego').onchange = preview;
  api('/api/v1/guardian/demo/scene').then(s => {
    scene = s;
    preview();
    const srcs = ui.querySelector('#srcs');
    for (const [c, label] of Object.entries(s.cameras)) {
      if (c.startsWith('CAM_FRONT')) continue;
      srcs.append(h(`<label class="row" style="gap:.25rem"><input type="checkbox" value="${c}" checked> ${esc(label.replace('RSU · ', 'RSU '))}</label>`));
    }
    srcs.append(h(`<label class="row" style="gap:.25rem"><input type="checkbox" value="V2V"> other vehicle</label>`));
  }).catch(e => { ui.querySelector('#alerts').innerHTML = `<p class="error">${esc(e.message)}</p>`; });

  function show(i) {
    const m = frames[i]; if (!m) return;
    cur = i;
    ui.querySelector('#scrub').value = i;
    const egoImg = ui.querySelector('#ego-img');
    const egoCam = m.cams[m.ego];
    egoImg.src = `/demo/v2xsim/frames/${egoCam.img}`;
    ui.querySelector('#ego-badge').textContent = `${camLabel(m.ego)} · t=${m.t.toFixed(1)} s`;
    const draw = () => drawCamView(ui.querySelector('#ego-ov'), egoImg, m.view, 800, { camLabel });
    if (egoImg.complete) draw(); else egoImg.onload = draw;
    const hidden = m.view.alerts.filter(a => a.status === 'occluded');
    const ban = ui.querySelector('#ego-banner');
    if (hidden.length) {
      ban.classList.remove('hidden');
      const vru = hidden.filter(a => a.vru);
      ban.textContent = `⚠ ${hidden.length} road user(s) hidden from this camera${vru.length ? `, ${vru.length} vulnerable (${[...new Set(vru.map(a => pretty(a.cls)))].join(', ')})` : ''}, revealed by the network`;
    } else ban.classList.add('hidden');
    // thumbnails
    const th = ui.querySelector('#thumbs');
    const others = Object.keys(m.cams).filter(c => c !== m.ego);
    if (th.children.length !== others.length) {
      th.innerHTML = '';
      for (const c of others) th.append(h(`<div class="stage" data-c="${c}"><img alt=""><canvas class="overlay"></canvas><span class="badge">${esc(camLabel(c))}</span></div>`));
    }
    for (const node of th.children) {
      const c = node.dataset.c, im = node.querySelector('img'), cv = node.querySelector('canvas');
      im.src = `/demo/v2xsim/frames/${m.cams[c].img}`;
      const d = () => drawTracks(cv, m.cams[c].tracks, 800);
      if (im.complete) d(); else im.onload = d;
    }
    // bird's-eye view
    if (!bev.view) {
      const pts = [...m.cams[m.ego].fov, ...others.map(c => m.cams[c].pos)];
      bev.fit(pts, 4);
    }
    const cams = Object.fromEntries(Object.entries(m.cams).map(([c, v]) => [c, { ...v, label: c === m.ego ? 'ego' : '' }]));
    drawWorld(bev, { cams, ego: m.ego, objects: m.view.objects, gt: m.gt, showGt: ui.querySelector('#gt').checked });
    ui.querySelector('#alerts').innerHTML = alertsHtml(m.view.alerts, camLabel);
    const tm = m.timing;
    ui.querySelector('#timing').innerHTML = `detector per camera: ${Object.entries(tm.detect_ms).map(([c, v]) => `${camLabel(c).split('·').pop().trim()} ${v}`).join(' · ')} ms<br>
      fusion + occlusion reasoning: ${tm.fusion_ms} ms<br>frame total (all cameras on one CPU): ${tm.frame_total_ms} ms<br>
      with one edge device per camera: ≈ ${Math.round(tm.detect_parallel_ms + tm.fusion_ms)} ms + network`;
    // run KPIs (over frames so far)
    const seenHidden = new Set(), seenVru = new Set();
    for (const f of frames.slice(0, i + 1)) for (const a of f.view.alerts) if (a.status === 'occluded') { seenHidden.add(a.gid); if (a.vru) seenVru.add(a.gid); }
    const egoOnly = m.view.own_tracks.length, fused = m.view.objects.filter(o => o.status !== 'out_of_view').length;
    ui.querySelector('#kpis').innerHTML = `
      <div class="kpi"><span class="v">${egoOnly}</span><span class="l">objects the ego camera sees itself</span></div>
      <div class="kpi"><span class="v">${fused}</span><span class="l">objects known after fusion (in/near view)</span></div>
      <div class="kpi"><span class="v">${seenHidden.size}</span><span class="l">distinct hidden objects revealed so far</span></div>
      <div class="kpi"><span class="v">${seenVru.size}</span><span class="l">of them vulnerable road users</span></div>`;
  }

  ui.querySelector('#gt').onchange = () => show(cur);
  ui.querySelector('#scrub').oninput = e => { stopPlay(); show(+e.target.value); };
  const stopPlay = () => { clearInterval(timer); timer = 0; ui.querySelector('#play').textContent = '▶ Replay'; };
  ui.querySelector('#play').onclick = () => {
    if (timer) return stopPlay();
    ui.querySelector('#play').textContent = '❚❚ Pause';
    let i = cur >= frames.length - 1 ? 0 : cur;
    timer = setInterval(() => { show(i); i++; if (i >= frames.length) stopPlay(); }, 200);
  };
  ui.querySelector('#run').onclick = async () => {
    stopPlay();
    if (job) job.cancel();
    frames = []; bev.view = null;
    const ego = ui.querySelector('#ego').value;
    const sources = [...ui.querySelectorAll('#srcs input:checked')].map(i => i.value)
      .flatMap(v => v === 'V2V' ? [ego === 'CAM_FRONT_id_1' ? 'CAM_FRONT_id_4' : 'CAM_FRONT_id_1'] : [v]);
    ui.querySelector('#run').disabled = true;
    ui.querySelector('#alerts').innerHTML = '<p class="muted">Processing on server…</p>';
    try {
      const r = await api('/api/v1/guardian/demo/run', { method: 'POST', json: { ego, sources } });
      job = streamJob(r.job_id, {
        frame: m => { frames.push(m); if (!timer) show(frames.length - 1); },
        progress: p => { ui.querySelector('#pbar').style.width = `${Math.round(p * 100)}%`; },
        done: () => {
          ui.querySelector('#run').disabled = false;
          const sc = ui.querySelector('#scrub'); sc.max = frames.length - 1; sc.disabled = false;
          ui.querySelector('#play').disabled = false;
        },
        error: e => { ui.querySelector('#run').disabled = false; ui.querySelector('#alerts').innerHTML = `<p class="error">${esc(e)}</p>`; },
      });
    } catch (e) { ui.querySelector('#run').disabled = false; ui.querySelector('#alerts').innerHTML = `<p class="error">${esc(e.message)}</p>`; }
  };
  const onResize = () => frames.length && show(cur);
  window.addEventListener('resize', onResize);
  return () => { stopPlay(); if (job) job.cancel(); window.removeEventListener('resize', onResize); };
}

// ------------------------------------------------------------------ calibration helper
// Lets the user tap 4 ground points (rectangle corners) on a frame; returns image coordinates in source px.
function calibrator(container, source, w, hgt, onChange) {
  const st = h(`<div class="stage" style="cursor:crosshair"><canvas></canvas></div>`);
  container.append(st);
  const cv = st.querySelector('canvas');
  cv.width = w; cv.height = hgt;
  const ctx = cv.getContext('2d');
  let pts = [];
  const redraw = () => {
    ctx.drawImage(source, 0, 0, w, hgt);
    ctx.lineWidth = Math.max(2, w / 400);
    if (pts.length) {
      ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(...p) : ctx.moveTo(...p));
      if (pts.length === 4) ctx.closePath();
      ctx.strokeStyle = '#ffb020'; ctx.stroke();
    }
    pts.forEach((p, i) => {
      ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(6, w / 160), 0, Math.PI * 2); ctx.fillStyle = '#ffb020'; ctx.fill();
      ctx.fillStyle = '#0b1220'; ctx.font = `bold ${Math.max(12, w / 80)}px system-ui`; ctx.fillText(String(i + 1), p[0] - 4, p[1] + 5);
    });
  };
  redraw();
  cv.addEventListener('click', e => {
    const r = cv.getBoundingClientRect();
    const p = [(e.clientX - r.left) * w / r.width, (e.clientY - r.top) * hgt / r.height];
    if (pts.length >= 4) pts = [];
    pts.push(p.map(v => Math.round(v * 10) / 10));
    redraw(); onChange(pts);
  });
  return { reset: () => { pts = []; redraw(); onChange(pts); }, points: () => pts, redraw };
}

const CALIB_HELP = `Tap 4 points on the <b>ground</b> that form a rectangle you can measure, e.g. parking-bay corners or floor tiles:
  1 = near-left, 2 = near-right, 3 = far-right, 4 = far-left. Use the <b>same physical points in the same order</b> on every camera.`;

// ------------------------------------------------------------------ two uploaded videos
function videos(pane) {
  const ui = h(`<div class="stack">
    <div class="card stack">
      <p class="note">${CALIB_HELP} The rectangle defines the shared ground frame (metres). The cameras must be static.</p>
      <div class="row"><label class="field">Rectangle width 1→2 (m)<input type="number" id="rw" value="2.5" step="0.1" min="0.2"></label>
        <label class="field">Rectangle length 2→3 (m)<input type="number" id="rl" value="5.0" step="0.1" min="0.2"></label>
        <label class="field">B starts later by (s)<input type="number" id="off" value="0" step="0.1"></label>
        <button class="btn primary" id="run" disabled>Fuse both videos on server</button></div>
      <div class="progress"><div id="pbar"></div></div>
    </div>
    <div class="grid g2">
      ${['A', 'B'].map(k => `<div class="card stack" id="cam${k}">
        <h2>Camera ${k}</h2>
        <label class="drop"><input type="file" accept="video/*"><b>Choose video ${k}</b></label>
        <div class="calib"></div>
        <div class="mono muted geo"></div>
        <div class="stage hidden play"><video playsinline muted></video><canvas class="overlay"></canvas><div class="banner hidden"></div></div>
        <div class="alerts"></div>
      </div>`).join('')}
    </div>
    <div class="grid g2"><canvas class="bev" id="bev"></canvas><div class="card flat"><h3>Fused objects</h3><div id="objs" class="mono muted">—</div></div></div>
  </div>`);
  pane.append(ui);
  const cams = { A: {}, B: {} };
  const bev = new BEV(ui.querySelector('#bev'));
  let frames = [], job = null, raf = 0;
  const ready = () => ui.querySelector('#run').disabled = !(cams.A.pts?.length === 4 && cams.B.pts?.length === 4);

  for (const k of ['A', 'B']) {
    const box = ui.querySelector(`#cam${k}`);
    const input = box.querySelector('input[type=file]');
    box.querySelector('.drop').onclick = () => input.click();
    input.onchange = async () => {
      const f = input.files[0]; if (!f) return;
      cams[k].file = f;
      const v = document.createElement('video');
      v.muted = true; v.playsInline = true; v.src = URL.createObjectURL(f);
      await new Promise(r => { v.onloadeddata = r; });
      v.currentTime = Math.min(0.5, v.duration / 2);
      await new Promise(r => { v.onseeked = r; });
      cams[k].w = v.videoWidth; cams[k].h = v.videoHeight;
      const cal = box.querySelector('.calib'); cal.innerHTML = '';
      calibrator(cal, v, v.videoWidth, v.videoHeight, async pts => {
        cams[k].pts = pts; ready();
        const g = box.querySelector('.geo');
        if (pts.length !== 4) { g.textContent = `${pts.length}/4 points`; return; }
        try {
          const d = await api('/api/v1/guardian/calibrate', { method: 'POST', json: calibJson(k) });
          cams[k].geo = d;
          g.innerHTML = `camera height ${d.height_m} m · pitch ${d.pitch_deg}° · focal ${d.focal_px}px · fit error ${d.calib_rmse_m} m${d.warnings.length ? `<br><span class="error">${d.warnings.join('; ')}</span>` : ''}`;
        } catch (e) { g.innerHTML = `<span class="error">${esc(e.message)}</span>`; }
      });
      const pv = box.querySelector('.play video'); pv.src = v.src;
    };
  }
  const calibJson = k => ({ width: cams[k].w, height: cams[k].h, image_points: cams[k].pts,
    rect_w: +ui.querySelector('#rw').value, rect_l: +ui.querySelector('#rl').value });

  ui.querySelector('#run').onclick = async () => {
    if (job) job.cancel();
    frames = []; bev.view = null;
    const fd = new FormData();
    fd.append('video_a', cams.A.file); fd.append('video_b', cams.B.file);
    fd.append('calib_a', JSON.stringify(calibJson('A'))); fd.append('calib_b', JSON.stringify(calibJson('B')));
    fd.append('offset_s', ui.querySelector('#off').value);
    ui.querySelector('#run').disabled = true;
    try {
      const r = await api('/api/v1/guardian/videos', { method: 'POST', body: fd });
      for (const k of ['A', 'B']) { const box = ui.querySelector(`#cam${k}`); box.querySelector('.calib').classList.add('hidden'); box.querySelector('.play').classList.remove('hidden'); }
      job = streamJob(r.job_id, {
        frame: m => frames.push(m),
        progress: p => { ui.querySelector('#pbar').style.width = `${Math.round(p * 100)}%`; },
        warning: w => { ui.querySelector('#objs').textContent = w; },
        done: () => {
          ui.querySelector('#run').disabled = false;
          for (const k of ['A', 'B']) {
            const v = ui.querySelector(`#cam${k} .play video`);
            v.currentTime = k === 'B' ? +ui.querySelector('#off').value : 0; v.play().catch(() => {});
          }
        },
        error: e => { ui.querySelector('#run').disabled = false; ui.querySelector('#objs').innerHTML = `<span class="error">${esc(e)}</span>`; },
      });
    } catch (e) { ui.querySelector('#run').disabled = false; ui.querySelector('#objs').innerHTML = `<span class="error">${esc(e.message)}</span>`; }
  };

  const loop = () => {
    const vA = ui.querySelector('#camA .play video');
    if (frames.length && vA && !vA.paused) {
      const t = vA.currentTime;
      let m = null; for (const f of frames) { if (f.t <= t + 0.05) m = f; else break; }
      if (m) {
        for (const k of ['A', 'B']) {
          const box = ui.querySelector(`#cam${k}`);
          drawCamView(box.querySelector('.play canvas'), null, m.views[k], cams[k].w);
          box.querySelector('.alerts').innerHTML = alertsHtml(m.views[k].alerts, c => `camera ${c}`);
          const hidden = m.views[k].alerts.filter(a => a.status === 'occluded');
          const ban = box.querySelector('.banner');
          ban.classList.toggle('hidden', !hidden.length);
          if (hidden.length) ban.textContent = `⚠ ${hidden.length} hidden: ${hidden.map(a => pretty(a.cls)).join(', ')}`;
        }
        if (!bev.view) bev.fit([...m.fov.A, ...m.fov.B], 3);
        const objs = m.views.A.objects.map(o => ({ ...o, status: o.status === 'visible' ? 'visible' : o.status }));
        drawWorld(bev, { cams: { A: { fov: m.fov.A, pos: m.pos.A, label: 'A' }, B: { fov: m.fov.B, pos: m.pos.B, label: 'B' } }, ego: 'A', objects: objs });
        ui.querySelector('#objs').innerHTML = m.views.A.objects.map(o => `${pretty(o.cls)} g${o.gid} · (${o.xy[0]}, ${o.xy[1]}) m · ±${o.r2s} m · seen by ${o.sources.join('+')} · ${o.status} for A`).join('<br>') || 'no objects';
      }
      const vB = ui.querySelector('#camB .play video');
      const want = t + +ui.querySelector('#off').value;
      if (vB && Math.abs(vB.currentTime - want) > 0.3) vB.currentTime = want;
    }
    raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);
  return () => { if (job) job.cancel(); cancelAnimationFrame(raf); };
}

// ------------------------------------------------------------------ live room with phones
function live(pane, args) {
  const room = (args[0] || Math.random().toString(36).slice(2, 6)).toUpperCase();
  const link = `${location.origin}${location.pathname}#/guardian/live/${room}`;
  const ui = h(`<div class="stack">
    <div class="card stack">
      <div class="row between">
        <div><h2>Room <span class="mono" style="color:var(--accent)">${room}</span></h2>
          <span class="muted">Open this link on every device: <a href="${link}" class="mono">${link}</a></span></div>
        <div class="row" id="roles">
          <button class="btn" data-r="A">This phone = camera A</button>
          <button class="btn" data-r="B">This phone = camera B</button>
          <button class="btn" data-r="viewer">Viewer (map)</button>
        </div>
      </div>
      <p class="note">Put two phones on stands looking at the same area from different angles, e.g. one view blocked by a parked car.
        ${CALIB_HELP} Each phone sends ~3–5 frames/s. Alerts show the measured end-to-end latency from the other camera's capture to this screen.</p>
      <div id="roster" class="mono muted">connecting…</div>
    </div>
    <div id="work"></div>
  </div>`);
  pane.append(ui);
  let ws = null, stream = null, timer = 0, offset = 0;
  const work = ui.querySelector('#work');

  function connect(onMsg) {
    ws = new WebSocket(wsUrl(`/ws/guardian/${room}`));
    const pings = [];
    ws.onopen = () => { for (let i = 0; i < 5; i++) setTimeout(() => ws?.readyState === 1 && ws.send(JSON.stringify({ type: 'ping', t: Date.now() })), i * 150); };
    ws.onmessage = ev => {
      const m = JSON.parse(ev.data);
      if (m.type === 'pong') {
        const now = Date.now(), rtt = now - m.t;
        pings.push({ rtt, off: m.server - (m.t + rtt / 2) });
        offset = pings.sort((a, b) => a.rtt - b.rtt)[0].off;  // min-RTT sample is the most reliable
      } else if (m.type === 'roster') {
        const cams = Object.values(m.info.cameras);
        ui.querySelector('#roster').textContent = `cameras: ${cams.map(c => `${c.cam}${c.calibrated ? ' ✓' : ' (not calibrated)'}`).join(', ') || 'none'} · viewers: ${m.info.viewers} · clock offset ${Math.round(offset)} ms`;
      }
      onMsg?.(m);
    };
    ws.onclose = () => { ui.querySelector('#roster').textContent = 'disconnected'; };
  }

  ui.querySelector('#roles').onclick = async e => {
    const r = e.target.dataset.r; if (!r) return;
    stop();
    ui.querySelectorAll('#roles button').forEach(b => b.classList.toggle('primary', b.dataset.r === r));
    if (r === 'viewer') return viewer();
    return camera(r);
  };

  function viewer() {
    work.innerHTML = '';
    const v = h(`<div class="grid g2"><canvas class="bev" id="bev"></canvas><div class="card flat"><h3>Alerts per camera</h3><div id="va" class="alerts"></div></div></div>`);
    work.append(v);
    const bev = new BEV(v.querySelector('#bev'));
    connect(m => {
      if (m.type === 'welcome') ws.send(JSON.stringify({ type: 'hello', role: 'viewer' }));
      if (m.type !== 'world') return;
      const pts = Object.values(m.cams).flatMap(c => c.fov || []);
      if (pts.length) { bev.fit(pts, 2); drawWorld(bev, { cams: Object.fromEntries(Object.entries(m.cams).map(([k, c]) => [k, { ...c, label: k }])), objects: m.objects }); }
      v.querySelector('#va').innerHTML = Object.entries(m.alerts).map(([c, a]) => `<b>camera ${c}</b>${alertsHtml(a, x => 'camera ' + x)}`).join('');
    });
  }

  async function camera(cam) {
    work.innerHTML = '';
    const v = h(`<div class="split"><div class="stack">
        <div class="stage" id="st"><video playsinline muted autoplay></video><canvas class="overlay"></canvas><div class="banner hidden"></div><span class="badge">camera ${cam}</span></div>
        <div class="row"><button class="btn" id="freeze">Calibrate: freeze frame &amp; tap 4 ground points</button>
          <label class="field">width 1→2 (m)<input type="number" id="rw" value="2.5" step="0.1"></label>
          <label class="field">length 2→3 (m)<input type="number" id="rl" value="5" step="0.1"></label>
          <button class="btn primary" id="send" disabled>Use calibration &amp; start streaming</button></div>
        <div id="cal"></div><div class="mono muted" id="geo"></div>
      </div>
      <div class="stack"><div class="card flat"><h3>Network alerts for camera ${cam}</h3><div class="alerts" id="al"><p class="muted">Calibrate to start.</p></div></div>
        <div class="card flat"><h3>Latency</h3><div class="mono muted" id="lat">—</div></div></div></div>`);
    work.append(v);
    const video = v.querySelector('video');
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } }, audio: false });
    } catch (e) { v.querySelector('#geo').innerHTML = `<span class="error">camera unavailable: ${esc(e.message)} (HTTPS and permission are required)</span>`; return; }
    video.srcObject = stream; await video.play();
    const W = 640, scale = () => W / Math.max(video.videoWidth, video.videoHeight);
    let pts = [], streaming = false, busy = false, lastView = null;
    const lat = [];
    connect(m => {
      if (m.type === 'welcome') ws.send(JSON.stringify({ type: 'hello', role: 'camera', cam }));
      if (m.type === 'calib_ok') { v.querySelector('#geo').textContent = `calibrated · height ${m.geometry.height_m} m · pitch ${m.geometry.pitch_deg}° · focal ${m.geometry.focal_px}px`; streaming = true; }
      if (m.type === 'calib_error') v.querySelector('#geo').innerHTML = `<span class="error">${esc(m.error)}</span>`;
      if (m.type === 'view') {
        if (m.source_cam === cam) busy = false;
        lastView = m;
        const now = Date.now() + offset;
        const e2e = now - m.source_capture_ms;
        if (m.source_cam !== cam) { lat.push(e2e); if (lat.length > 50) lat.shift(); }
        const med = lat.length ? [...lat].sort((a, b) => a - b)[Math.floor(lat.length / 2)] : null;
        v.querySelector('#lat').innerHTML = `server processing ${m.processing_ms} ms · upload ${m.upload_ms} ms<br>
          ${med != null ? `capture on other camera → alert here: median ${Math.round(med)} ms (n=${lat.length})` : 'waiting for observations from another camera'}`;
        const sw = Math.round(video.videoWidth * scale());
        drawCamView(v.querySelector('canvas'), null, m, sw, { camLabel: c => 'camera ' + c });
        v.querySelector('#al').innerHTML = alertsHtml(m.alerts, c => 'camera ' + c);
        const hidden = m.alerts.filter(a => a.status === 'occluded');
        const ban = v.querySelector('.banner');
        ban.classList.toggle('hidden', !hidden.length);
        if (hidden.length) ban.textContent = `⚠ Hidden ${pretty(hidden[0].cls)} ${hidden[0].distance_m} m · behind ${pretty(hidden[0].occluded_by?.cls)} · from camera ${hidden[0].sources.join(',')}`;
      }
    });
    v.querySelector('#freeze').onclick = () => {
      const c = document.createElement('canvas');
      const s = scale();
      c.width = Math.round(video.videoWidth * s); c.height = Math.round(video.videoHeight * s);
      c.getContext('2d').drawImage(video, 0, 0, c.width, c.height);
      const cal = v.querySelector('#cal'); cal.innerHTML = '';
      calibrator(cal, c, c.width, c.height, p => { pts = p; v.querySelector('#send').disabled = p.length !== 4; });
    };
    v.querySelector('#send').onclick = () => {
      const s = scale();
      ws.send(JSON.stringify({ type: 'calib', calib: { width: Math.round(video.videoWidth * s), height: Math.round(video.videoHeight * s), image_points: pts,
        rect_w: +v.querySelector('#rw').value, rect_l: +v.querySelector('#rl').value } }));
      v.querySelector('#cal').innerHTML = '';
    };
    const grab = document.createElement('canvas');
    let lastSent = 0;
    timer = setInterval(() => {
      if (!streaming || busy || !ws || ws.readyState !== 1 || Date.now() - lastSent < 200) return;
      const s = scale();
      grab.width = Math.round(video.videoWidth * s); grab.height = Math.round(video.videoHeight * s);
      grab.getContext('2d').drawImage(video, 0, 0, grab.width, grab.height);
      const tCap = Date.now();
      grab.toBlob(b => {
        if (!b || !ws || ws.readyState !== 1) return;
        busy = true; lastSent = Date.now();
        ws.send(JSON.stringify({ type: 'frame', t_capture: tCap, offset_ms: offset }));
        ws.send(b);
        setTimeout(() => { busy = false; }, 2500);  // never stall if a response is lost
      }, 'image/jpeg', 0.7);
    }, 50);
  }

  function stop() {
    clearInterval(timer); timer = 0;
    if (ws) { try { ws.close(); } catch {} ws = null; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  }
  if (args[1] === 'viewer') viewer();
  return stop;
}
