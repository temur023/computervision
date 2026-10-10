import { api, C, DAMAGE, deviceId, drawBox, dropZone, esc, fitCanvas, geolocate, h, pretty, streamJob, VRU, wsUrl } from './util.js';

export const TASHKENT = [41.3111, 69.2797];
export function baseLayer() {
  return L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '© OpenStreetMap contributors' });
}

// Draw one analysed frame onto the overlay canvas. Boxes come in analysis-resolution pixels (srcW wide).
function drawFrame(stage, ov, fr, srcW) {
  const r = stage.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const ctx = fitCanvas(ov, Math.round(r.width * dpr), Math.round(r.height * dpr));
  ctx.clearRect(0, 0, ov.width, ov.height);
  if (!fr) return;
  const s = ov.width / srcW;
  for (const d of fr.damage || []) drawBox(ctx, d.bbox, `${pretty(d.cls)} ${Math.round(d.conf * 100)}`, C.hazard, { scale: s, fill: true });
  for (const u of fr.users || []) {
    const lbl = `${pretty(u.cls)}${u.ttc ? ` · ttc ${u.ttc}s` : ''}`;
    drawBox(ctx, u.bbox, lbl, u.warn ? C.hazard : C.own, { scale: s, lw: u.warn ? 2.5 : 1.5 });
  }
}

export function mountScan(container) {
  const ui = h(`<div class="work">
    <div class="stack">
      <div class="row between">
        <div class="seg" id="mode"><button class="active" data-m="upload">Upload photo or video</button><button data-m="live">Live camera</button></div>
        <span class="mono faint" id="live-stats"></span>
      </div>
      <div id="upload-pane">
        <label class="drop" id="drop"><input type="file" accept="image/*,video/*">
          <b>Drop a road photo or video</b>
          <span class="faint">or click to choose · JPG, PNG, MP4, MOV, WebM · videos: first 60 s at 5 frames/s</span></label>
      </div>
      <div id="live-pane" class="hidden">
        <div class="row"><button class="btn primary" id="live-start">Start camera</button><button class="btn" id="live-stop" disabled>Stop</button></div>
        <p class="note" style="margin-top:.8rem">The phone sends 640 px frames to our server over a WebSocket and gets detections back.
          With GPS on, a damage track seen in three frames becomes a road-event candidate. Point the camera forward along the road.</p>
      </div>
      <div class="progress hidden" id="prog"><div></div></div>
      <div class="instrument hidden" id="stage">
        <video id="vid" playsinline muted controls class="hidden"></video>
        <img id="img" class="hidden" alt="analysed input">
        <canvas class="overlay" id="ov"></canvas>
        <span class="tag" id="tag">input</span>
        <div class="alertbar hidden" id="banner"><i></i><span></span></div>
      </div>
      <div class="legend"><span><i style="background:${C.own}; outline:1px solid #0001"></i>road user, this camera</span>
        <span><i style="background:${C.hazard}"></i>road damage · warning</span></div>
      <div id="results"></div>
    </div>
    <aside>
      <div class="panel">
        <span class="label">Location</span>
        <div id="minimap" class="map small"></div>
        <div class="row" style="margin-top:.7rem"><button class="btn" id="gps">Use device GPS</button>
          <span class="faint" style="font-size:.84rem">or click the map · photo GPS is read automatically</span></div>
        <p class="mono faint" id="geo" style="margin-top:.6rem;font-size:.78rem">No location: detections are shown, no road event is created.</p>
      </div>
      <div class="panel"><span class="label">Decisions</span><div id="decisions" class="list"><p class="faint">Run a scan to see the resulting actions.</p></div></div>
      <div class="panel"><span class="label">Measured on this server</span><div id="perf"><p class="faint">—</p></div></div>
    </aside>
  </div>`);
  container.append(ui);

  let geo = null, marker = null;
  const map = L.map(ui.querySelector('#minimap'), { zoomControl: false }).setView(TASHKENT, 12);
  baseLayer().addTo(map);
  const setGeo = (g, label) => {
    geo = g;
    if (marker) marker.remove();
    marker = L.circleMarker([g.lat, g.lon], { radius: 6, color: '#111316', weight: 2, fillColor: '#fff', fillOpacity: 1 }).addTo(map);
    map.setView([g.lat, g.lon], Math.max(map.getZoom(), 15));
    ui.querySelector('#geo').textContent = `${g.lat.toFixed(5)}, ${g.lon.toFixed(5)} · ±${Math.round(g.acc || 10)} m · ${label}`;
  };
  map.on('click', e => setGeo({ lat: e.latlng.lat, lon: e.latlng.lng, acc: 10 }, 'manual geotag'));
  ui.querySelector('#gps').onclick = async () => {
    const g = await geolocate();
    if (g) setGeo(g, 'device GPS'); else ui.querySelector('#geo').textContent = 'GPS unavailable: permission denied or no fix.';
  };

  const panes = { upload: ui.querySelector('#upload-pane'), live: ui.querySelector('#live-pane') };
  ui.querySelector('#mode').onclick = e => {
    const m = e.target.dataset.m; if (!m) return;
    ui.querySelectorAll('#mode button').forEach(b => b.classList.toggle('active', b.dataset.m === m));
    Object.entries(panes).forEach(([k, p]) => p.classList.toggle('hidden', k !== m));
    if (m !== 'live') stopLive();
  };

  const stage = ui.querySelector('#stage'), img = ui.querySelector('#img'), vid = ui.querySelector('#vid'), ov = ui.querySelector('#ov');
  const prog = ui.querySelector('#prog'), results = ui.querySelector('#results'), perf = ui.querySelector('#perf');
  let job = null, frames = [], raf = 0;
  const setProg = p => { prog.classList.toggle('hidden', p == null); if (p != null) prog.firstElementChild.style.width = `${Math.round(p * 100)}%`; };

  function showDecisions(dec, events) {
    const box = ui.querySelector('#decisions');
    box.innerHTML = '';
    if (!dec.length && !(events || []).length) { box.innerHTML = '<p class="faint">No hazard found in this input.</p>'; return; }
    for (const d of dec) {
      let t = '', m = '';
      if (d.kind === 'road_damage') { t = `${pretty(d.type)} ×${d.count} · ${d.severity} severity`; m = d.action; }
      else if (d.kind === 'driver_warning') { t = d.warning === 'forward_collision_risk' ? `Forward collision risk · ${pretty(d.cls)}` : `Vulnerable road user in path · ${pretty(d.cls)}`; m = `t = ${d.t} s${d.ttc_s ? ` · time to contact ${d.ttc_s} s` : ''} · ${d.action}`; }
      else if (d.kind === 'vru_presence') { t = `${d.count} vulnerable road user(s) in view`; m = d.action; }
      box.append(h(`<div class="item"><div><div class="t"><span class="dot" style="background:${d.kind === 'vru_presence' ? 'var(--ink-3)' : 'var(--hazard)'}"></span>${esc(t)}</div><div class="m">${esc(m)}</div></div></div>`));
    }
    for (const e of events || []) {
      box.append(h(`<a class="item" href="#/road/events/${e.id}"><div><div class="t">Road event #${e.id} · ${esc(pretty(e.type))}</div>
        <div class="m">${e.merged ? 'merged into existing event' : 'new candidate'} · <span class="status s-${e.status}">${e.status}</span>${e.confidence !== '' ? ` · confidence ${e.confidence}` : ''}${e.n_sources !== '' ? ` · ${e.n_sources} source(s)` : ''}</div></div><span class="mono faint">→</span></a>`));
    }
  }

  async function handleFile(file) {
    stopLive();
    if (job) job.cancel();
    cancelAnimationFrame(raf);
    results.innerHTML = '';
    stage.classList.remove('hidden');
    const isVideo = file.type.startsWith('video') || /\.(mp4|mov|webm|avi|mkv)$/i.test(file.name);
    const fd = new FormData();
    fd.append('file', file);
    fd.append('source_id', deviceId());
    if (geo) { fd.append('lat', geo.lat); fd.append('lon', geo.lon); fd.append('gps_accuracy_m', geo.acc || 10); }
    const t0 = performance.now();
    ui.querySelector('#tag').textContent = file.name.slice(0, 40);
    if (!isVideo) {
      vid.classList.add('hidden'); img.classList.remove('hidden');
      img.src = URL.createObjectURL(file);
      setProg(0.3);
      try {
        const r = await api('/api/v1/scan/image', { method: 'POST', body: fd });
        setProg(null);
        const rtt = performance.now() - t0;
        const draw = () => drawFrame(stage, ov, { users: r.road_users, damage: r.road_damage }, r.width);
        if (img.complete) draw(); else img.onload = draw;
        if (r.geotag?.lat && !geo) setGeo({ lat: r.geotag.lat, lon: r.geotag.lon, acc: r.geotag.accuracy_m }, 'photo EXIF');
        showDecisions(r.decisions, r.events);
        perf.innerHTML = `<table class="kv"><tbody>
          <tr><th>Road users</th><td>${r.timing_ms.road_users_ms} ms</td></tr>
          <tr><th>Road damage</th><td>${r.timing_ms.road_damage_ms} ms</td></tr>
          <tr><th>Server total</th><td>${r.server_ms} ms</td></tr>
          <tr><th>Round trip</th><td>${Math.round(rtt)} ms incl. upload</td></tr></tbody></table>`;
        const rows = [...r.road_damage.map(d => [pretty(d.cls), 'road damage', d.conf, d.severity_label]), ...r.road_users.map(u => [pretty(u.cls), 'road user', u.conf, '—'])];
        results.innerHTML = rows.length ? `<div class="scroll-x"><table class="tbl"><thead><tr><th>Object</th><th>Kind</th><th class="r">Confidence</th><th class="r">Severity</th></tr></thead><tbody>
          ${rows.map(x => `<tr><td>${esc(x[0])}</td><td class="faint">${x[1]}</td><td class="r">${x[2].toFixed(2)}</td><td class="r">${x[3]}</td></tr>`).join('')}</tbody></table></div>
          ${!geo && r.observations.length ? '<p class="note" style="margin-top:1rem">Damage found. Add a location (GPS or a map click) and run again to create a road event.</p>' : ''}`
          : '<p class="faint">Nothing detected.</p>';
      } catch (e) { setProg(null); results.innerHTML = `<p class="error">${esc(e.message)}</p>`; }
      return;
    }
    img.classList.add('hidden'); vid.classList.remove('hidden');
    vid.src = URL.createObjectURL(file);
    frames = [];
    setProg(0.02);
    perf.innerHTML = '<p class="faint">Uploading…</p>';
    let resp;
    try { resp = await api('/api/v1/scan/video', { method: 'POST', body: fd }); }
    catch (e) { results.innerHTML = `<p class="error">${esc(e.message)}</p>`; setProg(null); return; }
    perf.innerHTML = '<p class="faint">Processing on the server…</p>';
    const loop = () => {
      const t = vid.currentTime;
      let best = null;
      for (const f of frames) { if (f.t <= t + 0.05) best = f; else break; }
      if (best && t - best.t > 0.6) best = null;
      drawFrame(stage, ov, best, best?.w || 1280);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    let started = false;
    job = streamJob(resp.job_id, {
      frame: m => {
        frames.push(m.frame);
        if (!started && frames.length > 3) { started = true; vid.play().catch(() => {}); }
        perf.innerHTML = `<p class="mono faint">${frames.length} frames analysed · last ${m.frame.ms} ms</p>`;
      },
      progress: p => setProg(p),
      done: r => {
        setProg(null);
        showDecisions(r.decisions, r.events);
        perf.innerHTML = `<table class="kv"><tbody>
          <tr><th>Frames analysed</th><td>${r.perf.frames} at ${r.sample_fps.toFixed(1)} fps</td></tr>
          <tr><th>Per frame p50</th><td>${r.perf.ms_per_frame_p50} ms</td></tr>
          <tr><th>Per frame p95</th><td>${r.perf.ms_per_frame_p95} ms</td></tr>
          <tr><th>Wall time</th><td>${r.perf.wall_s} s for ${r.duration_s} s of video</td></tr></tbody></table>`;
        const users = Object.values(r.road_user_tracks).reduce((a, b) => a + b, 0);
        results.innerHTML = `<div class="cols c3" style="margin-top:.4rem">
          <div class="figure"><div class="v">${users}</div><div class="k">road users tracked</div></div>
          <div class="figure"><div class="v">${r.road_damage.reduce((a, b) => a + b.count, 0)}</div><div class="k">damage instances</div></div>
          <div class="figure"><div class="v">${r.warnings.length}</div><div class="k">driver warnings</div></div></div>
          <p class="mono faint" style="margin-top:1rem">${Object.entries(r.road_user_tracks).map(([k, v]) => `${pretty(k)} ${v}`).join(' · ') || 'no road users'}</p>
          ${r.observations.length && !r.events.length ? '<p class="note" style="margin-top:1rem">Damage found. Set a location and run again, or use Live camera with GPS, to create road events.</p>' : ''}`;
      },
      error: e => { setProg(null); results.innerHTML = `<p class="error">${esc(e)}</p>`; },
    });
  }
  dropZone(ui.querySelector('#drop'), 'image/*,video/*', fs => handleFile(fs[0]));

  // ---- live camera
  let stream = null, ws = null, liveTimer = 0, busy = false, watchId = null, lastSent = 0;
  const live = { n: 0, rtt: [], srv: [] };
  async function startLive() {
    stage.classList.remove('hidden'); img.classList.add('hidden'); vid.classList.remove('hidden');
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } }, audio: false });
    } catch (e) {
      ui.querySelector('#live-stats').textContent = `camera unavailable: ${e.message}`;
      return;
    }
    vid.srcObject = stream; vid.controls = false; await vid.play();
    ui.querySelector('#tag').textContent = 'live camera';
    ui.querySelector('#live-start').disabled = true; ui.querySelector('#live-stop').disabled = false;
    if (navigator.geolocation) watchId = navigator.geolocation.watchPosition(p => setGeo({ lat: p.coords.latitude, lon: p.coords.longitude, acc: p.coords.accuracy }, 'live GPS'), () => {}, { enableHighAccuracy: true });
    ws = new WebSocket(wsUrl('/ws/live/scan'));
    const grab = document.createElement('canvas');
    ws.onmessage = ev => {
      const r = JSON.parse(ev.data);
      if (r.type !== 'result') return;
      busy = false;
      live.n++; live.rtt.push(performance.now() - r.t); live.srv.push(r.server_ms);
      if (live.rtt.length > 30) { live.rtt.shift(); live.srv.shift(); }
      const med = a => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
      ui.querySelector('#live-stats').textContent = `${live.n} frames · round trip ${Math.round(med(live.rtt))} ms · server ${Math.round(med(live.srv))} ms`;
      drawFrame(stage, ov, { users: r.users, damage: r.damage }, r.w);
      const vru = r.users.filter(u => VRU.has(u.cls) && (u.bbox[3] - u.bbox[1]) > 0.12 * r.h);
      const ttc = r.users.filter(u => u.ttc && u.ttc < 3);
      const ban = ui.querySelector('#banner');
      if (ttc.length || vru.length) { ban.classList.remove('hidden'); ban.querySelector('span').textContent = ttc.length ? `Closing fast · ${pretty(ttc[0].cls)} · time to contact ${ttc[0].ttc} s` : `${pretty(vru[0].cls)} close ahead`; }
      else ban.classList.add('hidden');
      if (r.events.length) showDecisions([], r.events.map(e => ({ id: e.event_id, type: e.type, merged: e.merged, status: e.status, confidence: '', n_sources: '' })));
    };
    liveTimer = setInterval(() => {
      if (!ws || ws.readyState !== 1 || busy || !vid.videoWidth || performance.now() - lastSent < 150) return;
      const s = 640 / Math.max(vid.videoWidth, vid.videoHeight);
      grab.width = Math.round(vid.videoWidth * s); grab.height = Math.round(vid.videoHeight * s);
      grab.getContext('2d').drawImage(vid, 0, 0, grab.width, grab.height);
      grab.toBlob(b => {
        if (!b || !ws || ws.readyState !== 1) return;
        busy = true; lastSent = performance.now();
        ws.send(JSON.stringify({ type: 'meta', t: lastSent, lat: geo?.lat ?? null, lon: geo?.lon ?? null, acc: geo?.acc ?? null, source_id: deviceId() }));
        ws.send(b);
        setTimeout(() => { busy = false; }, 3000);
      }, 'image/jpeg', 0.7);
    }, 50);
  }
  function stopLive() {
    clearInterval(liveTimer); liveTimer = 0;
    if (ws) { try { ws.close(); } catch {} ws = null; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; vid.srcObject = null; vid.controls = true; }
    if (watchId != null) { navigator.geolocation.clearWatch(watchId); watchId = null; }
    ui.querySelector('#live-start').disabled = false; ui.querySelector('#live-stop').disabled = true;
    ui.querySelector('#banner').classList.add('hidden');
    busy = false;
  }
  ui.querySelector('#live-start').onclick = startLive;
  ui.querySelector('#live-stop').onclick = stopLive;

  return () => { stopLive(); if (job) job.cancel(); cancelAnimationFrame(raf); map.remove(); };
}
