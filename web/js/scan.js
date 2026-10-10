import { api, color, deviceId, drawBox, dropZone, esc, fitCanvas, geolocate, h, pretty, streamJob, VRU, wsUrl } from './util.js';

const TASHKENT = [41.3111, 69.2797];

export default async function scan(root) {
  root.append(h(`<section class="hero" style="padding-top:.5rem">
    <span class="pill amber">Scan · single camera</span>
    <h1 style="margin-top:.5rem">Photo or video in, decision out</h1>
    <p>Upload any road photo or dashcam/phone clip, or point your phone at the street. Our server detects road users and road
      damage, tracks them, and returns actions: road-event candidates for the city and warnings for the driver or fleet.</p>
  </section>`));

  const ui = h(`<div class="split">
    <div class="stack">
      <div class="card stack">
        <div class="tabs" id="mode">
          <button class="active" data-m="upload">Upload photo / video</button>
          <button data-m="live">Live camera</button>
        </div>
        <div id="upload-pane">
          <label class="drop" id="drop"><input type="file" accept="image/*,video/*">
            <b>Drop a photo or video here</b> or tap to choose<br>
            <small>JPG/PNG/HEIC→JPG, MP4/MOV/WebM · video: first 60 s analysed at 5 fps</small></label>
        </div>
        <div id="live-pane" class="hidden stack">
          <div class="row">
            <button class="btn primary" id="live-start">Start camera</button>
            <button class="btn" id="live-stop" disabled>Stop</button>
            <span class="muted mono" id="live-stats"></span>
          </div>
          <p class="note">Frames (640 px JPEG) go to our server over a WebSocket. Damage seen with GPS on becomes a road-event candidate.
            Point the camera forward along the road.</p>
        </div>
        <div class="progress hidden" id="prog"><div></div></div>
        <div class="stage hidden" id="stage">
          <video id="vid" playsinline muted controls class="hidden"></video>
          <img id="img" class="hidden" alt="">
          <canvas class="overlay" id="ov"></canvas>
          <div class="banner hidden" id="banner"></div>
        </div>
        <div class="legend">
          <span><i style="background:#f87171"></i>pedestrian</span><span><i style="background:#fb923c"></i>cyclist</span>
          <span><i style="background:#f472b6"></i>motorcyclist</span><span><i style="background:#38bdf8"></i>vehicle</span>
          <span><i style="background:#ef4444"></i>pothole</span><span><i style="background:#f97316"></i>alligator crack</span>
          <span><i style="background:#facc15"></i>transverse crack</span><span><i style="background:#eab308"></i>longitudinal crack</span>
        </div>
      </div>
      <div class="card stack hidden" id="results"></div>
    </div>
    <div class="stack">
      <div class="card stack">
        <h3>Location for road events</h3>
        <div class="row">
          <button class="btn" id="gps">Use my GPS</button>
          <span class="muted">or tap the map · photo EXIF GPS is used automatically</span>
        </div>
        <div id="minimap" class="map small"></div>
        <div class="mono muted" id="geo">no location: detections are shown, but no road event is created</div>
      </div>
      <div class="card stack">
        <h3>Decisions</h3>
        <div id="decisions" class="list"><p class="muted">Run a scan to see actions.</p></div>
      </div>
      <div class="card stack">
        <h3>Performance (measured)</h3>
        <div id="perf" class="mono muted">—</div>
      </div>
    </div>
  </div>`);
  root.append(ui);

  let geo = null;
  const map = L.map(ui.querySelector('#minimap'), { attributionControl: true }).setView(TASHKENT, 12);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '© OpenStreetMap contributors' }).addTo(map);
  let marker = null;
  const setGeo = (g, label) => {
    geo = g;
    if (marker) marker.remove();
    marker = L.marker([g.lat, g.lon]).addTo(map);
    map.setView([g.lat, g.lon], Math.max(map.getZoom(), 15));
    ui.querySelector('#geo').textContent = `${g.lat.toFixed(5)}, ${g.lon.toFixed(5)} · ±${Math.round(g.acc || 10)} m · ${label}`;
  };
  map.on('click', e => setGeo({ lat: e.latlng.lat, lon: e.latlng.lng, acc: 10 }, 'manual geotag'));
  ui.querySelector('#gps').onclick = async () => {
    const g = await geolocate();
    if (g) setGeo(g, 'device GPS'); else ui.querySelector('#geo').textContent = 'GPS unavailable (permission denied or no fix)';
  };

  // ---- mode switch
  const panes = { upload: ui.querySelector('#upload-pane'), live: ui.querySelector('#live-pane') };
  ui.querySelector('#mode').onclick = e => {
    const m = e.target.dataset.m; if (!m) return;
    ui.querySelectorAll('#mode button').forEach(b => b.classList.toggle('active', b.dataset.m === m));
    Object.entries(panes).forEach(([k, p]) => p.classList.toggle('hidden', k !== m));
    if (m !== 'live') stopLive();
  };

  const stage = ui.querySelector('#stage'), img = ui.querySelector('#img'), vid = ui.querySelector('#vid'), ov = ui.querySelector('#ov');
  const prog = ui.querySelector('#prog'), results = ui.querySelector('#results');
  let job = null, frames = [], srcW = 0, srcH = 0, raf = 0;

  function drawFrame(fr, W, Hh) {
    const r = stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const ctx = fitCanvas(ov, Math.round(r.width * dpr), Math.round(r.height * dpr));
    ctx.clearRect(0, 0, ov.width, ov.height);
    if (!fr) return;
    const s = ov.width / W;
    for (const d of fr.damage || []) drawBox(ctx, d.bbox, `${pretty(d.cls)} ${Math.round(d.conf * 100)}%`, color(d.cls), { scale: s, fill: true, lw: 2 * dpr });
    for (const u of fr.users || []) {
      const lbl = `${pretty(u.cls)}${u.id ? ' #' + u.id : ''}${u.ttc ? ` · TTC ${u.ttc}s` : ''}`;
      drawBox(ctx, u.bbox, lbl, u.warn ? '#ef4444' : color(u.cls), { scale: s, lw: (u.warn ? 3.5 : 2) * dpr });
    }
  }

  function showDecisions(dec, events) {
    const box = ui.querySelector('#decisions');
    box.innerHTML = '';
    if (!dec.length && !(events || []).length) { box.innerHTML = '<p class="muted">No hazard found in this input.</p>'; return; }
    for (const d of dec) {
      let t = '';
      if (d.kind === 'road_damage') t = `<b>${pretty(d.type)}</b> ×${d.count} · severity <b>${d.severity}</b><br><small>${esc(d.action)}</small>`;
      else if (d.kind === 'driver_warning') t = `<b style="color:var(--bad)">${d.warning === 'forward_collision_risk' ? 'Forward collision risk' : 'Vulnerable road user in path'}</b> · ${pretty(d.cls)} #${d.track} at ${d.t}s${d.ttc_s ? ` · TTC ${d.ttc_s}s` : ''}<br><small>${esc(d.action)}</small>`;
      else if (d.kind === 'vru_presence') t = `<b>${d.count} vulnerable road user(s)</b><br><small>${esc(d.action)}</small>`;
      box.append(h(`<div class="item"><span>${t}</span></div>`));
    }
    for (const e of events || []) {
      box.append(h(`<a class="item" href="#/road/${e.id}" style="text-decoration:none;color:inherit">
        <span>Road event <b>#${e.id}</b> · ${pretty(e.type)} · ${e.merged ? 'merged into existing event' : 'new candidate'}<br>
        <small>status <b>${e.status}</b> · confidence ${e.confidence} · ${e.n_sources} independent source(s)</small></span><span>→</span></a>`));
    }
  }

  async function handleFile(file) {
    stopLive();
    if (job) job.cancel();
    results.classList.add('hidden');
    stage.classList.remove('hidden');
    cancelAnimationFrame(raf);
    const isVideo = file.type.startsWith('video') || /\.(mp4|mov|webm|avi|mkv)$/i.test(file.name);
    const fd = new FormData();
    fd.append('file', file);
    fd.append('source_id', deviceId());
    if (geo) { fd.append('lat', geo.lat); fd.append('lon', geo.lon); fd.append('gps_accuracy_m', geo.acc || 10); }
    const t0 = performance.now();
    if (!isVideo) {
      vid.classList.add('hidden'); img.classList.remove('hidden');
      img.src = URL.createObjectURL(file);
      prog.classList.remove('hidden'); prog.firstElementChild.style.width = '30%';
      try {
        const r = await api('/api/v1/scan/image', { method: 'POST', body: fd });
        prog.classList.add('hidden');
        const rtt = performance.now() - t0;
        // server works on an orientation-corrected, resized copy; map boxes back onto the displayed image
        const draw = () => drawFrame({ users: r.road_users, damage: r.road_damage }, r.width, r.height);
        if (img.complete) draw(); else img.onload = draw;
        window.addEventListener('resize', draw, { once: true });
        if (r.geotag?.lat && !geo) setGeo({ lat: r.geotag.lat, lon: r.geotag.lon, acc: r.geotag.accuracy_m }, 'photo EXIF');
        showDecisions(r.decisions, r.events);
        ui.querySelector('#perf').innerHTML = `road users ${r.timing_ms.road_users_ms} ms · road damage ${r.timing_ms.road_damage_ms} ms<br>
          server total ${r.server_ms} ms · round trip incl. upload ${Math.round(rtt)} ms`;
        results.classList.remove('hidden');
        results.innerHTML = `<h3>Detections</h3><div class="table-wrap"><table class="t"><thead><tr><th>Type</th><th>Class</th><th class="num">Conf.</th><th class="num">Severity</th></tr></thead><tbody>
          ${r.road_damage.map(d => `<tr><td>damage</td><td>${pretty(d.cls)}</td><td class="num">${d.conf}</td><td class="num">${d.severity_label}</td></tr>`).join('')}
          ${r.road_users.map(u => `<tr><td>road user</td><td>${pretty(u.cls)}</td><td class="num">${u.conf}</td><td class="num">—</td></tr>`).join('')}
          </tbody></table></div>${!geo && r.observations.length ? '<p class="note">Add a location (GPS or map) and re-run to turn this damage into a road event.</p>' : ''}`;
      } catch (e) {
        prog.classList.add('hidden');
        results.classList.remove('hidden');
        results.innerHTML = `<p class="error">${esc(e.message)}</p>`;
      }
      return;
    }
    img.classList.add('hidden'); vid.classList.remove('hidden');
    vid.src = URL.createObjectURL(file);
    frames = [];
    prog.classList.remove('hidden'); prog.firstElementChild.style.width = '2%';
    ui.querySelector('#perf').textContent = 'uploading…';
    let resp;
    try { resp = await api('/api/v1/scan/video', { method: 'POST', body: fd }); }
    catch (e) { results.classList.remove('hidden'); results.innerHTML = `<p class="error">${esc(e.message)}</p>`; prog.classList.add('hidden'); return; }
    ui.querySelector('#perf').textContent = 'processing on server…';
    vid.onloadedmetadata = () => { srcW = vid.videoWidth; srcH = vid.videoHeight; };
    const loop = () => {
      const t = vid.currentTime;
      let best = null;
      for (const f of frames) { if (f.t <= t + 0.05) best = f; else break; }
      if (best && t - best.t > 0.6) best = null;
      // server boxes are in the (possibly downscaled) analysis resolution
      drawFrame(best, best?.w || frames.width || srcW || 1280);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    let started = false;
    job = streamJob(resp.job_id, {
      frame: m => {
        frames.push(m.frame);
        if (!started && frames.length > 3) { started = true; vid.play().catch(() => {}); }
        ui.querySelector('#perf').textContent = `analysed ${frames.length} frames · last frame ${m.frame.ms} ms on server`;
      },
      progress: p => { prog.firstElementChild.style.width = `${Math.round(p * 100)}%`; },
      done: r => {
        prog.classList.add('hidden');
        frames.width = r.width;
        showDecisions(r.decisions, r.events);
        ui.querySelector('#perf').innerHTML = `${r.perf.frames} frames analysed (${r.sample_fps.toFixed(1)} fps sampling)<br>
          per frame p50 ${r.perf.ms_per_frame_p50} ms · p95 ${r.perf.ms_per_frame_p95} ms · wall ${r.perf.wall_s}s for ${r.duration_s}s of video`;
        results.classList.remove('hidden');
        results.innerHTML = `<h3>Summary</h3>
          <div class="grid g3">
            <div class="kpi"><span class="v">${Object.values(r.road_user_tracks).reduce((a, b) => a + b, 0)}</span><span class="l">road users tracked</span></div>
            <div class="kpi"><span class="v">${r.road_damage.reduce((a, b) => a + b.count, 0)}</span><span class="l">damage instances</span></div>
            <div class="kpi"><span class="v">${r.warnings.length}</span><span class="l">driver warnings</span></div>
          </div>
          <p class="muted">${Object.entries(r.road_user_tracks).map(([k, v]) => `${pretty(k)}: ${v}`).join(' · ') || 'no road users'}</p>
          ${r.observations.length && !r.events.length ? '<p class="note">Damage found. Set a location and upload again to create road events, or use Live mode with GPS.</p>' : ''}`;
      },
      error: e => { prog.classList.add('hidden'); results.classList.remove('hidden'); results.innerHTML = `<p class="error">${esc(e)}</p>`; },
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
      ui.querySelector('#live-stats').textContent = 'camera unavailable: ' + e.message + ' (HTTPS and permission are required)';
      return;
    }
    vid.srcObject = stream; vid.controls = false; await vid.play();
    ui.querySelector('#live-start').disabled = true; ui.querySelector('#live-stop').disabled = false;
    if (navigator.geolocation) watchId = navigator.geolocation.watchPosition(p => setGeo({ lat: p.coords.latitude, lon: p.coords.longitude, acc: p.coords.accuracy }, 'live GPS'), () => {}, { enableHighAccuracy: true });
    ws = new WebSocket(wsUrl('/ws/live/scan'));
    const grab = document.createElement('canvas');
    let lastRes = null;
    ws.onmessage = ev => {
      const r = JSON.parse(ev.data);
      if (r.type !== 'result') return;
      busy = false;
      live.n++; live.rtt.push(performance.now() - r.t); live.srv.push(r.server_ms);
      if (live.rtt.length > 30) { live.rtt.shift(); live.srv.shift(); }
      lastRes = r;
      const med = a => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
      ui.querySelector('#live-stats').textContent = `${live.n} frames · round trip ${Math.round(med(live.rtt))} ms · server ${Math.round(med(live.srv))} ms`;
      drawFrame({ users: r.users, damage: r.damage }, r.w);
      const vru = r.users.filter(u => VRU.has(u.cls) && (u.bbox[3] - u.bbox[1]) > 0.12 * r.h);
      const ttc = r.users.filter(u => u.ttc && u.ttc < 3);
      const ban = ui.querySelector('#banner');
      if (ttc.length || vru.length) { ban.classList.remove('hidden'); ban.textContent = ttc.length ? `⚠ Closing fast: ${pretty(ttc[0].cls)} · TTC ${ttc[0].ttc}s` : `⚠ ${pretty(vru[0].cls)} close ahead`; }
      else ban.classList.add('hidden');
      if (r.events.length) showDecisions([], r.events.map(e => ({ id: e.event_id, type: e.type, merged: e.merged, status: e.status, confidence: '', n_sources: '' })));
    };
    const tick = () => {
      if (!ws || ws.readyState !== 1 || busy || !vid.videoWidth) return;
      if (performance.now() - lastSent < 150) return;
      const s = 640 / Math.max(vid.videoWidth, vid.videoHeight);
      grab.width = Math.round(vid.videoWidth * s); grab.height = Math.round(vid.videoHeight * s);
      grab.getContext('2d').drawImage(vid, 0, 0, grab.width, grab.height);
      grab.toBlob(b => {
        if (!b || !ws || ws.readyState !== 1) return;
        busy = true; lastSent = performance.now();
        ws.send(JSON.stringify({ type: 'meta', t: lastSent, lat: geo?.lat ?? null, lon: geo?.lon ?? null, acc: geo?.acc ?? null, source_id: deviceId() }));
        ws.send(b);
      }, 'image/jpeg', 0.7);
    };
    liveTimer = setInterval(tick, 50);
    setTimeout(() => { busy = false; }, 3000);
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
