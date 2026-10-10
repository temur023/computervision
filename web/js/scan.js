import { api, C, deviceId, drawBox, dropZone, esc, fitCanvas, geolocate, h, pretty, streamJob, VRU, wsUrl } from './util.js';

export const TASHKENT = [41.3111, 69.2797];
export function baseLayer() {
  return L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '© OpenStreetMap contributors' });
}

// Draw one analysed frame onto the overlay canvas. Boxes come in analysis-resolution pixels (srcW wide).
// Road damage is the subject here: it gets the colour and the labels. Road users are context, drawn as faint
// brackets, unless one is closing in (time to contact), which is a warning.
function drawFrame(stage, ov, fr, srcW) {
  const r = stage.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const ctx = fitCanvas(ov, Math.round(r.width * dpr), Math.round(r.height * dpr));
  ctx.clearRect(0, 0, ov.width, ov.height);
  if (!fr) return;
  const s = ov.width / srcW;
  for (const u of fr.users || []) {
    if (u.warn) drawBox(ctx, u.bbox, `${pretty(u.cls)}${u.ttc ? ` · contact in ${u.ttc} s` : ''}`, C.hazard, { scale: s, lw: 2.5 });
    else { ctx.save(); ctx.globalAlpha = 0.45; drawBox(ctx, u.bbox, '', C.own, { scale: s, lw: 1.2 }); ctx.restore(); }
  }
  for (const d of fr.damage || []) drawBox(ctx, d.bbox, `${pretty(d.cls)} ${Math.round(d.conf * 100)}%`, C.hazard, { scale: s, fill: true, lw: 2 });
}

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const damageSummary = list => list.map(([t, n]) => plural(n, pretty(t).toLowerCase())).join(', ');

export function mountScan(container) {
  const ui = h(`<div class="work">
    <div class="stack">
      <div class="row between">
        <div class="seg" id="mode"><button class="active" data-m="upload">Photo or video</button><button data-m="live">Live camera</button></div>
        <span class="mono faint" id="live-stats" style="font-size:.78rem"></span>
      </div>
      <div id="upload-pane">
        <label class="drop" id="drop"><input type="file" accept="image/*,video/*">
          <b>Drop a road photo or video</b><span class="again">Drop another photo or video, or click to choose</span>
          <span class="faint">or click to choose · JPG, PNG, MP4, MOV · videos: first 60 s at 5 frames per second</span></label>
      </div>
      <div id="live-pane" class="hidden">
        <div class="row"><button class="btn primary" id="live-start">Start camera</button><button class="btn" id="live-stop" disabled>Stop</button></div>
        <p class="note" style="margin-top:.8rem">Point the phone forward along the road. Frames go to our server and detections come back.
          With GPS on, damage seen in three frames becomes a road-event candidate.</p>
      </div>
      <div class="progress hidden" id="prog"><div></div></div>
      <div class="instrument hidden" id="stage">
        <video id="vid" playsinline muted controls class="hidden"></video>
        <img id="img" class="hidden" alt="analysed input">
        <canvas class="overlay" id="ov"></canvas>
        <span class="tag" id="tag">input</span>
        <div class="alertbar hidden" id="verdict"><i></i><span></span></div>
      </div>
      <p class="mono faint hidden" id="timing" style="font-size:.76rem"></p>
      <div id="samples-box">
        <div class="row between" style="margin-bottom:.6rem"><span class="label">Or try a Tashkent street · captions are our labels</span>
          <span class="mono faint" style="font-size:.72rem">Mapillary · CC BY-SA 4.0</span></div>
        <div class="samples" id="samples"></div>
      </div>
    </div>
    <aside>
      <div class="panel"><span class="label">Result</span><div id="result" class="list"><p class="faint">Drop a photo or pick a street to see what the system would do.</p></div></div>
      <div class="panel">
        <span class="label">Location</span>
        <div id="minimap" class="map small"></div>
        <div class="row" style="margin-top:.7rem"><button class="btn" id="gps">Use device GPS</button>
          <span class="faint" style="font-size:.84rem">or click the map</span></div>
        <p class="mono faint" id="geo" style="margin-top:.6rem;font-size:.76rem">No location: detections only, no road event.</p>
      </div>
    </aside>
  </div>`);
  container.append(ui);

  let geo = null, geoByUser = false, marker = null;
  const map = L.map(ui.querySelector('#minimap'), { zoomControl: false }).setView(TASHKENT, 12);
  baseLayer().addTo(map);
  const setGeo = (g, label, byUser = false) => {
    geo = g; geoByUser = byUser;
    if (marker) marker.remove();
    marker = L.circleMarker([g.lat, g.lon], { radius: 6, color: '#111316', weight: 2, fillColor: '#fff', fillOpacity: 1 }).addTo(map);
    map.setView([g.lat, g.lon], Math.max(map.getZoom(), 15));
    ui.querySelector('#geo').textContent = `${g.lat.toFixed(5)}, ${g.lon.toFixed(5)} · ±${Math.round(g.acc || 10)} m · ${label}`;
  };
  const clearGeo = () => {
    geo = null; geoByUser = false;
    if (marker) { marker.remove(); marker = null; }
    ui.querySelector('#geo').textContent = 'No location: detections only, no road event.';
  };
  map.on('click', e => setGeo({ lat: e.latlng.lat, lon: e.latlng.lng, acc: 10 }, 'set on the map', true));
  ui.querySelector('#gps').onclick = async () => {
    const g = await geolocate();
    if (g) setGeo(g, 'device GPS', true); else ui.querySelector('#geo').textContent = 'GPS unavailable: permission denied or no fix.';
  };

  const panes = { upload: ui.querySelector('#upload-pane'), live: ui.querySelector('#live-pane') };
  ui.querySelector('#mode').onclick = e => {
    const m = e.target.dataset.m; if (!m) return;
    ui.querySelectorAll('#mode button').forEach(b => b.classList.toggle('active', b.dataset.m === m));
    Object.entries(panes).forEach(([k, p]) => p.classList.toggle('hidden', k !== m));
    ui.querySelector('#samples-box').classList.toggle('hidden', m === 'live');
    if (m !== 'live') stopLive();
  };

  const stage = ui.querySelector('#stage'), img = ui.querySelector('#img'), vid = ui.querySelector('#vid'), ov = ui.querySelector('#ov');
  const prog = ui.querySelector('#prog'), result = ui.querySelector('#result'), timing = ui.querySelector('#timing');
  let job = null, frames = [], raf = 0, truth = null, damageModel = true;
  api('/api/v1/health').then(hl => { damageModel = !!hl.models?.road_damage; }).catch(() => {});
  const setProg = p => { prog.classList.toggle('hidden', p == null); if (p != null) prog.firstElementChild.style.width = `${Math.round(p * 100)}%`; };
  const setTiming = t => { timing.classList.toggle('hidden', !t); timing.textContent = t || ''; };
  const setVerdict = (html, kind) => {
    const v = ui.querySelector('#verdict');
    v.classList.toggle('hidden', !html);
    v.classList.toggle('calm', kind === 'calm');
    v.classList.toggle('hazard', kind === 'hazard');
    v.querySelector('span').innerHTML = html || '';
  };

  // The result panel: what was found, what the system does with it, and (for our test images) our own label.
  function showResult({ damage, warnings = [], events = [], extra = '' }) {
    result.innerHTML = '';
    const row = (t, m, col) => result.append(h(`<div class="item"><div><div class="t">${col ? `<span class="dot" style="background:${col}"></span>` : ''}${t}</div>${m ? `<div class="m">${m}</div>` : ''}</div></div>`));
    for (const d of damage) row(`${esc(plural(d.count, pretty(d.type).toLowerCase()))}`, `${esc(d.severity)} severity · ${esc(d.action.replace(/^Create road event candidate -> /, ''))}`, 'var(--hazard)');
    for (const w of warnings) row(`${w.warning === 'forward_collision_risk' ? 'Forward collision risk' : 'Rider or pedestrian in path'} · ${esc(pretty(w.cls))}`, `at ${w.t} s${w.ttc_s ? ` · contact in ${w.ttc_s} s` : ''}`, 'var(--hazard)');
    if (!damage.length && !warnings.length) row(damageModel ? 'No road damage found' : 'Road-damage model not loaded yet', damageModel ? '' : 'road users only');
    for (const e of events) {
      result.append(h(`<a class="item" href="#/road/events/${e.id}"><div><div class="t">${e.merged ? 'Added to' : 'New'} road event · ${esc(pretty(e.type))}</div>
        <div class="m"><span class="status s-${e.status}">${e.status}</span>${e.confidence !== '' ? ` · confidence ${e.confidence}` : ''}${e.n_sources !== '' ? ` · ${e.n_sources} source(s)` : ''}</div></div><span class="mono faint">→</span></a>`));
    }
    if (damage.length && !events.length && !geo) row('No road event created', 'Add a location (GPS or a click on the map) and run again.');
    if (truth) row(`Our label: ${truth.not_road ? 'not a road' : truth.damage ? esc(truth.types.map(pretty).join(', ')) : 'no damage'}`, 'made by eye before the model was run; see Data & evidence');
    if (extra) result.append(h(`<p class="mono faint" style="font-size:.76rem;margin-top:.6rem">${extra}</p>`));
  }

  async function handleFile(file, opts = {}) {
    stopLive();
    if (job) job.cancel();
    cancelAnimationFrame(raf);
    truth = opts.truth || null;
    // a location read from the previous photo or sample must not travel with the next input
    if (!geoByUser) clearGeo();
    if (opts.geo) setGeo(opts.geo, opts.geo.label);
    setVerdict(null); setTiming(null);
    result.innerHTML = '<p class="faint">Analysing on the server…</p>';
    stage.classList.remove('hidden');
    ui.querySelector('#drop').classList.add('compact');
    const isVideo = file.type.startsWith('video') || /\.(mp4|mov|webm|avi|mkv)$/i.test(file.name);
    const fd = new FormData();
    fd.append('file', file);
    fd.append('source_id', deviceId());
    if (geo) { fd.append('lat', geo.lat); fd.append('lon', geo.lon); fd.append('gps_accuracy_m', geo.acc || 10); }
    const t0 = performance.now();
    ui.querySelector('#tag').textContent = (opts.tag || file.name).slice(0, 48);
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
        if (r.geotag?.lat && !geo) setGeo({ lat: r.geotag.lat, lon: r.geotag.lon, acc: r.geotag.accuracy_m }, 'from the photo');
        const dmg = r.decisions.filter(d => d.kind === 'road_damage');
        const ev = r.events || [];
        if (dmg.length) {
          const worst = ['high', 'medium', 'low'].find(s => dmg.some(d => d.severity === s));
          setVerdict(`<b>${esc(damageSummary(dmg.map(d => [d.type, d.count])))}</b> · ${worst} severity${ev.length ? ` · ${ev.some(e => !e.merged) ? 'new road event' : 'added to an existing road event'}` : geo ? '' : ' · add a location to report it'}`, 'hazard');
        } else setVerdict(damageModel ? 'No road damage found in this image.' : 'The road-damage model is not loaded on this server yet: road users only.', 'calm');
        showResult({ damage: dmg, events: ev });
        setTiming(`analysed on this server in ${Math.round(r.server_ms)} ms (road users ${Math.round(r.timing_ms.road_users_ms)} ms${damageModel && r.timing_ms.road_damage_ms != null ? ` · damage ${Math.round(r.timing_ms.road_damage_ms)} ms` : ''}) · ${Math.round(rtt)} ms with upload`);
      } catch (e) { setProg(null); result.innerHTML = `<p class="error">${esc(e.message)}</p>`; }
      return;
    }
    img.classList.add('hidden'); vid.classList.remove('hidden');
    vid.src = URL.createObjectURL(file);
    frames = [];
    setProg(0.02);
    let resp;
    try { resp = await api('/api/v1/scan/video', { method: 'POST', body: fd }); }
    catch (e) { result.innerHTML = `<p class="error">${esc(e.message)}</p>`; setProg(null); return; }
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
        setTiming(`${frames.length} frames analysed · last ${Math.round(m.frame.ms)} ms`);
      },
      progress: p => setProg(p),
      done: r => {
        setProg(null);
        const dmg = r.decisions.filter(d => d.kind === 'road_damage');
        const warn = r.decisions.filter(d => d.kind === 'driver_warning');
        if (dmg.length) setVerdict(`<b>${esc(damageSummary(dmg.map(d => [d.type, d.count])))}</b> in ${r.duration_s} s of video${warn.length ? ` · ${plural(warn.length, 'driver warning')}` : ''}`, 'hazard');
        else setVerdict(`${damageModel ? 'No road damage found' : 'Road-damage model not loaded: road users only'} in ${r.duration_s} s of video${warn.length ? ` · ${plural(warn.length, 'driver warning')}` : ''}.`, warn.length ? 'hazard' : 'calm');
        const users = Object.entries(r.road_user_tracks).map(([k, v]) => `${pretty(k)} ${v}`).join(' · ');
        showResult({ damage: dmg, warnings: warn, events: r.events, extra: users ? `road users tracked: ${esc(users)}` : '' });
        setTiming(`${r.perf.frames} frames at ${r.sample_fps.toFixed(1)} fps · ${r.perf.ms_per_frame_p50} ms per frame (p50) · ${r.perf.wall_s} s on this server for ${r.duration_s} s of video`);
      },
      error: e => { setProg(null); result.innerHTML = `<p class="error">${esc(e)}</p>`; },
    });
  }
  dropZone(ui.querySelector('#drop'), 'image/*,video/*', fs => handleFile(fs[0]));

  // Tashkent test images, chosen by our label only (one per damage type, plus clean roads), never by model output.
  api('/api/v1/labels/tashkent').then(d => {
    const lab = it => it.review || it.first_pass;
    const ok = d.items.filter(it => !lab(it).uncertain && !lab(it).not_road);
    const pick = [], why = [];
    for (const t of ['pothole', 'alligator_crack', 'transverse_crack', 'longitudinal_crack']) {
      const it = ok.find(x => lab(x).damage && lab(x).types.includes(t) && !pick.includes(x));
      if (it) { pick.push(it); why.push(t); }
    }
    for (const it of ok.filter(x => !lab(x).damage).slice(0, 6 - pick.length)) { pick.push(it); why.push(null); }
    const caption = (it, t) => t ? `${pretty(t)}${lab(it).types.length > 1 ? ` +${lab(it).types.length - 1}` : ''}` : 'No damage';
    const box = ui.querySelector('#samples');
    box.innerHTML = pick.map((it, i) => `<button class="sample" data-i="${i}" title="our label: ${esc(lab(it).damage ? lab(it).types.map(pretty).join(', ') : 'no damage')}">
      <img src="${esc(it.image)}" alt="Tashkent street" loading="lazy"><span>${esc(caption(it, why[i]))}</span></button>`).join('');
    box.onclick = async e => {
      const b = e.target.closest('.sample'); if (!b) return;
      const it = pick[+b.dataset.i];
      box.querySelectorAll('.sample').forEach(x => x.classList.toggle('active', x === b));
      const blob = await (await fetch(it.image)).blob();
      const g = it.lat != null ? { lat: it.lat, lon: it.lon, acc: 10, label: 'where Mapillary captured it' } : null;
      handleFile(new File([blob], `${it.id}.jpg`, { type: 'image/jpeg' }), { geo: g, truth: lab(it), tag: `Tashkent · Mapillary ${it.id}` });
    };
  }).catch(() => ui.querySelector('#samples-box').classList.add('hidden'));

  // ---- live camera
  let stream = null, ws = null, liveTimer = 0, busy = false, watchId = null, lastSent = 0;
  const live = { n: 0, rtt: [], srv: [] };
  async function startLive() {
    stage.classList.remove('hidden'); img.classList.add('hidden'); vid.classList.remove('hidden');
    truth = null;
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
      if (ttc.length) setVerdict(`<b>Closing fast:</b> ${esc(pretty(ttc[0].cls))}, contact in ${ttc[0].ttc} s`, 'hazard');
      else if (r.damage.length) setVerdict(`<b>${esc(damageSummary(Object.entries(r.damage.reduce((a, d) => ({ ...a, [d.cls]: (a[d.cls] || 0) + 1 }), {}))))}</b> ahead`, 'hazard');
      else if (vru.length) setVerdict(`${esc(pretty(vru[0].cls))} close ahead`, 'calm');
      else setVerdict(null);
      if (r.events.length) showResult({ damage: [], events: r.events.map(e => ({ id: e.event_id, type: e.type, merged: e.merged, status: e.status, confidence: '', n_sources: '' })) });
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
    busy = false;
  }
  ui.querySelector('#live-start').onclick = startLive;
  ui.querySelector('#live-stop').onclick = () => { stopLive(); setVerdict(null); };

  return () => { stopLive(); if (job) job.cancel(); cancelAnimationFrame(raf); map.remove(); };
}
