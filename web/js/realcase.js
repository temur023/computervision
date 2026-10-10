// Real case: a crash at a Tashkent crossing, reconstructed from one dashcam, and what CrossGuard would have done.
import { api, C, drawBox, esc, fitCanvas, h } from './util.js';

const FPS = 8;
const fmt = t => `${t.toFixed(2)} s`;

export function realCase(pane) {
  const ui = h(`<div class="console" tabindex="-1">
    <div class="bar">
      <div class="ctl"><span class="title">Real case</span><span class="lbl">Tashkent · 16 July 2026 · 18:28 · one dashcam</span></div>
      <div class="ctl">
        <button class="netswitch" id="cf" aria-pressed="false" title="With a camera at the crossing (C)"><span class="knob"></span><span id="cflbl">Crossing camera off</span></button>
        <span class="ctl hidden" id="ranges"></span>
        <button class="btn" id="present">Present</button>
      </div>
    </div>
    <div class="stage2">
      <div class="pane cam-pane"><img id="frame" alt="Dashcam frame, faces and plates blurred"><canvas class="overlay" id="hud"></canvas>
        <span class="cap">What the dashcam car sees</span>
        <div class="alertbar calm" id="bar"><i></i><span>Loading…</span></div></div>
      <div class="pane map-pane2"><div id="map" style="position:absolute;inset:0"></div><span class="cap right" style="z-index:500">Top view · the real crossing</span></div>
    </div>
    <div class="tl" id="tl"><button class="playbtn" id="play" aria-label="Play">▶</button>
      <div class="track" id="track"><div class="bars" id="marks"></div><div class="head" id="head"></div></div>
      <span class="time" id="time">—</span></div>
    <div class="foot2" style="grid-template-columns:1fr">
      <div class="nums" id="nums"></div>
    </div>
    <div class="legend dark">
      <span><i style="background:#e9e7e2"></i>our car (GPS)</span>
      <span><i style="background:#7b9bff"></i>the car ahead (tracked)</span>
      <span><i style="background:#ff5a4a"></i>motorcycle: seen by the dashcam · dashed: before it was visible</span>
      <span><i style="background:#f0a030"></i>crossing camera and CrossGuard</span>
    </div>
    <p class="statusline" id="status">Reconstructing on the server…</p>
  </div>`);
  pane.append(ui);
  const $ = s => ui.querySelector(s);
  let data = null, i = 0, playing = false, cf = false, raf = 0, last = 0, acc = 0, map = null, layers = null, range = 100;
  const scn = () => data.counterfactual.scenarios.find(x => x.range_m === range) || data.counterfactual.scenarios[0];
  const imgs = [];

  function buildMap() {
    map = L.map($('#map'), { zoomControl: false, attributionControl: true, zoomSnap: 0.25 });
    const P = data.paths;
    map.fitBounds(L.latLngBounds([...P.ego.slice(-24), ...P.moto_before.slice(-5), ...P.lead, data.impact, ...P.camera_cover['60']]), { padding: [36, 36] });
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 21, maxNativeZoom: 19, className: 'tiles-dark', attribution: '© OpenStreetMap contributors' }).addTo(map);
    L.polyline(P.ego, { color: '#e9e7e2', weight: 2, opacity: 0.55, dashArray: '2 5' }).addTo(map);
    L.polyline(P.lead, { color: '#7b9bff', weight: 2.5, opacity: 0.8 }).addTo(map);
    L.polyline(P.moto_before, { color: '#ff5a4a', weight: 2.5, opacity: 0.85, dashArray: '6 5' }).addTo(map);
    L.polyline(P.moto_seen, { color: '#ff5a4a', weight: 3.5, opacity: 1 }).addTo(map);
    L.circleMarker(data.impact, { radius: 9, color: '#ff5a4a', weight: 2, fillColor: '#ff5a4a', fillOpacity: 0.2 }).addTo(map)
      .bindTooltip('impact', { permanent: true, direction: 'right', className: 'maptag crit' });
    layers = {
      cam: L.layerGroup(),
      stop: null,
      ego: L.circleMarker(P.ego[0], { radius: 6, color: '#0d0f12', weight: 2, fillColor: '#e9e7e2', fillOpacity: 1 }).addTo(map),
      lead: L.circleMarker(P.lead[0], { radius: 6, color: '#0d0f12', weight: 2, fillColor: '#7b9bff', fillOpacity: 1 }).addTo(map),
      moto: L.circleMarker(data.impact, { radius: 6, color: '#0d0f12', weight: 2, fillColor: '#ff5a4a', fillOpacity: 1 }),
    };
  }

  // the crossing camera's view of the cross street and where the warned car would be when the motorcycle crosses
  function drawScenario() {
    layers.cam.clearLayers();
    const sc = scn(), cover = data.paths.camera_cover[String(range)];
    L.polyline(cover, { color: '#f0a030', weight: 16, opacity: 0.16 }).addTo(layers.cam);
    L.circleMarker(cover[0], { radius: 7, color: '#f0a030', weight: 2, fillColor: '#0d0f12', fillOpacity: 1 }).addTo(layers.cam)
      .bindTooltip(`camera · sees ${range} m`, { permanent: true, direction: 'top', className: 'maptag amber' });
    const b = sc.braking?.[0];
    if (b) {
      L.circleMarker(b.car_at, { radius: 7, color: b.avoided ? '#3fbf86' : '#f0a030', weight: 2, fillColor: b.avoided ? '#3fbf86' : '#f0a030', fillOpacity: 0.35 }).addTo(layers.cam)
        .bindTooltip(b.avoided ? `braked: stops ${b.clear_m} m short` : 'braked: still in the path', { permanent: true, direction: 'bottom', className: 'maptag amber' });
    }
  }

  // position of the motorcycle on its path at time t (linear in time between the extrapolated and seen segments)
  function motoAt(t) {
    const m = data.motorcycle, P = data.paths;
    const t0 = data.impact_t - 3, tv = m.visible_from_t;
    if (t < t0) return null;
    const seg = t < tv ? P.moto_before : P.moto_seen;
    const a = t < tv ? t0 : tv, b = t < tv ? tv : data.impact_t;
    const u = Math.max(0, Math.min(1, (t - a) / (b - a))) * (seg.length - 1);
    const k = Math.min(seg.length - 2, Math.floor(u)), f = u - k;
    return [seg[k][0] + (seg[k + 1][0] - seg[k][0]) * f, seg[k][1] + (seg[k + 1][1] - seg[k][1]) * f];
  }

  function narrative(t) {
    const m = data.motorcycle, c = scn();
    if (cf && c.t_warn != null && t >= c.t_warn - 1e-6 && t < data.impact_t + 0.01) {
      const b = c.braking[0];
      return [`<b>CrossGuard → car ahead: ${esc(c.action)}.</b> ${esc(c.warning.reason)} ${b.avoided ? `Braking now, it stays ${b.clear_m} m short of the motorcycle's path.` : 'Even braking now, it is still in the path.'}`, 'alertbar'];
    }
    if (cf && t >= c.t_track && t < data.impact_t) return [`<b>Crossing camera:</b> motorcycle tracked on the cross street, ~${m.speed_kmh} km/h. CrossGuard predicts both paths.`, 'alertbar hazard'];
    if (cf && t >= c.t_detect && t < data.impact_t) return ['<b>Crossing camera:</b> a fast road user enters its view on the cross street.', 'alertbar hazard'];
    if (t >= data.impact_t - 0.01) return ['<b>Impact.</b> The replay stops here: we show the conflict, not its outcome.', 'alertbar'];
    if (t >= m.visible_from_t) return [`<b>Motorcycle visible to this car only now:</b> ${m.visible_s.toFixed(2)} s before impact. No driver reacts that fast (about 1 s).`, 'alertbar'];
    if (t >= 26) return [`Our car moves off at ${data.frames[i].ego_kmh} km/h behind the dark car. The cross street on the left is hidden by the corner.`, 'alertbar calm'];
    return ['Both cars wait at the traffic officer. The dark car pulls out first.', 'alertbar calm'];
  }

  function show() {
    const f = data.frames[i];
    const im = imgs[i];
    if ($('#frame').dataset.i !== String(i)) { $('#frame').src = im.src; $('#frame').dataset.i = String(i); }
    const cv = $('#hud'), r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1;
    const ctx = fitCanvas(cv, Math.round(r.width * d), Math.round(r.height * d));
    ctx.clearRect(0, 0, cv.width, cv.height);
    const s = cv.width / 960;
    const lead = data.lead_car.track_id;
    for (const tr of f.tracks) {
      if (tr.cls === 'person') continue;
      if (tr.id === lead) drawBox(ctx, tr.bbox, 'car ahead', C.network, { scale: s, lw: 1.6 });
      else { ctx.save(); ctx.globalAlpha = 0.45; drawBox(ctx, tr.bbox, '', C.own, { scale: s, lw: 1.1 }); ctx.restore(); }
    }
    if (f.moto_box) drawBox(ctx, f.moto_box, 'motorcycle · annotated', C.critical, { scale: s, lw: 2, dashed: true });
    const [text, cls] = narrative(f.t);
    $('#bar').className = cls;
    $('#bar').querySelector('span').innerHTML = text;
    layers.ego.setLatLng(f.ego);
    if (f.lead) layers.lead.setLatLng(f.lead);
    const mp = motoAt(f.t);
    if (mp) { layers.moto.setLatLng(mp); if (!map.hasLayer(layers.moto)) layers.moto.addTo(map); } else if (map.hasLayer(layers.moto)) layers.moto.remove();
    const n = data.frames.length;
    $('#head').style.left = `calc(${(100 * i) / Math.max(1, n - 1)}% - 1px)`;
    $('#time').textContent = `${fmt(f.t)} · ${f.ego_kmh} km/h`;
  }

  function marks() {
    const t0 = data.frames[0].t, t1 = data.frames[data.frames.length - 1].t, x = t => `${(100 * (t - t0)) / (t1 - t0)}%`;
    const m = data.motorcycle, c = scn();
    const list = [[m.visible_from_t, data.impact_t, C.critical, 'motorcycle visible to this car']];
    if (cf && c.t_warn != null) list.push([c.t_warn, data.impact_t, C.hazard, 'CrossGuard warning with a crossing camera']);
    $('#marks').innerHTML = list.map(([a, b, col, title]) => `<span title="${esc(title)}" style="left:${x(a)};width:max(4px, calc(${x(b)} - ${x(a)}));height:${col === C.hazard ? '45%' : '100%'};background:${col};${col === C.hazard ? 'bottom:auto;top:0' : ''}"></span>`).join('');
  }

  function numbers() {
    const m = data.motorcycle, c = scn(), b = c.braking?.[0], b2 = c.braking?.[1];
    const need = data.counterfactual.min_range_avoided_m;
    $('#nums').innerHTML = [
      [m.visible_s.toFixed(2), 's', 'how long the dashcam car could see the motorcycle before impact: the corner hid the cross street'],
      [cf && c.lead_s != null ? c.lead_s.toFixed(1) : '—', 's', cf ? `CrossGuard warning before impact, with a crossing camera that sees ${range} m of the cross street` : 'turn the crossing camera on to see when CrossGuard would warn'],
      [cf && b ? (b.avoided ? b.clear_m.toFixed(1) : 'no') : '—', cf && b && b.avoided ? 'm' : '', cf && b ? (b.avoided ? `the car ahead stays this far short of the motorcycle's path (reaction 1.0 s${b2 ? `; 1.5 s: ${b2.avoided ? `${b2.clear_m} m` : 'too late'}` : ''})` : `the car ahead cannot get out of the path with ${range} m of view${need ? `; it needs about ${need} m` : ''}`) : (need ? `metres of cross street a camera must see to prevent this crash: about ${need}` : '')],
    ].map(([v, u, k]) => `<div><div class="v">${v}<small>${u}</small></div><div class="k">${k}</div></div>`).join('');
    $('#status').textContent = `Computed on this server in ${data.compute_s ?? '—'} s: car ahead tracked in ${data.lead_car.observations} frames (~${data.lead_car.speed_kmh} km/h at the crossing), motorcycle ~${m.speed_kmh} km/h (${m.range_kmh[0]}–${m.range_kmh[1]}). `
      + `Measured: ${data.measured.join('; ')}. Estimated: ${data.estimated.join('; ')}. Assumed: ${data.assumed.join('; ')}. ${data.source}.`;
  }

  const setCf = v => {
    cf = v;
    $('#cf').classList.toggle('on', cf); $('#cf').setAttribute('aria-pressed', String(cf));
    $('#cflbl').textContent = cf ? 'Crossing camera on' : 'Crossing camera off';
    $('#ranges').classList.toggle('hidden', !cf);
    if (cf) { drawScenario(); layers.cam.addTo(map); } else layers.cam.remove();
    marks(); numbers(); show();
  };
  const setRange = r => {
    range = r;
    ui.querySelectorAll('#ranges .chip').forEach(b => b.classList.toggle('on', +b.dataset.r === r));
    drawScenario(); marks(); numbers(); show();
  };
  function tick(ts) {
    const dt = last ? (ts - last) / 1000 : 0; last = ts;
    if (playing) {
      acc += dt * FPS * (data.frames[i].t > data.impact_t - 1.2 ? 0.35 : 1);  // slow motion over the last second
      while (acc >= 1) { acc -= 1; if (i < data.frames.length - 1) i++; else { playing = false; $('#play').textContent = '▶'; } }
      show();
    }
    raf = requestAnimationFrame(tick);
  }
  $('#play').onclick = () => { if (i >= data.frames.length - 1) i = 0; playing = !playing; $('#play').textContent = playing ? '❚❚' : '▶'; };
  $('#track').onclick = e => { const r = $('#track').getBoundingClientRect(); i = Math.round(((e.clientX - r.left) / r.width) * (data.frames.length - 1)); playing = false; $('#play').textContent = '▶'; show(); };
  $('#cf').onclick = () => setCf(!cf);
  $('#present').onclick = () => (document.fullscreenElement ? document.exitFullscreen() : ui.requestFullscreen?.());
  const onKey = e => {
    if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
    if (e.code === 'Space') { e.preventDefault(); $('#play').click(); }
    else if (e.key === 'ArrowRight') { i = Math.min(data.frames.length - 1, i + 1); show(); }
    else if (e.key === 'ArrowLeft') { i = Math.max(0, i - 1); show(); }
    else if (e.key === 'c' || e.key === 'C') setCf(!cf);
  };

  let poll = 0;
  const load = async () => {
    const r = await api('/api/v1/guardian/incident').catch(e => ({ ready: false, error: e.message }));
    if (!r.ready) { $('#status').textContent = r.error ? `Error: ${r.error}` : 'Reconstructing on the server…'; poll = setTimeout(load, 3000); return; }
    data = r;
    data.frames.forEach((f, k) => { const im = new Image(); im.src = `/demo/incident/frames/${f.img}`; imgs[k] = im; });
    buildMap();
    $('#ranges').innerHTML = [60, 100, 150].map(r => `<button class="chip ${r === range ? 'on' : ''}" data-r="${r}">${r} m</button>`).join('');
    $('#ranges').onclick = e => { const b = e.target.closest('[data-r]'); if (b) setRange(+b.dataset.r); };
    i = data.frames.findIndex(f => f.t >= 24.5);
    marks(); numbers(); setCf(false); show();
    window.addEventListener('keydown', onKey);
    raf = requestAnimationFrame(tick);
  };
  load();
  return () => { clearTimeout(poll); cancelAnimationFrame(raf); window.removeEventListener('keydown', onKey); if (map) map.remove(); };
}
