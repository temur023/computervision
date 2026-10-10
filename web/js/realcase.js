// Real case: a crash at a Tashkent crossing, reconstructed from one dashcam, and what CrossGuard would have done.
// Left: the anonymised dashcam video with the reconstruction projected into it (the hidden motorcycle through the
// corner, the conflict point on the road). Right: the same moment from above on the real map.
import { api, C, drawBox, esc, fitCanvas, h } from './util.js';

const R_EARTH = 6378137;
const MONO = '"Geist Mono", ui-monospace, monospace';

// a short code per browser: a phone that opens #/driver/<code> receives this console's warnings
function channelCode() {
  try {
    let c = localStorage.getItem('swarmsight.channel');
    if (!c) { c = Math.random().toString(36).slice(2, 6); localStorage.setItem('swarmsight.channel', c); }
    return c;
  } catch { return 'demo'; }
}

export function realCase(pane) {
  const ui = h(`<div class="console" tabindex="-1">
    <div class="bar">
      <div class="ctl"><span class="title">Real case</span><span class="lbl">Tashkent · 16 July 2026 · 18:28 · one dashcam</span></div>
      <div class="ctl">
        <button class="netswitch" id="cf" aria-pressed="false" title="With a camera at the crossing (C)"><span class="knob"></span><span id="cflbl">Crossing camera off</span></button>
        <span class="ctl hidden" id="ranges"></span>
        <a class="chip" id="phone" target="_blank" rel="noopener" title="Open on a phone: it beeps when the warning goes out"></a>
        <button class="btn" id="present">Present</button>
      </div>
    </div>
    <div class="stage2">
      <div class="pane cam-pane"><video id="vid" muted playsinline preload="auto"></video><canvas class="overlay" id="hud"></canvas>
        <span class="cap">What the dashcam car sees</span>
        <div class="alertbar calm" id="bar"><i></i><span>Loading…</span></div></div>
      <div class="pane map-pane2"><div id="map" style="position:absolute;inset:0"></div><span class="cap right" style="z-index:500" id="mapcap">Top view · the real crossing</span></div>
    </div>
    <div class="tl" id="tl"><button class="playbtn" id="play" aria-label="Play">▶</button>
      <div class="track" id="track"><div class="bars" id="marks"></div><div class="head" id="head"></div></div>
      <span class="time" id="time">—</span></div>
    <div class="nums" id="nums"></div>
    <div class="legend dark">
      <span><i style="background:#e9e7e2"></i>our car (GPS)</span>
      <span><i style="background:#7b9bff"></i>the car ahead (tracked)</span>
      <span><i style="background:#ff5a4a"></i>motorcycle · solid: known to the system · dashed: reconstructed, nobody sees it yet</span>
      <span><i style="background:#f0a030"></i>crossing camera and CrossGuard</span>
    </div>
    <p class="statusline" id="status">Reconstructing on the server…</p>
  </div>`);
  pane.append(ui);
  const $ = s => ui.querySelector(s);
  const vid = $('#vid');
  let data = null, cf = false, range = 100, raf = 0, map = null, sentAt = '', poll = 0;
  const L_ = {};
  const channel = channelCode();
  $('#phone').href = `#/driver/${channel}`;
  $('#phone').textContent = `Phone: ${location.host}/#/driver/${channel}`;

  const scn = () => data.counterfactual.scenarios.find(x => x.range_m === range) || data.counterfactual.scenarios[0];
  const T = () => data.video.t0 + vid.currentTime;
  const ll = xy => [data.origin[0] + (xy[1] / R_EARTH) * 180 / Math.PI, data.origin[1] + (xy[0] / (R_EARTH * Math.cos(data.origin[0] * Math.PI / 180))) * 180 / Math.PI];
  const lerp = (a, b, f) => [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];

  // our car's position at any time (track sampled at 8 Hz)
  function egoAt(t) {
    const tr = data.ego_track;
    let k = tr.findIndex(r => r[0] > t);
    if (k <= 0) k = k === 0 ? 1 : tr.length - 1;
    const a = tr[k - 1], b = tr[k], f = Math.max(0, Math.min(1, (t - a[0]) / (b[0] - a[0] || 1)));
    return { xy: lerp([a[1], a[2]], [b[1], b[2]], f), kmh: Math.round(a[3] + (b[3] - a[3]) * f) };
  }
  function frameAt(t) {
    const fr = data.frames;
    const k = fr.findIndex(f => f.t > t);
    if (k === -1) return [fr[fr.length - 1], fr[fr.length - 1], 0];
    if (k === 0) return [fr[0], fr[0], 0];
    return [fr[k - 1], fr[k], (t - fr[k - 1].t) / (fr[k].t - fr[k - 1].t)];
  }
  const motoAt = t => { const m = data.motorcycle; return [m.impact_xy[0] - m.dir[0] * m.speed_ms * (data.impact_t - t), m.impact_xy[1] - m.dir[1] * m.speed_ms * (data.impact_t - t)]; };
  const leadAt = t => { const [a, b, f] = frameAt(t); return a.lead_xy && b.lead_xy ? lerp(a.lead_xy, b.lead_xy, f) : a.lead_xy || b.lead_xy; };
  const known = t => (cf && t >= scn().t_detect) || t >= data.motorcycle.visible_from_t;  // does the system know about the motorcycle?

  // project a ground point (local metres) into the dashcam image (1280-wide pixels)
  function project(p, t) {
    const e = egoAt(t).xy, c = data.calibration;
    const d = [p[0] - e[0], p[1] - e[1]];
    const Z = d[0] * data.ego_dir[0] + d[1] * data.ego_dir[1], X = d[0] * data.ego_right[0] + d[1] * data.ego_right[1];
    if (Z < 3) return null;
    return { x: c.vanishing_x + X * c.focal_px / Z, y: c.horizon_y + c.focal_px * c.camera_height_m / Z, Z, X, s: c.focal_px / Z };
  }

  // ---------- map
  function rect(xy, ang, Lm, Wm) {  // oriented footprint (ang: unit direction vector)
    const r = [ang[1], -ang[0]];
    return [[1, 1], [1, -1], [-1, -1], [-1, 1]].map(([a, b]) => ll([xy[0] + ang[0] * Lm / 2 * a + r[0] * Wm / 2 * b, xy[1] + ang[1] * Lm / 2 * a + r[1] * Wm / 2 * b]));
  }
  function buildMap() {
    map = L.map($('#map'), { zoomControl: false, zoomSnap: 0.25 });
    const P = data.paths;
    map.fitBounds(L.latLngBounds([...P.ego.slice(-20), ...P.lead, data.impact, P.moto_before[P.moto_before.length - 4]]), { padding: [30, 30] });
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 21, maxNativeZoom: 19, className: 'tiles-dark', attribution: '© OpenStreetMap contributors' }).addTo(map);
    L_.cone = L.polygon([[0, 0]], { stroke: false, fillColor: '#e9e7e2', fillOpacity: 0.08, interactive: false }).addTo(map);
    L_.cover = L.layerGroup().addTo(map);
    L_.motoPath = L.polyline([], { color: '#ff5a4a', weight: 2, opacity: 0.7, dashArray: '5 6' }).addTo(map);
    L_.egoTrail = L.polyline([], { color: '#e9e7e2', weight: 2, opacity: 0.45 }).addTo(map);
    L_.leadTrail = L.polyline([], { color: '#7b9bff', weight: 2, opacity: 0.55 }).addTo(map);
    L_.sight = L.polyline([], { color: '#ff5a4a', weight: 1.5, opacity: 0.9, dashArray: '2 5' }).addTo(map);
    L.circleMarker(data.impact, { radius: 8, color: '#ff5a4a', weight: 2, fillOpacity: 0.12 }).addTo(map)
      .bindTooltip('impact', { permanent: true, direction: 'right', className: 'maptag crit' });
    L_.ego = L.polygon(rect([0, 0], data.ego_dir, 4.6, 1.9), { color: '#0d0f12', weight: 1.5, fillColor: '#e9e7e2', fillOpacity: 1 }).addTo(map);
    L_.lead = L.polygon(rect([0, 0], data.ego_dir, 4.6, 1.9), { color: '#0d0f12', weight: 1.5, fillColor: '#7b9bff', fillOpacity: 1 }).addTo(map);
    L_.moto = L.polygon(rect([0, 0], data.motorcycle.dir, 2.2, 0.9), { color: '#ff5a4a', weight: 2, fillColor: '#ff5a4a', fillOpacity: 1 }).addTo(map);
    L_.motoTag = L.tooltip({ permanent: true, direction: 'top', className: 'maptag crit', offset: [0, -8] }).setLatLng(data.impact).setContent('');
  }
  function drawCover() {
    L_.cover.clearLayers();
    if (!cf) return;
    const cover = data.paths.camera_cover[String(range)], b = scn().braking?.[0];
    L.polyline(cover, { color: '#f0a030', weight: 18, opacity: 0.14 }).addTo(L_.cover);
    L.circleMarker(cover[0], { radius: 7, color: '#f0a030', weight: 2, fillColor: '#0d0f12', fillOpacity: 1 }).addTo(L_.cover)
      .bindTooltip(`crossing camera · sees ${range} m`, { permanent: true, direction: 'top', className: 'maptag amber' });
    if (b) L.circleMarker(b.car_at, { radius: 7, color: b.avoided ? '#3fbf86' : '#f0a030', weight: 2, fillColor: b.avoided ? '#3fbf86' : '#f0a030', fillOpacity: 0.3 }).addTo(L_.cover)
      .bindTooltip(b.avoided ? `with the warning it stops ${b.clear_m} m short` : 'with the warning: still in the path', { permanent: true, direction: 'bottom', className: 'maptag amber' });
  }
  function drawMap(t) {
    const e = egoAt(t).xy, lead = leadAt(t), m = motoAt(t), kn = known(t);
    L_.ego.setLatLngs(rect([e[0] - data.ego_dir[0] * 1.5, e[1] - data.ego_dir[1] * 1.5], data.ego_dir, 4.6, 1.9));
    const hf = Math.atan(640 / data.calibration.focal_px), d = data.ego_dir;
    const rot = a => [d[0] * Math.cos(a) - d[1] * Math.sin(a), d[0] * Math.sin(a) + d[1] * Math.cos(a)];
    L_.cone.setLatLngs([ll(e), ...[hf, 0, -hf].map(a => { const v = rot(a); return ll([e[0] + v[0] * 70, e[1] + v[1] * 70]); })]);
    L_.egoTrail.setLatLngs(data.ego_track.filter(r => r[0] <= t).map(r => ll([r[1], r[2]])));
    if (lead) { L_.lead.setLatLngs(rect(lead, data.ego_dir, 4.6, 1.9)); L_.leadTrail.setLatLngs(data.frames.filter(f => f.t <= t && f.lead_xy).map(f => ll(f.lead_xy))); }
    const show = t >= data.impact_t - 3.2;
    L_.moto.setLatLngs(show ? rect(m, data.motorcycle.dir, 2.2, 0.9) : [[0, 0]]);
    L_.moto.setStyle({ dashArray: kn ? null : '3 3', fillOpacity: kn ? 1 : 0.15 });
    if (show) {
      L_.motoTag.setLatLng(ll(m)).setContent(`motorcycle · ${data.motorcycle.speed_kmh} km/h${kn ? '' : ' · nobody sees it'}`);
      if (!map.hasLayer(L_.motoTag)) L_.motoTag.addTo(map);
    } else if (map.hasLayer(L_.motoTag)) L_.motoTag.remove();
    L_.motoPath.setLatLngs(show ? [ll(m), data.impact] : []);
    L_.sight.setLatLngs(show && t < data.motorcycle.visible_from_t ? [ll(e), ll(m)] : []);
    $('#mapcap').textContent = `Top view · T−${Math.max(0, data.impact_t - t).toFixed(1)} s to impact`;
  }

  // ---------- dashcam overlay: the reconstruction projected into the image
  function label(ctx, text, x, y, col, fs, dpr) {
    ctx.font = `600 ${fs}px ${MONO}`;
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = 'rgba(13,15,18,.85)'; ctx.fillRect(x, y, tw + 12 * dpr, fs * 1.9);
    ctx.fillStyle = col; ctx.fillRect(x, y, 2 * dpr, fs * 1.9);
    ctx.fillStyle = '#e9e7e2'; ctx.fillText(text, x + 7 * dpr, y + fs * 1.3);
    return tw + 12 * dpr;
  }
  function drawHud(t) {
    const cv = $('#hud'), r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const ctx = fitCanvas(cv, Math.round(r.width * dpr), Math.round(r.height * dpr));
    ctx.clearRect(0, 0, cv.width, cv.height);
    const s960 = cv.width / 960, s1280 = cv.width / 1280, fs = 11 * dpr;
    const [a, b, f] = frameAt(t);
    const lead = data.lead_car.track_ids;
    for (const ta of a.tracks) {
      if (!lead.includes(ta.id)) continue;
      const tb = b.tracks.find(x => x.id === ta.id);
      const box = tb ? ta.bbox.map((v, i) => v + (tb.bbox[i] - v) * f) : ta.bbox;
      drawBox(ctx, box, 'car ahead', C.network, { scale: s960, lw: 1.6 });
    }
    const m = data.motorcycle;
    if (t >= m.visible_from_t) {  // seen by this camera: our annotation
      const cur = data.frames.filter(x => x.moto_box && x.t <= t + 0.07).pop();
      if (cur) drawBox(ctx, cur.moto_box, 'motorcycle · seen now', C.critical, { scale: s960, lw: 2.2 });
    } else if (known(t)) {  // known through the crossing camera: drawn where it really is, through the corner
      const p = project(motoAt(t), t);
      if (p && p.x >= 0 && p.x <= 1280) {
        const w = 2.0 * p.s, hh = 1.5 * p.s;
        drawBox(ctx, [p.x - w / 2, p.y - hh, p.x + w / 2, p.y], `motorcycle behind the corner · ${m.speed_kmh} km/h`, C.critical, { scale: s1280, lw: 2, dashed: true });
      } else if (p) {  // off-screen: an arrow at the edge it will come from
        const y = Math.max(cv.height * 0.42, Math.min(cv.height * 0.7, (data.calibration.horizon_y + 30) * s1280));
        const left = p.x < 0, x = left ? 12 * dpr : cv.width - 12 * dpr, dir = left ? -1 : 1;
        const mp = motoAt(t), e = egoAt(t).xy, dist = Math.hypot(mp[0] - e[0], mp[1] - e[1]);
        const pulse = 0.6 + 0.4 * Math.sin(performance.now() / 120);
        ctx.globalAlpha = pulse; ctx.fillStyle = C.critical;
        ctx.beginPath(); ctx.moveTo(x + dir * 2 * dpr, y); ctx.lineTo(x - dir * 18 * dpr, y - 14 * dpr); ctx.lineTo(x - dir * 18 * dpr, y + 14 * dpr); ctx.closePath(); ctx.fill();
        ctx.globalAlpha = 1;
        const text = `MOTORCYCLE ${Math.round(dist)} M · ${m.speed_kmh} KM/H · BEHIND THE CORNER`;
        ctx.font = `600 ${fs}px ${MONO}`;
        const tw = ctx.measureText(text).width + 12 * dpr;
        label(ctx, text, left ? x + 24 * dpr : x - 24 * dpr - tw, y - fs, C.critical, fs, dpr);
      }
    }
    // the predicted conflict point on the road, with a countdown
    const sc = scn();
    if (cf && sc.t_warn != null && t >= sc.t_warn && t < data.impact_t) {
      const p = project(m.impact_xy, t);
      if (p) {
        const cx = p.x * s1280, cy = p.y * s1280, rx = Math.max(10 * dpr, 2.2 * p.s * s1280), ry = Math.max(3 * dpr, rx * 0.2);
        const ph = (performance.now() / 700) % 1;
        ctx.strokeStyle = C.critical; ctx.lineWidth = 2.2 * dpr;
        ctx.beginPath(); ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2); ctx.stroke();
        ctx.globalAlpha = 1 - ph; ctx.beginPath(); ctx.ellipse(cx, cy, rx * (1 + ph), ry * (1 + ph), 0, 0, Math.PI * 2); ctx.stroke(); ctx.globalAlpha = 1;
        const text = `CONFLICT IN ${(data.impact_t - t).toFixed(1)} S`;
        ctx.font = `600 ${fs}px ${MONO}`;
        const tw = ctx.measureText(text).width + 12 * dpr;
        label(ctx, text, cx - tw / 2, cy + ry + 8 * dpr, C.critical, fs, dpr);
      }
    }
  }

  function narrative(t) {
    const m = data.motorcycle, c = scn(), kmh = egoAt(t).kmh;
    if (t >= data.impact_t - 0.02) return ['<b>Impact.</b> The replay stops here: we show the conflict, not its outcome.', 'alertbar'];
    if (cf && c.t_warn != null && t >= c.t_warn) {
      const b = c.braking[0];
      return [`<b>CrossGuard → car ahead: ${esc(c.action)}.</b> ${esc(c.warning.reason)} ${b.avoided ? `Braking now, it stays ${b.clear_m} m short of the motorcycle's path.` : 'Even braking now, it is still in the path.'}`, 'alertbar'];
    }
    if (t >= m.visible_from_t) return [`<b>Motorcycle visible to this car only now:</b> ${(data.impact_t - t).toFixed(2)} s before impact. No driver reacts that fast.`, 'alertbar'];
    if (cf && t >= c.t_detect) return [`<b>Crossing camera:</b> motorcycle on the cross street, ~${m.speed_kmh} km/h. This car cannot see it: the corner is in the way.`, 'alertbar hazard'];
    if (t >= 26) return [`Our car moves off (${kmh} km/h) behind the dark car. The cross street on the left is hidden by the corner.`, 'alertbar calm'];
    return ['Both cars wait at the traffic officer. The dark car pulls out first.', 'alertbar calm'];
  }

  // the warning goes out over the same channel a crossing camera would use: /api/v1/alerts/{channel}
  function publish(t) {
    const m = data.motorcycle, c = scn();
    let a = null, key = '';
    if (cf && c.t_warn != null && t >= c.t_warn && t < data.impact_t) {
      key = `cf-${range}`;
      a = { kind: 'collision_risk', level: 'high', cls: 'motorcyclist', ttc_s: Math.max(0, data.impact_t - t), bearing_deg: 270, speed_kmh: m.speed_kmh,
        ttl_s: 3, action: c.action || 'Brake now', text: c.warning.reason, lat: data.impact[0], lon: data.impact[1], source: `crossing camera, ${range} m` };
    } else if (!cf && t >= m.visible_from_t && t < data.impact_t) {
      key = 'dashcam';
      a = { kind: 'collision_risk', level: 'high', cls: 'motorcyclist', ttc_s: Math.max(0, data.impact_t - t), bearing_deg: 270, speed_kmh: m.speed_kmh,
        ttl_s: 2, action: 'Brake', text: 'Seen by this car only now: too late to react.', source: 'own dashcam' };
    }
    if (!a || sentAt === key) return;
    sentAt = key;
    api(`/api/v1/alerts/${channel}`, { method: 'POST', json: a }).catch(() => {});
  }

  function marks() {
    const t0 = data.video.t0, t1 = data.impact_t, x = t => `${(100 * (t - t0)) / (t1 - t0)}%`;
    const m = data.motorcycle, c = scn();
    const list = [[m.visible_from_t, t1, C.critical, 'motorcycle visible to this car', '100%', 'bottom:0']];
    if (cf && c.t_warn != null) list.push([c.t_warn, t1, C.hazard, 'CrossGuard warning with a crossing camera', '45%', 'top:0']);
    $('#marks').innerHTML = list.map(([a, b, col, title, hh, pos]) => `<span title="${esc(title)}" style="left:${x(a)};width:max(4px, calc(${x(b)} - ${x(a)}));height:${hh};background:${col};${pos}"></span>`).join('');
  }
  function numbers() {
    const m = data.motorcycle, c = scn(), b = c.braking?.[0], b2 = c.braking?.[1], need = data.counterfactual.min_range_avoided_m;
    $('#nums').innerHTML = [
      [m.visible_s.toFixed(2), 's', 'how long the dashcam car could see the motorcycle before impact: the corner hid the cross street'],
      [cf && c.lead_s != null ? c.lead_s.toFixed(1) : '—', 's', cf ? `CrossGuard warning before impact, with a crossing camera that sees ${range} m of the cross street` : 'turn the crossing camera on to see when CrossGuard would warn'],
      [cf && b ? (b.avoided ? b.clear_m.toFixed(1) : 'no') : '—', cf && b && b.avoided ? 'm' : '', cf && b ? (b.avoided ? `the car ahead stays this far short of the motorcycle's path (reaction 1.0 s${b2 ? `; 1.5 s: ${b2.avoided ? `${b2.clear_m} m` : 'too late'}` : ''})` : `the car ahead cannot get out of the path with ${range} m of view${need ? `; it needs about ${need} m` : ''}`) : (need ? `metres of cross street a camera must see to prevent this crash: about ${need}` : '')],
    ].map(([v, u, k]) => `<div><div class="v">${v}<small>${u}</small></div><div class="k">${k}</div></div>`).join('');
    $('#status').textContent = `Computed on this server in ${data.compute_s ?? '—'} s: car ahead tracked in ${data.lead_car.observations} frames (~${data.lead_car.speed_kmh} km/h at the crossing), motorcycle ~${m.speed_kmh} km/h (${m.range_kmh[0]}–${m.range_kmh[1]}). `
      + `Measured: ${data.measured.join('; ')}. Estimated: ${data.estimated.join('; ')}. Assumed: ${data.assumed.join('; ')}. ${data.source}.`;
  }

  function render() {
    const t = T();
    // slow motion over the last 1.3 s, so the moment the motorcycle appears can be seen
    const rate = t > data.impact_t - 1.3 ? 0.3 : 1;
    if (vid.playbackRate !== rate) vid.playbackRate = rate;
    if (!vid.paused) publish(t);
    if (t < 25) sentAt = '';
    drawHud(t); drawMap(t);
    const [text, cls] = narrative(t);
    if ($('#bar').className !== cls) $('#bar').className = cls;
    const span = $('#bar').querySelector('span');
    if (span.innerHTML !== text) span.innerHTML = text;
    const frac = (t - data.video.t0) / (data.impact_t - data.video.t0);
    $('#head').style.left = `calc(${Math.min(100, 100 * frac)}% - 1px)`;
    $('#time').textContent = `T−${Math.max(0, data.impact_t - t).toFixed(1)} s · ${egoAt(t).kmh} km/h`;
    const want = vid.paused ? '▶' : '❚❚';
    if ($('#play').textContent !== want) $('#play').textContent = want;
  }
  const loop = () => { if (data) render(); raf = requestAnimationFrame(loop); };

  const setCf = v => {
    cf = v;
    $('#cf').classList.toggle('on', cf); $('#cf').setAttribute('aria-pressed', String(cf));
    $('#cflbl').textContent = cf ? 'Crossing camera on' : 'Crossing camera off';
    $('#ranges').classList.toggle('hidden', !cf);
    drawCover(); marks(); numbers();
  };
  const setRange = r => {
    range = r;
    ui.querySelectorAll('#ranges .chip').forEach(b => b.classList.toggle('on', +b.dataset.r === r));
    drawCover(); marks(); numbers();
  };
  const seekT = t => { vid.currentTime = Math.max(0, Math.min((vid.duration || 99) - 0.01, t - data.video.t0)); };
  $('#play').onclick = () => {
    if (!data) return;
    if (vid.ended || T() >= data.impact_t - 0.03) seekT(24.5);
    if (vid.paused) vid.play().catch(() => {}); else vid.pause();
  };
  $('#track').onclick = e => { if (!data) return; const r = $('#track').getBoundingClientRect(); vid.pause(); seekT(data.video.t0 + ((e.clientX - r.left) / r.width) * (data.impact_t - data.video.t0)); };
  $('#cf').onclick = () => data && setCf(!cf);
  $('#present').onclick = () => (document.fullscreenElement ? document.exitFullscreen() : ui.requestFullscreen?.());
  const onKey = e => {
    if (!data || ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
    const r = ui.getBoundingClientRect();
    if (r.bottom < 0 || r.top > window.innerHeight) return;
    if (e.code === 'Space') { e.preventDefault(); $('#play').click(); }
    else if (e.key === 'ArrowRight') { vid.pause(); seekT(T() + 0.125); }
    else if (e.key === 'ArrowLeft') { vid.pause(); seekT(T() - 0.125); }
    else if (e.key === 'c' || e.key === 'C') setCf(!cf);
  };

  const load = async () => {
    const r = await api('/api/v1/guardian/incident').catch(e => ({ ready: false, error: e.message }));
    if (!r.ready) { $('#status').textContent = r.error ? `Error: ${r.error}` : 'Reconstructing on the server…'; poll = setTimeout(load, 3000); return; }
    data = r;
    vid.poster = `/demo/incident/${data.video.poster}`;
    vid.innerHTML = `<source src="/demo/incident/${data.video.file}" type="video/mp4">${data.video.webm ? `<source src="/demo/incident/${data.video.webm}" type="video/webm">` : ''}`;
    vid.load();
    buildMap();
    $('#ranges').innerHTML = [60, 100, 150].map(x => `<button class="chip ${x === range ? 'on' : ''}" data-r="${x}">${x} m</button>`).join('');
    $('#ranges').onclick = e => { const b = e.target.closest('[data-r]'); if (b) setRange(+b.dataset.r); };
    vid.addEventListener('loadeddata', () => { if (vid.currentTime < 0.01) seekT(24.5); }, { once: true });
    setCf(false);
    window.addEventListener('keydown', onKey);
  };
  raf = requestAnimationFrame(loop);
  load();
  return () => { clearTimeout(poll); cancelAnimationFrame(raf); window.removeEventListener('keydown', onKey); vid.pause(); if (map) map.remove(); };
}
