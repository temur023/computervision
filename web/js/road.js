import { api, esc, fmtTime, h, pretty } from './util.js';
import { baseLayer, mountScan, TASHKENT } from './scan.js';

const STATUS_COL = { candidate: '#d4800f', corroborated: '#2350d1', verified: '#1d7f55', rejected: '#c23a2e', expired: '#8b9099' };
const FILTERS = { open: 'candidate,corroborated', verified: 'verified', all: 'candidate,corroborated,verified,rejected,expired' };

export default async function road(root, args) {
  const [metrics, data] = await Promise.all([api('/api/v1/metrics').catch(() => ({})), api('/api/v1/data').catch(() => ({ datasets: [] }))]);
  const ds = Object.fromEntries((data.datasets || []).map(d => [d.id, d]));
  const rd = metrics.road_damage_eval, tk = metrics.tashkent_eval;
  const rdd = ds.rdd2022, tash = ds.tashkent;
  const nRdd = rdd ? rdd.used.train + rdd.used.val + rdd.used.test : null;

  root.append(h(`<div class="wrap">
    <section class="page-head">
      <span class="label">Module 02 · Road Intelligence · first revenue</span>
      <h1>Road damage, verified before anyone acts on it</h1>
      <p class="lead">Phone and dashcam footage in. A deduplicated list of road defects out, each with a position, evidence and a confidence.</p>
    </section>
    <div class="spec">
      <div><span class="label">Problem · users</span><p>Defects reach the road service as complaints, late and duplicated.
        <b>City road service, Yandex Maps, taxi fleets.</b></p></div>
      <div><span class="label">Data</span><p><b>${nRdd ? nRdd.toLocaleString('en') : '—'}</b> RDD2022 images, 6 countries.
        <b>${tash?.used?.test ?? 120}</b> Tashkent images labelled by us.</p></div>
      <div><span class="label">Model</span><p><b>YOLO11n trained by us</b> for 4 damage types. Runs on a CPU.</p></div>
      <div><span class="label">Measured</span><p>${rd ? `<b>mAP@.5 ${rd.test.map50.toFixed(2)}</b> on ${rd.n_test_images.toLocaleString('en')} unseen images` : 'Held-out evaluation runs when training finishes.'}
        ${tk ? `<br><b>Tashkent AP ${tk.ap_image.toFixed(2)}</b> (random ${(tk.n_positive / tk.n_images).toFixed(2)})` : ''}</p></div>
    </div>

    <section class="section" style="border-top:0">
      <div class="row between" style="margin-bottom:1.2rem">
        <div class="seg" id="tabs"><button data-t="scan">Scan</button><button data-t="events">Event map</button><button data-t="review">Label review</button></div>
        <span class="faint" style="font-size:.86rem" id="tabnote"></span>
      </div>
      <div id="pane"></div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">How it works</span><div><h2>From one report to a verified event</h2></div></div>
      <div class="pipeline p4">
        <div><span class="label">01</span><b>Detect</b><p>Our model finds potholes and three kinds of crack in each frame.</p></div>
        <div><span class="label">02</span><b>Track</b><p>A pothole seen in five frames is one report, not five.</p></div>
        <div><span class="label">03</span><b>Merge</b><p>A report of the same type within 12–15 m joins the existing event.</p></div>
        <div><span class="label">04</span><b>Trust</b><p>Two independent sources make an event corroborated. A moderator verifies or rejects it.</p></div>
      </div>
      <div class="flow">
        <span class="status s-candidate">candidate</span><span class="arr">→</span>
        <span class="status s-corroborated">corroborated</span><span class="arr">→</span>
        <span class="status s-verified">verified</span><span class="faint">or</span><span class="status s-rejected">rejected</span>
        <span class="faint">· expires after 30 days (potholes) or 90 days (cracks) without a new report</span>
      </div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">Data &amp; limits</span><div><h2>What it learned from, and where it is weak</h2></div></div>
      <div class="cols c2">
        <div class="facts">
          ${rdd ? `<div><b>RDD2022</b><span>${rdd.used.train.toLocaleString('en')} train · ${rdd.used.val} val · ${rdd.used.test.toLocaleString('en')} test images · Japan, India, Czechia, Norway, USA, China · ${esc(rdd.license)}</span></div>` : ''}
          ${tash ? `<div><b>Tashkent, labelled by us</b><span>${tash.used?.test ?? 120} Mapillary street images from map tiles kept out of any training · ${esc(tash.license)}</span></div>` : ''}
          <a class="mono" style="font-size:.8rem" href="#/data">Every number and its source →</a>
        </div>
        <ul class="limits">
          <li>RDD2022 has no Central Asian roads. The Tashkent set measures that gap.</li>
          <li>Thin cracks far from the camera and wet asphalt cause most errors.</li>
          <li>Night and heavy rain are tested only with synthetic perturbations so far.</li>
        </ul>
      </div>
    </section>
  </div>`));

  const pane = root.querySelector('#pane');
  const notes = { scan: 'Analysed on our server. Nothing is pre-computed.', events: 'Every event here was created by a scan on this deployment.', review: 'Human check of the Tashkent test labels.' };
  let cleanup = null;
  const open = (t, rest = []) => {
    if (cleanup) { cleanup(); cleanup = null; }
    root.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('active', b.dataset.t === t));
    root.querySelector('#tabnote').textContent = notes[t];
    pane.innerHTML = '';
    cleanup = t === 'events' ? mountEvents(pane, rest[0] ? +rest[0] : null) : t === 'review' ? mountReview(pane) : mountScan(pane);
  };
  root.querySelector('#tabs').onclick = e => e.target.dataset.t && open(e.target.dataset.t);
  open(['events', 'review'].includes(args[0]) ? args[0] : 'scan', args.slice(1));
  return () => cleanup && cleanup();
}

export function mountEvents(container, selected) {
  const ui = h(`<div class="work">
    <div class="stack">
      <div class="row between">
        <div class="seg" id="filters"><button data-f="open" class="active">Open</button><button data-f="verified">Verified</button><button data-f="all">All</button></div>
        <div class="legend">${['candidate', 'corroborated', 'verified'].map(s => `<span><b class="pin" style="background:${STATUS_COL[s]}"></b>${s}</span>`).join('')}</div>
      </div>
      <div id="map" class="map"></div>
      <p class="mono faint" style="font-size:.78rem">Partner feed: <a href="/api/v1/road-events?status=verified&format=geojson" target="_blank" rel="noopener">verified events, GeoJSON</a>
        · <a href="/docs" target="_blank" rel="noopener">API reference</a></p>
    </div>
    <aside>
      <div class="panel"><span class="label">Events</span><div class="list" id="list"><p class="faint">Loading…</p></div></div>
      <div class="panel hidden" id="detail"></div>
      <details class="panel more"><summary>Source reliability, learned from moderation</summary><div id="sources" style="margin-top:.8rem"><p class="faint">—</p></div></details>
    </aside>
  </div>`);
  container.append(ui);
  const map = L.map(ui.querySelector('#map')).setView(TASHKENT, 12);
  baseLayer().addTo(map);
  const layer = L.layerGroup().addTo(map);
  let events = [], filter = 'open', zone = null;

  async function load(fit = false) {
    events = (await api(`/api/v1/road-events?status=${FILTERS[filter]}&format=json`)).events;
    layer.clearLayers();
    const list = ui.querySelector('#list');
    list.innerHTML = events.length ? '' : '<p class="faint">No events yet. Scan a road photo that has a location, or one of the Tashkent images on the Scan tab.</p>';
    for (const e of events) {
      if (e.lat != null) {
        L.circleMarker([e.lat, e.lon], { radius: 6 + 3 * e.severity, color: '#ffffff', weight: 2, fillColor: STATUS_COL[e.status], fillOpacity: 1 })
          .addTo(layer).on('click', () => showDetail(e.id)).bindTooltip(`${pretty(e.type)} · ${e.status}`);
      }
      const it = h(`<div class="item"><div><div class="t">${esc(pretty(e.type))}</div>
        <div class="m">${e.n_sources} source${e.n_sources === 1 ? '' : 's'} · confidence ${e.confidence.toFixed(2)} · ${fmtTime(e.last_seen)}</div></div>
        <span class="status s-${e.status}">${e.status}</span></div>`);
      it.onclick = () => showDetail(e.id);
      list.append(it);
    }
    if (fit && events.some(e => e.lat != null)) map.fitBounds(L.latLngBounds(events.filter(e => e.lat != null).map(e => [e.lat, e.lon])).pad(0.3), { maxZoom: 16 });
    const s = await api('/api/v1/sources');
    ui.querySelector('#sources').innerHTML = s.sources.length ? `<table class="tbl"><thead><tr><th>Source</th><th class="r">Verified</th><th class="r">Rejected</th><th class="r">Weight</th></tr></thead><tbody>
      ${s.sources.slice(0, 10).map(x => `<tr><td class="mono" style="font-size:.8rem">${esc(x.source_id.slice(0, 12))}</td><td class="r">${Math.round(x.alpha - 2)}</td><td class="r">${Math.round(x.beta - 1)}</td><td class="r">${x.reliability.toFixed(2)}</td></tr>`).join('')}</tbody></table>
      <p class="faint" style="font-size:.8rem;margin-top:.6rem">Every source starts at weight 0.67 (Beta(2, 1) prior). A verified report raises it, a rejected one lowers it.</p>`
      : '<p class="faint">No sources yet.</p>';
  }

  async function showDetail(id) {
    const e = await api(`/api/v1/road-events/${id}`);
    if (zone) zone.remove();
    if (e.lat != null) {
      zone = L.circle([e.lat, e.lon], { radius: e.radius_m, color: STATUS_COL[e.status], weight: 1, fillOpacity: .06, interactive: false }).addTo(map);
      map.setView([e.lat, e.lon], Math.max(map.getZoom(), 16));
    }
    const d = ui.querySelector('#detail');
    d.classList.remove('hidden');
    d.innerHTML = `<div class="row between"><h3>${esc(pretty(e.type))}</h3><span class="status s-${e.status}">${e.status}</span></div>
      ${e.evidence ? `<img class="evidence" style="margin-top:.8rem" src="/media/${esc(e.evidence)}" alt="evidence, people and vehicles blurred">` : ''}
      <table class="kv" style="margin-top:.6rem"><tbody>
        <tr><th>Confidence</th><td>${e.confidence.toFixed(2)} from ${e.n_sources} independent source${e.n_sources === 1 ? '' : 's'}</td></tr>
        <tr><th>Position</th><td>${e.lat?.toFixed(5)}, ${e.lon?.toFixed(5)} · ±${e.radius_m} m</td></tr>
        <tr><th>Last seen</th><td>${fmtTime(e.last_seen)}</td></tr>
        <tr><th>Expires</th><td>${fmtTime(e.expires_at)} unless reported again</td></tr></tbody></table>
      ${['candidate', 'corroborated'].includes(e.status) ? `<div class="row" style="margin-top:1rem"><input type="text" id="note" placeholder="Moderator note" style="flex:1;min-width:140px">
        <button class="btn" data-a="reject">Reject</button><button class="btn primary" data-a="verify">Verify</button></div>` : ''}
      <details class="more" style="margin-top:1rem"><summary>Reports and history (${e.observations.length})</summary>
        <div class="scroll-x" style="margin-top:.6rem"><table class="tbl"><thead><tr><th>Time</th><th>Source</th><th class="r">Conf</th><th class="r">Weight</th></tr></thead><tbody>
          ${e.observations.map(o => `<tr><td class="mono" style="font-size:.78rem">${fmtTime(o.observed_at)}</td><td class="mono" style="font-size:.78rem">${esc(o.source_id)}</td><td class="r">${o.conf.toFixed(2)}</td><td class="r">${o.source_reliability.toFixed(2)}</td></tr>`).join('')}</tbody></table></div>
        <div class="mono faint" style="font-size:.76rem;line-height:1.7;margin-top:.6rem">${e.history.map(x => `${fmtTime(x.t)} · ${esc(x.action)} · ${esc(x.by)}${x.note ? ` · ${esc(x.note)}` : ''}`).join('<br>')}</div>
      </details>`;
    d.querySelectorAll('button[data-a]').forEach(b => b.onclick = async () => {
      await api(`/api/v1/road-events/${id}/review`, { method: 'PATCH', json: { action: b.dataset.a, reviewer: 'web-moderator', note: d.querySelector('#note')?.value || '' } });
      await load(); showDetail(id);
    });
  }

  ui.querySelector('#filters').onclick = e => {
    if (!e.target.dataset.f) return;
    filter = e.target.dataset.f;
    ui.querySelectorAll('#filters button').forEach(b => b.classList.toggle('active', b.dataset.f === filter));
    load(true);
  };
  load(true).then(() => selected && showDetail(selected));
  return () => map.remove();
}

// Human verification of the Tashkent test labels (first pass made without model predictions).
export function mountReview(container) {
  const ui = h(`<div class="work">
    <div class="stack">
      <div class="row between">
        <div class="seg" id="filter"><button data-f="todo" class="active">To review</button><button data-f="all">All</button><button data-f="disagree">Disagreements</button></div>
        <span class="mono faint" id="pos" style="font-size:.8rem"></span>
      </div>
      <div class="instrument"><img id="img" alt="Tashkent street image" style="min-height:200px"><span class="tag" id="tag"></span></div>
      <p class="note">Is road damage visible on the drivable surface ahead? Patches, sealed cracks, manholes and wet spots are not damage.
        Mark "uncertain" when it cannot be judged (night, blur, rain on the glass) and "not a road" for shops, facades and interiors.
        Keys: <span class="mono">Y</span> confirm first pass · <span class="mono">←/→</span> previous/next.</p>
    </div>
    <aside>
      <div class="panel"><span class="label">Progress</span><div id="prog"></div></div>
      <div class="panel"><span class="label">First pass</span><div id="first"></div></div>
      <div class="panel"><span class="label">Your verdict</span>
        <div class="stack" style="margin-top:.3rem">
          <div class="seg" id="dmg"><button data-v="1">Damage</button><button data-v="0">No damage</button></div>
          <div id="types" class="row"></div>
          <div class="row"><label class="check"><input type="checkbox" id="unc"> uncertain</label><label class="check"><input type="checkbox" id="nr"> not a road</label></div>
          <label class="field"><span>Reviewer</span><input type="text" id="who" placeholder="your name"></label>
          <div class="row"><button class="btn" id="agree">Confirm first pass</button><button class="btn primary" id="save">Save verdict</button></div>
          <div class="row"><button class="btn" id="prev">← Previous</button><button class="btn" id="next">Next →</button></div>
          <p class="mono faint" id="msg" style="font-size:.78rem"></p>
        </div>
      </div>
      <div class="panel"><span class="label">Export</span><p class="faint" style="font-size:.86rem">Merged labels (verified where reviewed) for the repository.</p>
        <a class="btn" style="margin-top:.6rem" href="/api/v1/labels/tashkent/export">Download labels_test.json</a></div>
    </aside>
  </div>`);
  container.append(ui);
  let data = null, list = [], idx = 0, filter = 'todo', verdict = null;
  const who = ui.querySelector('#who');
  try { who.value = localStorage.getItem('swarmsight.reviewer') || ''; } catch {}
  who.onchange = () => { try { localStorage.setItem('swarmsight.reviewer', who.value); } catch {} };

  async function load(keepId) {
    data = await api('/api/v1/labels/tashkent');
    list = data.items.filter(i => filter === 'all' || (filter === 'todo' ? !i.review : i.review && !i.review.agrees));
    idx = Math.max(0, keepId ? list.findIndex(i => i.id === keepId) : 0);
    const p = data.progress;
    ui.querySelector('#prog').innerHTML = `<div class="progress" style="margin:.2rem 0 .7rem"><div style="width:${100 * p.reviewed / p.total}%"></div></div>
      <table class="kv"><tbody><tr><th>Reviewed</th><td>${p.reviewed} / ${p.total}</td></tr>
      <tr><th>Agreement with first pass</th><td>${p.agreement == null ? '—' : (p.agreement * 100).toFixed(0) + '%'}</td></tr></tbody></table>`;
    ui.querySelector('#types').innerHTML = data.types.map(t => `<label class="check"><input type="checkbox" value="${t}">${esc(pretty(t))}</label>`).join('');
    ui.querySelectorAll('#types input').forEach(c => c.onchange = () => { verdict.types = [...ui.querySelectorAll('#types input:checked')].map(x => x.value); });
    show();
  }
  function show() {
    const it = list[idx];
    ui.querySelector('#pos').textContent = list.length ? `${idx + 1} / ${list.length}` : 'nothing in this filter';
    if (!it) { ui.querySelector('#img').removeAttribute('src'); ui.querySelector('#first').innerHTML = '<p class="faint">All done.</p>'; return; }
    ui.querySelector('#img').src = it.image;
    ui.querySelector('#tag').textContent = `${it.id} · ${it.captured_at ? new Date(it.captured_at * 1000).getFullYear() : ''}`;
    const f = it.first_pass;
    ui.querySelector('#first').innerHTML = `<table class="kv"><tbody>
      <tr><th>Damage</th><td>${f.not_road ? 'not a road' : f.damage ? esc(f.types.map(pretty).join(', ')) : 'none'}</td></tr>
      <tr><th>Uncertain</th><td>${f.uncertain ? 'yes' : 'no'}</td></tr>
      <tr><th>Conditions</th><td>${esc(f.conditions.join(', ') || '—')}</td></tr>
      ${it.review ? `<tr><th>Reviewed</th><td>${esc(it.review.reviewer)} · ${it.review.agrees ? 'agrees' : 'corrected'}</td></tr>` : ''}</tbody></table>
      <a class="mono faint" style="font-size:.74rem" href="${esc(it.source)}" target="_blank" rel="noopener">source on Mapillary ↗</a>`;
    const base = it.review || f;
    verdict = { damage: base.damage, types: [...base.types], uncertain: base.uncertain, not_road: base.not_road };
    paint();
  }
  function paint() {
    ui.querySelectorAll('#dmg button').forEach(b => b.classList.toggle('active', (b.dataset.v === '1') === verdict.damage));
    ui.querySelectorAll('#types input').forEach(c => { c.checked = verdict.types.includes(c.value); c.disabled = !verdict.damage; });
    ui.querySelector('#unc').checked = verdict.uncertain;
    ui.querySelector('#nr').checked = verdict.not_road;
  }
  async function save(v) {
    const it = list[idx]; if (!it) return;
    if (!who.value.trim()) { ui.querySelector('#msg').textContent = 'Enter your name as reviewer first.'; who.focus(); return; }
    const r = await api(`/api/v1/labels/tashkent/${it.id}`, { method: 'PUT', json: { ...v, reviewer: who.value.trim() } });
    ui.querySelector('#msg').textContent = r.agrees ? 'Saved: agrees with first pass.' : 'Saved: correction recorded.';
    const nextId = list[idx + 1]?.id;
    await load(filter === 'todo' ? nextId : it.id);
    if (filter !== 'todo' && idx < list.length - 1) { idx++; show(); }
  }
  ui.querySelector('#dmg').onclick = e => { if (!e.target.dataset.v) return; verdict.damage = e.target.dataset.v === '1'; if (!verdict.damage) verdict.types = []; paint(); };
  ui.querySelector('#unc').onchange = e => { verdict.uncertain = e.target.checked; };
  ui.querySelector('#nr').onchange = e => { verdict.not_road = e.target.checked; };
  ui.querySelector('#agree').onclick = () => { const f = list[idx]?.first_pass; if (f) save({ damage: f.damage, types: f.types, uncertain: f.uncertain, not_road: f.not_road }); };
  ui.querySelector('#save').onclick = () => save(verdict);
  ui.querySelector('#prev').onclick = () => { if (idx > 0) { idx--; show(); } };
  ui.querySelector('#next').onclick = () => { if (idx < list.length - 1) { idx++; show(); } };
  ui.querySelector('#filter').onclick = e => {
    if (!e.target.dataset.f) return;
    filter = e.target.dataset.f;
    ui.querySelectorAll('#filter button').forEach(b => b.classList.toggle('active', b.dataset.f === filter));
    load();
  };
  const onKey = e => {
    if (['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
    if (e.key === 'ArrowRight') ui.querySelector('#next').click();
    else if (e.key === 'ArrowLeft') ui.querySelector('#prev').click();
    else if (e.key === 'y' || e.key === 'Y') ui.querySelector('#agree').click();
  };
  window.addEventListener('keydown', onKey);
  load();
  return () => window.removeEventListener('keydown', onKey);
}
