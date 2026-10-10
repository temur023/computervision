import { api, esc, fmtTime, h, pretty } from './util.js';
import { baseLayer, mountScan, TASHKENT } from './scan.js';

const STATUS_COL = { candidate: '#d4800f', corroborated: '#2350d1', verified: '#1d7f55', rejected: '#c23a2e', expired: '#8b9099' };

export default async function road(root, args) {
  const [metrics, data] = await Promise.all([api('/api/v1/metrics').catch(() => ({})), api('/api/v1/data').catch(() => ({ datasets: [] }))]);
  const ds = Object.fromEntries((data.datasets || []).map(d => [d.id, d]));
  const rd = metrics.road_damage_eval, tk = metrics.tashkent_eval;
  const rdd = ds.rdd2022, tash = ds.tashkent;

  root.append(h(`<div class="wrap">
    <section class="page-head">
      <span class="label">Module 02 · Road Intelligence · first revenue</span>
      <h1>Road damage, verified before anyone acts on it</h1>
      <p class="lead">Ordinary phone and dashcam footage goes in. A ranked, deduplicated list of road defects comes out, with position,
        evidence and confidence, for the city road service and for map platforms.</p>
    </section>
    <div class="spec">
      <div><span class="label">Problem · user</span><p>Defects reach road services through complaints, weeks late and duplicated.
        <b>Users:</b> city road service, Yandex Maps, taxi fleets.</p></div>
      <div><span class="label">Data</span><p><b>${rdd ? (rdd.used.train + rdd.used.val + rdd.used.test).toLocaleString('en') : '—'}</b> RDD2022 images from 6 countries (${esc(rdd?.license || '')}).
        <b>Tashkent:</b> ${tash?.used ? `${tash.used.total ?? ''} Mapillary images labelled by us` : 'Mapillary street imagery, labelling in progress'}.</p></div>
      <div><span class="label">Model</span><p>YOLO11n <b>fine-tuned by us</b> on 4 damage classes, 512 px, ONNX Runtime on CPU.
        ByteTrack across frames, then the trust engine.</p></div>
      <div><span class="label">Measured</span><p>${rd ? `<b>mAP@.5 ${rd.test.map50.toFixed(3)}</b> on ${rd.n_test_images} held-out images · pothole AP ${rd.test.per_class.pothole.ap50.toFixed(3)}` : 'Held-out evaluation runs when training finishes.'}
        ${tk ? `<br><b>Tashkent mAP@.5 ${tk.map50.toFixed(3)}</b> on ${tk.n_images} images` : '<br>Tashkent test set: in labelling.'}</p></div>
    </div>

    <section class="section" style="border-top:0">
      <div class="row between" style="margin-bottom:1.2rem">
        <div class="seg" id="tabs"><button data-t="scan">Scan</button><button data-t="events">Events &amp; moderation</button></div>
        <span class="faint" style="font-size:.86rem">Test it with your own photo or video. Nothing is pre-computed.</span>
      </div>
      <div id="pane"></div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">Data</span><div><h2>Where the training and test images come from</h2></div></div>
      <div class="scroll-x"><table class="tbl"><thead><tr><th>Dataset</th><th>Role</th><th>Licence</th><th class="r">Train</th><th class="r">Val</th><th class="r">Test</th></tr></thead><tbody>
        ${[rdd, tash].filter(Boolean).map(d => `<tr><td><b>${esc(d.name)}</b><br><span class="faint" style="font-size:.84rem">${esc(d.content)}</span></td>
          <td style="font-size:.88rem">${esc(d.role)}</td><td class="mono" style="font-size:.8rem">${esc(d.license)}</td>
          <td class="r">${d.used?.train ?? '—'}</td><td class="r">${d.used?.val ?? '—'}</td><td class="r">${d.used?.test ?? '—'}</td></tr>`).join('')}
      </tbody></table></div>
      <p class="note" style="margin-top:1rem">Sampling: ${esc(rdd?.sampling || '')}. Tashkent: ${esc(tash?.sampling || '')}.</p>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">Model &amp; pipeline</span><div><h2>From pixels to a road event</h2></div></div>
      <div class="pipeline p5">
        <div><span class="label">01</span><b>Detect</b><p>Damage model (4 classes) and the road-user model run on every sampled frame.</p></div>
        <div><span class="label">02</span><b>Track</b><p>A pothole seen in consecutive frames is one track, not five reports.</p></div>
        <div><span class="label">03</span><b>Score</b><p>Severity = damage type × apparent size × proximity × confidence.</p></div>
        <div><span class="label">04</span><b>Merge</b><p>Same type within 12–15 m (pothole, crack) joins the existing event.</p></div>
        <div><span class="label">05</span><b>Trust</b><p>Noisy-OR over independent sources, each weighted by its moderation record.</p></div>
      </div>
      <div class="cols c3" style="margin-top:2rem">
        <div><h3>Lifecycle</h3><p class="muted" style="margin-top:.4rem">candidate → corroborated (≥ 2 independent sources and confidence ≥ 0.6) → verified or rejected by a moderator.
          An event expires if no one re-observes it within its time-to-live: 30 days for potholes, 90 for cracks.</p></div>
        <div><h3>Source reliability</h3><p class="muted" style="margin-top:.4rem">Each device has a Beta(α, β) record. A verified report raises α and a rejected one raises β.
          A phone that keeps reporting shadows loses weight automatically.</p></div>
        <div><h3>Privacy</h3><p class="muted" style="margin-top:.4rem">Evidence snapshots are blurred (heads, plate band) before storage. The partner feed never exposes
          device identities or tracks, only aggregated events.</p></div>
      </div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">Quality &amp; limits</span><div><h2>Where it is weak</h2></div></div>
      <div id="quality"></div>
    </section>
  </div>`));

  const q = root.querySelector('#quality');
  q.innerHTML = `${rd ? `<div class="scroll-x"><table class="tbl" style="margin-bottom:1.5rem"><thead><tr><th>Class</th><th class="r">AP@.5</th><th class="r">AP@.5:.95</th></tr></thead><tbody>
      ${Object.entries(rd.test.per_class).map(([k, v]) => `<tr><td>${pretty(k)}</td><td class="r">${v.ap50.toFixed(3)}</td><td class="r">${v.ap.toFixed(3)}</td></tr>`).join('')}</tbody></table></div>` : ''}
    <div class="cols c2">
      <ul class="muted" style="margin:0;padding-left:1.1rem">
        <li>Thin cracks far from the camera and wet asphalt reflections are the main error sources.</li>
        <li>RDD2022 has no Central Asian roads. The Tashkent test set measures that gap instead of assuming it away.</li>
      </ul>
      <ul class="muted" style="margin:0;padding-left:1.1rem">
        <li>Night and heavy rain: only synthetic perturbations so far (see Data &amp; evidence). That is a lower bound, not a substitute for real footage.</li>
        <li>An uploaded clip carries one geotag, so its damage is reported per type at that location, not per pothole.</li>
      </ul>
    </div>`;

  const pane = root.querySelector('#pane');
  let cleanup = null;
  const open = (t, rest = []) => {
    if (cleanup) { cleanup(); cleanup = null; }
    root.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('active', b.dataset.t === t));
    pane.innerHTML = '';
    cleanup = t === 'events' ? mountEvents(pane, rest[0] ? +rest[0] : null) : mountScan(pane);
  };
  root.querySelector('#tabs').onclick = e => e.target.dataset.t && open(e.target.dataset.t);
  open(args[0] === 'events' ? 'events' : 'scan', args.slice(1));
  return () => cleanup && cleanup();
}

export function mountEvents(container, selected) {
  const ui = h(`<div class="work">
    <div class="stack">
      <div class="row between">
        <div class="row" id="filters">${Object.keys(STATUS_COL).map(s => `<label class="check"><input type="checkbox" value="${s}" ${['rejected', 'expired'].includes(s) ? '' : 'checked'}>
          <span class="status s-${s}">${s}</span></label>`).join('')}</div>
        <button class="btn quiet" id="refresh">Refresh</button>
      </div>
      <div id="map" class="map"></div>
      <p class="note">Every event on this map was created by a scan on this deployment. Nothing is pre-loaded.</p>
      <div class="panel"><span class="label">Partner API</span>
        <pre class="mono" style="margin:0;white-space:pre-wrap;font-size:.8rem;color:var(--ink-2)">GET   /api/v1/road-events?status=verified&amp;format=geojson
GET   /api/v1/road-events?status=verified,corroborated&amp;bbox=69.1,41.2,69.4,41.4
POST  /api/v1/road-events                    device or manual observation
PATCH /api/v1/road-events/{id}/review        {"action": "verify" | "reject"}</pre>
        <div class="row" style="margin-top:.8rem"><a class="btn" href="/api/v1/road-events?status=verified&format=geojson" target="_blank" rel="noopener">Verified feed (GeoJSON)</a>
          <a class="btn quiet" href="/docs" target="_blank" rel="noopener">OpenAPI reference →</a></div></div>
    </div>
    <aside>
      <div class="panel"><span class="label">Events</span><div class="list" id="list"><p class="faint">Loading…</p></div></div>
      <div class="panel hidden" id="detail"></div>
      <div class="panel"><span class="label">Source reliability · learned from moderation</span><div id="sources"><p class="faint">—</p></div></div>
    </aside>
  </div>`);
  container.append(ui);
  const map = L.map(ui.querySelector('#map')).setView(TASHKENT, 12);
  baseLayer().addTo(map);
  const layer = L.layerGroup().addTo(map);
  let events = [];

  async function load(fit = false) {
    const st = [...ui.querySelectorAll('#filters input:checked')].map(i => i.value).join(',') || 'verified';
    events = (await api(`/api/v1/road-events?status=${st}&format=json`)).events;
    layer.clearLayers();
    const list = ui.querySelector('#list');
    list.innerHTML = events.length ? '' : '<p class="faint">No events yet. Scan a road photo with a location to create one.</p>';
    for (const e of events) {
      if (e.lat != null) {
        L.circle([e.lat, e.lon], { radius: e.radius_m, color: STATUS_COL[e.status], weight: 1, fillOpacity: .08 }).addTo(layer);
        L.circleMarker([e.lat, e.lon], { radius: 5 + 4 * e.severity, color: '#ffffff', weight: 2, fillColor: STATUS_COL[e.status], fillOpacity: 1 })
          .addTo(layer).on('click', () => showDetail(e.id)).bindTooltip(`#${e.id} ${pretty(e.type)} · ${e.status}`);
      }
      const it = h(`<div class="item"><div><div class="t">#${e.id} ${esc(pretty(e.type))}</div>
        <div class="m">${e.n_sources} source(s) · ${e.n_obs} obs · conf ${e.confidence.toFixed(2)} · ${fmtTime(e.last_seen)}</div></div>
        <span class="status s-${e.status}">${e.status}</span></div>`);
      it.onclick = () => showDetail(e.id);
      list.append(it);
    }
    if (fit && events.some(e => e.lat != null)) map.fitBounds(L.latLngBounds(events.filter(e => e.lat != null).map(e => [e.lat, e.lon])).pad(0.3), { maxZoom: 16 });
    const s = await api('/api/v1/sources');
    ui.querySelector('#sources').innerHTML = s.sources.length ? `<table class="tbl"><thead><tr><th>Source</th><th class="r">α</th><th class="r">β</th><th class="r">Weight</th></tr></thead><tbody>
      ${s.sources.slice(0, 10).map(x => `<tr><td class="mono" style="font-size:.8rem">${esc(x.source_id.slice(0, 12))}</td><td class="r">${x.alpha}</td><td class="r">${x.beta}</td><td class="r">${x.reliability.toFixed(2)}</td></tr>`).join('')}</tbody></table>`
      : '<p class="faint">No sources yet.</p>';
  }

  async function showDetail(id) {
    const e = await api(`/api/v1/road-events/${id}`);
    if (e.lat != null) map.setView([e.lat, e.lon], Math.max(map.getZoom(), 16));
    const d = ui.querySelector('#detail');
    d.classList.remove('hidden');
    d.innerHTML = `<div class="row between"><span class="label" style="margin:0">Event #${e.id}</span><span class="status s-${e.status}">${e.status}</span></div>
      <h3 style="margin:.5rem 0 .8rem">${esc(pretty(e.type))}</h3>
      ${e.evidence ? `<img class="evidence" src="/media/${esc(e.evidence)}" alt="anonymised evidence">` : ''}
      <table class="kv" style="margin-top:.6rem"><tbody>
        <tr><th>Confidence</th><td>${e.confidence.toFixed(3)} · ${e.n_sources} independent source(s)</td></tr>
        <tr><th>Severity</th><td>${e.severity.toFixed(2)}</td></tr>
        <tr><th>Position</th><td>${e.lat?.toFixed(5)}, ${e.lon?.toFixed(5)} · ±${e.radius_m} m</td></tr>
        <tr><th>Seen</th><td>${fmtTime(e.first_seen)} → ${fmtTime(e.last_seen)}</td></tr>
        <tr><th>Expires</th><td>${fmtTime(e.expires_at)} unless re-observed</td></tr></tbody></table>
      <span class="label" style="display:block;margin:1rem 0 .4rem">Provenance</span>
      <div class="scroll-x"><table class="tbl"><thead><tr><th>Time</th><th>Source</th><th class="r">Conf</th><th class="r">Weight</th></tr></thead><tbody>
        ${e.observations.map(o => `<tr><td class="mono" style="font-size:.78rem">${fmtTime(o.observed_at)}</td><td class="mono" style="font-size:.78rem">${esc(o.source_id)}</td><td class="r">${o.conf.toFixed(2)}</td><td class="r">${o.source_reliability.toFixed(2)}</td></tr>`).join('')}</tbody></table></div>
      <span class="label" style="display:block;margin:1rem 0 .4rem">History</span>
      <div class="mono faint" style="font-size:.76rem;line-height:1.7">${e.history.map(x => `${fmtTime(x.t)} · ${esc(x.action)} · ${esc(x.by)}${x.note ? ` · ${esc(x.note)}` : ''}`).join('<br>')}</div>
      ${['candidate', 'corroborated'].includes(e.status) ? `<div class="row" style="margin-top:1rem"><input type="text" id="note" placeholder="Moderator note" style="flex:1;min-width:140px">
        <button class="btn" data-a="reject">Reject</button><button class="btn primary" data-a="verify">Verify</button></div>` : ''}`;
    d.querySelectorAll('button[data-a]').forEach(b => b.onclick = async () => {
      await api(`/api/v1/road-events/${id}/review`, { method: 'PATCH', json: { action: b.dataset.a, reviewer: 'web-moderator', note: d.querySelector('#note')?.value || '' } });
      await load(); showDetail(id);
    });
  }

  ui.querySelector('#filters').onchange = () => load();
  ui.querySelector('#refresh').onclick = () => load(true);
  load(true).then(() => selected && showDetail(selected));
  return () => map.remove();
}
