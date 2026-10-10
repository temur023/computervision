import { api, esc, fmtTime, h, pretty } from './util.js';

const STATUS = { candidate: '#fbbf24', corroborated: '#38bdf8', verified: '#34d399', rejected: '#f87171', expired: '#64748b' };
const TASHKENT = [41.3111, 69.2797];

export default async function road(root, args) {
  root.append(h(`<section class="hero" style="padding-top:.5rem">
    <span class="pill amber">Road Intelligence · trust engine</span>
    <h1 style="margin-top:.5rem">From noisy sightings to verified road events</h1>
    <p>Reports of the same damage within a type-specific radius merge into one event. Confidence combines <b>independent sources</b>
      (noisy-OR), each weighted by that device's track record. Repeat reports from one phone do not count twice. Moderators verify or reject,
      and each decision updates source reliability. Partners only receive verified events.</p>
  </section>`));
  const ui = h(`<div class="split">
    <div class="stack">
      <div class="card stack">
        <div class="row between">
          <div class="row" id="filters">
            ${Object.entries(STATUS).map(([s, c]) => `<label class="row" style="gap:.3rem"><input type="checkbox" value="${s}" ${s !== 'rejected' && s !== 'expired' ? 'checked' : ''}><span class="pill" style="color:${c}">${s}</span></label>`).join('')}
          </div>
          <button class="btn ghost" id="refresh">Refresh</button>
        </div>
        <div id="map" class="map"></div>
        <p class="note">Events appear here when a scan with a location (photo EXIF, device GPS or a map click) finds road damage.
          No events are pre-loaded: everything on this map came from scans on this deployment.</p>
      </div>
      <div class="card stack">
        <h3>Partner API (Appendix A2 of our concept)</h3>
        <pre class="mono" style="white-space:pre-wrap;margin:0">GET  ${location.origin}/api/v1/road-events?status=verified&amp;format=geojson
GET  ${location.origin}/api/v1/road-events?status=verified,corroborated&amp;bbox=69.1,41.2,69.4,41.4
POST ${location.origin}/api/v1/road-events          (device or manual observation)
PATCH ${location.origin}/api/v1/road-events/{id}/review   {"action":"verify"|"reject"}</pre>
        <div class="row"><a class="btn" href="/api/v1/road-events?status=verified&format=geojson" target="_blank">Open verified GeoJSON feed</a>
          <a class="btn ghost" href="/docs" target="_blank">OpenAPI docs</a></div>
      </div>
    </div>
    <div class="stack">
      <div class="card stack"><h3>Events</h3><div class="list" id="list"><p class="muted">Loading…</p></div></div>
      <div class="card stack hidden" id="detail"></div>
      <div class="card stack"><h3>Source reliability (learned from moderation)</h3><div id="sources" class="mono muted">—</div></div>
    </div>
  </div>`);
  root.append(ui);
  const map = L.map(ui.querySelector('#map')).setView(TASHKENT, 12);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '© OpenStreetMap contributors' }).addTo(map);
  const layer = L.layerGroup().addTo(map);
  let events = [], selected = args[0] ? +args[0] : null;

  async function load(fit = false) {
    const st = [...ui.querySelectorAll('#filters input:checked')].map(i => i.value).join(',') || 'verified';
    const r = await api(`/api/v1/road-events?status=${st}&format=json`);
    events = r.events;
    layer.clearLayers();
    const list = ui.querySelector('#list');
    list.innerHTML = events.length ? '' : '<p class="muted">No events yet. Scan a road photo with a location to create one.</p>';
    for (const e of events) {
      if (e.lat != null) {
        L.circle([e.lat, e.lon], { radius: e.radius_m, color: STATUS[e.status], weight: 1, fillOpacity: .15 }).addTo(layer);
        const mk = L.circleMarker([e.lat, e.lon], { radius: 7 + 5 * e.severity, color: '#0b1220', weight: 2, fillColor: STATUS[e.status], fillOpacity: .95 }).addTo(layer);
        mk.on('click', () => showDetail(e.id));
        mk.bindTooltip(`#${e.id} ${pretty(e.type)} · ${e.status}`);
      }
      const it = h(`<div class="item ${e.id === selected ? 'sel' : ''}"><span><b>#${e.id} ${pretty(e.type)}</b><br>
        <small>${e.n_sources} source(s) · ${e.n_obs} obs · conf ${e.confidence.toFixed(2)} · last ${fmtTime(e.last_seen)}</small></span>
        <span class="pill" style="color:${STATUS[e.status]}">${e.status}</span></div>`);
      it.onclick = () => showDetail(e.id);
      list.append(it);
    }
    if (fit && events.some(e => e.lat != null)) map.fitBounds(L.latLngBounds(events.filter(e => e.lat != null).map(e => [e.lat, e.lon])).pad(0.3), { maxZoom: 16 });
    const s = await api('/api/v1/sources');
    ui.querySelector('#sources').innerHTML = s.sources.length ? `<table class="t"><thead><tr><th>source</th><th>kind</th><th class="num">α</th><th class="num">β</th><th class="num">reliability</th></tr></thead><tbody>
      ${s.sources.slice(0, 12).map(x => `<tr><td>${esc(x.source_id.slice(0, 10))}…</td><td>${x.kind}</td><td class="num">${x.alpha}</td><td class="num">${x.beta}</td><td class="num">${x.reliability}</td></tr>`).join('')}</tbody></table>` : 'no sources yet';
  }

  async function showDetail(id) {
    selected = id;
    ui.querySelectorAll('#list .item').forEach((n, i) => n.classList.toggle('sel', events[i]?.id === id));
    const e = await api(`/api/v1/road-events/${id}`);
    if (e.lat != null) map.setView([e.lat, e.lon], Math.max(map.getZoom(), 16));
    const d = ui.querySelector('#detail');
    d.classList.remove('hidden');
    d.innerHTML = `<div class="row between"><h2>#${e.id} ${pretty(e.type)}</h2><span class="pill" style="color:${STATUS[e.status]}">${e.status}</span></div>
      ${e.evidence ? `<img class="evidence" src="/media/${e.evidence}" alt="evidence (anonymised)">` : ''}
      <table class="t"><tbody>
        <tr><th>confidence</th><td class="mono">${e.confidence.toFixed(3)} (noisy-OR over ${e.n_sources} independent source(s))</td></tr>
        <tr><th>severity</th><td class="mono">${e.severity.toFixed(2)}</td></tr>
        <tr><th>position</th><td class="mono">${e.lat?.toFixed(5)}, ${e.lon?.toFixed(5)} · merge radius ${e.radius_m} m</td></tr>
        <tr><th>first / last seen</th><td class="mono">${fmtTime(e.first_seen)} / ${fmtTime(e.last_seen)}</td></tr>
        <tr><th>expires if not re-seen</th><td class="mono">${fmtTime(e.expires_at)}</td></tr>
      </tbody></table>
      <h3>Observations (provenance)</h3>
      <div class="table-wrap"><table class="t"><thead><tr><th>time</th><th>source</th><th class="num">conf</th><th class="num">reliab.</th><th>model</th></tr></thead><tbody>
      ${e.observations.map(o => `<tr><td>${fmtTime(o.observed_at)}</td><td class="mono">${esc(o.source_id)}</td><td class="num">${o.conf.toFixed(2)}</td><td class="num">${o.source_reliability}</td><td class="mono">${esc(o.model_version || '')}</td></tr>`).join('')}
      </tbody></table></div>
      <h3>History</h3>
      <div class="mono muted" style="font-size:.8rem">${e.history.map(x => `${fmtTime(x.t)} · ${x.action} · ${esc(x.by)} · ${esc(x.note || '')}`).join('<br>')}</div>
      <div class="row">${['candidate', 'corroborated'].includes(e.status) ? `<button class="btn okb" data-a="verify">Verify</button><button class="btn danger" data-a="reject">Reject</button>
        <input type="text" id="note" placeholder="moderator note" style="flex:1">` : ''}</div>`;
    d.querySelectorAll('button[data-a]').forEach(b => b.onclick = async () => {
      await api(`/api/v1/road-events/${id}/review`, { method: 'PATCH', json: { action: b.dataset.a, reviewer: 'web-moderator', note: d.querySelector('#note')?.value || '' } });
      await load(); showDetail(id);
    });
  }

  ui.querySelector('#filters').onchange = () => load();
  ui.querySelector('#refresh').onclick = () => load(true);
  await load(true);
  if (selected) showDetail(selected);
  return () => map.remove();
}
