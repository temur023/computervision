import { api, C, drawBox, dropZone, esc, fitCanvas, h, streamJob } from './util.js';

const VEH = ['car', 'truck', 'bus', 'motorcycle'];

export default async function charge(root) {
  root.append(h(`<div class="wrap">
    <section class="page-head">
      <span class="label">Module 03 · ChargePulse · add-on</span>
      <h1>Which charging bays are free, and since when</h1>
      <p class="lead">One camera over a charging site and the bays drawn once. ChargePulse reports visual occupancy per bay with timestamps
        and utilisation. It never reports "charging": that state belongs to the operator's telemetry.</p>
    </section>
    <div class="spec">
      <div><span class="label">Problem · user</span><p>EV drivers arrive at bays that are blocked, and operators cannot see bay use between sessions.
        <b>Users:</b> charging operators, dealer service centres, parking owners.</p></div>
      <div><span class="label">Data</span><p>No extra training data: it reuses the <b>road-user detector</b> (COCO classes car, truck, bus, motorcycle).</p></div>
      <div><span class="label">Model</span><p>Bay polygons × the <b>ground-contact band</b> of each vehicle box (lower 40%). A state changes only after
        <b>3 agreeing samples</b>.</p></div>
      <div><span class="label">Measured</span><p><b>Not evaluated yet.</b> The pilot's first step is two weeks of footage at one site with hourly manual counts.</p></div>
    </div>

    <section class="section" style="border-top:0">
      <div class="work">
        <div class="stack">
          <label class="drop" id="drop"><input type="file" accept="image/*,video/*"><b>Drop a photo or video of the bays</b><span class="faint">camera looking down at a parking or charging area</span></label>
          <div class="instrument hidden" id="stage" style="cursor:crosshair"><canvas class="base" id="cv"></canvas><span class="tag">draw bays: tap corners</span></div>
          <div class="row"><button class="btn" id="close" disabled>Close bay</button><button class="btn quiet" id="undo" disabled>Undo last bay</button>
            <button class="btn primary" id="run" disabled>Analyse on server <span class="arr">→</span></button></div>
          <div class="progress"><div id="pbar"></div></div>
        </div>
        <aside>
          <div class="panel"><span class="label">Bays</span><div id="bays"><p class="faint">Tap three or more corners of a bay, then "Close bay". Repeat for each bay.</p></div></div>
          <div class="panel"><span class="label">State changes</span><div id="events" class="mono faint" style="font-size:.8rem">—</div></div>
        </aside>
      </div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">Semantics</span><div><h2>What the API promises, and what it does not</h2></div></div>
      <div class="cols c3">
        <div><h3>Visual occupancy</h3><p class="muted" style="margin-top:.4rem">A vehicle's contact band covers at least 25% of the bay for three consecutive samples.
          Reported with the time the state began.</p></div>
        <div><h3>Not charging status</h3><p class="muted" style="margin-top:.4rem">"Occupied" can mean a combustion car blocking the bay, or a finished session.
          Charging state comes from the operator (OCPP) and is joined in the operator's system.</p></div>
        <div><h3>API</h3><p class="muted" style="margin-top:.4rem"><span class="mono">GET /api/v1/charger-observations?site=…</span> returns state changes with timestamps.
          No images or plates are stored.</p></div>
      </div>
    </section>
  </div>`));
  const cv = root.querySelector('#cv');
  let file = null, frame = null, W = 0, H = 0, bays = [], cur = [], states = {}, vehicles = [], job = null;

  function redraw() {
    if (!frame) return;
    const ctx = fitCanvas(cv, W, H);
    ctx.drawImage(frame, 0, 0, W, H);
    const lw = Math.max(2, W / 500);
    for (const v of vehicles) drawBox(ctx, v.bbox, v.cls, C.own, { lw: lw / (window.devicePixelRatio || 1) });
    bays.forEach((b, i) => {
      const st = states[`bay${i + 1}`]?.state;
      ctx.beginPath(); b.forEach((p, j) => j ? ctx.lineTo(...p) : ctx.moveTo(...p)); ctx.closePath();
      ctx.fillStyle = st === 'occupied' ? 'rgba(240,160,48,.32)' : st === 'free' ? 'rgba(63,191,134,.28)' : 'rgba(233,231,226,.14)';
      ctx.fill(); ctx.strokeStyle = st === 'occupied' ? C.hazard : st === 'free' ? C.verified : '#e9e7e2'; ctx.lineWidth = lw; ctx.stroke();
      const c = b.reduce((a, p) => [a[0] + p[0] / b.length, a[1] + p[1] / b.length], [0, 0]);
      const fs = Math.max(13, W / 70);
      ctx.font = `500 ${fs}px "Geist Mono", monospace`; ctx.textAlign = 'center';
      ctx.fillStyle = 'rgba(13,15,18,.8)'; const t = `BAY ${i + 1}${st ? ' · ' + st.toUpperCase() : ''}`; const tw = ctx.measureText(t).width + fs;
      ctx.fillRect(c[0] - tw / 2, c[1] - fs, tw, fs * 1.6); ctx.fillStyle = '#e9e7e2'; ctx.fillText(t, c[0], c[1] + fs * 0.25); ctx.textAlign = 'start';
    });
    if (cur.length) {
      ctx.beginPath(); cur.forEach((p, j) => j ? ctx.lineTo(...p) : ctx.moveTo(...p)); ctx.strokeStyle = '#ffffff'; ctx.lineWidth = lw; ctx.stroke();
      cur.forEach(p => { ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(5, W / 220), 0, 7); ctx.fillStyle = '#ffffff'; ctx.fill(); });
    }
  }
  function renderBays() {
    const box = root.querySelector('#bays');
    if (!bays.length) { box.innerHTML = '<p class="faint">Tap three or more corners of a bay, then "Close bay". Repeat for each bay.</p>'; }
    else box.innerHTML = `<div class="list">${bays.map((b, i) => {
      const s = states[`bay${i + 1}`];
      const tl = s?.timeline ? `<div class="timeline" style="margin-top:.45rem">${s.timeline.map(([a, b2, st]) => `<span style="flex:${Math.max(0.01, b2 - a)};background:${st === 'occupied' ? 'var(--hazard)' : 'var(--verified)'}"></span>`).join('')}</div>` : '';
      return `<div class="item" style="display:block"><div class="row between"><span class="t">Bay ${i + 1}</span>
        <span class="status ${s?.state === 'occupied' ? 's-candidate' : s?.state === 'free' ? 's-verified' : 's-expired'}">${s?.state || 'not analysed'}</span></div>
        <div class="m">${s?.since != null ? `since t ${(+s.since).toFixed(1)} s` : `${b.length} corners`}${s?.utilisation != null ? ` · utilisation ${(s.utilisation * 100).toFixed(0)}%` : ''}</div>${tl}</div>`;
    }).join('')}</div>`;
    root.querySelector('#undo').disabled = !bays.length;
    root.querySelector('#run').disabled = !(bays.length && file);
  }
  cv.addEventListener('click', e => {
    if (!frame) return;
    const r = cv.getBoundingClientRect();
    cur.push([(e.clientX - r.left) * W / r.width, (e.clientY - r.top) * H / r.height].map(Math.round));
    root.querySelector('#close').disabled = cur.length < 3;
    redraw();
  });
  root.querySelector('#close').onclick = () => { bays.push(cur); cur = []; root.querySelector('#close').disabled = true; redraw(); renderBays(); };
  root.querySelector('#undo').onclick = () => { bays.pop(); redraw(); renderBays(); };

  dropZone(root.querySelector('#drop'), 'image/*,video/*', async fs => {
    file = fs[0]; bays = []; cur = []; states = {}; vehicles = [];
    root.querySelector('#stage').classList.remove('hidden');
    if (file.type.startsWith('video')) {
      const v = document.createElement('video'); v.muted = true; v.playsInline = true; v.src = URL.createObjectURL(file);
      await new Promise(r => { v.onloadeddata = r; }); v.currentTime = 0.1; await new Promise(r => { v.onseeked = r; });
      W = v.videoWidth; H = v.videoHeight; frame = v;
    } else {
      const im = new Image(); im.src = URL.createObjectURL(file); await im.decode();
      W = im.naturalWidth; H = im.naturalHeight; frame = im;
    }
    redraw(); renderBays();
  });

  root.querySelector('#run').onclick = async () => {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('bays', JSON.stringify(Object.fromEntries(bays.map((b, i) => [`bay${i + 1}`, b]))));
    root.querySelector('#run').disabled = true;
    try {
      const r = await api('/api/v1/chargepulse/analyze', { method: 'POST', body: fd });
      if (r.kind === 'image') {
        states = r.bays; vehicles = r.vehicles.filter(v => VEH.includes(v.cls));
        redraw(); renderBays(); root.querySelector('#run').disabled = false;
        root.querySelector('#events').textContent = r.note;
        return;
      }
      job = streamJob(r.job_id, {
        frame: m => { states = m.bays; vehicles = m.vehicles.filter(v => VEH.includes(v.cls)); redraw(); renderBays(); },
        progress: p => { root.querySelector('#pbar').style.width = `${Math.round(p * 100)}%`; },
        done: res => {
          states = res.summary; renderBays(); redraw(); root.querySelector('#run').disabled = false;
          root.querySelector('#events').innerHTML = res.events.map(e => `t ${e.t.toFixed(1)} s · ${esc(e.bay)} → ${esc(e.state)}${e.vehicle_cls ? ` (${esc(e.vehicle_cls)})` : ''}`).join('<br>') + `<br><br>${esc(res.note)}`;
        },
        error: e => { root.querySelector('#events').innerHTML = `<span class="error">${esc(e)}</span>`; root.querySelector('#run').disabled = false; },
      });
    } catch (e) { root.querySelector('#events').innerHTML = `<span class="error">${esc(e.message)}</span>`; root.querySelector('#run').disabled = false; }
  };
  return () => job && job.cancel();
}
