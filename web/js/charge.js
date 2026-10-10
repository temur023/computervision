import { api, drawBox, dropZone, esc, fitCanvas, h, streamJob } from './util.js';

export default async function charge(root) {
  root.append(h(`<section class="hero" style="padding-top:.5rem">
    <span class="pill violet">ChargePulse · EV bays</span>
    <h1 style="margin-top:.5rem">Which charging bays are free, and since when</h1>
    <p>Upload a photo or video from a camera over a parking or charging area and draw the bays once. We report
      <b>visual occupancy</b> with timestamps and utilisation. We do not infer "charging": that needs operator telemetry (OCPP).</p>
  </section>`));
  const ui = h(`<div class="split">
    <div class="card stack">
      <label class="drop" id="drop"><input type="file" accept="image/*,video/*"><b>Choose a photo or video of the bays</b></label>
      <div class="stage hidden" id="stage" style="cursor:crosshair"><canvas id="cv"></canvas></div>
      <div class="row"><button class="btn" id="close" disabled>Close bay polygon</button><button class="btn ghost" id="undo" disabled>Undo bay</button>
        <button class="btn primary" id="run" disabled>Analyse on server</button></div>
      <p class="note">Tap the corners of a bay (3+ points), then "Close bay polygon". Repeat for each bay.</p>
      <div class="progress"><div id="pbar"></div></div>
    </div>
    <div class="stack">
      <div class="card stack"><h3>Bays</h3><div id="bays" class="list"><p class="muted">No bays drawn yet.</p></div></div>
      <div class="card stack"><h3>Events</h3><div id="events" class="mono muted">—</div></div>
    </div>
  </div>`);
  root.append(ui);
  const cv = ui.querySelector('#cv');
  let file = null, frame = null, W = 0, H = 0, bays = [], cur = [], states = {}, vehicles = [], job = null;

  function redraw() {
    if (!frame) return;
    const ctx = fitCanvas(cv, W, H);
    ctx.drawImage(frame, 0, 0, W, H);
    for (const v of vehicles) drawBox(ctx, v.bbox, `${v.cls} ${Math.round(v.conf * 100)}%`, '#38bdf8', { lw: Math.max(2, W / 500) });
    bays.forEach((b, i) => {
      const st = states[`bay${i + 1}`]?.state;
      ctx.beginPath(); b.forEach((p, j) => j ? ctx.lineTo(...p) : ctx.moveTo(...p)); ctx.closePath();
      ctx.fillStyle = st === 'occupied' ? 'rgba(248,113,113,.35)' : st === 'free' ? 'rgba(52,211,153,.35)' : 'rgba(167,139,250,.25)';
      ctx.fill(); ctx.strokeStyle = '#a78bfa'; ctx.lineWidth = Math.max(2, W / 500); ctx.stroke();
      const c = b.reduce((a, p) => [a[0] + p[0] / b.length, a[1] + p[1] / b.length], [0, 0]);
      ctx.fillStyle = '#fff'; ctx.font = `bold ${Math.max(14, W / 60)}px system-ui`; ctx.fillText(`bay${i + 1}${st ? ' · ' + st : ''}`, c[0] - 30, c[1]);
    });
    if (cur.length) {
      ctx.beginPath(); cur.forEach((p, j) => j ? ctx.lineTo(...p) : ctx.moveTo(...p)); ctx.strokeStyle = '#ffb020'; ctx.lineWidth = Math.max(2, W / 500); ctx.stroke();
      cur.forEach(p => { ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(5, W / 200), 0, 7); ctx.fillStyle = '#ffb020'; ctx.fill(); });
    }
  }
  function renderBays() {
    const box = ui.querySelector('#bays');
    box.innerHTML = bays.length ? '' : '<p class="muted">No bays drawn yet.</p>';
    bays.forEach((b, i) => {
      const s = states[`bay${i + 1}`];
      box.append(h(`<div class="item"><span><b>bay${i + 1}</b> · ${b.length} pts${s?.since != null ? ` · since t=${(+s.since).toFixed(1)} s` : ''}${s?.utilisation != null ? ` · utilisation ${(s.utilisation * 100).toFixed(0)}%` : ''}</span>
        <span class="pill ${s?.state === 'occupied' ? 'bad' : s?.state === 'free' ? 'ok' : ''}">${s?.state || 'not analysed'}</span></div>`));
      if (s?.timeline) {
        const t0 = s.timeline[0][0], t1 = s.timeline[s.timeline.length - 1][1] || 1;
        box.append(h(`<div class="timeline">${s.timeline.map(([a, b2, st]) => `<span style="flex:${Math.max(0.01, b2 - a)};background:${st === 'occupied' ? '#f87171' : '#34d399'}" title="${st} ${a}-${b2}s"></span>`).join('')}</div>`));
      }
    });
    ui.querySelector('#undo').disabled = !bays.length;
    ui.querySelector('#run').disabled = !(bays.length && file);
  }
  cv.addEventListener('click', e => {
    if (!frame) return;
    const r = cv.getBoundingClientRect();
    cur.push([(e.clientX - r.left) * W / r.width, (e.clientY - r.top) * H / r.height].map(Math.round));
    ui.querySelector('#close').disabled = cur.length < 3;
    redraw();
  });
  ui.querySelector('#close').onclick = () => { bays.push(cur); cur = []; ui.querySelector('#close').disabled = true; redraw(); renderBays(); };
  ui.querySelector('#undo').onclick = () => { bays.pop(); redraw(); renderBays(); };

  dropZone(ui.querySelector('#drop'), 'image/*,video/*', async fs => {
    file = fs[0]; bays = []; cur = []; states = {}; vehicles = [];
    ui.querySelector('#stage').classList.remove('hidden');
    if (file.type.startsWith('video')) {
      const v = document.createElement('video'); v.muted = true; v.playsInline = true; v.src = URL.createObjectURL(file);
      await new Promise(r => { v.onloadeddata = r; }); v.currentTime = 0.1; await new Promise(r => { v.onseeked = r; });
      W = v.videoWidth; H = v.videoHeight; frame = v;
    } else {
      const im = new Image(); im.src = URL.createObjectURL(file); await im.decode();
      // the server applies EXIF orientation; browsers do too when decoding <img>
      W = im.naturalWidth; H = im.naturalHeight; frame = im;
    }
    redraw(); renderBays();
  });

  ui.querySelector('#run').onclick = async () => {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('bays', JSON.stringify(Object.fromEntries(bays.map((b, i) => [`bay${i + 1}`, b]))));
    ui.querySelector('#run').disabled = true;
    try {
      const r = await api('/api/v1/chargepulse/analyze', { method: 'POST', body: fd });
      if (r.kind === 'image') {
        states = r.bays; vehicles = r.vehicles.filter(v => ['car', 'truck', 'bus', 'motorcycle'].includes(v.cls));
        redraw(); renderBays(); ui.querySelector('#run').disabled = false;
        ui.querySelector('#events').textContent = r.note;
        return;
      }
      job = streamJob(r.job_id, {
        frame: m => { states = m.bays; vehicles = m.vehicles.filter(v => ['car', 'truck', 'bus', 'motorcycle'].includes(v.cls)); redraw(); renderBays(); },
        progress: p => { ui.querySelector('#pbar').style.width = `${Math.round(p * 100)}%`; },
        done: res => {
          states = res.summary; renderBays(); redraw(); ui.querySelector('#run').disabled = false;
          ui.querySelector('#events').innerHTML = res.events.map(e => `t=${e.t.toFixed(1)}s · ${e.bay} → ${e.state}${e.vehicle_cls ? ` (${e.vehicle_cls})` : ''}`).join('<br>') + `<br><br>${esc(res.note)}`;
        },
        error: e => { ui.querySelector('#events').innerHTML = `<span class="error">${esc(e)}</span>`; ui.querySelector('#run').disabled = false; },
      });
    } catch (e) { ui.querySelector('#events').innerHTML = `<span class="error">${esc(e.message)}</span>`; ui.querySelector('#run').disabled = false; }
  };
  return () => job && job.cancel();
}
