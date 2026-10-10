import { api, esc, h, pretty } from './util.js';

const pct = v => v == null ? '—' : `${v}%`;

export default async function metrics(root) {
  root.append(h(`<section class="hero" style="padding-top:.5rem">
    <span class="pill ok">Measured, not claimed</span>
    <h1 style="margin-top:.5rem">Quality, speed and cost</h1>
    <p>All numbers come from scripts in our repository (<span class="mono">benchmarks/</span>, <span class="mono">training/</span>)
      run on test data the models never saw during training. Live latencies on this server are at the bottom.</p>
  </section>`));
  const box = h('<div class="stack"></div>');
  root.append(box);
  let m;
  try { m = await api('/api/v1/metrics'); } catch (e) { box.append(h(`<p class="error">${esc(e.message)}</p>`)); return; }

  const coop = m.coop_v2xsim;
  if (coop) {
    const c = coop.configs;
    const rows = [
      ['Recall of road users in the ego camera\'s field of view', c.rsu.overall['recall_single_camera_%'], c.rsu.overall['recall_fused_%'], c['rsu+v2v'].overall['recall_fused_%']],
      ['Recall of objects <b>occluded</b> for the ego camera', 0, c.rsu.overall['occluded_recall_fused_%'], c['rsu+v2v'].overall['occluded_recall_fused_%']],
      ['… of which vulnerable road users (bicycles, motorcycles)', 0, c.rsu.overall['occluded_vru_recall_fused_%'], c['rsu+v2v'].overall['occluded_vru_recall_fused_%']],
      ['Alert precision (alert matches a real object)', '—', c.rsu.overall['alert_precision_%'], c['rsu+v2v'].overall['alert_precision_%']],
    ];
    box.append(h(`<div class="card stack">
      <h2>Guardian: cooperative perception <span class="pill cyan">${esc(coop.dataset)}</span></h2>
      <div class="table-wrap"><table class="t"><thead><tr><th>Metric (ego vehicles 1, 2, 4 · ${coop.configs.rsu.overall.frames} frames · range ${coop.range_m} m)</th>
        <th class="num">Own camera only</th><th class="num">+ road-side unit</th><th class="num">+ RSU + other vehicles</th></tr></thead><tbody>
        ${rows.map(r => `<tr><td>${r[0]}</td><td class="num">${pct(r[1])}</td><td class="num">${pct(r[2])}</td><td class="num">${pct(r[3])}</td></tr>`).join('')}
        <tr><td>Median localisation error of fused objects</td><td class="num">${c.rsu.overall.loc_err_single_m_median} m</td><td class="num">${c.rsu.overall.loc_err_fused_m_median} m</td><td class="num">${c['rsu+v2v'].overall.loc_err_fused_m_median} m</td></tr>
        <tr><td>Alert message size</td><td class="num">—</td><td class="num">${c.rsu.overall.alert_msg_bytes_mean} B</td><td class="num">${c['rsu+v2v'].overall.alert_msg_bytes_mean} B</td></tr>
      </tbody></table></div>
      <p class="note">Ground-truth occlusion per camera is computed from the dataset's semantic segmentation and depth maps: an object is
        "occluded" when fewer than 400 depth-consistent object pixels are visible. Detector: ${esc(coop.detector)} at
        ${coop.detector_ms_per_image_p50 ?? '—'} ms/image (p50) on the benchmark machine. The scene is synthetic (CARLA). Real-world transfer
        is measured in the pilot (see README, failure cases).</p>
    </div>`));
  }

  const rd = m.road_damage_eval;
  if (rd) {
    box.append(h(`<div class="card stack">
      <h2>Road damage detector <span class="pill amber">RDD2022 held-out test split</span></h2>
      <div class="grid g4">
        <div class="kpi"><span class="v">${rd.test.map50.toFixed(3)}</span><span class="l">mAP@0.5 (all classes)</span></div>
        <div class="kpi"><span class="v">${rd.test.per_class.pothole?.ap50?.toFixed(3) ?? '—'}</span><span class="l">AP@0.5 pothole</span></div>
        <div class="kpi"><span class="v">${rd.test.precision.toFixed(2)} / ${rd.test.recall.toFixed(2)}</span><span class="l">precision / recall</span></div>
        <div class="kpi"><span class="v">${rd.n_test_images}</span><span class="l">test images · ${rd.countries.length} countries</span></div>
      </div>
      <div class="table-wrap"><table class="t"><thead><tr><th>Class</th><th class="num">AP@0.5</th><th class="num">AP@0.5:0.95</th></tr></thead><tbody>
        ${Object.entries(rd.test.per_class).map(([k, v]) => `<tr><td>${pretty(k)}</td><td class="num">${v.ap50.toFixed(3)}</td><td class="num">${v.ap.toFixed(3)}</td></tr>`).join('')}
      </tbody></table></div>
      ${rd.robustness ? `<h3>Robustness (synthetic perturbations of the same test images)</h3>
      <div class="table-wrap"><table class="t"><thead><tr><th>Condition</th><th class="num">mAP@0.5</th><th class="num">vs clean</th></tr></thead><tbody>
        ${Object.entries(rd.robustness).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${v.map50.toFixed(3)}</td><td class="num">${((v.map50 / rd.test.map50 - 1) * 100).toFixed(0)}%</td></tr>`).join('')}
      </tbody></table></div>` : ''}
      ${rd.per_country ? `<h3>By country of capture</h3><div class="table-wrap"><table class="t"><thead><tr><th>Country</th><th class="num">images</th><th class="num">mAP@0.5</th></tr></thead><tbody>
        ${Object.entries(rd.per_country).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${v.n}</td><td class="num">${v.map50.toFixed(3)}</td></tr>`).join('')}</tbody></table></div>` : ''}
      <p class="note">${esc(rd.note || '')}</p>
    </div>`));
  }

  const sp = m.speed_cost;
  if (sp) {
    box.append(h(`<div class="card stack">
      <h2>Speed and cost per inference</h2>
      <div class="table-wrap"><table class="t"><thead><tr><th>Stage</th><th class="num">p50 ms</th><th class="num">p95 ms</th></tr></thead><tbody>
        ${Object.entries(sp.stages).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${v.p50}</td><td class="num">${v.p95}</td></tr>`).join('')}
      </tbody></table></div>
      <div class="table-wrap"><table class="t"><thead><tr><th>Deployment</th><th class="num">$/hour</th><th class="num">frames/s</th><th class="num">$ per 1000 frames</th><th class="num">$ per camera-month (1 fps, 10 h/day)</th></tr></thead><tbody>
        ${sp.cost.map(c => `<tr><td>${esc(c.name)}</td><td class="num">${c.usd_per_hour}</td><td class="num">${c.fps}</td><td class="num">${c.usd_per_1000_frames}</td><td class="num">${c.usd_per_camera_month}</td></tr>`).join('')}
      </tbody></table></div>
      <p class="note">${esc(sp.note)}</p>
    </div>`));
  }

  const hl = await api('/api/v1/health');
  box.append(h(`<div class="card stack"><h2>This deployment, live</h2>
    <div class="table-wrap"><table class="t"><thead><tr><th>Path</th><th class="num">n</th><th class="num">p50 ms</th><th class="num">p95 ms</th></tr></thead><tbody>
    ${Object.entries(hl.live_latency).map(([k, v]) => `<tr><td>${k}</td><td class="num">${v?.n ?? 0}</td><td class="num">${v?.p50_ms ?? '—'}</td><td class="num">${v?.p95_ms ?? '—'}</td></tr>`).join('')}
    </tbody></table></div><p class="note">${esc(hl.inference)} · ${hl.hardware.cpu_count} vCPU. Reset on restart.</p></div>`));

  box.append(h(`<div class="card stack"><h2>Known failure cases</h2><ul>
    <li><b>Moving cameras without pose:</b> phone/dashcam fusion needs a static camera or an accurate pose (GNSS+IMU). The V2X-Sim demo uses simulator poses.</li>
    <li><b>Truncated or very distant objects:</b> foot point is unknown or the projection error explodes, so the uncertainty grows and association may fail.</li>
    <li><b>Detector class confusion:</b> a rider whose bike is cut off by the image edge becomes "person"; jeeps become "truck". Groups (person / two-wheeler / vehicle) absorb most of it.</li>
    <li><b>Road damage:</b> thin cracks at distance and wet asphalt reflections. Night and heavy rain were not captured locally yet; the synthetic perturbation results above are a lower bound, not a substitute.</li>
    <li><b>Uzbek roads and plates:</b> not evaluated yet. RDD2022 has no Central Asian imagery. The first pilot month collects a local held-out set.</li>
  </ul></div>`));
}
