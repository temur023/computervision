import { api, esc, h, pretty } from './util.js';

const pc = v => v == null ? '—' : `${(v * 100).toFixed(0)}%`;
const COUNTRY = { Czech: 'Czech Republic', United_States: 'United States', China_MotorBike: 'China (motorbike)' };

export default async function data(root) {
  const [m, reg, hl] = await Promise.all([api('/api/v1/metrics').catch(() => ({})), api('/api/v1/data').catch(() => ({ datasets: [] })), api('/api/v1/health').catch(() => null)]);
  const rd = m.road_damage_eval, tk = m.tashkent_eval, coop = m.coop_v2xsim, sp = m.speed_cost;
  const o = coop?.configs?.['rsu+v2v']?.overall;
  const scanStage = sp && Object.entries(sp.stages).find(([k]) => k.startsWith('Scan'))?.[1];
  const fuseStage = sp && Object.entries(sp.stages).find(([k]) => k.startsWith('Guardian'))?.[1];
  const pending = 'runs when training finishes';
  const cgr = m.conflict_v2xsim;
  const inc = await api('/api/v1/guardian/incident').then(r => (r.ready ? r : null)).catch(() => null);
  const inc100 = inc?.counterfactual?.scenarios?.find(x => x.range_m === 100);

  // The claims ledger: one row per number shown anywhere on the site.
  const claims = [
    ['Guardian', 'Hidden cyclists and motorcyclists found by the network', o && `${o['occluded_vru_recall_fused_%']}%`, o && `V2X-Sim 2.0 · ${o.gt_occluded_vru} hidden instances`, 'own camera: 0%', 'benchmarks/coop_v2xsim.py'],
    ['Guardian', 'All hidden road users found', o && `${o['occluded_recall_fused_%']}%`, o && `${o.gt_occluded_for_ego} hidden instances`, 'own camera: 0%', 'benchmarks/coop_v2xsim.py'],
    ['Guardian', 'Road users in view known to the car', o && `${o['recall_fused_%']}%`, o && `${o.gt_objects_in_fov.toLocaleString('en')} object-frames`, o && `own camera: ${o['recall_single_camera_%']}%`, 'benchmarks/coop_v2xsim.py'],
    ['Guardian', 'Median position error', o && `${o.loc_err_fused_m_median} m`, o && `fused objects · p90 ${o.loc_err_fused_m_p90} m`, o && `own camera: ${o.loc_err_single_m_median} m`, 'benchmarks/coop_v2xsim.py'],
    ['Guardian', 'Alerts that sit on a real road user', o && `${o['alert_precision_%']}%`, o && `${o.alerts} alerts`, '', 'benchmarks/coop_v2xsim.py'],
    ...(cgr ? [
      ['CrossGuard', 'Prediction error 2 s ahead (median)', `${cgr.prediction_error_m_median['2s'].ours} m`, `V2X-Sim · ${cgr.prediction_error_m_median['2s'].n} predictions, real perception`, `constant velocity: ${cgr.prediction_error_m_median['2s'].cv} m`, 'benchmarks/conflict_v2xsim.py'],
      ['CrossGuard', '"Safe to start?" check, prediction logic', `AUROC ${cgr.start_check_prediction_only.auroc}`, `${cgr.start_check_prediction_only.positives} dangerous starts in ${cgr.start_check_prediction_only.checks.toLocaleString('en')} checks, ground-truth tracks`, `warns on ${cgr.start_check_prediction_only.caution_or_high['recall_%']}% of them`, 'benchmarks/conflict_v2xsim.py'],
      ['CrossGuard', '"Safe to start?" check, end to end', `AUROC ${cgr.start_check.auroc}`, `same checks on our perception: ${cgr.start_check.positives_not_perceived} of ${cgr.start_check.positives} dangerous road users not localised within 3 m`, 'limited by distant cross traffic', 'benchmarks/conflict_v2xsim.py'],
    ] : []),
    ...(inc ? [
      ['Real case', 'Motorcycle visible from the dashcam car before impact', `${inc.motorcycle.visible_s} s`, 'Tashkent crossing, frame by frame (measured)', '', 'swarmsight/incident.py'],
      ['Real case', 'CrossGuard warning with a crossing camera (100 m view)', inc100?.lead_s != null ? `${inc100.lead_s} s` : null, 'reconstruction: estimated speeds, stated assumptions', inc100?.braking?.[0] ? (inc100.braking[0].avoided ? `car stops ${inc100.braking[0].clear_m} m short (1 s reaction)` : 'too late') : '', 'swarmsight/incident.py'],
    ] : []),
    ['Road', 'Road-damage detection, mAP@0.5', rd && rd.test.map50.toFixed(3), rd ? `RDD2022 · ${rd.n_test_images.toLocaleString('en')} held-out images, 6 countries` : pending, '', 'training/eval_road_damage.py'],
    ['Road', 'Tashkent streets, image-level AP', tk && tk.ap_image.toFixed(2), tk ? `${tk.n_images} images labelled by us` : pending, tk && `random ranking: ${(tk.n_positive / tk.n_images).toFixed(2)}`, 'training/eval_tashkent.py'],
    ['Speed', 'One frame, both detectors, CPU', scanStage && `${scanStage.p50} ms`, sp ? esc(sp.hardware.split(' · ')[0]) : 'benchmark pending', scanStage && `p95 ${scanStage.p95} ms`, 'benchmarks/speed_cost.py'],
    fuseStage ? ['Speed', 'Guardian tracking, fusion and reasoning', `${fuseStage.p50} ms`, '5 cameras per frame, detection excluded', `p95 ${fuseStage.p95} ms`, 'benchmarks/speed_cost.py']
      : ['Speed', 'Guardian fusion and reasoning', o && `${o.fusion_frame_ms_p50} ms`, '7 cameras per frame, detections cached', '', 'benchmarks/coop_v2xsim.py'],
    ['ChargePulse', 'Bay-state accuracy', null, 'not measured yet: needs footage of a real site', '', '—'],
  ];

  root.append(h(`<div class="wrap">
    <section class="page-head">
      <span class="label">Data &amp; evidence</span>
      <h1>Every number on this site, with its source</h1>
      <p class="lead">What each module learned from, how we labelled it, and how it scores on data it never saw.</p>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">01 · Claims</span><div><h2>What we measured</h2>
        <p class="muted">Each row is produced by the script named on it. Test data never took part in training or model selection.</p></div></div>
      <div class="scroll-x"><table class="tbl ledger"><thead><tr><th>Module</th><th>Claim</th><th class="r">Value</th><th>Measured on</th><th>Compared with</th><th>Script</th></tr></thead><tbody>
        ${claims.map(([mod, k, v, on, base, script]) => `<tr><td class="mono faint">${mod}</td><td>${k}</td>
          <td class="r val">${v ?? '<span class="faint">—</span>'}</td><td class="muted">${on || ''}</td><td class="muted">${base || ''}</td><td class="mono faint">${script}</td></tr>`).join('')}
      </tbody></table></div>
      ${coop ? `<p class="note" style="margin-top:1.2rem">${esc(coop.dataset)}. Ego vehicles 1, 2 and 4; sources: the road-side unit (4 cameras) and the other vehicles' front cameras.
        Occlusion labels are ours, computed from the simulator's segmentation and depth.</p>` : ''}
    </section>

    <section class="section">
      <div class="section-head"><span class="label">02 · Datasets</span><div><h2>Where the data comes from</h2>
        <p class="muted">${esc(reg.no_personal_data || '')}</p></div></div>
      <div class="scroll-x"><table class="tbl"><thead><tr><th>Dataset</th><th>Used for</th><th>Size used</th><th>Licence</th></tr></thead><tbody>
        ${(reg.datasets || []).map(d => `<tr><td><b>${esc(d.name)}</b><br><span class="faint" style="font-size:.8rem">${esc(d.origin)}</span></td>
          <td style="font-size:.88rem">${esc(d.role)}</td><td class="mono" style="font-size:.8rem">${esc(sizeOf(d))}</td>
          <td class="mono" style="font-size:.78rem">${esc(d.license)}</td></tr>`).join('')}
      </tbody></table></div>
      <details class="more" style="margin-top:1rem"><summary>How labels were made, and the scripts that fetch each dataset</summary>
        <div class="facts" style="margin-top:.8rem">${(reg.datasets || []).map(d => `<div><b>${esc(d.name)}</b><span>${esc(d.labels)}${d.sampling ? ` · Sampling: ${esc(d.sampling)}` : ''} · <span class="mono">${esc(d.script)}</span></span></div>`).join('')}</div>
      </details>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">03 · Road damage</span><div><h2>Per class, per country, under bad conditions</h2>
        <p class="muted">RDD2022 held-out test split, evaluated as the exported ONNX file that the server runs.</p></div></div>
      <div id="rdd"></div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">04 · Tashkent</span><div><h2>Our own test set on Uzbek streets</h2>
        <p class="muted">Mapillary images from 31 map tiles kept out of any training, labelled image by image before any model output was seen.
        People and vehicles are blurred by us.</p></div></div>
      <div id="tk"></div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">05 · Cost</span><div><h2>What a frame costs</h2></div></div>
      <div id="sp"></div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">06 · Limits</span><div><h2>Known failure cases</h2></div></div>
      <div class="cols c2">
        <ul class="limits">
          <li><b>Crowdsourced input is noisy.</b> ${tk ? `${tk.not_road_images} of ${tk.n_labelled}` : '11 of 120'} Tashkent images were not road scenes. The feed needs a road-scene gate.</li>
          <li><b>Night.</b> ${tk ? `${tk.uncertain_images} of ${tk.n_labelled}` : '32 of 120'} Tashkent images were too dark, blurred or wet to label. They are excluded from the metric, not counted as correct.</li>
          <li><b>Thin cracks at distance</b> and wet-asphalt reflections cause most damage errors.</li>
        </ul>
        <ul class="limits">
          <li><b>Moving cameras</b> need an accurate pose (GNSS + IMU) before Guardian can fuse them.</li>
          <li><b>Guardian is measured in simulation.</b> V2X-Sim is a CARLA scene; a real crossing is the first pilot step.</li>
          <li><b>Class confusion:</b> a rider whose bike is cut off becomes a pedestrian. Fusion works on groups to absorb this.</li>
        </ul>
      </div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">07 · Privacy &amp; licences</span><div><h2>What we store, and under which terms</h2></div></div>
      <div class="cols c3">
        <div><h3>No personal data</h3><p class="muted" style="margin-top:.4rem">Evidence snapshots are blurred before storage. Partner feeds expose events, never devices or tracks.</p></div>
        <div><h3>Open data</h3><p class="muted" style="margin-top:.4rem">RDD2022 and Mapillary images are CC BY-SA 4.0, and so are our labels. V2X-Sim is Apache-2.0. Geist fonts: OFL. Leaflet: BSD-2.</p></div>
        <div><h3>AGPL-3.0</h3><p class="muted" style="margin-top:.4rem">Ultralytics YOLO is AGPL-3.0, so this repository is too. A commercial deployment needs an Ultralytics licence or an Apache-2.0 detector; the pipeline does not depend on the detector.</p></div>
      </div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">08 · This server</span><div><h2>Measured live</h2></div></div>
      <div id="live"></div>
    </section>
  </div>`));

  const $ = s => root.querySelector(s);
  if (rd) {
    $('#rdd').innerHTML = `<div class="cols c3">
        <div class="scroll-x"><table class="tbl mini"><thead><tr><th>Class</th><th class="r">AP@.5</th></tr></thead><tbody>
          ${Object.entries(rd.test.per_class).map(([k, v]) => `<tr><td>${pretty(k)}</td><td class="r">${v.ap50.toFixed(3)}</td></tr>`).join('')}
          <tr class="sum"><td>All classes</td><td class="r">${rd.test.map50.toFixed(3)}</td></tr></tbody></table></div>
        ${rd.per_country ? `<div class="scroll-x"><table class="tbl mini"><thead><tr><th>Country</th><th class="r">Images</th><th class="r">mAP@.5</th></tr></thead><tbody>
          ${Object.entries(rd.per_country).map(([k, v]) => `<tr><td>${esc(COUNTRY[k] || k.replace(/_/g, ' '))}</td><td class="r">${v.n}</td><td class="r">${v.map50.toFixed(3)}</td></tr>`).join('')}</tbody></table></div>` : ''}
        ${rd.robustness ? `<div class="scroll-x"><table class="tbl mini"><thead><tr><th>Synthetic condition</th><th class="r">mAP@.5</th><th class="r">Change</th></tr></thead><tbody>
          ${Object.entries(rd.robustness).map(([k, v]) => `<tr><td>${esc(k.replace(/ \(.*\)$/, ''))}</td><td class="r">${v.map50.toFixed(3)}</td><td class="r">${((v.map50 / rd.test.map50 - 1) * 100).toFixed(0)}%</td></tr>`).join('')}</tbody></table></div>` : ''}
      </div>
      <p class="note" style="margin-top:1.2rem">Precision ${rd.test.precision.toFixed(2)} and recall ${rd.test.recall.toFixed(2)} (mean over classes). ${esc(rd.note)}</p>`;
  } else $('#rdd').innerHTML = `<p class="faint">The detector is still training. This section fills from benchmarks/results/road_damage_eval.json once the held-out evaluation has run.</p>`;

  if (tk) {
    const shown = tk.per_image.filter(r => !r.excluded).slice(0, 8);
    $('#tk').innerHTML = `<div class="cols c4">
        ${fig(tk.ap_image.toFixed(2), 'image-level average precision', `random ranking: ${(tk.n_positive / tk.n_images).toFixed(2)}`)}
        ${fig(pc(tk.precision_at_thr), `of flagged images truly show damage`, `threshold ${tk.threshold}`)}
        ${fig(pc(tk.recall_at_thr), 'of damaged images get flagged', `${tk.n_positive} damaged of ${tk.n_images}`)}
        ${fig(pc(tk.not_road_false_alarm_rate), `false alarms on ${tk.not_road_images} non-road images`, 'shops, facades, interiors')}
      </div>
      <div class="strip" style="grid-template-columns:repeat(auto-fill,minmax(210px,1fr));margin-top:2rem">
        ${shown.map(r => `<figure style="margin:0"><div class="instrument"><img src="/datasets/tashkent/images/${esc(r.id)}.jpg" alt="Tashkent street" loading="lazy"></div>
          <figcaption class="mono" style="font-size:.72rem;margin-top:.4rem;color:var(--ink-2)">our label: ${r.damage ? esc(r.types.map(pretty).join(', ')) : 'no damage'}<br>
          model: ${r.score >= tk.threshold ? `<span style="color:var(--hazard)">flagged, ${r.score.toFixed(2)}</span>` : `not flagged, ${r.score.toFixed(2)}`}</figcaption></figure>`).join('')}
      </div>
      <p class="note" style="margin-top:1.2rem">Labels: ${esc(tk.labels)}. Left out of the metric: ${tk.not_road_images} non-road and ${tk.uncertain_images} uncertain images
        (night, motion blur, rain on the glass). Model: ${esc(tk.model)}.</p>`;
  } else $('#tk').innerHTML = '<p class="faint">Fills from benchmarks/results/tashkent_eval.json once the final model is evaluated.</p>';

  if (sp) {
    $('#sp').innerHTML = `<div class="scroll-x"><table class="tbl"><thead><tr><th>Where it runs</th><th class="r">$ / hour</th><th class="r">Frames / s</th><th class="r">$ / 1,000 frames</th><th class="r">$ / camera-month</th></tr></thead><tbody>
        ${sp.cost.map(c => `<tr><td>${esc(c.name)}<br><span class="faint" style="font-size:.8rem">${esc(c.basis)}</span></td><td class="r">${c.usd_per_hour}</td><td class="r">${c.fps}</td><td class="r">${c.usd_per_1000_frames}</td><td class="r">${c.usd_per_camera_month}</td></tr>`).join('')}</tbody></table></div>
      <details class="more" style="margin-top:1rem"><summary>Time per stage on ${esc(sp.hardware)}</summary>
        <div class="scroll-x" style="margin-top:.6rem"><table class="tbl mini"><thead><tr><th>Stage</th><th class="r">p50 ms</th><th class="r">p95 ms</th></tr></thead><tbody>
        ${Object.entries(sp.stages).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="r">${v.p50}</td><td class="r">${v.p95}</td></tr>`).join('')}</tbody></table></div></details>
      <p class="note" style="margin-top:1.2rem">${esc(sp.note)}</p>`;
  } else $('#sp').innerHTML = '<p class="faint">Fills from benchmarks/results/speed_cost.json, measured after training on this hardware.</p>';

  if (hl) {
    const rows = Object.entries(hl.live_latency).filter(([, v]) => v?.n);
    $('#live').innerHTML = `<p class="muted">${esc(hl.inference)} · ${hl.hardware.cpu_count} vCPU · up ${Math.round(hl.uptime_s / 60)} min ·
        models: ${Object.values(hl.models).map(x => esc(x.file)).join(', ')}</p>
      ${rows.length ? `<div class="scroll-x" style="margin-top:1rem"><table class="tbl mini"><thead><tr><th>Request</th><th class="r">Count</th><th class="r">p50 ms</th><th class="r">p95 ms</th></tr></thead><tbody>
        ${rows.map(([k, v]) => `<tr><td>${esc(LIVE[k] || k)}</td><td class="r">${v.n}</td><td class="r">${v.p50_ms}</td><td class="r">${v.p95_ms}</td></tr>`).join('')}</tbody></table></div>`
        : '<p class="faint" style="margin-top:.6rem">No requests since start-up.</p>'}
      <p class="faint" style="font-size:.8rem;margin-top:.8rem">Counters reset when the server restarts.</p>`;
  }
}

const LIVE = { scan_image: 'Photo scan', live_scan: 'Live camera frame', live_guardian: 'Guardian, live phones', video_frame: 'Video frame', guardian_frame: 'Guardian demo frame (5 cameras)' };

function fig(v, k, src) {
  return `<div class="figure"><div class="v">${v ?? '—'}</div><div class="k">${k}</div>${src ? `<div class="src">${src}</div>` : ''}</div>`;
}

function sizeOf(d) {
  const u = d.used;
  if (!u) return d.id === 'coco' ? 'weights only' : 'map tiles';
  if (u.frames) return `${u.frames} frames`;
  if (u.train != null || u.test != null) {
    const parts = [u.train && `${u.train.toLocaleString('en')} train`, u.val && `${u.val} val`, u.test && `${u.test.toLocaleString('en')} test`].filter(Boolean);
    if (u.collected) parts.push(`${u.collected.toLocaleString('en')} collected`);
    return parts.join(' · ');
  }
  if (u.benchmark_frames) return `${u.benchmark_frames} frames · ${u.demo_frames} in the demo`;
  return '';
}
