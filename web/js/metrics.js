import { api, esc, figure, h, pretty } from './util.js';

const pc = v => v == null ? '—' : `${(v * 100).toFixed(0)}%`;

export default async function data(root) {
  const [m, reg, hl] = await Promise.all([api('/api/v1/metrics').catch(() => ({})), api('/api/v1/data').catch(() => ({ datasets: [] })), api('/api/v1/health').catch(() => null)]);
  const rd = m.road_damage_eval, tk = m.tashkent_eval, coop = m.coop_v2xsim, sp = m.speed_cost;

  root.append(h(`<div class="wrap">
    <section class="page-head">
      <span class="label">Data &amp; evidence</span>
      <h1>Every number on this site, with its source</h1>
      <p class="lead">Datasets, licences and splits; how labels were made; quality measured on data the models never saw during training;
        speed and cost on real hardware; and where the system fails.</p>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">01 · Datasets</span><div><h2>Where the data comes from</h2>
        <p class="muted">${esc(reg.no_personal_data || '')}</p></div></div>
      <div class="scroll-x"><table class="tbl"><thead><tr><th>Dataset</th><th>Module</th><th>Role</th><th>Licence</th><th>Labels</th><th>Script</th></tr></thead><tbody>
        ${(reg.datasets || []).map(d => `<tr><td><b>${esc(d.name)}</b><br><span class="faint" style="font-size:.82rem">${esc(d.origin)}</span></td>
          <td class="mono" style="font-size:.8rem">${esc(d.module)}</td><td style="font-size:.86rem">${esc(d.role)}</td>
          <td class="mono" style="font-size:.78rem">${esc(d.license)}</td><td style="font-size:.86rem">${esc(d.labels)}</td>
          <td class="mono" style="font-size:.78rem">${esc(d.script)}</td></tr>`).join('')}
      </tbody></table></div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">02 · Road damage</span><div><h2>RDD2022 held-out test split, six countries</h2>
        <p class="muted">Our YOLO11n fine-tune, evaluated as the exported ONNX file that the server runs.</p></div></div>
      <div id="rdd"></div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">03 · Tashkent</span><div><h2>Our own test set on Uzbek streets</h2>
        <p class="muted">Mapillary images from 31 held-out map tiles, labelled image by image before any model prediction was seen.
        Images shown here are anonymised by us: every person and vehicle is blurred.</p></div></div>
      <div id="tk"></div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">04 · Guardian</span><div><h2>Cooperative perception on V2X-Sim</h2></div></div>
      <div id="coop"></div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">05 · Speed &amp; cost</span><div><h2>Where inference runs and what it costs</h2></div></div>
      <div id="sp"></div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">06 · Limits</span><div><h2>Known failure cases</h2></div></div>
      <div class="cols c2">
        <ul class="muted" style="margin:0;padding-left:1.1rem;line-height:1.8">
          <li><b>Crowdsourced input is noisy.</b> ${tk ? `${tk.not_road_images} of ${tk.n_labelled}` : 'About 1 in 10'} Tashkent images were not road scenes (shop shelves, facades). The feed needs a road-scene gate.</li>
          <li><b>Night.</b> ${tk ? `${tk.uncertain_images} of ${tk.n_labelled}` : 'A quarter of'} Tashkent images were too dark, blurred or wet to label reliably. These are excluded from the metric, not counted as correct.</li>
          <li><b>Thin cracks at distance</b> and wet-asphalt reflections cause most damage misses and false alarms.</li>
        </ul>
        <ul class="muted" style="margin:0;padding-left:1.1rem;line-height:1.8">
          <li><b>Moving cameras</b> need accurate pose (GNSS + IMU) before Guardian can fuse them. The live demo uses static phones.</li>
          <li><b>Truncated objects</b> at the image edge have no visible ground contact. Their uncertainty grows and association may fail.</li>
          <li><b>Class confusion:</b> a rider whose bike is cut off becomes "pedestrian"; jeeps become "truck". Fusion works on groups to absorb this.</li>
        </ul>
      </div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">07 · Privacy &amp; licences</span><div><h2>What we store and under which terms</h2></div></div>
      <div class="cols c3">
        <div><h3>No personal data</h3><p class="muted" style="margin-top:.4rem">No customer records, no non-public data. Evidence snapshots are blurred before storage.
          Partner feeds expose events, never devices or tracks.</p></div>
        <div><h3>Open licences</h3><p class="muted" style="margin-top:.4rem">RDD2022 and Mapillary imagery are CC BY-SA 4.0, and our derived labels follow the same terms.
          V2X-Sim is Apache-2.0. Geist fonts are OFL. Leaflet is BSD-2.</p></div>
        <div><h3>AGPL-3.0</h3><p class="muted" style="margin-top:.4rem">Ultralytics YOLO weights are AGPL-3.0, so this repository is AGPL-3.0. A commercial deployment
          needs an Ultralytics licence or an Apache-2.0 detector (e.g. RT-DETR); the pipeline is detector-agnostic.</p></div>
      </div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">08 · This deployment</span><div><h2>Measured live on this server</h2></div></div>
      <div id="live"></div>
    </section>
  </div>`));

  const $ = s => root.querySelector(s);
  if (rd) {
    $('#rdd').innerHTML = `<div class="cols c4">
        ${figure(rd.test.map50.toFixed(3), '', 'mAP@0.5, all four classes', `${rd.n_test_images} test images`)}
        ${figure(rd.test.per_class.pothole?.ap50?.toFixed(3), '', 'AP@0.5 for potholes', 'D40')}
        ${figure(rd.test.precision.toFixed(2), '', 'precision (mean over classes)', 'at the F1-optimal threshold')}
        ${figure(rd.test.recall.toFixed(2), '', 'recall (mean over classes)', 'at the F1-optimal threshold')}
      </div>
      <div class="cols c3" style="margin-top:2rem">
        <div class="scroll-x"><table class="tbl"><thead><tr><th>Class</th><th class="r">AP@.5</th><th class="r">AP@.5:.95</th></tr></thead><tbody>
          ${Object.entries(rd.test.per_class).map(([k, v]) => `<tr><td>${pretty(k)}</td><td class="r">${v.ap50.toFixed(3)}</td><td class="r">${v.ap.toFixed(3)}</td></tr>`).join('')}</tbody></table></div>
        ${rd.per_country ? `<div class="scroll-x"><table class="tbl"><thead><tr><th>Country</th><th class="r">Images</th><th class="r">mAP@.5</th></tr></thead><tbody>
          ${Object.entries(rd.per_country).map(([k, v]) => `<tr><td>${esc(k.replace('_', ' '))}</td><td class="r">${v.n}</td><td class="r">${v.map50.toFixed(3)}</td></tr>`).join('')}</tbody></table></div>` : ''}
        ${rd.robustness ? `<div class="scroll-x"><table class="tbl"><thead><tr><th>Synthetic condition</th><th class="r">mAP@.5</th><th class="r">Δ</th></tr></thead><tbody>
          <tr><td>clean</td><td class="r">${rd.test.map50.toFixed(3)}</td><td class="r">—</td></tr>
          ${Object.entries(rd.robustness).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="r">${v.map50.toFixed(3)}</td><td class="r">${((v.map50 / rd.test.map50 - 1) * 100).toFixed(0)}%</td></tr>`).join('')}</tbody></table></div>` : ''}
      </div>
      <p class="note" style="margin-top:1.2rem">${esc(rd.note)}</p>`;
  } else $('#rdd').innerHTML = '<p class="faint">Evaluation runs when training finishes; this section fills from benchmarks/results/road_damage_eval.json.</p>';

  if (tk) {
    const base = tk.n_positive / tk.n_images;
    const shown = tk.per_image.filter(r => !r.excluded).slice(0, 12);
    $('#tk').innerHTML = `<div class="cols c4">
        ${figure(tk.ap_image.toFixed(2), '', `image-level average precision (random ranking would score ${base.toFixed(2)})`, `${tk.n_images} images · ${tk.n_positive} with damage`)}
        ${figure(pc(tk.precision_at_thr), '', `precision at the event threshold ${tk.threshold}`, 'share of flagged images that truly show damage')}
        ${figure(pc(tk.recall_at_thr), '', 'recall at the same threshold', 'share of damaged images that get flagged')}
        ${figure(pc(tk.not_road_false_alarm_rate), '', `false alarms on ${tk.not_road_images} non-road images`, 'shop shelves, facades, interiors')}
      </div>
      <div class="strip" style="grid-template-columns:repeat(auto-fill,minmax(180px,1fr));margin-top:2rem">
        ${shown.map(r => `<figure style="margin:0"><div class="instrument"><img src="/datasets/tashkent/images/${esc(r.id)}.jpg" alt="Tashkent street" loading="lazy"></div>
          <figcaption class="mono" style="font-size:.72rem;margin-top:.35rem;color:var(--ink-2)">label: ${r.damage ? esc(r.types.map(pretty).join(', ')) : 'no damage'}<br>
          model: ${r.score >= tk.threshold ? `<span style="color:var(--hazard)">flagged ${r.score.toFixed(2)}</span>` : `not flagged ${r.score.toFixed(2)}`}</figcaption></figure>`).join('')}
      </div>
      <p class="note" style="margin-top:1.2rem">Labels: ${esc(tk.labels)}. Excluded from metrics: ${tk.not_road_images} non-road and ${tk.uncertain_images} uncertain images (night, motion blur, rain on glass).
        Model: ${esc(tk.model)}.</p>`;
  } else $('#tk').innerHTML = '<p class="faint">Fills from benchmarks/results/tashkent_eval.json once the final model is evaluated.</p>';

  if (coop) {
    const o = coop.configs['rsu+v2v'].overall, r = coop.configs.rsu.overall;
    $('#coop').innerHTML = `<div class="cols c4">
        ${figure(o['occluded_recall_fused_%'], '%', 'of road users hidden from the ego camera recovered (own camera: 0%)', `${o.gt_occluded_for_ego} occluded instances`)}
        ${figure(o['occluded_vru_recall_fused_%'], '%', 'of hidden bicycles and motorcycles recovered', `${o.gt_occluded_vru} instances`)}
        ${figure(`${r['recall_single_camera_%']}→${o['recall_fused_%']}`, '%', 'recall in the field of view, own camera → cooperative', `${o.gt_objects_in_fov} object-frames`)}
        ${figure(o.loc_err_fused_m_median, 'm', 'median position error of fused objects', `p90 ${o.loc_err_fused_m_p90} m`)}
      </div>
      <p class="note" style="margin-top:1.2rem">${esc(coop.dataset)}. Ego vehicles 1, 2 and 4; sources: road-side unit (4 cameras) and the other vehicles' front cameras.
        Alert precision ${o['alert_precision_%']}%. Script: benchmarks/coop_v2xsim.py.</p>`;
  }

  if (sp) {
    $('#sp').innerHTML = `<div class="scroll-x"><table class="tbl"><thead><tr><th>Stage · ${esc(sp.hardware)}</th><th class="r">p50 ms</th><th class="r">p95 ms</th></tr></thead><tbody>
        ${Object.entries(sp.stages).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="r">${v.p50}</td><td class="r">${v.p95}</td></tr>`).join('')}</tbody></table></div>
      <div class="scroll-x" style="margin-top:1.5rem"><table class="tbl"><thead><tr><th>Deployment</th><th class="r">$ / hour</th><th class="r">frames / s</th><th class="r">$ / 1000 frames</th><th class="r">$ / camera-month</th></tr></thead><tbody>
        ${sp.cost.map(c => `<tr><td>${esc(c.name)}<br><span class="faint" style="font-size:.8rem">${esc(c.basis)}</span></td><td class="r">${c.usd_per_hour}</td><td class="r">${c.fps}</td><td class="r">${c.usd_per_1000_frames}</td><td class="r">${c.usd_per_camera_month}</td></tr>`).join('')}</tbody></table></div>
      <p class="note" style="margin-top:1.2rem">${esc(sp.note)}</p>`;
  } else $('#sp').innerHTML = '<p class="faint">Fills from benchmarks/results/speed_cost.json.</p>';

  if (hl) {
    $('#live').innerHTML = `<div class="scroll-x"><table class="tbl"><thead><tr><th>Path</th><th class="r">Requests</th><th class="r">p50 ms</th><th class="r">p95 ms</th></tr></thead><tbody>
      ${Object.entries(hl.live_latency).map(([k, v]) => `<tr><td class="mono" style="font-size:.84rem">${esc(k)}</td><td class="r">${v?.n ?? 0}</td><td class="r">${v?.p50_ms ?? '—'}</td><td class="r">${v?.p95_ms ?? '—'}</td></tr>`).join('')}
    </tbody></table></div><p class="note" style="margin-top:1rem">${esc(hl.inference)} · ${hl.hardware.cpu_count} vCPU · counters reset on restart.</p>`;
  }
}
