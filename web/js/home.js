import { api, esc, figure, h } from './util.js';

export default async function home(root) {
  const [metrics, data, health] = await Promise.all([
    api('/api/v1/metrics').catch(() => ({})), api('/api/v1/data').catch(() => ({ datasets: [] })), api('/api/v1/health').catch(() => null),
  ]);
  const coop = metrics.coop_v2xsim?.configs;
  const rd = metrics.road_damage_eval;
  const tk = metrics.tashkent_eval;
  const ds = Object.fromEntries((data.datasets || []).map(d => [d.id, d]));

  root.append(h(`<div class="wrap">
    <section class="page-head">
      <span class="label">SwarmSight · Startup · Tashkent</span>
      <h1>Every car sees part of the road. Together they see all of it.</h1>
      <p class="lead">SwarmSight joins the cameras that are already on the road (taxis, fleets, intersections) into one perception
        network. It shows a driver the road users their own camera cannot see. It turns road damage into verified, deduplicated events
        that a city or a map can act on.</p>
      <div class="row" style="margin-top:1.8rem">
        <a class="btn primary" href="#/road">Test with your own photo or video <span class="arr">→</span></a>
        <a class="btn" href="#/guardian">See a hidden cyclist appear</a>
      </div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">Measured</span>
        <div><h2>What the network adds, in numbers</h2>
        <p class="muted">Every figure is produced by a script in our repository, on data the models did not see in training.</p></div></div>
      <div class="cols c4">
        ${figure(coop ? `${coop['rsu+v2v'].overall['occluded_recall_fused_%']}` : null, '%', 'of road users <b>hidden from a car\'s own camera</b> recovered by the network (own camera alone: 0%)', 'V2X-Sim 2.0 · 285 ego-frames')}
        ${figure(coop ? `${coop.rsu.overall['recall_single_camera_%']}→${coop['rsu+v2v'].overall['recall_fused_%']}` : null, '%', 'road users known in the field of view: own camera → cooperative', 'V2X-Sim 2.0 · RSU + other vehicles')}
        ${figure(rd ? rd.test.map50.toFixed(2) : null, 'mAP@.5', 'road-damage detector we trained, on the held-out test split of 6 countries', rd ? `RDD2022 · ${rd.n_test_images} test images` : 'training in progress')}
        ${figure(tk ? tk.ap_image.toFixed(2) : null, 'AP', `the same detector on <b>Tashkent streets</b>, image level${tk ? ` (random: ${(tk.n_positive / tk.n_images).toFixed(2)})` : ''}`, tk ? `${tk.n_images} images labelled by us · Mapillary` : 'local test set labelled, evaluation pending')}
      </div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">Modules</span>
        <div><h2>One principle: never trust a single camera</h2>
        <p class="muted">Guardian fuses observations within milliseconds. Road Intelligence fuses them over days. Both rely on independent sources agreeing.</p></div></div>
      <div class="modules">
        <a class="module" href="#/guardian">
          <span class="label">01</span>
          <div><h3>Guardian</h3><div class="stage" style="color:var(--network)">Research core</div></div>
          <p>Projects every camera's detections onto a shared ground plane with uncertainty, then associates and fuses them.
            Tells each camera which road users are hidden from it, and behind what.</p>
          <p class="mono" style="font-size:.8rem">${esc(ds.v2xsim?.name || '')}<br>${esc(ds.v2xsim?.license || '')}</p>
          <span class="go">→</span>
        </a>
        <a class="module" href="#/road">
          <span class="label">02</span>
          <div><h3>Road Intelligence</h3><div class="stage" style="color:var(--hazard)">First revenue · pilot</div></div>
          <p>Finds potholes and cracks in ordinary phone and dashcam footage. Merges repeat reports, weighs independent sources and
            routes events through moderation to a partner feed (GeoJSON) for Yandex Maps, the city road service and fleets.</p>
          <p class="mono" style="font-size:.8rem">${esc(ds.rdd2022?.name || '')}<br>${esc(ds.tashkent?.name || '')}</p>
          <span class="go">→</span>
        </a>
        <a class="module" href="#/charge">
          <span class="label">03</span>
          <div><h3>ChargePulse</h3><div class="stage" style="color:var(--ink-3)">Add-on</div></div>
          <p>Visual occupancy of EV charging bays, with timestamps and utilisation. It reports what the camera sees and never claims
            "charging"; that requires operator telemetry (OCPP).</p>
          <p class="mono" style="font-size:.8rem">same road-user detector<br>no extra training</p>
          <span class="go">→</span>
        </a>
      </div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">Pipeline</span>
        <div><h2>From a frame to an action</h2></div></div>
      <div class="pipeline">
        <div><span class="label">01</span><b>Capture</b><p>Phone, dashcam or road-side camera; timestamp, GPS when available.</p></div>
        <div><span class="label">02</span><b>Detect</b><p>YOLO11n road users and our road-damage model, ONNX Runtime on CPU.</p></div>
        <div><span class="label">03</span><b>Track</b><p>Kalman + two-stage IoU matching. Riders merged with their bike.</p></div>
        <div><span class="label">04</span><b>Locate</b><p>Image→ground homography. Pose from 4 points. Uncertainty propagated.</p></div>
        <div><span class="label">05</span><b>Fuse</b><p>Gated Hungarian on Mahalanobis distance; Kalman filter in metres.</p></div>
        <div><span class="label">06</span><b>Reason</b><p>Visible, occluded (by what), missed or out of view, per camera.</p></div>
        <div><span class="label">07</span><b>Act</b><p>Alerts with TTL, event candidates, moderation, partner API.</p></div>
      </div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">Built for</span><div><h2>Who acts on the output</h2></div></div>
      <div class="cols c3">
        <div><h3>Map platforms</h3><p class="muted" style="margin-top:.4rem">Fresh, verified road events with position accuracy and
          provenance. They add what drivers cannot report from behind the wheel. Partner: Yandex.</p></div>
        <div><h3>City road service</h3><p class="muted" style="margin-top:.4rem">A ranked defect list per street, re-observed as
          vehicles pass, instead of complaints and periodic manual surveys.</p></div>
        <div><h3>Fleets and dealers</h3><p class="muted" style="margin-top:.4rem">Taxi and service fleets are the camera network. Risk
          events (time-to-contact, vulnerable road users ahead) feed fleet safety and service intake.</p></div>
      </div>
    </section>

    <section class="section">
      <div class="section-head"><span class="label">This deployment</span><div><h2>Live server</h2></div></div>
      <div id="srv"></div>
    </section>
  </div>`));

  if (health) {
    const lat = Object.entries(health.live_latency).filter(([, v]) => v);
    root.querySelector('#srv').innerHTML = `<div class="cols c2"><table class="kv"><tbody>
      <tr><th>Inference</th><td>${esc(health.inference)}</td></tr>
      <tr><th>Hardware</th><td>${health.hardware.cpu_count} vCPU · ${esc(health.hardware.machine)}</td></tr>
      ${Object.entries(health.models).map(([k, m]) => `<tr><th>${esc(k.replace('_', ' '))}</th><td>${esc(m.file)} · ${m.classes} classes · ${m.imgsz} px</td></tr>`).join('')}
    </tbody></table>
    <table class="kv"><tbody>
      ${lat.length ? lat.map(([k, v]) => `<tr><th>${esc(k.replace('_', ' '))}</th><td>p50 ${v.p50_ms} ms · p95 ${v.p95_ms} ms · n ${v.n}</td></tr>`).join('') : '<tr><th>Latency</th><td>no requests since start-up</td></tr>'}
      <tr><th>Uptime</th><td>${Math.round(health.uptime_s / 60)} min</td></tr>
    </tbody></table></div>`;
  }
}
