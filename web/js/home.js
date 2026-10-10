import { api, h } from './util.js';

export default async function home(root) {
  root.append(h(`<section class="hero">
    <span class="pill amber">Startup · Computer Vision track</span>
    <h1 style="margin-top:.6rem">Every car sees only part of the road.<br>SwarmSight connects the cameras.</h1>
    <p>Independent cameras on cars, taxis and intersections become one perception network. It reveals road users hidden from
      a driver and turns road damage into verified, deduplicated events that a city or map provider can act on.</p>
    <div class="row" style="margin-top:1rem">
      <a class="btn primary" href="#/scan">Try it on your photo or video</a>
      <a class="btn" href="#/guardian">See a hidden cyclist appear</a>
      <a class="btn ghost" href="#/metrics">Measured results</a>
    </div>
  </section>`));

  root.append(h(`<section class="grid g3">
    <a class="card module" href="#/guardian">
      <span class="pill cyan tag">Guardian · R&amp;D core</span>
      <h2>Cooperative perception</h2>
      <p class="muted">Detections from several cameras are projected to a shared ground plane, associated and fused.
        Each subscriber learns what is <b>hidden from it</b>, what is hiding it, the source and the age.</p>
    </a>
    <a class="card module" href="#/scan">
      <span class="pill amber tag">Road Intelligence · first revenue</span>
      <h2>Verified road events</h2>
      <p class="muted">A road-damage model we trained on 6 countries finds potholes and cracks. The trust engine merges repeat
        reports, weighs independent sources and sends events through moderation to a partner GeoJSON API.</p>
    </a>
    <a class="card module" href="#/charge">
      <span class="pill violet tag">ChargePulse · add-on</span>
      <h2>EV bay occupancy</h2>
      <p class="muted">A camera over a charging site reports which bays are visually occupied and since when.
        It never claims "charging": that needs operator telemetry (OCPP).</p>
    </a>
  </section>`));

  root.append(h(`<section class="grid g2" style="margin-top:1rem">
    <div class="card">
      <h3>How a frame becomes an action</h3>
      <ol class="how">
        <li><span><b>Capture</b> on a phone, dashcam or roadside camera, with a timestamp and GPS if available</span></li>
        <li><span><b>Detect</b>: YOLO11n road users (COCO) and our RDD2022 road-damage model, run with ONNX Runtime on CPU</span></li>
        <li><span><b>Track</b>: ByteTrack-style Kalman tracker. Riders are merged with their bike; time to contact comes from box growth</span></li>
        <li><span><b>Geometry</b>: image to ground homography, with camera pose recovered from 4 clicked points and uncertainty propagated</span></li>
        <li><span><b>Associate &amp; fuse</b>: gated Hungarian matching on Mahalanobis distance, then a Kalman filter in metres</span></li>
        <li><span><b>Reason</b>: each object is visible, occluded (by which object), undetected or out of view, per subscriber</span></li>
        <li><span><b>Act</b>: alerts with TTL, road-event candidates, moderation, and a partner API</span></li>
      </ol>
    </div>
    <div class="card stack" id="srv">
      <h3>Live server</h3>
      <p class="muted">Loading…</p>
    </div>
  </section>`));

  root.append(h(`<section class="card" style="margin-top:1rem">
    <h3>For the jury: test with your own input</h3>
    <div class="grid g3">
      <div><b>Any road photo or video</b><br><span class="muted">Scan: upload it, or use your phone camera live.
        Road users, road damage and warnings come back from our server.</span></div>
      <div><b>Two views of one scene</b><br><span class="muted">Guardian, "Your two videos": mark 4 ground points per camera
        and the system fuses both views. "Live" lets two phones join one room.</span></div>
      <div><b>A parking / charging lot</b><br><span class="muted">ChargePulse: draw the bays on your image or video and get
        free/occupied with timestamps.</span></div>
    </div>
  </section>`));

  try {
    const s = await api('/api/v1/health');
    const srv = root.querySelector('#srv');
    const lat = Object.entries(s.live_latency).filter(([, v]) => v);
    srv.innerHTML = `<h3>Live server</h3>
      <table class="t"><tbody>
      <tr><th>Inference</th><td>${s.inference}</td></tr>
      <tr><th>Hardware</th><td>${s.hardware.cpu_count} vCPU · ${s.hardware.machine}</td></tr>
      ${Object.entries(s.models).map(([k, m]) => `<tr><th>${k}</th><td class="mono">${m.file} · ${m.classes} classes · ${m.imgsz}px</td></tr>`).join('')}
      ${lat.map(([k, v]) => `<tr><th>${k}</th><td class="mono">p50 ${v.p50_ms} ms · p95 ${v.p95_ms} ms (n=${v.n})</td></tr>`).join('')}
      <tr><th>Uptime</th><td class="mono">${Math.round(s.uptime_s / 60)} min</td></tr>
      </tbody></table>
      <p class="note">Latencies above are measured live on this deployment since start-up.</p>`;
  } catch (e) { /* status chip already shows the error */ }
}
