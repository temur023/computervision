import { api, esc, h } from './util.js';

// Overview: one sentence per module and the one measured number that justifies it.
export default async function home(root) {
  const m = await api('/api/v1/metrics').catch(() => ({}));
  const coop = m.coop_v2xsim?.configs?.['rsu+v2v']?.overall;
  const rd = m.road_damage_eval, tk = m.tashkent_eval;

  const num = (v, unit, k, src) => `<div class="mnum"><div class="v${v == null ? ' none' : ''}">${v ?? '—'}${unit && v != null ? `<small>${unit}</small>` : ''}</div>
    <div class="k">${k}</div>${src ? `<div class="src">${src}</div>` : ''}</div>`;

  const rows = [
    {
      n: '01', href: '#/guardian', name: 'Guardian', stage: 'Research core', col: 'var(--network)',
      what: 'Shows a driver the cyclists and pedestrians their own camera cannot see, predicts whose path will cross theirs, and warns in time.',
      who: 'Fleets · carmakers\' safety teams · city traffic centres',
      fig: coop ? num(coop['occluded_vru_recall_fused_%'], '%', 'of cyclists and motorcyclists hidden from the car\'s own camera are found by the network',
        `V2X-Sim 2.0 · ${coop.gt_occluded_vru} hidden instances · own camera alone: 0%`) : num(null, '', 'benchmark result missing'),
    },
    {
      n: '02', href: '#/road', name: 'Road Intelligence', stage: 'First revenue', col: 'var(--hazard)',
      what: 'Turns ordinary phone and dashcam footage into verified, deduplicated road-damage events with a position and evidence.',
      who: 'Map platforms (Yandex Maps) · city road service · taxi fleets',
      fig: rd ? num(rd.test.map50.toFixed(2), 'mAP@.5', 'road-damage detector we trained, on images from 6 countries it never saw',
        `RDD2022 · ${rd.n_test_images.toLocaleString('en')} held-out images${tk ? ` · Tashkent AP ${tk.ap_image.toFixed(2)}` : ''}`)
        : num(null, '', 'Detector in training. The number appears here when the held-out evaluation has run.'),
    },
    {
      n: '03', href: '#/charge', name: 'ChargePulse', stage: 'Add-on', col: 'var(--ink-3)',
      what: 'Tells drivers and operators which EV charging bays are free, and since when, from one camera over the site.',
      who: 'Charging operators · dealer service centres · parking owners',
      fig: num(null, '', 'Not measured yet. First pilot step: two weeks of footage at one site with hourly manual counts.'),
    },
  ];

  root.append(h(`<div class="wrap">
    <section class="page-head hero">
      <span class="label">SwarmSight · Startup · Tashkent</span>
      <h1>Every car sees part of the road. Together they see all of it.</h1>
      <p class="lead">SwarmSight joins the cameras already on the road (taxis, fleets, intersections) into one perception network.</p>
      <div class="row" style="margin-top:1.8rem">
        <a class="btn primary" href="#/guardian">Watch a hidden rider appear <span class="arr">→</span></a>
        <a class="btn" href="#/road">Scan your own road photo</a>
      </div>
    </section>

    <section class="mods">
      ${rows.map(r => `<a class="mod" href="${r.href}">
        <span class="label">${r.n}</span>
        <div class="mod-what"><h3>${esc(r.name)}</h3><div class="stage" style="color:${r.col}">${esc(r.stage)}</div>
          <p>${esc(r.what)}</p><p class="who">${esc(r.who)}</p></div>
        ${r.fig}
        <span class="go">→</span>
      </a>`).join('')}
    </section>

    <section class="section">
      <div class="section-head"><span class="label">How it works</span>
        <div><h2>One rule: never trust a single camera</h2></div></div>
      <div class="pipeline p4">
        <div><span class="label">01</span><b>See</b><p>Phones, dashcams and intersection cameras that are already on the road. No new hardware.</p></div>
        <div><span class="label">02</span><b>Place</b><p>Every detection is put on the ground in metres, with its uncertainty, so that different cameras can be compared.</p></div>
        <div><span class="label">03</span><b>Agree</b><p>A hidden cyclist needs another camera that sees it. A pothole needs reports from independent sources.</p></div>
        <div><span class="label">04</span><b>Act</b><p>A warning for the driver. A verified, deduplicated event for the map and the road service.</p></div>
      </div>
      <p class="note" style="margin-top:1.6rem">Every number on this site is computed by our scripts or by this server, on data the models did not see in training.
        Sources, splits and licences: <a href="#/data">Data &amp; evidence</a>.</p>
    </section>
  </div>`));
}
