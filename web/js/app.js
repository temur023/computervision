import { $, $$, api } from './util.js';
import home from './home.js';
import scan from './scan.js';
import guardian from './guardian.js';
import road from './road.js';
import charge from './charge.js';
import metrics from './metrics.js';

const routes = { '': home, scan, guardian, road, charge, metrics };
let cleanup = null;

async function render() {
  const [route, ...rest] = location.hash.replace(/^#\/?/, '').split('/');
  const view = routes[route] || home;
  if (cleanup) { try { cleanup(); } catch {} cleanup = null; }
  $$('#nav a').forEach(a => a.classList.toggle('active', a.dataset.route === route));
  const root = $('#view');
  root.innerHTML = '';
  window.scrollTo(0, 0);
  cleanup = (await view(root, rest)) || null;
}

async function status() {
  const chip = $('#status-chip');
  try {
    const s = await api('/api/v1/health');
    const n = Object.keys(s.models).length;
    chip.textContent = `server ok · ${n} model${n === 1 ? '' : 's'} · ${s.hardware.cpu_count} CPU`;
    chip.className = 'status-chip ok';
    chip.title = Object.entries(s.models).map(([k, m]) => `${k}: ${m.file}`).join('\n');
  } catch {
    chip.textContent = 'server unreachable';
    chip.className = 'status-chip bad';
  }
}

window.addEventListener('hashchange', render);
render();
status();
setInterval(status, 30000);
