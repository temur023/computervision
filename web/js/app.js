import { $, $$, api } from './util.js';
import home from './home.js';
import guardian from './guardian.js';
import road from './road.js';
import charge from './charge.js';
import data from './metrics.js';
import driver from './driver.js';

const routes = { '': home, guardian, road, charge, data, driver };
const aliases = { scan: 'road', metrics: 'data' };
let cleanup = null;

async function render() {
  let [route, ...rest] = location.hash.replace(/^#\/?/, '').split('/');
  route = aliases[route] ?? route;
  const view = routes[route] || home;
  if (cleanup) { try { cleanup(); } catch {} cleanup = null; }
  $$('#nav a').forEach(a => a.classList.toggle('active', a.dataset.route === (routes[route] ? route : '')));
  const root = $('#view');
  root.innerHTML = '';
  window.scrollTo(0, 0);
  cleanup = (await view(root, rest)) || null;
}

async function status() {
  const el = $('#server');
  try {
    const s = await api('/api/v1/health');
    el.className = 'server ok';
    el.querySelector('span').textContent = `server · ${s.hardware.cpu_count} vCPU · ${Object.keys(s.models).length} models`;
    el.title = Object.entries(s.models).map(([k, m]) => `${k}: ${m.file}`).join('\n');
  } catch {
    el.className = 'server bad';
    el.querySelector('span').textContent = 'server unreachable';
  }
}

window.addEventListener('hashchange', render);
render();
status();
setInterval(status, 30000);
