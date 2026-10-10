// Shared helpers: API calls, job streaming, DOM and canvas drawing.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function h(html) {
  const t = document.createElement('template');
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

export function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

export async function api(path, opts = {}) {
  const o = { ...opts };
  if (o.json !== undefined) {
    o.body = JSON.stringify(o.json);
    o.headers = { ...(o.headers || {}), 'Content-Type': 'application/json' };
    delete o.json;
  }
  const r = await fetch(path, o);
  const ct = r.headers.get('content-type') || '';
  const body = ct.includes('json') ? await r.json() : await r.text();
  if (!r.ok) throw new Error(typeof body === 'object' ? (body.detail ? JSON.stringify(body.detail) : JSON.stringify(body)) : body);
  return body;
}

export function wsUrl(path) {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${location.host}${path}`;
}

// Stream a server job: frame/progress/done/error messages. Falls back to polling if WebSockets fail.
export function streamJob(jobId, handlers) {
  let closed = false, gotDone = false;
  const ws = new WebSocket(wsUrl(`/ws/jobs/${jobId}`));
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.type === 'frame') handlers.frame?.(m);
    else if (m.type === 'progress') handlers.progress?.(m.progress);
    else if (m.type === 'warning') handlers.warning?.(m.message);
    else if (m.type === 'done') { gotDone = true; handlers.done?.(m.result); }
    else if (m.type === 'error') { gotDone = true; handlers.error?.(m.error); }
  };
  ws.onerror = () => { if (!gotDone && !closed) poll(); };
  async function poll() {
    closed = true;
    for (;;) {
      const s = await api(`/api/v1/jobs/${jobId}`);
      handlers.progress?.(s.progress);
      if (s.status === 'done') { handlers.done?.(s.result); return; }
      if (s.status === 'error') { handlers.error?.(s.error); return; }
      await new Promise(r => setTimeout(r, 800));
    }
  }
  return { cancel: () => { try { ws.close(); } catch {} api(`/api/v1/jobs/${jobId}`, { method: 'DELETE' }).catch(() => {}); } };
}

// Colour encodes the kind of information, never the object class (the class is in the label).
export const C = {
  own: '#e9e7e2',        // seen by this camera
  hazard: '#f0a030',     // road damage, time-to-contact warnings
  network: '#7b9bff',    // observation delivered by another camera
  critical: '#ff5a4a',   // hidden vulnerable road user
  verified: '#3fbf86',
  muted: '#8b9099',
};
export const VRU = new Set(['person', 'cyclist', 'motorcyclist', 'bicycle', 'motorcycle']);
export const DAMAGE = new Set(['pothole', 'alligator_crack', 'transverse_crack', 'longitudinal_crack']);
export const PRETTY = {
  pothole: 'Pothole', alligator_crack: 'Alligator crack', transverse_crack: 'Transverse crack', longitudinal_crack: 'Longitudinal crack',
  person: 'Pedestrian', cyclist: 'Cyclist', motorcyclist: 'Motorcyclist', car: 'Car', truck: 'Truck', bus: 'Bus',
  bicycle: 'Bicycle', motorcycle: 'Motorcycle', vehicle: 'Vehicle', two_wheeler: 'Two-wheeler',
};
export const pretty = c => PRETTY[c] || c;
// Back-compat for modules that colour by class: map classes onto information kinds.
export const color = c => DAMAGE.has(c) ? C.hazard : C.own;

// Box styles: 'corners' (own detection), 'dashed' (network observation), 'fill' (hazard area).
export function drawBox(ctx, box, label, col, { dashed = false, scale = 1, lw = 1.5, fill = false, corners = !dashed && !fill } = {}) {
  const dpr = window.devicePixelRatio || 1;
  const [x1, y1, x2, y2] = box.map(v => v * scale);
  const w = x2 - x1, hgt = y2 - y1;
  ctx.save();
  ctx.strokeStyle = col;
  ctx.lineWidth = lw * dpr;
  if (fill) { ctx.fillStyle = col + '2e'; ctx.fillRect(x1, y1, w, hgt); }
  if (corners) {
    const k = Math.max(6 * dpr, Math.min(w, hgt) * 0.22);
    ctx.beginPath();
    for (const [px, py, dx, dy] of [[x1, y1, 1, 1], [x2, y1, -1, 1], [x2, y2, -1, -1], [x1, y2, 1, -1]]) {
      ctx.moveTo(px + dx * k, py); ctx.lineTo(px, py); ctx.lineTo(px, py + dy * k);
    }
    ctx.stroke();
  } else {
    if (dashed) ctx.setLineDash([5 * dpr, 4 * dpr]);
    ctx.strokeRect(x1, y1, w, hgt);
  }
  if (label) {
    ctx.setLineDash([]);
    const fs = 10.5 * dpr;
    ctx.font = `500 ${fs}px "Geist Mono", ui-monospace, monospace`;
    const text = label.toUpperCase();
    const tw = ctx.measureText(text).width + 10 * dpr;
    const th = fs + 8 * dpr;
    const ty = y1 > th + 2 ? y1 - th - 2 : y1 + 2;
    ctx.fillStyle = 'rgba(13,15,18,.82)';
    ctx.fillRect(x1, ty, tw, th);
    ctx.fillStyle = col;
    ctx.fillRect(x1, ty, 2 * dpr, th);
    ctx.fillStyle = '#e9e7e2';
    ctx.fillText(text, x1 + 6 * dpr, ty + fs + 2 * dpr);
  }
  ctx.restore();
}

export function fitCanvas(canvas, w, h) {
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  return canvas.getContext('2d');
}

export function geolocate(timeout = 8000) {
  return new Promise(resolve => {
    if (!navigator.geolocation) return resolve(null);
    navigator.geolocation.getCurrentPosition(
      p => resolve({ lat: p.coords.latitude, lon: p.coords.longitude, acc: p.coords.accuracy }),
      () => resolve(null), { enableHighAccuracy: true, timeout, maximumAge: 10000 });
  });
}

export function deviceId() {
  try {
    let id = localStorage.getItem('swarmsight.device');
    if (!id) { id = 'web-' + Math.random().toString(36).slice(2, 10); localStorage.setItem('swarmsight.device', id); }
    return id;
  } catch { return 'web-anon'; }
}

export function figure(v, unit, k, src) {
  return `<div class="figure"><div class="v">${v ?? '—'}${unit && v != null ? `<small>${unit}</small>` : ''}</div><div class="k">${k}</div>${src ? `<div class="src">${src}</div>` : ''}</div>`;
}

export function fmtTime(ts) {
  if (!ts) return '—';
  return new Date(ts * 1000).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' });
}

export function dropZone(node, accept, onFiles) {
  const input = node.querySelector('input[type=file]');
  node.addEventListener('click', () => input.click());
  input.addEventListener('change', () => input.files.length && onFiles([...input.files]));
  node.addEventListener('dragover', e => { e.preventDefault(); node.classList.add('over'); });
  node.addEventListener('dragleave', () => node.classList.remove('over'));
  node.addEventListener('drop', e => { e.preventDefault(); node.classList.remove('over'); if (e.dataTransfer.files.length) onFiles([...e.dataTransfer.files]); });
}

// Draws a bird's-eye view (ground plane, metres) of cameras, FOVs, ground truth and fused objects.
export class BEV {
  constructor(canvas) { this.c = canvas; this.view = null; }
  fit(points, pad = 6) {
    if (!points.length) return;
    const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) + 2 * pad;
    this.view = { cx, cy, span: Math.max(span, 12) };
  }
  begin() {
    const r = this.c.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(200, Math.round(r.width * dpr));
    const ctx = fitCanvas(this.c, W, W);
    ctx.fillStyle = '#0d0f12';
    ctx.fillRect(0, 0, W, W);
    this.ctx = ctx; this.W = W; this.s = W / this.view.span;
    // 5 m grid
    ctx.strokeStyle = '#1b1f25'; ctx.lineWidth = 1;
    const x0 = this.view.cx - this.view.span / 2, y0 = this.view.cy - this.view.span / 2;
    for (let x = Math.ceil(x0 / 5) * 5; x < x0 + this.view.span; x += 5) { const [px] = this.p([x, 0]); ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, W); ctx.stroke(); }
    for (let y = Math.ceil(y0 / 5) * 5; y < y0 + this.view.span; y += 5) { const [, py] = this.p([0, y]); ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(W, py); ctx.stroke(); }
    ctx.fillStyle = '#5d626b'; ctx.font = `500 ${10 * (window.devicePixelRatio || 1)}px "Geist Mono", monospace`;
    ctx.fillText('GRID 5 M · NORTH UP', 10, W - 10);
    return ctx;
  }
  p(xy) { return [(xy[0] - this.view.cx) * this.s + this.W / 2, this.W / 2 - (xy[1] - this.view.cy) * this.s]; }
  poly(pts, stroke, fill, dash) {
    const ctx = this.ctx; ctx.save(); ctx.beginPath();
    pts.forEach((q, i) => { const [x, y] = this.p(q); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; if (dash) ctx.setLineDash(dash); ctx.stroke(); }
    ctx.restore();
  }
  dot(xy, r, col, label, ring) {
    const ctx = this.ctx, [x, y] = this.p(xy), d = window.devicePixelRatio || 1;
    if (ring) { ctx.beginPath(); ctx.arc(x, y, Math.max(ring * this.s, 3), 0, Math.PI * 2); ctx.strokeStyle = col + '66'; ctx.lineWidth = d; ctx.setLineDash([3 * d, 3 * d]); ctx.stroke(); ctx.setLineDash([]); }
    ctx.beginPath(); ctx.arc(x, y, r * d, 0, Math.PI * 2); ctx.fillStyle = col; ctx.fill();
    if (label) { ctx.fillStyle = '#e9e7e2'; ctx.font = `500 ${10 * d}px "Geist Mono", monospace`; ctx.fillText(label, x + 8 * d, y - 7 * d); }
  }
  box(xy, yaw, l, w, stroke) {
    const c = Math.cos(yaw), s = Math.sin(yaw);
    const pts = [[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]].map(([a, b]) => [xy[0] + a * c - b * s, xy[1] + a * s + b * c]);
    this.poly(pts, stroke, null, [2, 2]);
  }
  trail(pts, col) {
    if (!pts || pts.length < 2) return;
    const ctx = this.ctx; ctx.save(); ctx.beginPath();
    pts.forEach((q, i) => { const [x, y] = this.p(q); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.strokeStyle = col + '80'; ctx.lineWidth = 1.2 * (window.devicePixelRatio || 1); ctx.stroke(); ctx.restore();
  }
}
