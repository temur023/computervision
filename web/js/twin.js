// Digital-twin rendering for Guardian: the bird's-eye map and the camera HUD.
// Colour encodes information only: light = seen by this camera, cobalt = known through the network,
// red = a hidden vulnerable road user. Labels never overlap; displaced labels get a leader line.
import { C, VRU, pretty } from './util.js';

const DPR = () => window.devicePixelRatio || 1;
const VEH = { car: [4.6, 1.9], vehicle: [4.6, 1.9], truck: [7.5, 2.5], bus: [11.0, 2.6] };
const MONO = '"Geist Mono", ui-monospace, SFMono-Regular, Menlo, monospace';
export const isVRU = c => VRU.has(c);

// ------------------------------------------------------------------ primitives
function rrect(ctx, x, y, w, h, r) {
  r = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath();
}

function alpha(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

class LabelLayer {
  constructor() { this.boxes = []; }
  hit(b) { return this.boxes.some(q => b.x < q.x + q.w && b.x + b.w > q.x && b.y < q.y + q.h && b.y + b.h > q.y); }
  add(b) { this.boxes.push(b); }
}

function tagSize(ctx, text, fs) {
  ctx.font = `500 ${fs}px ${MONO}`;
  return { w: ctx.measureText(text).width + fs * 1.25, h: Math.round(fs * 1.9) };
}

function drawTag(ctx, b, text, fs, accent, ink = '#e9e7e2') {
  ctx.fillStyle = 'rgba(13,15,18,.9)';
  ctx.fillRect(b.x, b.y, b.w, b.h);
  if (accent) { ctx.fillStyle = accent; ctx.fillRect(b.x, b.y, Math.max(2, fs * 0.2), b.h); }
  ctx.font = `500 ${fs}px ${MONO}`;
  ctx.fillStyle = ink;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, b.x + fs * 0.7, b.y + b.h / 2 + fs * 0.06);
  ctx.textBaseline = 'alphabetic';
}

// Place a tag near (ax, ay) without overlapping earlier tags; leader line when it had to move away.
function placeTag(ctx, layer, ax, ay, text, fs, accent, bounds, prefer = 'above') {
  const { w, h } = tagSize(ctx, text, fs);
  const g = fs * 0.9;
  const base = prefer === 'above'
    ? [[g * 0.2, -h - g], [-w / 2, -h - g], [g, g * 0.6], [-w - g, -h - g], [-w - g, g * 0.6]]
    : [[g, -h / 2], [-w - g, -h / 2], [g, -h - g], [g, g]];
  const clamp = b => ({ ...b, x: Math.max(bounds.x0, Math.min(bounds.x1 - b.w, b.x)), y: Math.max(bounds.y0, Math.min(bounds.y1 - b.h, b.y)) });
  let best = null;
  for (let k = 0; k < 4 && !best; k++) {
    for (const [dx, dy] of base) {
      const b = clamp({ x: ax + dx, y: ay + dy - (dy < 0 ? k * (h + g * 0.5) : -k * (h + g * 0.5)), w, h });
      if (!layer.hit(b)) { best = b; break; }
    }
  }
  if (!best) best = clamp({ x: ax + g, y: ay - h - g, w, h });
  const cx = Math.max(best.x, Math.min(best.x + best.w, ax));
  const cy = Math.max(best.y, Math.min(best.y + best.h, ay));
  if (Math.hypot(cx - ax, cy - ay) > fs * 1.6) {
    ctx.save();
    ctx.strokeStyle = accent ? alpha(accent, 0.8) : 'rgba(233,231,226,.6)';
    ctx.lineWidth = Math.max(1, fs * 0.1);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(cx, cy); ctx.stroke();
    ctx.restore();
  }
  drawTag(ctx, best, text, fs, accent);
  layer.add(best);
  return best;
}

// ------------------------------------------------------------------ silhouettes for the camera HUD
function silhouette(ctx, cls, x, y, w, h, col) {
  ctx.save();
  ctx.fillStyle = alpha(col, 0.34);
  ctx.strokeStyle = alpha(col, 0.95);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  if (cls === 'person') {
    const cx = x + w / 2, s = Math.min(w / 0.42, h);
    const top = y + h - s;
    ctx.beginPath(); ctx.arc(cx, top + 0.085 * s, 0.075 * s, 0, Math.PI * 2); ctx.fill();
    rrect(ctx, cx - 0.12 * s, top + 0.18 * s, 0.24 * s, 0.4 * s, 0.07 * s); ctx.fill();
    ctx.lineWidth = 0.085 * s;
    ctx.beginPath(); ctx.moveTo(cx - 0.06 * s, top + 0.55 * s); ctx.lineTo(cx - 0.09 * s, top + 0.97 * s);
    ctx.moveTo(cx + 0.06 * s, top + 0.55 * s); ctx.lineTo(cx + 0.1 * s, top + 0.97 * s); ctx.strokeStyle = alpha(col, 0.34); ctx.stroke();
  } else if (cls === 'cyclist' || cls === 'motorcyclist' || cls === 'bicycle' || cls === 'motorcycle') {
    const s = Math.min(w, h);
    const ox = x + (w - s) / 2, oy = y + h - s;
    const r = 0.2 * s;
    const lw = Math.max(1.5, 0.05 * s);
    const w1 = [ox + 0.22 * s, oy + s - r], w2 = [ox + 0.78 * s, oy + s - r];
    ctx.lineWidth = lw;
    ctx.beginPath(); ctx.arc(w1[0], w1[1], r, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(w2[0], w2[1], r, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(w1[0], w1[1]); ctx.lineTo(ox + 0.46 * s, oy + 0.6 * s); ctx.lineTo(w2[0], w2[1]);
    ctx.moveTo(ox + 0.46 * s, oy + 0.6 * s); ctx.lineTo(ox + 0.7 * s, oy + 0.42 * s); ctx.stroke();
    ctx.beginPath(); ctx.arc(ox + 0.56 * s, oy + 0.1 * s, 0.075 * s, 0, Math.PI * 2); ctx.fill();
    ctx.lineWidth = 0.09 * s; ctx.strokeStyle = alpha(col, 0.34);
    ctx.beginPath(); ctx.moveTo(ox + 0.52 * s, oy + 0.2 * s); ctx.lineTo(ox + 0.44 * s, oy + 0.56 * s); ctx.stroke();
  } else {
    rrect(ctx, x, y + 0.36 * h, w, 0.48 * h, 0.12 * h); ctx.fill();
    ctx.beginPath(); ctx.moveTo(x + 0.18 * w, y + 0.4 * h); ctx.lineTo(x + 0.3 * w, y + 0.06 * h); ctx.lineTo(x + 0.72 * w, y + 0.06 * h);
    ctx.lineTo(x + 0.86 * w, y + 0.4 * h); ctx.closePath(); ctx.fill();
    ctx.fillStyle = 'rgba(13,15,18,.75)';
    for (const fx of [0.2, 0.8]) { ctx.beginPath(); ctx.arc(x + fx * w, y + 0.86 * h, 0.12 * h, 0, Math.PI * 2); ctx.fill(); }
  }
  ctx.restore();
}

function brackets(ctx, x1, y1, x2, y2, col, lw) {
  const k = Math.max(5, Math.min(x2 - x1, y2 - y1) * 0.22);
  ctx.strokeStyle = col; ctx.lineWidth = lw;
  ctx.beginPath();
  for (const [px, py, dx, dy] of [[x1, y1, 1, 1], [x2, y1, -1, 1], [x2, y2, -1, -1], [x1, y2, 1, -1]]) {
    ctx.moveTo(px + dx * k, py); ctx.lineTo(px, py); ctx.lineTo(px, py + dy * k);
  }
  ctx.stroke();
}

// HUD over a camera frame. view: Guardian view_for() payload; srcW: pixel width the boxes are in.
export function drawHUD(canvas, view, srcW, { network = true, camLabel = c => c, compact = false, highlight = null, clean = false } = {}) {
  const d = DPR();
  const r = canvas.getBoundingClientRect();
  const W = Math.round(r.width * d), H = Math.round(r.height * d);
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  if (!view) return;
  const s = W / srcW;
  const fs = (compact ? 9.5 : 10.5) * d;
  const layer = new LabelLayer();
  const bounds = { x0: 4 * d, y0: 34 * d, x1: W - 4 * d, y1: H - 46 * d };
  // own detections: corner brackets; label only vulnerable road users (the ones that matter)
  for (const tr of view.own_tracks || []) {
    const [x1, y1, x2, y2] = tr.bbox.map(v => v * s);
    brackets(ctx, x1, y1, x2, y2, alpha(C.own, clean ? 0.5 : 0.9), (clean ? 1.1 : 1.4) * d);
    if (!clean && VRU.has(tr.cls) && !compact) placeTag(ctx, layer, x1, y1, pretty(tr.cls).toUpperCase(), fs * 0.92, C.own, bounds);
  }
  if (!network) return;
  const ghosts = (view.objects || []).filter(o => o.ghost_box && (clean ? o.status === 'occluded' && VRU.has(o.cls) : (o.status === 'occluded' || o.status === 'undetected')))
    .sort((a, b) => (b.ghost_box[3] - b.ghost_box[1]) - (a.ghost_box[3] - a.ghost_box[1]));
  for (const o of ghosts) {
    const [x1, y1, x2, y2] = o.ghost_box.map(v => v * s);
    const hidden = o.status === 'occluded';
    const col = hidden && VRU.has(o.cls) ? C.critical : C.network;
    const focus = highlight == null || highlight === o.gid;
    ctx.globalAlpha = focus ? 1 : 0.45;
    silhouette(ctx, o.cls, x1, y1, x2 - x1, y2 - y1, col);
    ctx.save(); ctx.setLineDash([4 * d, 3 * d]); ctx.strokeStyle = alpha(col, 0.9); ctx.lineWidth = 1.3 * d;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1); ctx.restore();
    const text = clean
      ? `${pretty(o.cls).toUpperCase()} · ${Math.round(o.dist_m)} M`
      : hidden
        ? `HIDDEN ${pretty(o.cls).toUpperCase()} · ${Math.round(o.dist_m)} M${o.occluded_by ? ` · BEHIND ${pretty(o.occluded_by.cls).toUpperCase()}` : ''}`
        : `${pretty(o.cls).toUpperCase()} · ${Math.round(o.dist_m)} M · NOT DETECTED HERE`;
    placeTag(ctx, layer, (x1 + x2) / 2, y1, text, fs, col, bounds);
    ctx.globalAlpha = 1;
  }
}

// Thumbnails of the other cameras: their detections; the ones shared with the ego as hidden objects in red.
export function drawThumbHUD(canvas, tracks, srcW, sharedIds = new Set(), dim = false) {
  const d = DPR();
  const r = canvas.getBoundingClientRect();
  const W = Math.round(r.width * d), H = Math.round(r.height * d);
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);
  if (dim) { ctx.fillStyle = 'rgba(13,15,18,.62)'; ctx.fillRect(0, 0, W, H); return; }
  const s = W / srcW;
  for (const tr of tracks || []) {
    const [x1, y1, x2, y2] = tr.bbox.map(v => v * s);
    const shared = sharedIds.has(tr.id);
    brackets(ctx, x1, y1, x2, y2, shared ? C.critical : alpha(C.network, 0.9), (shared ? 2 : 1.2) * d);
  }
}

// ------------------------------------------------------------------ the map
export class TwinMap {
  constructor(canvas) {
    this.c = canvas;
    this.view = null;
    this.base = null;
    this.headings = new Map();
  }
  setBase(src, bbox) {
    const im = new Image();
    im.onload = () => { this.base = { im, bbox }; };
    im.src = src;
  }
  setView(cx, cy, span) { this.view = { cx, cy, span }; }
  fit(points, pad = 6) {
    if (!points.length) return;
    const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
    const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) + 2 * pad;
    this.view = { cx: (Math.min(...xs) + Math.max(...xs)) / 2, cy: (Math.min(...ys) + Math.max(...ys)) / 2, span: Math.max(span, 12) };
  }
  p(x, y) { return [(x - this.view.cx) * this.s + this.W / 2, this.H / 2 - (y - this.view.cy) * this.s]; }

  heading(o) {
    const v = o.v || [0, 0];
    if (Math.hypot(v[0], v[1]) > 0.8) this.headings.set(o.gid, Math.atan2(v[1], v[0]));
    return this.headings.get(o.gid) ?? null;
  }

  footprint(ctx, xy, yaw, L, Wd, fill, stroke, lw) {
    const [px, py] = this.p(xy[0], xy[1]);
    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(-yaw);
    rrect(ctx, -L * this.s / 2, -Wd * this.s / 2, L * this.s, Wd * this.s, Math.min(0.5 * this.s, Wd * this.s / 3));
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
    ctx.restore();
  }

  // st: { cams: [{id, kind: ego|rsu|veh, pos, yaw, fov, label}], objects, gt, network, clock, chip, timeLabel, focusGid }
  render(st) {
    const d = DPR();
    const r = this.c.getBoundingClientRect();
    const W = Math.max(200, Math.round(r.width * d)), H = Math.max(200, Math.round(r.height * d));
    if (this.c.width !== W || this.c.height !== H) { this.c.width = W; this.c.height = H; }
    if (!this.view) this.view = { cx: 0, cy: 0, span: 60 };
    this.W = W; this.H = H; this.s = Math.min(W, H) / this.view.span;
    const ctx = this.c.getContext('2d');
    const fs = 10.5 * d;
    const clock = st.clock || 0;
    ctx.fillStyle = '#0d0f12';
    ctx.fillRect(0, 0, W, H);

    // base layer: orthophoto stitched from the road-side cameras, or a quiet grid
    if (this.base) {
      const [x0, y0, x1, y1] = this.base.bbox;
      const [ax, ay] = this.p(x0, y1), [bx, by] = this.p(x1, y0);
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(this.base.im, ax, ay, bx - ax, by - ay);
    } else {
      ctx.strokeStyle = '#181b20'; ctx.lineWidth = 1;
      const step = 5;
      const x0 = this.view.cx - this.view.span, y0 = this.view.cy - this.view.span;
      for (let x = Math.ceil(x0 / step) * step; x < this.view.cx + this.view.span; x += step) { const [px] = this.p(x, 0); ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke(); }
      for (let y = Math.ceil(y0 / step) * step; y < this.view.cy + this.view.span; y += step) { const [, py] = this.p(0, y); ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(W, py); ctx.stroke(); }
    }

    const cams = st.cams || [];
    const own = c => c.kind === 'ego' || c.kind === 'fixed-ego';
    const ego = cams.find(own);
    const net = st.network !== false;

    // fields of view: soft wedges fading with distance
    for (const cam of cams) {
      if (!cam.fov || cam.fov.length < 3 || (!net && !own(cam))) continue;
      const col = own(cam) ? C.own : C.network;
      const [cx, cy] = this.p(cam.pos[0], cam.pos[1]);
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, 46 * this.s);
      g.addColorStop(0, alpha(col, own(cam) ? 0.2 : (st.clean ? 0.07 : 0.14)));
      g.addColorStop(1, alpha(col, 0));
      ctx.beginPath();
      cam.fov.forEach((q, i) => { const [x, y] = this.p(q[0], q[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.closePath();
      ctx.fillStyle = g; ctx.fill();
      if (st.clean && !own(cam)) continue;
      ctx.strokeStyle = alpha(col, 0.35); ctx.lineWidth = d;
      const a = this.p(cam.fov[1][0], cam.fov[1][1]), b = this.p(cam.fov[cam.fov.length - 1][0], cam.fov[cam.fov.length - 1][1]);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(cx, cy); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }

    // ground truth (optional): thin dashed outlines
    if (st.gt) {
      ctx.save(); ctx.setLineDash([3 * d, 3 * d]);
      for (const g of st.gt) this.footprint(ctx, g.xy, g.yaw, g.l, g.w, null, 'rgba(198,201,207,.55)', d);
      ctx.restore();
    }

    const objs = (st.objects || []).filter(o => net || o.status === 'visible');
    const key = o => o.status === 'occluded' && VRU.has(o.cls);  // the only thing a clean view highlights
    if (st.clean) {
      this.renderClean(ctx, st, objs, cams, ego, own, net, key, d, fs, clock);
      return;
    }
    const colour = o => o.status === 'visible' ? C.own : (o.status === 'occluded' && VRU.has(o.cls) ? C.critical : C.network);
    // trails
    for (const o of objs) {
      if (!o.trail || o.trail.length < 2) continue;
      ctx.beginPath();
      o.trail.forEach((q, i) => { const [x, y] = this.p(q[0], q[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      const [hx, hy] = this.p(o.xy[0], o.xy[1]); ctx.lineTo(hx, hy);
      ctx.strokeStyle = alpha(colour(o), 0.45); ctx.lineWidth = 1.2 * d; ctx.stroke();
    }
    // sight lines from the ego camera through the occluder to each hidden road user
    if (ego && net) {
      const [ex, ey] = this.p(ego.pos[0], ego.pos[1]);
      for (const o of objs) {
        if (o.status !== 'occluded') continue;
        const [ox, oy] = this.p(o.xy[0], o.xy[1]);
        ctx.save(); ctx.setLineDash([5 * d, 4 * d]); ctx.strokeStyle = alpha(colour(o), 0.55); ctx.lineWidth = 1.2 * d;
        ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(ox, oy); ctx.stroke(); ctx.restore();
      }
    }
    // objects: vehicles as oriented footprints, vulnerable road users as dots; hidden ones last
    const order = [...objs].sort((a, b) => (a.status === 'occluded') - (b.status === 'occluded'));
    for (const o of order) {
      const col = colour(o);
      const [px, py] = this.p(o.xy[0], o.xy[1]);
      const hid = o.status === 'occluded';
      if (hid && o.r2s) {
        ctx.save(); ctx.setLineDash([2 * d, 3 * d]); ctx.strokeStyle = alpha(col, 0.7); ctx.lineWidth = d;
        ctx.beginPath(); ctx.arc(px, py, Math.max(o.r2s * this.s, 6 * d), 0, Math.PI * 2); ctx.stroke(); ctx.restore();
      }
      if (VRU.has(o.cls)) {
        const rad = Math.max(0.55 * this.s, 3.6 * d);
        ctx.beginPath(); ctx.arc(px, py, rad, 0, Math.PI * 2);
        ctx.fillStyle = col; ctx.fill(); ctx.lineWidth = 1.5 * d; ctx.strokeStyle = '#0d0f12'; ctx.stroke();
      } else {
        const hdg = this.heading(o);
        const [L, Wd] = VEH[o.cls] || VEH.car;
        if (hdg == null) {
          ctx.beginPath(); ctx.arc(px, py, Math.max(1.2 * this.s, 4 * d), 0, Math.PI * 2);
          ctx.fillStyle = alpha(col, 0.28); ctx.fill(); ctx.strokeStyle = col; ctx.lineWidth = 1.5 * d; ctx.stroke();
        } else {
          this.footprint(ctx, o.xy, hdg, L, Wd, alpha(col, 0.28), col, 1.5 * d);
        }
      }
      if (hid) {
        const ph = (clock * 1.1) % 1;
        ctx.beginPath(); ctx.arc(px, py, (7 + 16 * ph) * d, 0, Math.PI * 2);
        ctx.strokeStyle = alpha(col, 0.85 * (1 - ph)); ctx.lineWidth = 2 * d; ctx.stroke();
      }
    }
    // cameras: the ego vehicle, the road-side mast, other connected vehicles
    const drawn = new Set();
    for (const cam of cams) {
      if (!cam.pos || (!net && !own(cam))) continue;
      const key = `${cam.pos[0].toFixed(1)},${cam.pos[1].toFixed(1)}`;
      if (cam.kind === 'ego' || cam.kind === 'veh') {
        const back = [cam.pos[0] - 1.5 * Math.cos(cam.yaw), cam.pos[1] - 1.5 * Math.sin(cam.yaw)];
        this.footprint(ctx, back, cam.yaw, 4.6, 1.9, cam.kind === 'ego' ? 'rgba(233,231,226,.95)' : alpha(C.network, 0.35),
          cam.kind === 'ego' ? '#0d0f12' : C.network, 1.5 * d);
        const [px, py] = this.p(cam.pos[0], cam.pos[1]);
        ctx.beginPath(); ctx.arc(px, py, 2.6 * d, 0, Math.PI * 2); ctx.fillStyle = cam.kind === 'ego' ? '#0d0f12' : C.network; ctx.fill();
      } else if (!drawn.has(key)) {
        drawn.add(key);
        const col = own(cam) ? C.own : C.network;
        const [px, py] = this.p(cam.pos[0], cam.pos[1]);
        ctx.beginPath(); ctx.arc(px, py, 7 * d, 0, Math.PI * 2); ctx.fillStyle = '#0d0f12'; ctx.fill();
        ctx.strokeStyle = col; ctx.lineWidth = 2 * d; ctx.stroke();
        ctx.beginPath(); ctx.arc(px, py, 2.6 * d, 0, Math.PI * 2); ctx.fillStyle = col; ctx.fill();
      }
    }
    // labels, collision-free
    const layer = new LabelLayer();
    const bounds = { x0: 6 * d, y0: 36 * d, x1: W - 6 * d, y1: H - 40 * d };
    const labelled = new Set();
    for (const cam of cams) {
      if (!cam.label || !cam.pos || (!net && !own(cam))) continue;
      const key = `${cam.pos[0].toFixed(1)},${cam.pos[1].toFixed(1)}`;
      if (labelled.has(key)) continue;
      labelled.add(key);
      const [px, py] = this.p(cam.pos[0], cam.pos[1]);
      placeTag(ctx, layer, px, py, cam.label, fs, own(cam) ? C.own : C.network, bounds, 'side');
    }
    for (const o of order.filter(o => o.status === 'occluded')) {
      const [px, py] = this.p(o.xy[0], o.xy[1]);
      placeTag(ctx, layer, px, py, `HIDDEN ${pretty(o.cls).toUpperCase()}${o.dist_m != null ? ` · ${Math.round(o.dist_m)} M` : ''}`, fs, colour(o), bounds, 'side');
    }

    // chrome: mode chip, clock, scale bar, north arrow
    const chip = st.chip || 'DIGITAL TWIN';
    drawTag(ctx, { x: 10 * d, y: 10 * d, ...tagSize(ctx, chip, fs) }, chip, fs, net ? C.network : C.muted);
    if (st.timeLabel) {
      const sz = tagSize(ctx, st.timeLabel, fs);
      drawTag(ctx, { x: W - sz.w - 10 * d, y: 10 * d, ...sz }, st.timeLabel, fs, null);
    }
    const bar = 10 * this.s;
    const bx = 14 * d, by = H - 16 * d;
    ctx.strokeStyle = '#e9e7e2'; ctx.lineWidth = 1.5 * d;
    ctx.beginPath(); ctx.moveTo(bx, by - 5 * d); ctx.lineTo(bx, by); ctx.lineTo(bx + bar, by); ctx.lineTo(bx + bar, by - 5 * d); ctx.stroke();
    ctx.font = `500 ${fs}px ${MONO}`; ctx.fillStyle = '#e9e7e2'; ctx.fillText('10 M', bx + bar + 8 * d, by + 1 * d);
    const nx = W - 22 * d, ny = H - 30 * d;
    ctx.beginPath(); ctx.moveTo(nx, ny - 9 * d); ctx.lineTo(nx + 6 * d, ny + 7 * d); ctx.lineTo(nx, ny + 3 * d); ctx.lineTo(nx - 6 * d, ny + 7 * d); ctx.closePath();
    ctx.fillStyle = '#e9e7e2'; ctx.fill();
    ctx.fillText('N', nx - 4 * d, ny + 22 * d);
  }

  // Clean view: neutral dots for everyone, one highlighted class of object (hidden pedestrians and riders),
  // a dashed line of sight from the car to each of them, and only three kinds of labels.
  renderClean(ctx, st, objs, cams, ego, own, net, key, d, fs, clock) {
    const { W, H } = this;
    if (ego && net) {
      const [ex, ey] = this.p(ego.pos[0], ego.pos[1]);
      for (const o of objs.filter(key)) {
        const [ox, oy] = this.p(o.xy[0], o.xy[1]);
        ctx.save(); ctx.setLineDash([5 * d, 4 * d]); ctx.strokeStyle = alpha(C.critical, 0.6); ctx.lineWidth = 1.3 * d;
        ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(ox, oy); ctx.stroke(); ctx.restore();
      }
    }
    for (const o of objs) {
      if (key(o)) continue;
      const [px, py] = this.p(o.xy[0], o.xy[1]);
      const seen = o.status === 'visible';
      ctx.beginPath(); ctx.arc(px, py, (VRU.has(o.cls) ? 3 : 4) * d, 0, Math.PI * 2);
      ctx.fillStyle = seen ? 'rgba(233,231,226,.9)' : 'rgba(123,155,255,.75)'; ctx.fill();
    }
    for (const o of objs.filter(key)) {
      const [px, py] = this.p(o.xy[0], o.xy[1]);
      const ph = (clock * 1.1) % 1;
      ctx.beginPath(); ctx.arc(px, py, (8 + 16 * ph) * d, 0, Math.PI * 2);
      ctx.strokeStyle = alpha(C.critical, 0.8 * (1 - ph)); ctx.lineWidth = 2 * d; ctx.stroke();
      ctx.beginPath(); ctx.arc(px, py, 5.5 * d, 0, Math.PI * 2);
      ctx.fillStyle = C.critical; ctx.fill(); ctx.lineWidth = 1.5 * d; ctx.strokeStyle = '#0d0f12'; ctx.stroke();
    }
    const drawn = new Set();
    for (const cam of cams) {
      if (!cam.pos || (!net && !own(cam))) continue;
      const k = `${cam.pos[0].toFixed(1)},${cam.pos[1].toFixed(1)}`;
      if (cam.kind === 'ego' || cam.kind === 'veh') {
        const back = [cam.pos[0] - 1.5 * Math.cos(cam.yaw), cam.pos[1] - 1.5 * Math.sin(cam.yaw)];
        this.footprint(ctx, back, cam.yaw, 4.6, 1.9, cam.kind === 'ego' ? 'rgba(233,231,226,.95)' : alpha(C.network, 0.35),
          cam.kind === 'ego' ? '#0d0f12' : C.network, 1.5 * d);
      } else if (!drawn.has(k)) {
        drawn.add(k);
        const col = own(cam) ? C.own : C.network;
        const [px, py] = this.p(cam.pos[0], cam.pos[1]);
        ctx.beginPath(); ctx.arc(px, py, 7 * d, 0, Math.PI * 2); ctx.fillStyle = '#0d0f12'; ctx.fill();
        ctx.strokeStyle = col; ctx.lineWidth = 2 * d; ctx.stroke();
        ctx.beginPath(); ctx.arc(px, py, 2.6 * d, 0, Math.PI * 2); ctx.fillStyle = col; ctx.fill();
      }
    }
    const layer = new LabelLayer();
    const bounds = { x0: 6 * d, y0: 8 * d, x1: W - 6 * d, y1: H - 40 * d };
    const done = new Set();
    for (const cam of cams) {
      if (!cam.label || !cam.pos || (!net && !own(cam))) continue;
      const k = `${cam.pos[0].toFixed(1)},${cam.pos[1].toFixed(1)}`;
      if (done.has(k)) continue;
      done.add(k);
      const [px, py] = this.p(cam.pos[0], cam.pos[1]);
      placeTag(ctx, layer, px, py, cam.label, fs, own(cam) ? C.own : C.network, bounds, 'side');
    }
    for (const o of objs.filter(key)) {
      const [px, py] = this.p(o.xy[0], o.xy[1]);
      placeTag(ctx, layer, px, py, `${pretty(o.cls).toUpperCase()}${o.dist_m != null ? ` · ${Math.round(o.dist_m)} M` : ''}`, fs, C.critical, bounds, 'side');
    }
    if (st.chip) drawTag(ctx, { x: 10 * d, y: 10 * d, ...tagSize(ctx, st.chip, fs) }, st.chip, fs, C.muted);
    const bar = 10 * this.s, bx = 14 * d, by = H - 16 * d;
    ctx.strokeStyle = 'rgba(233,231,226,.8)'; ctx.lineWidth = 1.5 * d;
    ctx.beginPath(); ctx.moveTo(bx, by - 5 * d); ctx.lineTo(bx, by); ctx.lineTo(bx + bar, by); ctx.lineTo(bx + bar, by - 5 * d); ctx.stroke();
    ctx.font = `500 ${fs}px ${MONO}`; ctx.fillStyle = 'rgba(233,231,226,.8)'; ctx.fillText('10 M', bx + bar + 8 * d, by + d);
  }
}
