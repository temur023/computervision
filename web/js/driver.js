// Driver display: what a phone on the dashboard (or a paired Bluetooth device) shows and sounds when a warning arrives.
import { esc, h, wsUrl } from './util.js';

const WORDS = { motorcyclist: 'Motorcyclist', cyclist: 'Cyclist', person: 'Pedestrian', car: 'Car', truck: 'Truck', bus: 'Bus' };
const side = b => b == null ? '' : b >= 225 && b <= 315 ? 'from the left' : b >= 45 && b <= 135 ? 'from the right' : b < 45 || b > 315 ? 'ahead' : 'behind';

export default function driver(root, args) {
  const channel = (args[0] || 'demo').toLowerCase().replace(/[^a-z0-9-]/g, '').slice(0, 32) || 'demo';
  document.body.classList.add('bare');
  const ui = h(`<div class="driver">
    <div class="dtop"><span>SwarmSight · driver display</span><span id="conn">connecting · channel ${esc(channel)}</span></div>
    <div class="dmain" id="main">
      <div class="dstate" id="state">CLEAR</div>
      <div class="dsub" id="sub">Listening for warnings from the crossing.</div>
    </div>
    <div class="dfoot">
      <div><span class="lbl">Bluetooth frame · 16 bytes</span><code id="ble">—</code></div>
      <div><span class="lbl">V2X (DENM)</span><code id="denm">—</code></div>
    </div>
    <button class="dunlock" id="unlock">Tap to enable sound and vibration</button>
  </div>`);
  root.append(ui);
  const $ = s => ui.querySelector(s);
  let audio = null, ws = null, timer = 0, ping = 0, closed = false;

  $('#unlock').onclick = () => {
    try { audio = new (window.AudioContext || window.webkitAudioContext)(); audio.resume(); } catch {}
    try { speechSynthesis.speak(new SpeechSynthesisUtterance('')); } catch {}
    $('#unlock').remove();
  };
  function beep(n, hz) {
    if (!audio) return;
    for (let k = 0; k < n; k++) {
      const o = audio.createOscillator(), g = audio.createGain(), t0 = audio.currentTime + k * 0.18;
      o.frequency.value = hz; o.type = 'square';
      g.gain.setValueAtTime(0.0001, t0); g.gain.exponentialRampToValueAtTime(0.25, t0 + 0.01); g.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.12);
      o.connect(g).connect(audio.destination); o.start(t0); o.stop(t0 + 0.13);
    }
  }
  function show(m) {
    const a = m.alert, high = a.level === 'high';
    const who = WORDS[a.cls] || 'Road user';
    $('#main').className = `dmain ${high ? 'high' : 'caution'}`;
    $('#state').textContent = (a.action || (high ? 'Brake' : 'Caution')).toUpperCase();
    $('#sub').innerHTML = `${esc(who)} ${esc(side(a.bearing_deg))}${a.speed_kmh != null ? ` · ${Math.round(a.speed_kmh)} km/h` : ''}${a.ttc_s != null ? ` · <b>${a.ttc_s.toFixed(1)} s</b>` : ''}${a.text ? `<br><span class="dtext">${esc(a.text)}</span>` : ''}`;
    $('#ble').textContent = m.ble_hex;
    $('#denm').textContent = `cause ${m.denm.situation.eventType.causeCode} · quality ${m.denm.situation.informationQuality} · valid ${m.denm.management.validityDuration} s`;
    beep(high ? 3 : 1, high ? 1800 : 1100);
    try { navigator.vibrate?.(high ? [220, 90, 220, 90, 420] : [200]); } catch {}
    try { if (audio && a.action) { speechSynthesis.cancel(); speechSynthesis.speak(Object.assign(new SpeechSynthesisUtterance(`${a.action}. ${who} ${side(a.bearing_deg)}.`), { rate: 1.1 })); } } catch {}
    clearTimeout(timer);
    timer = setTimeout(() => { $('#main').className = 'dmain'; $('#state').textContent = 'CLEAR'; $('#sub').textContent = 'Listening for warnings from the crossing.'; }, Math.max(2500, (a.ttl_s || 3) * 1000));
  }
  function connect() {
    if (closed) return;
    ws = new WebSocket(wsUrl(`/ws/alerts/${channel}`));
    ws.onopen = () => { $('#conn').textContent = `connected · channel ${channel}`; ping = setInterval(() => ws.readyState === 1 && ws.send('ping'), 20000); };
    ws.onmessage = ev => { const m = JSON.parse(ev.data); if (m.type === 'alert') show(m); };
    ws.onclose = () => { clearInterval(ping); $('#conn').textContent = 'reconnecting…'; setTimeout(connect, 1500); };
  }
  connect();
  return () => { closed = true; clearTimeout(timer); clearInterval(ping); try { ws.close(); } catch {} document.body.classList.remove('bare'); };
}
