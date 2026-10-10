"""Warning delivery: one alert, the formats different devices need.

* JSON over WebSocket (/ws/alerts/{channel}): the driver display on a phone, or any app.
* 16-byte binary frame for Bluetooth Low Energy (one GATT notification): small devices such as radar-detector-
  class displays, mirrors and helmets can beep and show it without parsing JSON.  Layout below; CRC-16/CCITT.
* ETSI DENM-style content for a V2X road-side unit: cause codes from ETSI TS 102 894-2 (97 collision risk,
  12 human presence on the road, 9 hazardous location: surface condition).  A certified ITS stack encodes and
  signs it (ASN.1 UPER); we produce the content, not the radio frame.
* GPX waypoints of verified road hazards: the format that navigators and radar-detector point-of-interest
  databases import.

BLE frame, little endian, 16 bytes:
  0  u8   0x53 ('S')                       8  u16  distance, 0.1 m (0xFFFF unknown)
  1  u8   version (1)                      10 u8   bearing relative to the vehicle heading, 2 degrees per unit
  2  u8   kind << 4 | level << 2           11 u8   speed of the other road user, km/h (255 unknown)
  3  u8   road-user class                  12 u16  validity, 0.1 s
  4  u16  alert id                         14 u16  CRC-16/CCITT-FALSE over bytes 0-13
  6  u16  time to conflict, 0.01 s (0xFFFF unknown)
"""
from __future__ import annotations

import struct
import time
from xml.sax.saxutils import escape

KINDS = {"collision_risk": 1, "hidden_road_user": 2, "road_damage": 3, "test": 15}
LEVELS = {"info": 0, "caution": 1, "high": 2}
CLASSES = {"unknown": 0, "person": 1, "cyclist": 2, "bicycle": 2, "motorcyclist": 3, "motorcycle": 3, "car": 4, "truck": 5,
           "bus": 6, "pothole": 10, "alligator_crack": 11, "transverse_crack": 12, "longitudinal_crack": 13}
DENM_CAUSE = {"collision_risk": 97, "hidden_road_user": 12, "road_damage": 9, "test": 0}
_INV = lambda d: {v: k for k, v in reversed(list(d.items()))}


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def normalise(a: dict) -> dict:
    """Validate and complete an alert published by a camera, a console or a road-side unit."""
    kind = a.get("kind", "collision_risk")
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind}")
    level = a.get("level", "high")
    if level not in LEVELS:
        raise ValueError(f"unknown level {level}")
    num = lambda k, lo, hi: None if a.get(k) is None else max(lo, min(hi, float(a[k])))
    return {
        "id": int(a.get("id", int(time.time() * 1000))) & 0xFFFF,
        "kind": kind, "level": level, "cls": a.get("cls", "unknown") if a.get("cls") in CLASSES else "unknown",
        "ttc_s": num("ttc_s", 0, 600), "distance_m": num("distance_m", 0, 6000), "bearing_deg": num("bearing_deg", 0, 359.9),
        "speed_kmh": num("speed_kmh", 0, 254), "ttl_s": num("ttl_s", 0.1, 6000) or 3.0,
        "lat": num("lat", -90, 90), "lon": num("lon", -180, 180),
        "action": str(a.get("action", ""))[:60], "text": str(a.get("text", ""))[:240], "source": str(a.get("source", ""))[:60],
        "t": time.time(),
    }


def encode_ble(a: dict) -> bytes:
    u16 = lambda v, scale: 0xFFFF if v is None else min(0xFFFE, int(round(v * scale)))
    body = struct.pack("<BBBBHHHBBH", 0x53, 1, (KINDS[a["kind"]] << 4) | (LEVELS[a["level"]] << 2), CLASSES.get(a["cls"], 0),
                       a["id"] & 0xFFFF, u16(a.get("ttc_s"), 100), u16(a.get("distance_m"), 10),
                       255 if a.get("bearing_deg") is None else int(a["bearing_deg"] // 2) % 180,
                       255 if a.get("speed_kmh") is None else min(254, int(round(a["speed_kmh"]))), u16(a.get("ttl_s"), 10))
    return body + struct.pack("<H", crc16(body))


def decode_ble(frame: bytes) -> dict:
    if len(frame) != 16 or frame[0] != 0x53:
        raise ValueError("not a SwarmSight alert frame")
    if crc16(frame[:14]) != struct.unpack("<H", frame[14:])[0]:
        raise ValueError("CRC mismatch")
    _, ver, kl, cls, aid, ttc, dist, brg, spd, ttl, _ = struct.unpack("<BBBBHHHBBHH", frame)
    return {"version": ver, "kind": _INV(KINDS)[kl >> 4], "level": _INV(LEVELS)[(kl >> 2) & 3], "cls": _INV(CLASSES).get(cls, "unknown"),
            "id": aid, "ttc_s": None if ttc == 0xFFFF else ttc / 100, "distance_m": None if dist == 0xFFFF else dist / 10,
            "bearing_deg": None if brg == 255 else brg * 2, "speed_kmh": None if spd == 255 else spd, "ttl_s": ttl / 10}


def to_denm(a: dict, station_id: int = 1) -> dict:
    """DENM content (ETSI EN 302 637-3 structure; positions in 1e-7 degrees, times in ms)."""
    now_ms = int(a["t"] * 1000)
    return {
        "header": {"protocolVersion": 2, "messageID": 1, "stationID": station_id},
        "management": {"actionID": {"originatingStationID": station_id, "sequenceNumber": a["id"]},
                       "detectionTime": now_ms, "referenceTime": now_ms,
                       "eventPosition": None if a["lat"] is None else {"latitude": int(a["lat"] * 1e7), "longitude": int(a["lon"] * 1e7)},
                       "relevanceDistance": "lessThan200m", "validityDuration": max(1, int(round(a["ttl_s"]))), "stationType": 15},
        "situation": {"informationQuality": {"info": 1, "caution": 4, "high": 7}[a["level"]],
                      "eventType": {"causeCode": DENM_CAUSE[a["kind"]], "subCauseCode": 0}},
    }


def to_gpx(events: list[dict], name: str = "SwarmSight verified road hazards") -> str:
    pts = []
    for e in events:
        if e.get("lat") is None:
            continue
        label = f"{e['type'].replace('_', ' ').capitalize()} ({e['status']}, confidence {e['confidence']:.2f})"
        pts.append(f'  <wpt lat="{e["lat"]:.6f}" lon="{e["lon"]:.6f}"><name>{escape(label)}</name><type>{escape(e["type"])}</type></wpt>')
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="SwarmSight" xmlns="http://www.topografix.com/GPX/1/1">\n'
            f"  <metadata><name>{escape(name)}</name></metadata>\n" + "\n".join(pts) + "\n</gpx>\n")
