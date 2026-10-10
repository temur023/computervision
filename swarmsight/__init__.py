"""SwarmSight AI: collective vision for connected mobility.

Modules
-------
detector   ONNX Runtime YOLO inference (no torch at runtime)
tracker    ByteTrack-style multi-object tracker (Kalman + two-stage IoU matching)
geometry   ground-plane homographies, uncertainty propagation, camera pose from a homography
guardian   cooperative perception: cross-camera association, fusion, occlusion reasoning
trust      Road Intelligence trust engine: corroboration, source reliability, lifecycle
store      SQLite persistence for road events, observations and provenance
pipeline   single-camera analysis (road damage, road users, time-to-contact warnings)
chargepulse  EV bay visual occupancy
privacy    anonymisation of evidence snapshots
"""

__version__ = "0.1.0"
