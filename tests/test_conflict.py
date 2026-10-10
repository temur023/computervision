import numpy as np

from swarmsight.conflict import Agent, ConflictMonitor, assess, assess_all, assess_for


def mover(id, cls, xy, v, t_hist=1.2):
    xy, v = np.array(xy, float), np.array(v, float)
    hist = [(-t_hist + k * 0.2, *(xy - v * (t_hist - k * 0.2)), *v) for k in range(int(t_hist / 0.2) + 1)]
    return Agent(id, cls, xy, v, 0.3, 0.3, hist, age_s=2.0)


def test_crossing_paths_meeting_at_the_same_time_is_a_high_conflict():
    car = mover("car", "car", [-20, 0], [10, 0])  # reaches the origin in 2 s
    moto = mover("moto", "motorcycle", [0, -24], [0, 12])  # also in 2 s
    r = assess(car, moto)
    assert r is not None and r["level"] == "high" and r["type"] == "crossing"
    assert 1.2 <= r["ttc_s"] <= 2.4
    assert abs(r["point"][0]) < 3 and abs(r["point"][1]) < 3


def test_crossing_with_a_safe_gap_is_not_a_conflict():
    car = mover("car", "car", [-20, 0], [10, 0])  # at the origin at 2 s
    moto = mover("moto", "motorcycle", [0, -60], [0, 12])  # at the origin at 5 s
    r = assess(car, moto)
    assert r is None or r["level"] == "low"


def test_parallel_lanes_do_not_conflict():
    a = mover("a", "car", [0, 0], [12, 0])
    b = mover("b", "car", [-6, 3.5], [12, 0])
    r = assess(a, b)
    assert r is None or r["level"] == "low"


def test_stopped_car_start_check_and_message():
    ego = Agent("ego", "car", [0, 0], [0, 0], 0.2, 0.2, [], ego=True, heading=np.pi / 2)  # facing north
    moto = mover("m", "motorcycle", [-30, 8], [12, 0])  # crosses 8 m ahead in ~2.5 s
    res = assess_for(ego, [moto])
    assert res and res[0]["hyp"][0] == "start"
    assert res[0]["action"] in ("Do not start", "Wait before starting")
    assert "from the left" in res[0]["reason"]


def test_road_side_mode_and_monitor_hysteresis():
    car = mover("car", "car", [-20, 0], [10, 0])
    moto = mover("moto", "motorcycle", [0, -24], [0, 12])
    mon = ConflictMonitor(raise_n=2, clear_n=3)
    first = mon.update(assess_all([car, moto]), 0.0)
    second = mon.update(assess_all([car, moto]), 0.2)
    assert first == [] or first[0]["new"]
    assert second and second[0]["alert_id"] == 1
    for k in range(3):
        out = mon.update([], 0.4 + 0.2 * k)
    assert out == []
