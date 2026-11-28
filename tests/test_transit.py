import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kkc.commute.subway import SubwayGraph
from kkc.commute.transit import TransitGraph, legs, ride_minutes, route_lines, wait_minutes


def small_city():
    """지하철 A–B 30분. 버스 7번은 A 근처 → B 근처 한 방향(10분), 정류장은 역에서 약 100m."""
    sub = SubwayGraph()
    sub.add_segment("1", "A", "B", 30)
    coords = pd.DataFrame({"lat": [37.50, 37.60], "lon": [127.0, 127.0]}, index=["A", "B"])
    g = TransitGraph(sub, coords)
    stops = pd.DataFrame({"route": [7, 7], "seq": [1, 2], "stop": [101, 102],
                          "lat": [37.5009, 37.6009], "lon": [127.0, 127.0]})
    g.add_bus(stops, ride_min={(7, 1): 10.0}, wait_min={(7, 101): 3.0})
    g.link_walks()
    return g


def test_bus_beats_subway_in_its_direction_only():
    g = small_city()
    to_b = g.times_to(g.access(37.6009, 127.0))  # 목적지 = B 정류장 앞
    t = g.time_from(37.50, 127.0, to_b)  # A 역 앞에서 출발
    assert t == pytest.approx(1.9 + 3 + 10, abs=0.5)  # 정류장까지 도보 ~1.9분 + 대기 3 + 버스 10 < 지하철 30
    to_a = g.times_to(g.access(37.5009, 127.0))
    assert g.time_from(37.60, 127.0, to_a) > 30  # 반대 방향 버스는 없어서 지하철(30분) + 도보


def test_wait_is_half_headway_clipped():
    assert wait_minutes(6) == pytest.approx(5)  # 시간당 6대 → 10분 간격 → 평균 5분
    assert wait_minutes(60) == 1 and wait_minutes(1) == 15


def test_ride_minutes_caps_terminal_layover():
    assert ride_minutes(120, km=1.0) == pytest.approx(2)  # 실측 2분은 그대로
    assert ride_minutes(3600, km=1.0) == pytest.approx(7.5)  # 종점 대기가 섞인 1시간은 시속 8km 기준으로 자른다


def test_subway_transfer_cannot_shortcut_through_bus_stop():
    sub = SubwayGraph()
    sub.add_segment("1", "A", "B", 5)
    sub.add_segment("2", "A", "C", 5)
    sub.add_transfer("A", "1", "2", 4)  # 공식 환승 4분
    coords = pd.DataFrame({"lat": [37.50, 37.55, 37.45], "lon": [127.0, 127.0, 127.0]}, index=["A", "B", "C"])
    g = TransitGraph(sub, coords)
    g.add_bus(pd.DataFrame({"route": [9, 9], "seq": [1, 2], "stop": [1, 2], "lat": [37.5003, 37.7], "lon": [127.0, 127.0]}),
              ride_min={}, wait_min={})
    g.link_walks()
    assert g.times_to({("A", "2"): 0})[("A", "1")] == pytest.approx(4)  # 역 밖 정류장으로 나갔다 들어오는 게 더 싸면 안 된다


def test_trip_follows_bus_and_legs_name_it():
    g = small_city()
    g.route_name, g.stop_name = {7: "273"}, {101: "A역 앞", 102: "B역 앞"}
    egress = g.access(37.6009, 127.0)
    nxt = {}
    to_b = g.times_to(egress, nxt)
    minutes, walk0, path = g.trip(37.50, 127.0, to_b, nxt=nxt)
    assert minutes == pytest.approx(g.time_from(37.50, 127.0, to_b))
    assert walk0 + to_b[path[0]] == pytest.approx(minutes)
    assert path[-1] in egress
    assert legs(path, g.route_name, g.stop_name) == ["273번 버스 A역 앞 → B역 앞 (1정거장)"]


def test_subway_route_and_legs_drop_transfer_only_hops():
    sub = SubwayGraph()
    sub.add_segment("1", "A", "B", 5)
    sub.add_segment("1", "B", "C", 5)
    sub.add_segment("2", "B", "D", 5)
    sub.add_transfer("B", "1", "2", 3)
    path = sub.route("A", "D")
    assert path[0][0] == "A" and path[-1] == ("D", "2")
    assert legs(path) == ["1호선 A → B (1정거장)", "2호선 B → D (1정거장)"]
    assert legs(sub.route("B", "D")) == ["2호선 B → D (1정거장)"]  # 출발역에서 갈아타는 건 구간이 아님
    assert sub.route("A", "없는역") == []


def test_route_lines_split_rides_and_walks():
    g = small_city()
    egress = g.access(37.6009, 127.0)
    nxt = {}
    to_b = g.times_to(egress, nxt)
    _, _, path = g.trip(37.50, 127.0, to_b, nxt=nxt)
    lines = route_lines(path, g.xy.get)
    assert [ln["kind"] for ln in lines] == [("R", 7)]  # 정류장 앞에서 타고 내리는 거라 도보 구간은 길이 0 → 뺌
    assert lines[0]["pts"] == [[37.5009, 127.0], [37.6009, 127.0]]
    sub = SubwayGraph()
    sub.add_segment("1", "A", "B", 5)
    sub.add_segment("2", "B", "C", 5)
    sub.add_transfer("B", "1", "2", 3)
    xy = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0)}
    lines = route_lines(sub.route("A", "C"), lambda n: xy[n[0]])
    assert [(ln["kind"], ln["pts"]) for ln in lines] == [(("M", "1"), [[0.0, 0.0], [1.0, 0.0]]), (("M", "2"), [[1.0, 0.0], [1.0, 1.0]])]
