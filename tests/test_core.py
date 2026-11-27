import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kkc.commute.subway import SubwayGraph
from kkc.ranking.rank import score_regions, tradeoff_suggestions
from kkc.vision.fov import crop_ratio, fov_from_focal_35mm, restore_standard_view


def test_dijkstra_with_transfer():
    g = SubwayGraph()
    g.add_segment("2", "신림", "서울대입구", 2)
    g.add_segment("2", "서울대입구", "사당", 4)
    g.add_segment("4", "사당", "총신대입구", 2)
    g.add_transfer("사당", "2", "4", 3)
    t = g.times_from("신림")
    assert t["사당"] == 6
    assert t["총신대입구"] == 6 + 3 + 2


def test_fov_math():
    assert fov_from_focal_35mm(26) == pytest.approx(69.4, abs=0.1)
    assert fov_from_focal_35mm(13) == pytest.approx(108.3, abs=0.2)
    assert crop_ratio(69.4) == 1.0
    assert crop_ratio(108.3) == pytest.approx(0.5, abs=0.01)  # 0.5x → 절반 크롭


def test_restore_keeps_size():
    img = Image.new("RGB", (400, 300))
    assert restore_standard_view(img, 108).size == (400, 300)


def test_ranking_and_tradeoff():
    df = pd.DataFrame({"region": ["A", "B", "C"], "pred_rent": [70, 60, 50],
                       "commute_min": [20, 28, 38], "amenity_cnt": [10, 10, 10]})
    r = score_regions(df, budget=80, max_commute=30)
    assert list(r.region) == ["B", "A"]  # C는 통학 초과
    s = tradeoff_suggestions(df, budget=80, max_commute=30)
    assert s[0] == {"extra_minutes": 10, "monthly_saving": 10.0, "region": "C"}



def test_osm_subway_map_dedups_ways_clips_and_merges_stops():
    from kkc.commute.subway import osm_subway_map
    way = {"type": "way", "ref": 1, "role": "", "geometry": [{"lat": 37.5, "lon": 127.0}, {"lat": 37.51, "lon": 127.01}]}
    far = {"type": "way", "ref": 2, "role": "", "geometry": [{"lat": 36.8, "lon": 127.1}, {"lat": 36.81, "lon": 127.1}]}
    stop = {"type": "node", "ref": 9, "role": "stop"}
    els = [{"type": "relation", "tags": {"ref": "2"}, "members": [way, far, stop]},
           {"type": "relation", "tags": {"ref": "2"}, "members": [way]},  # 반대 방향 노선이 같은 선로를 또 적는다
           {"type": "node", "lat": 37.50, "lon": 127.0, "tags": {"name": "가"}},
           {"type": "node", "lat": 37.52, "lon": 127.0, "tags": {"name": "가"}},  # 방향별 정차 위치 → 평균
           {"type": "node", "lat": 36.80, "lon": 127.1, "tags": {"name": "천안"}}]  # 서울 밖
    m = osm_subway_map(els)
    assert m["lines"] == {"2": [[[37.5, 127.0], [37.51, 127.01]]]}
    assert m["stations"] == [{"name": "가", "lat": 37.51, "lon": 127.0}]


def test_osm_subway_map_routes_keep_stop_order_and_skip_express():
    from kkc.commute.subway import osm_subway_map
    stops = [{"type": "node", "ref": i, "role": "stop", "lat": 37.5 + i / 100, "lon": 127.0} for i in (1, 2, 3)]
    els = [{"type": "relation", "tags": {"ref": "1", "name": "1호선: 가 → 다"}, "members": stops},
           {"type": "relation", "tags": {"ref": "1", "name": "1호선: 가 → 다 급행"}, "members": [stops[0], stops[2]]},
           {"type": "relation", "tags": {"ref": "1", "name": "1호선: 가 → 다"}, "members": stops}]  # 같은 순서는 한 번만
    els += [{"type": "node", "id": i, "lat": 37.5 + i / 100, "lon": 127.0, "tags": {"name": n}}
            for i, n in ((1, "가"), (2, "나"), (3, "다"))]
    assert osm_subway_map(els)["routes"] == {"1": [[["가", 37.51, 127.0], ["나", 37.52, 127.0], ["다", 37.53, 127.0]]]}


def test_add_osm_routes_fills_missing_stations_by_position():
    from kkc.commute.subway import add_osm_routes
    g = SubwayGraph()
    g.add_segment("1", "청량리", "제기동", 2)  # 서울교통공사 구간: 약 1.11km에 2분
    g.add_segment("6", "석계", "돌곶이", 2)
    coords = pd.DataFrame({"lat": [37.580, 37.590, 37.615, 37.610], "lon": [127.0, 127.0, 127.066, 127.056]},
                          index=["청량리", "제기동", "석계", "돌곶이"])
    route = [["청량리역", 37.5801, 127.0], ["회기", 37.600, 127.0], ["석계", 37.6151, 127.066]]  # 이름이 달라도 300m 안이면 같은 역
    coords = add_osm_routes(g, coords, {"1": [route]})
    assert "회기" in coords.index and "청량리역" not in coords.index
    t = g.times_from("청량리")
    assert t["회기"] == pytest.approx(2 / 1.11 * 2.22, rel=0.05)  # 2.22km × 서울교통공사 평균 분/km
    assert t["돌곶이"] > t["석계"]  # 새 1호선 석계 ↔ 기존 6호선 석계 환승으로 이어진다


def test_nearest_station_by_walk_uses_route_distance_and_falls_back():
    from kkc.commute.subway import nearest_station_by_walk
    coords = pd.DataFrame({"lat": [37.600, 37.590], "lon": [127.0, 127.0]}, index=["산너머", "회기"])
    by_road = lambda lat, lon, pts: [3000 if p[0] == 37.600 else 900 for p in pts]  # noqa: E731
    assert nearest_station_by_walk(37.597, 127.0, coords, by_road) == ("회기", pytest.approx(900 / 67))

    def offline(*_):
        raise OSError
    st, _ = nearest_station_by_walk(37.597, 127.0, coords, offline)
    assert st == "산너머"  # 경로를 못 받으면 직선 거리로


def test_add_osm_routes_matches_parenthesized_name():
    from kkc.commute.subway import add_osm_routes
    g = SubwayGraph()
    g.add_segment("8", "잠실", "석촌", 2)
    coords = pd.DataFrame({"lat": [37.5133, 37.5056], "lon": [127.1001, 127.1069]}, index=["잠실", "석촌"])
    coords = add_osm_routes(g, coords, {"8": [[["잠실(송파구청)", 37.5160, 127.1030], ["석촌", 37.5056, 127.1069]]]})
    assert "잠실(송파구청)" not in coords.index  # 400m 떨어져 있어도 괄호 뗀 이름이 같으면 같은 역


def test_commute_relax_finds_smallest_extra():
    from kkc.ranking.rank import commute_relax
    df = pd.DataFrame({"region": ["A", "B", "C"], "commute_min": [38, 41, 55]})
    assert commute_relax(df, max_commute=30) == {"extra_minutes": 10, "count": 1}  # 40분이면 A
    assert commute_relax(df, max_commute=35, steps=(5, 10)) == {"extra_minutes": 5, "count": 1}
    assert commute_relax(df, max_commute=5) is None  # 30분 늘려도 35분 < 38분
    assert commute_relax(df.iloc[:0], max_commute=30) is None
