"""지하철 그래프 + 다익스트라로 통학 시간 계산.

데이터: 서울교통공사 역간거리·운행시간(1~8호선), 환승역거리 소요시간(환승역 145곳).
노드 = (역명, 호선). 같은 역의 다른 호선끼리는 환승 시간으로 연결한다.
"""
from __future__ import annotations

import heapq
import re
from collections import defaultdict

import pandas as pd

from kkc.data import col, read_table

Node = tuple[str, str]  # (역명, 호선)

# 자치구 → 대표역(구청·중심 상권 근처, 서울교통공사 1~8호선 구간). 지역 통학 시간 = 대표역까지 시간.
# 지선 첫 역 → 실제로 갈라지는 역. 역간 파일은 지선을 본선 끝에 이어 적어서 그대로 이으면 틀린다.
BRANCH_FROM = {("2", "용답"): "성수", ("2", "도림천"): "신도림", ("5", "둔촌동"): "강동"}
# 같은 역인데 파일·호선마다 이름이 다른 곳 (4호선 총신대입구 = 7호선 이수, 좌표 파일의 옛 역명 뚝섬유원지 = 자양).
ALIAS = {"이수": "총신대입구", "신내역": "신내", "뚝섬유원지": "자양"}
# shortcut: 구 안의 역이 여러 개여도 한 역만 봄. 구별 역 목록 데이터가 생기면 구 안 역들의 최솟값으로.
GU_STATION = {
    "종로구": "종로3가", "중구": "시청", "용산구": "삼각지", "성동구": "왕십리", "광진구": "건대입구",
    "동대문구": "청량리", "중랑구": "상봉", "성북구": "성신여대입구", "강북구": "미아사거리", "도봉구": "창동",
    "노원구": "노원", "은평구": "연신내", "서대문구": "신촌", "마포구": "합정", "양천구": "오목교",
    "강서구": "발산", "구로구": "신도림", "금천구": "가산디지털단지", "영등포구": "영등포구청", "동작구": "총신대입구",
    "관악구": "서울대입구", "서초구": "교대", "강남구": "강남", "송파구": "잠실", "강동구": "천호",
}


def _minutes(v) -> float:
    """'2:30'(분:초) 또는 2.5 → 분."""
    s = str(v).strip()
    if ":" in s:
        m, sec = s.split(":")[-2:]
        return int(m) + int(sec) / 60
    return float(s)


def _line(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)")[0]  # "2호선" / "2" → "2"


def _station(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\(.*\)", "", regex=True).str.strip().replace(ALIAS)  # "사당(대림)" → "사당"


def load_seoul_metro(dist_csv, transfer_csv) -> "SubwayGraph":
    """서울교통공사 '역간거리 및 소요시간' + '환승역거리 소요시간 정보' CSV → 그래프.
    역간 파일은 호선별로 역이 순서대로 나열되고, 각 행의 시간 = 앞 역에서 이 역까지."""
    g = SubwayGraph()
    d = read_table(dist_csv, must="역명")
    line, name = _line(d[col(d, "호선")]), _station(d[col(d, "역명")])
    t = d[col(d, "시간")].map(_minutes)
    for i in range(1, len(d)):
        if line.iat[i] == line.iat[i - 1]:
            prev = BRANCH_FROM.get((line.iat[i], name.iat[i]), name.iat[i - 1])
            g.add_segment(line.iat[i], prev, name.iat[i], t.iat[i])
    tr = read_table(transfer_csv, must="환승")
    a, b = _line(tr[col(tr, "호선")]), _line(tr[col(tr, "노선")])
    st, tt = _station(tr[col(tr, "역명")]), tr[col(tr, "시간")].map(_minutes)
    for i in range(len(tr)):
        if pd.isna(a.iat[i]) or pd.isna(b.iat[i]):  # 공항철도·경의중앙선 등 1~8호선 밖 노선은 그래프에 없음
            continue
        g.add_transfer(st.iat[i], a.iat[i], b.iat[i], tt.iat[i])
    return g


def load_station_coords(csv) -> pd.DataFrame:
    """서울교통공사 '1_8호선 역사 좌표(위경도)' CSV → 역명 index, lat/lon 컬럼. 환승역은 호선마다 적혀 있어 평균."""
    d = read_table(csv, must="위도")
    name = _station(d[col(d, "역명")]).replace({"서울": "서울역"})  # 좌표 파일은 '서울', 역간 파일은 '서울역'
    return pd.DataFrame({"station": name, "lat": d[col(d, "위도")], "lon": d[col(d, "경도")]}).groupby("station").mean()


WALK_M_PER_MIN = 67  # 시속 4km


def nearest_station(lat: float, lon: float, coords: pd.DataFrame) -> tuple[str, float]:
    """가장 가까운 역과 도보 분. 거리 = 직선 × 1.3(골목 우회). 서울 위도에서 위도 1° ≈ 111km, 경도 1° ≈ 88km."""
    m = (((coords.lat - lat) * 111_000) ** 2 + ((coords.lon - lon) * 88_000) ** 2) ** 0.5
    return m.idxmin(), float(m.min() * 1.3 / WALK_M_PER_MIN)


def region_commute(g: "SubwayGraph", dest_station: str) -> dict[str, float]:
    """자치구별 목적지(통학역)까지 분. 대표역이 그래프에 없으면 NaN."""
    t = g.times_from(dest_station)
    return {gu: t.get(s, float("nan")) for gu, s in GU_STATION.items()}


class SubwayGraph:
    def __init__(self) -> None:
        self.adj: dict[Node, list[tuple[Node, float]]] = defaultdict(list)

    def add_segment(self, line: str, a: str, b: str, minutes: float) -> None:
        """같은 호선의 인접 역 (양방향)."""
        u, v = (a, line), (b, line)
        self.adj[u].append((v, minutes))
        self.adj[v].append((u, minutes))

    def add_transfer(self, station: str, line_a: str, line_b: str, minutes: float) -> None:
        u, v = (station, line_a), (station, line_b)
        self.adj[u].append((v, minutes))
        self.adj[v].append((u, minutes))

    def nodes_of(self, station: str) -> list[Node]:
        return [n for n in self.adj if n[0] == station]

    def times_from(self, station: str) -> dict[str, float]:
        """출발역(모든 호선)에서 각 역까지 최소 소요시간(분)."""
        dist: dict[Node, float] = {}
        pq = [(0.0, n) for n in self.nodes_of(station)]
        heapq.heapify(pq)
        while pq:
            d, u = heapq.heappop(pq)
            if u in dist:
                continue
            dist[u] = d
            for v, w in self.adj[u]:
                if v not in dist:
                    heapq.heappush(pq, (d + w, v))
        best: dict[str, float] = {}
        for (name, _), d in dist.items():
            best[name] = min(d, best.get(name, float("inf")))
        return best

    def route(self, a: str, b: str) -> list[Node]:
        """a역 → b역 최단 경로 노드 [(역, 호선), ...]. 못 가면 []."""
        prev: dict[Node, Node | None] = {}
        pq = [(0.0, i, n, None) for i, n in enumerate(self.nodes_of(a))]
        heapq.heapify(pq)
        i = len(pq)  # 같은 시간일 때 노드끼리 비교하지 않게 하는 순번
        while pq:
            d, _, u, p = heapq.heappop(pq)
            if u in prev:
                continue
            prev[u] = p
            if u[0] == b:
                path = [u]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            for v, w in self.adj[u]:
                if v not in prev:
                    i += 1
                    heapq.heappush(pq, (d + w, i, v, u))
        return []



SEOUL_BBOX = (37.38, 126.70, 37.75, 127.27)  # 남, 서, 북, 동 — 앱 지도가 보여 주는 범위


def osm_subway_map(elements: list, bbox=SEOUL_BBOX) -> dict:
    """Overpass 결과(노선 relation `out geom` + 정차 위치 node) → 지도용 {lines: {호선: [선로 좌표열]}, stations: [{name, lat, lon}]}
    + 그래프 보강용 routes: {호선: [[[역명, lat, lon], ...정차 순서]]} (급행은 역을 건너뛰어서 뺀다).
    같은 선로(way)는 한 번만, 범위 밖 선로·역은 뺀다. 같은 이름의 정차 위치(방향별)는 평균 좌표 한 점으로."""
    s, w, n, e = bbox
    inside = lambda p: s <= p["lat"] <= n and w <= p["lon"] <= e  # noqa: E731
    lines, seen, stops = defaultdict(list), set(), defaultdict(list)
    for el in elements:
        if el["type"] == "relation":
            for m in el["members"]:
                geo = m.get("geometry") or []
                if m["type"] == "way" and m["role"] == "" and m["ref"] not in seen and any(map(inside, geo)):
                    seen.add(m["ref"])
                    lines[el["tags"]["ref"]].append([[round(p["lat"], 5), round(p["lon"], 5)] for p in geo])
        elif el["type"] == "node" and "name" in el.get("tags", {}) and inside(el):
            stops[el["tags"]["name"]].append((el["lat"], el["lon"]))
    stations = [{"name": k, "lat": round(sum(p[0] for p in v) / len(v), 5), "lon": round(sum(p[1] for p in v) / len(v), 5)}
                for k, v in stops.items()]
    names = {el.get("id"): el["tags"]["name"] for el in elements if el["type"] == "node" and "name" in el.get("tags", {})}
    routes = defaultdict(list)
    for el in elements:
        if el["type"] == "relation" and "급행" not in el["tags"].get("name", ""):
            seq = [[names[m["ref"]], round(m["lat"], 5), round(m["lon"], 5)] for m in el["members"]
                   if m["type"] == "node" and m["role"].startswith("stop") and m.get("ref") in names and "lat" in m]
            if len(seq) > 1 and seq not in routes[el["tags"]["ref"]]:
                routes[el["tags"]["ref"]].append(seq)
    return {"lines": dict(lines), "stations": stations, "routes": dict(routes)}


def _km(lat1, lon1, lat2, lon2):
    return (((lat1 - lat2) * 111) ** 2 + ((lon1 - lon2) * 88) ** 2) ** 0.5  # 서울 위도: 위도 1° ≈ 111km, 경도 1° ≈ 88km


TRANSFER_MIN = 4  # 새로 붙인 역의 환승 시간 (서울교통공사 환승 파일에 없는 곳)


def add_osm_routes(g: SubwayGraph, coords: pd.DataFrame, routes: dict, same_m: float = 300) -> pd.DataFrame:
    """서울교통공사 파일에 없는 구간(1호선 코레일 구간 회기·용산 등, 9호선)을 OSM 정차 순서로 그래프에 붙이고 늘어난 역 좌표를 돌려준다.
    같은 역 판단은 이름이 아니라 위치(same_m 안)로 — '서울'/'서울역'처럼 표기가 달라도 묶인다.
    이미 그래프에 있는 역끼리의 구간은 건드리지 않는다 (공식 시간이 우선)."""
    # shortcut: 새 구간 시간 = 거리 × 서울교통공사 구간 평균 분/km 추정. 코레일·9호선 역간 시간 데이터가 생기면 그걸로.
    per_km = pd.Series([w / _km(*coords.loc[a], *coords.loc[b]) for (a, la), nb in g.adj.items() for (b, lb), w in nb
                        if la == lb and a in coords.index and b in coords.index and _km(*coords.loc[a], *coords.loc[b]) > 0]).median()
    coords, official = coords.copy(), set(g.adj)  # 처음부터 있던 (역, 호선)

    def match(name, lat, lon):
        d = _km(coords.lat, coords.lon, lat, lon)
        base = re.sub(r"\s*\(.*?\)|역$", "", name)  # '잠실(송파구청)' → '잠실'
        if base in d.index and d[base] < 1:
            return base
        if len(d) and d.min() * 1000 <= same_m:
            return d.idxmin()
        coords.loc[name] = [lat, lon]
        return name

    for line, seqs in routes.items():
        for seq in seqs:
            ids = [match(*st) for st in seq]
            for a, b in zip(ids, ids[1:]):
                for x in (a, b):
                    if (x, line) not in g.adj:
                        for other in g.nodes_of(x):
                            g.add_transfer(x, line, other[1], TRANSFER_MIN)
                if a != b and not ((a, line) in official and (b, line) in official):
                    g.add_segment(line, a, b, round(_km(*coords.loc[a], *coords.loc[b]) * per_km, 1))
    return coords


def nearest_station_by_walk(lat: float, lon: float, coords: pd.DataFrame, route, k: int = 6) -> tuple[str, float]:
    """직선으로 가까운 역 k곳 중 실제 도보 경로가 가장 짧은 역과 도보 분. 산·강·철길을 돌아가는 걸 반영한다.
    route(lat, lon, [(lat, lon), ...]) → 각 역까지 도보 m. 경로를 못 받으면(OSError) 직선 거리 방식으로."""
    cand = coords.loc[_km(coords.lat, coords.lon, lat, lon).nsmallest(k).index]
    try:
        m = pd.Series(route(lat, lon, list(zip(cand.lat, cand.lon))), index=cand.index, dtype=float)
    except (OSError, ValueError, KeyError):
        return nearest_station(lat, lon, coords)
    return m.idxmin(), float(m.min() / WALK_M_PER_MIN)
