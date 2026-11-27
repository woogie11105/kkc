"""공공데이터 → 앱이 쓰는 파일 만들기.

    python scripts/build_data.py --rent data/raw/rent_*.csv --dist data/raw/역간거리.csv \
        --transfer data/raw/환승.csv --coords data/raw/역좌표.csv --stores data/raw/상가_서울.csv

만드는 것:
  models/price.joblib          월세 GBR 모델 + 검증 MAE + 학습 데이터 기준월
  models/subway.pkl            지하철 그래프 (앱에서 통학역 바꾸면 즉시 다익스트라)
  models/transit.pkl           버스+지하철 그래프 (--bus-* 를 줬을 때)
  data/processed/stations.csv  역명, 위도, 경도 (지도·목적지 근처 역 찾기)
  data/processed/regions.csv   region, pred_rent, amenity_cnt, 대표역 좌표
  data/processed/deals.csv     구 × 원룸/투룸 × 월세/전세 보증금·월세 통계(최근 12개월)
  data/processed/dongs.csv     gu, dong, 동 위치(상가 좌표 중앙값), 편의시설 수, 가까운 역·도보 분, AI 예측 시세
  data/processed/deals_dong.csv  동 × 원룸/투룸 × 월세/전세 통계
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kkc.commute.subway import (GU_STATION, add_osm_routes, load_seoul_metro, load_station_coords,  # noqa: E402
                                nearest_station)
from kkc.commute.transit import TransitGraph, load_bus  # noqa: E402
from kkc.price.model import load_rent, region_deal_stats, region_rents, train  # noqa: E402
from kkc.ranking.rank import dong_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rent", nargs="+", required=True, help="국토부 전월세 실거래가 CSV (여러 개 가능)")
ap.add_argument("--dist", required=True, help="서울교통공사 역간거리 및 소요시간 CSV")
ap.add_argument("--transfer", required=True, help="서울교통공사 환승역거리 소요시간 정보 CSV")
ap.add_argument("--coords", required=True, help="서울교통공사 1_8호선 역사 좌표(위경도) CSV")
ap.add_argument("--stores", required=True, help="소상공인시장진흥공단 상가(상권)정보 서울 CSV")
ap.add_argument("--bus-routes", help="서울시 버스노선별 정류소정보 xlsx (셋 다 주면 버스+지하철 통학 그래프)")
ap.add_argument("--bus-speed", help="TOPIS 노선별 정류장 구간 평균 운행시간 CSV (tpss_route_section_speedh_*)")
ap.add_argument("--bus-turn", help="TOPIS 노선별 정류장별 버스 운행횟수 CSV (tpss_sta_route_hturn_*)")
args = ap.parse_args()

(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "data/processed").mkdir(parents=True, exist_ok=True)

deals = pd.concat([load_rent(p) for p in args.rent], ignore_index=True)
model, mae, band = train(deals)
period = f"{deals.contract_ym.min()}~{deals.contract_ym.max()}"
joblib.dump({"model": model, "mae": mae, "band": band, "period": period, "latest_ym": int(deals.contract_ym.max())},ROOT / "models/price.joblib")
print(f"월세 모델: 거래 {len(deals):,}건 ({period}), 검증 MAE {mae:.1f}만원")

graph = load_seoul_metro(args.dist, args.transfer)
coords = load_station_coords(args.coords)
osm = ROOT / "data/processed/subway_map.json"  # scripts/fetch_subway_map.py — 1호선 코레일 구간(회기·용산 등)·9호선 보강
if osm.exists():
    n = len(coords)
    coords = add_osm_routes(graph, coords, json.loads(osm.read_text(encoding="utf-8"))["routes"])
    print(f"OSM 노선으로 서울교통공사 파일에 없는 역 {len(coords) - n}곳 추가 (역간 시간은 거리로 추정)")
else:
    print("⚠ data/processed/subway_map.json 없음 → 1호선 코레일 구간·9호선 역 빠짐 (python scripts/fetch_subway_map.py 먼저)")
with open(ROOT / "models/subway.pkl", "wb") as f:
    pickle.dump(graph, f)
if args.bus_routes and args.bus_speed and args.bus_turn:
    stops, ride_min, wait_min = load_bus(args.bus_routes, args.bus_speed, args.bus_turn)
    transit = TransitGraph(graph, coords)
    transit.add_bus(stops, ride_min, wait_min)
    transit.link_walks()
    with open(ROOT / "models/transit.pkl", "wb") as f:
        pickle.dump(transit, f)
    print(f"버스: 노선 {stops.route.nunique()}개, 정류장 {stops.stop.nunique():,}곳 (실측 구간 {len(ride_min):,}/{len(stops):,}) "
          f"→ 버스+지하철 그래프 노드 {len(transit.adj):,}개")
else:
    print("버스 파일(--bus-routes/--bus-speed/--bus-turn)이 없어 통학은 지하철만")
coords.to_csv(ROOT / "data/processed/stations.csv")
no_xy = sorted({n for n, _ in graph.adj} - set(coords.index))
print(f"역 좌표: {len(coords)}곳" + (f", 그래프 역 중 좌표 없음 {no_xy}" if no_xy else ""))
missing = [s for s in GU_STATION.values() if not graph.nodes_of(s)]
print(f"지하철: 역 {len({n for n, _ in graph.adj}):,}곳" + (f", 대표역 이름 불일치 {missing}" if missing else ""))

dongs = dong_table(args.stores)
regions = pd.DataFrame({"pred_rent": region_rents(model, deals), "amenity_cnt": dongs.groupby("gu").amenity_cnt.sum()})
stats = region_deal_stats(deals).reset_index()
stats.to_csv(ROOT / "data/processed/deals.csv", index=False)
print(f"deals.csv: 구×방×계약 {len(stats)}칸 (최근 12개월 ~{deals.contract_ym.max()})")
regions["station"] = regions.index.map(GU_STATION)
regions = regions.join(coords, on="station").rename_axis("region").reset_index().dropna()
regions.to_csv(ROOT / "data/processed/regions.csv", index=False)
print(f"regions.csv: {len(regions)}개 구")

# 동: 동 위치에서 가장 가까운 역 + 도보 분 → 앱에서 통학역까지 지하철 시간만 더하면 된다
dongs["pred_rent"] = region_rents(model, deals, by=("gu", "dong"))
dongs = dongs.dropna(subset=["pred_rent"])  # 전월세 거래가 없는 동(상가만 있는 곳)은 뺀다
dongs["station"], dongs["walk_min"] = zip(*(nearest_station(la, lo, coords) for la, lo in zip(dongs.lat, dongs.lon)))
dongs.round({"walk_min": 1}).reset_index().to_csv(ROOT / "data/processed/dongs.csv", index=False)
dstats = region_deal_stats(deals, by=("gu", "dong")).reset_index()
dstats.to_csv(ROOT / "data/processed/deals_dong.csv", index=False)
print(f"dongs.csv: {len(dongs)}개 동 (가까운 역까지 도보 중앙값 {dongs.walk_min.median():.0f}분), deals_dong.csv: {len(dstats)}칸")
