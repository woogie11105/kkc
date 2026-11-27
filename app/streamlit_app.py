"""웹 데모 (Streamlit). 실행: streamlit run app/streamlit_app.py

직장·학교 입력 → 가장 가까운 역 → 자치구별 통학 시간 + 원룸 보증금·월세 → 추천 순위와 서울 지도.
scripts/build_data.py 가 만든 파일을 쓰고, 없으면 예시 데이터로 동작한다.
  data/processed/regions.csv, data/processed/stations.csv
  models/price.joblib, models/subway.pkl, models/fov.pt (scripts/train_fov.py)

화면은 다방(dabangapp.com) 데스크톱 구조를 따른다: 위 헤더(로고·검색·내비) + 왼쪽 화면 레일 + 드롭다운 필터 줄
+ 카드 목록 + 지도. 지도는 한글 라벨(원 + 이름표 마커)을 위해 pydeck 대신 Leaflet(HTML)으로 그린다.
"""
import base64
import html
import io
import json
import pickle
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kkc.commute.subway import nearest_station_by_walk, region_commute  # noqa: E402
from kkc.price.model import BTYPES, ROOMS, band_for, converted_rent, price_gap  # noqa: E402
from kkc.ranking.rank import commute_relax, score_regions, tradeoff_suggestions  # noqa: E402
from kkc.ui import SORTS, gap_label, pick_photo, sort_regions, tradeoff_lines  # noqa: E402
from kkc.vision.fov import (STD_FOV_DEG, crop_ratio, load_fov_model, predict_fov,  # noqa: E402
                            restore_standard_view, wide_angle_flag)

REGIONS, STATIONS, DEALS, DONGS, DEALS_DONG, PRICE, SUBWAY, FOV = (ROOT / p for p in (
    "data/processed/regions.csv", "data/processed/stations.csv", "data/processed/deals.csv", "data/processed/dongs.csv",
    "data/processed/deals_dong.csv", "models/price.joblib", "models/subway.pkl", "models/fov.pt"))
TRANSIT = ROOT / "models/transit.pkl"  # 버스+지하철 그래프 (build_data.py --bus-*), 없으면 지하철만
PHOTOS = ROOT / "data/raw/listing_photos"  # scripts/unsplash_listings.py 가 받은 예시 매물 사진 (없으면 일러스트)
MIN_DEALS = 10  # 동 단위는 최근 12개월 거래가 이보다 적으면 통계가 들쭉날쭉해서 뺀다
DONG_SHOWN = 15  # 동 단위 목록·지도에 보여줄 추천 상위 개수 (400곳을 다 그리면 지도가 덮인다)
SAMPLE = pd.DataFrame({
    "region": ["관악구", "동작구", "광진구", "마포구", "성북구", "강남구"],
    "pred_rent": [55, 60, 62, 68, 52, 102],
    "commute_min": [35, 28, 40, 30, 45, 20],
    "amenity_cnt": [120, 95, 110, 140, 80, 200],
    "deposit_med": [500, 1000, 1000, 1000, 500, 1000], "rent_med": [45, 50, 52, 58, 43, 70],
})  # 예시 값 — 실제 데이터로 교체
EXAMPLE_LISTINGS = pd.DataFrame({  # 매물 비교 첫 화면용 예시 입력 (사용자가 표에서 바꾼다)
    "name": ["매물 A", "매물 B", "매물 C"], "gu": ["동대문구", "중랑구", "중랑구"], "dong": ["회기동", "상봉동", "상봉동"],
    "area_m2": [20.0, 18.0, 18.0], "floor": [3, 2, 3], "build_year": [2015, 2009, 2010],
    "btype": ["단독다가구", "연립다세대", "연립다세대"], "deposit": [1000, 500, 500], "rent": [62, 40, 58]})
SEOUL_BOX = "126.76,37.70,127.19,37.42"  # 위치 검색을 서울 안으로 한정 (서,북,동,남)
PAGES = ["동네 찾기", "매물 비교", "사진 화각"]
PAGE_ICON = {"동네 찾기": ":material/home:", "매물 비교": ":material/view_column:", "사진 화각": ":material/photo_camera:"}
BLUE = "#326CF9"
LINE_COLOR = {"1": "#0052A4", "2": "#00A84D", "3": "#EF7C1C", "4": "#00A5DE",  # 서울 지하철 호선 표준색
              "5": "#996CAC", "6": "#CD7C2F", "7": "#747F00", "8": "#E6186C", "9": "#BDB092"}
SUBWAY_MAP = ROOT / "data/processed/subway_map.json"  # scripts/fetch_subway_map.py (OSM 실제 선로 + 역 이름)
FILTER_KEYS = ["level", "kind", "room", "dep_m", "rent_원룸", "rent_투룸", "dep_j_원룸", "dep_j_투룸", "commute", "w",
               "order2", "pick3"]

st.set_page_config(page_title="KKC 자취방 추천", page_icon="🏠", layout="wide")
ss = st.session_state
for k in FILTER_KEYS + [k for k in ss if k.startswith(("sort_", "desc_"))]:  # 화면을 옮겨도 필터 값이 사라지지 않게 (그리지 않은 위젯 값은 Streamlit이 지운다)
    if k in ss:
        ss[k] = ss[k]
ss.setdefault("page", PAGES[0])
ss.setdefault("listings_df", EXAMPLE_LISTINGS)

# 다방 데스크톱 치수: 헤더 80px, 레일 80px, 필터 줄 60px, 목록 400px. 흰 바탕 · #E1E1E1 선 · #326CF9 강조 · 보라 태그.
st.html(f"""<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable.min.css');
.stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button p, .stApp li, .stApp td, .stApp th
  {{ font-family: 'Pretendard Variable', Pretendard, -apple-system, sans-serif; letter-spacing: -.2px; }}
header[data-testid="stHeader"], [data-testid="stToolbar"] {{ display: none; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
.stMain [data-testid="stVerticalBlock"] {{ gap: 0; }}
[data-testid="stHorizontalBlock"] {{ gap: 0 !important; }}
/* 헤더 */
.st-key-hd {{ height: 80px; border-bottom: 1px solid #E1E1E1; padding: 0 24px; align-items: center; gap: 0 !important; }}
.db-logo {{ font-size: 30px; font-weight: 900; color: {BLUE}; letter-spacing: -1.5px; width: 92px; }}
.st-key-search {{ width: 340px !important; flex: none !important; height: 40px; border: 1px solid #DADADA; border-radius: 20px;
  padding: 0 16px; align-items: center; gap: 6px !important; }}
.st-key-search > * {{ flex: 0 0 auto !important; width: auto !important; }}
.st-key-search [data-testid="stElementContainer"], .st-key-search [data-testid="stTextInput"],
.st-key-search [data-testid="stTextInput"] > div, .st-key-search [data-testid="stTextInputRootElement"] {{ width: auto !important; }}
.st-key-search input {{ field-sizing: content; min-width: 40px; max-width: 180px; }}
.st-key-search [data-baseweb="input"], .st-key-search [data-baseweb="base-input"] {{ border: none !important; background: transparent !important; }}
.st-key-search input {{ padding: 0 !important; font-size: 14px; font-weight: 500; color: #222; background: transparent; }}
.db-hint {{ font-size: 14px; color: #A0A0A0; white-space: nowrap; }}
.st-key-nav {{ margin-left: auto; width: auto !important; flex: none !important; }}
.st-key-nav [data-testid="stButtonGroup"] > div {{ gap: 40px; }}
.st-key-nav button {{ border: none !important; background: none !important; box-shadow: none !important; padding: 0 !important; min-height: 0 !important; }}
.st-key-nav button p {{ font-size: 16px !important; color: #222; }}
.st-key-nav button[aria-checked="true"] p {{ color: {BLUE} !important; font-weight: 500; }}
.st-key-hd > *, .st-key-fb > * {{ flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }}
.st-key-hd > :nth-child(3) {{ margin-left: auto; }}
.st-key-search [data-testid="stTextInputRootElement"], .st-key-search [data-testid="stTextInputRootElement"] * {{
  background: transparent !important; border: none !important; box-shadow: none !important; }}
.db-ico {{ font-size: 14px; opacity: .55; }}
.st-key-search input {{ min-width: 150px; }}
/* 왼쪽 레일 */
.st-key-rail {{ width: 80px; min-height: calc(100vh - 80px); border-right: 1px solid #E1E1E1; padding-top: 12px; }}
.st-key-rail [role="radiogroup"] {{ flex-direction: column !important; gap: 6px; align-items: center; }}
.st-key-rail [data-testid="stRadioOption"] {{ width: 64px; height: 68px; border-radius: 6px; border: 1px solid transparent;
  display: flex; align-items: center; justify-content: center; margin: 0 !important; padding: 0 !important; cursor: pointer; }}
.st-key-rail [data-testid="stRadioOption"] > div > div:first-child {{ display: none; }}
.st-key-rail [data-testid="stRadioOption"] p {{ display: flex; flex-direction: column; align-items: center; gap: 4px;
  font-size: 12px; font-weight: 700; color: #656565; text-align: center; line-height: 1.1; }}
.st-key-rail [data-testid="stRadioOption"] p span[role="img"] {{ font-size: 26px !important; line-height: 1; }}
.st-key-rail [data-testid="stRadioOption"][data-selected="true"] {{ background: #F0F5FF; border-color: #C6D6FD; }}
.st-key-rail [data-testid="stRadioOption"][data-selected="true"] p {{ color: {BLUE}; }}
/* 드롭다운 필터 줄 */
.st-key-fb {{ height: 60px; border-bottom: 1px solid #E1E1E1; padding: 0 16px; align-items: center; gap: 8px !important; }}
.st-key-fb [data-testid="stPopover"] {{ width: auto !important; }}
.st-key-fb [data-testid="stPopover"] button {{ height: 36px; min-height: 36px; border: 1px solid #DADADA; border-radius: 4px; padding: 0 12px;
  background: #fff; box-shadow: none; }}
.st-key-fb [data-testid="stPopover"] button p {{ font-size: 14px; color: #222; }}
.st-key-fb [data-testid="stPopover"] button [data-testid="stIconMaterial"] {{ display: none; }}
.st-key-fb [data-testid="stPopover"] button p {{ display: flex; align-items: center; gap: 7px; line-height: 1; }}
.st-key-fb [data-testid="stPopover"] button p::after {{ content: ""; width: 6px; height: 6px; margin-top: -4px; flex: none;
  border-right: 1.6px solid currentColor; border-bottom: 1.6px solid currentColor; transform: rotate(45deg); }}
.st-key-fb [data-testid="stPopover"] button[aria-expanded="true"] p::after {{ margin-top: 3px; transform: rotate(-135deg); }}
.st-key-fb [class*="st-key-on_"] {{ width: auto !important; }}
.st-key-fb [class*="st-key-on_"] button {{ border-color: #222 !important; }} .st-key-fb [class*="st-key-on_"] button p {{ font-weight: 700; }}
[data-testid="stPopoverBody"] {{ min-width: 320px; }}
.st-key-pop_add [data-testid="stPopoverBody"], div[data-testid="stPopoverBody"]:has(.st-key-editor) {{ min-width: 980px; }}
/* 목록 */
.st-key-list {{ width: 400px; border-right: 1px solid #E1E1E1; }}
.db-pills {{ padding: 16px 16px 8px; display: flex; gap: 6px; }}
.db-pl {{ font-size: 14px; border-radius: 18px; padding: 7px 15px; border: 1px solid #DADADA; color: #656565; background: #fff; }}
.db-pl.on {{ background: #222; color: #fff; border-color: #222; font-weight: 700; }}
.st-key-sort, .st-key-sort2 {{ padding: 16px 16px 8px; }}
.st-key-sortdir {{ padding: 0 16px 8px; }}
.st-key-sort button, .st-key-sortdir button, .st-key-sort2 button {{ border-radius: 18px !important; min-height: 34px; padding: 0 15px; border: 1px solid #DADADA; }}
.st-key-sort button p, .st-key-sortdir button p, .st-key-sort2 button p {{ font-size: 14px !important; color: #656565; }}
.st-key-sort button[aria-checked="true"], .st-key-sortdir button[aria-checked="true"], .st-key-sort2 button[aria-checked="true"] {{ background: #222 !important; border-color: #222 !important; }}
.st-key-sort button[aria-checked="true"] p, .st-key-sortdir button[aria-checked="true"] p, .st-key-sort2 button[aria-checked="true"] p {{ color: #fff !important; font-weight: 700; }}
.st-key-sort [data-testid="stButtonGroup"] > div, .st-key-sortdir [data-testid="stButtonGroup"] > div, .st-key-sort2 [data-testid="stButtonGroup"] > div {{ gap: 6px; }}
.db-it {{ display: flex; gap: 16px; padding: 16px; border-bottom: 1px solid #EEE; }}
.db-sq {{ width: 140px; height: 140px; flex: none; border-radius: 4px; background: #F0F5FF; color: {BLUE}; display: flex;
  flex-direction: column; align-items: center; justify-content: center; font-weight: 800; font-size: 34px; line-height: 1.15; }}
.db-sq small {{ font-size: 13px; font-weight: 600; color: #656565; }}
.db-room {{ flex: none; border-radius: 4px; display: block; }}
.db-pic {{ width: 140px; height: 140px; object-fit: cover; border-radius: 4px; display: block; }}
.db-cr {{ display: block; width: 140px; margin-top: 4px; font-size: 11px; color: #979797 !important; text-decoration: none;
  overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }}
.db-ph {{ width: 140px; height: 140px; flex: none; border-radius: 4px; background: #F3F3F3; color: #B0B0B0; display: flex;
  flex-direction: column; align-items: center; justify-content: center; font-size: 12px; gap: 6px; }}
.db-pr {{ font-size: 20px; font-weight: 700; color: #222; margin: 2px 0 6px; }}
.db-l1 {{ font-size: 14px; color: #656565; line-height: 24px; }} .db-l2 {{ font-size: 14px; color: #979797; line-height: 24px; }}
.db-tg {{ display: inline-block; margin: 6px 4px 0 0; font-size: 12px; font-weight: 700; color: #3E26FD; background: #F1EFFF;
  padding: 3px 7px; border-radius: 3px; }}
.db-tg.good {{ color: #12A150; background: #EAF7EF; }} .db-tg.bad {{ color: #E42939; background: #FFF0F0; }}
.db-tg.neutral {{ color: #656565; background: #F3F3F3; }} .db-tg.warn {{ color: #B26A00; background: #FFF4DB; }}
.db-sortby {{ font-size: 13px; color: #656565; padding: 0 16px 6px; }}
.db-sortby b {{ color: #222; }}
.db-tip {{ font-size: 13px; color: #333; background: #F1F6FF; border-radius: 8px; padding: 10px 14px; margin: 8px 12px; line-height: 1.6; }}
.db-tip small {{ color: #979797; }}
.db-note {{ font-size: 12px; color: #979797; line-height: 1.6; padding: 14px 16px; }}
/* 사진 화각 팝업 */
.st-key-dim {{ background: rgba(0,0,0,.45); min-height: calc(100vh - 80px); padding: 32px 110px; }}
.st-key-modal {{ background: #fff; border-radius: 6px; overflow: hidden; }}
.db-ph2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }} .db-ph2 div {{ position: relative; height: 400px; overflow: hidden; background: #F3F3F3; }}
.db-ph2 img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
.db-ph2 span {{ position: absolute; left: 14px; top: 14px; font-size: 13px; font-weight: 700; color: #fff; background: rgba(0,0,0,.6);
  border-radius: 3px; padding: 4px 9px; }} .db-ph2 .o {{ background: {BLUE}; }}
.db-md {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 28px; padding: 22px 28px 26px; }}
.db-md h2 {{ font-size: 26px; font-weight: 800; color: #222; margin: 0; }} .db-md .s {{ font-size: 14px; color: #656565; margin-top: 4px; }}
.db-tbl {{ width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 14px; }}
.db-tbl td {{ border-top: 1px solid #EEE; padding: 10px 4px; }} .db-tbl td:first-child {{ color: #656565; width: 130px; }}
.db-fv {{ border: 1px solid #E1E1E1; border-radius: 6px; padding: 16px 18px; }} .db-fv h3 {{ font-size: 16px; font-weight: 800; margin: 0; }}
.db-fv .d {{ display: flex; align-items: center; gap: 10px; margin-top: 8px; }} .db-fv .d b {{ font-size: 34px; color: #B26A00; }}
.db-fv p {{ font-size: 13px; color: #656565; line-height: 1.6; margin-top: 10px; }}
.st-key-pick {{ padding: 0 28px; }}
.st-key-tools {{ background: #F7F7F7; border-top: 1px solid #EEE; padding: 10px 28px; gap: 12px; }}
.st-key-tools > * {{ flex: 0 0 auto !important; width: auto !important; }} .st-key-tools > :last-child {{ flex: 1 1 auto !important; }}
.db-tl {{ font-size: 14px; font-weight: 700; color: #444; white-space: nowrap; }}
.st-key-tools [data-testid="stFileUploaderDropzone"] {{ padding: 6px 12px; background: #fff; border: 1px dashed #C9C9C9; border-radius: 4px; }}
.st-key-tools [data-testid="stFileUploaderDropzoneInstructions"] svg {{ display: none; }}
.db-ph2 svg {{ display: block; }}
.st-key-foot {{ padding: 16px 24px; }}
</style>""")


@st.cache_resource
def load():
    price = joblib.load(PRICE) if PRICE.exists() else None
    subway = pickle.load(open(SUBWAY, "rb")) if SUBWAY.exists() else None
    fov = load_fov_model(FOV) if FOV.exists() else None
    transit = pickle.load(open(TRANSIT, "rb")) if TRANSIT.exists() else None
    return price, subway, fov, transit


@st.cache_data(show_spinner="위치 찾는 중…")
def geocode(query: str):
    """장소 이름 → (위도, 경도, 주소). OpenStreetMap Nominatim(키 불필요). 못 찾거나 오프라인이면 None."""
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "limit": 1, "countrycodes": "kr", "viewbox": SEOUL_BOX, "bounded": 1})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "kkc-room-finder-demo"})
        hit = json.load(urllib.request.urlopen(req, timeout=8))
    except (OSError, ValueError):
        return None
    return (float(hit[0]["lat"]), float(hit[0]["lon"]), hit[0]["display_name"]) if hit else None


def walk_m(lat: float, lon: float, pts: list) -> list:
    """(lat, lon) → 각 역까지 실제 도보 경로 길이(m). OSM 도보 라우팅(routing.openstreetmap.de, 키 불필요)."""
    coords = ";".join(f"{lo},{la}" for la, lo in [(lat, lon), *pts])
    req = urllib.request.Request(f"https://routing.openstreetmap.de/routed-foot/table/v1/foot/{coords}?sources=0"
                                 "&annotations=distance", headers={"User-Agent": "kkc-room-finder-demo"})
    r = json.load(urllib.request.urlopen(req, timeout=8))
    if r.get("code") != "Ok" or None in r["distances"][0]:
        raise ValueError(r.get("code"))
    return r["distances"][0][1:]


@st.cache_data(show_spinner=False)
def walk_path(lat: float, lon: float, slat: float, slon: float) -> list | None:
    """목적지 → 역 실제 도보 경로 [[lat, lon], ...] (지도에 '가는 길'로). 못 받으면 None → 직선 점선."""
    req = urllib.request.Request(f"https://routing.openstreetmap.de/routed-foot/route/v1/foot/{lon},{lat};{slon},{slat}"
                                 "?overview=full&geometries=geojson", headers={"User-Agent": "kkc-room-finder-demo"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=8))
        return [[la, lo] for lo, la in r["routes"][0]["geometry"]["coordinates"]]
    except (OSError, ValueError, KeyError, IndexError):
        return None


@st.cache_data(show_spinner="가까운 역까지 걷는 길 찾는 중…")
def walk_station(lat: float, lon: float, coords: pd.DataFrame) -> tuple[str, float]:
    """직선이 아니라 실제 걷는 길로 가장 가까운 역 (산·철길 너머 역을 고르지 않게). 오프라인이면 직선 거리로."""
    return nearest_station_by_walk(lat, lon, coords, walk_m)


@st.cache_data(show_spinner="버스+지하철 통학 시간 계산 중…")
def bus_commute(lat: float, lon: float, station: str, walk: float, origins: pd.DataFrame) -> pd.Series:
    """origins(lat, lon, station, walk_min) 각각에서 목적지까지 버스+지하철 최소 분. 버스 그래프가 없으면 NaN.
    목적지 쪽: 실제 걷는 길로 닿는 정류장·역 + 고른 통학역(도보 walk). 출발 쪽: 걸어서 닿는 정류장·역 + 그 동의 가까운 역."""
    if transit is None:
        return pd.Series(float("nan"), index=origins.index)
    egress = transit.access(lat, lon, walk_m)
    egress.update({n: walk for n in subway.nodes_of(station)})
    to = transit.times_to(egress)
    return pd.Series([transit.time_from(o.lat, o.lon, to, {n: o.walk_min for n in subway.nodes_of(o.station)})
                      for o in origins.itertuples()], index=origins.index)


def with_bus(df: pd.DataFrame, subway_min: pd.Series, mixed: pd.Series) -> pd.DataFrame:
    """통학 = 지하철만 / 버스 섞기 중 빠른 쪽. bus = 버스를 타야 더 빠른 곳."""
    return df.assign(commute_min=pd.concat([subway_min, mixed], axis=1).min(axis=1).round(),
                     bus=(mixed < subway_min - 0.5).fillna(False))


def won(v: float) -> str:
    """만원 단위 → '1,000만' / '2.5억'. 1000만 이상은 100만 단위로 반올림(실제 계약값 6,793만 → 6,800만)."""
    if v >= 10000:
        return f"{v / 10000:.1f}억".replace(".0억", "억")
    return f"{round(v, -2) if v >= 1000 else v:,.0f}만"


def deposit_text(r) -> str:
    """보증금은 1000만에 몰려 있어 한 값보다 '보통 범위(가운데 50%)'가 구별 차이를 보여준다."""
    if "deposit_lo" not in r or r.deposit_lo == r.deposit_hi:
        return won(r.deposit_med)
    return f"{won(r.deposit_lo)}~{won(r.deposit_hi)}"


def price_text(r) -> str:
    return f"전세 {deposit_text(r)}" if r.rent_med == 0 else f"보증금 {deposit_text(r)} / 월세 {r.rent_med:,.0f}만"


def esc(v) -> str:
    return html.escape(str(v))


def map_css(height: int) -> str:
    """두 지도(카카오·Leaflet)가 같이 쓰는 마커·이름표 모양."""
    return f"""<style>
html,body,#m{{height:{height}px;margin:0;font-family:'Pretendard Variable',Pretendard,'Malgun Gothic',sans-serif}}
.mk{{display:flex;align-items:center;white-space:nowrap;font-family:inherit}}
.mk .c{{width:44px;height:44px;border-radius:22px;background:{BLUE};color:#fff;font-weight:700;font-size:15px;display:flex;
  align-items:center;justify-content:center;border:2px solid #fff;box-shadow:0 1px 4px rgba(0,0,0,.2);position:relative;z-index:2;
  letter-spacing:-.5px}}
.mk .n{{margin-left:-14px;padding:5px 14px 5px 22px;background:#fff;border:1.4px solid {BLUE};border-radius:15px;font-size:14px;color:#222}}
.mk.top .n{{border-width:2px}} .mk.off .c{{background:#A3A3A3}} .mk.off .n{{border-color:#A3A3A3;color:#888}}
.dst{{display:flex;align-items:flex-start}} .dst svg{{flex:none;filter:drop-shadow(0 1px 2px rgba(0,0,0,.3))}}
.mk .c,.mk .n{{flex:none}}
.dst b{{background:#222;color:#fff;font-size:13px;font-weight:700;padding:4px 9px;border-radius:4px;margin:2px 0 0 4px;white-space:nowrap}}
.stn{{display:flex;align-items:center;gap:5px;white-space:nowrap;font-size:13px;font-weight:700;color:#0052A4}}
.stn i{{width:12px;height:12px;border-radius:6px;background:#0052A4;border:3px solid #fff;box-shadow:0 0 2px rgba(0,0,0,.4)}}
.note{{position:absolute;left:16px;bottom:16px;z-index:1000;background:#fff;border:1px solid #DADADA;border-radius:4px;padding:9px 13px;
  font-size:12px;color:#656565}}
.leaflet-bar a{{width:40px!important;height:40px!important;line-height:40px!important;font-size:20px!important;color:#444!important}}
.leaflet-bar{{border:1px solid #DADADA!important;border-radius:4px!important;box-shadow:none!important}}
.leaflet-tooltip{{font-family:inherit;font-size:12.5px;border-radius:4px}}
.leaflet-tile-pane{{filter:saturate(.35) brightness(1.04)}}  /* 배경을 흐리게 해서 호선 색이 눈에 띄게 */
.sl{{position:absolute;left:7px;top:-8px;white-space:nowrap;font-size:11.5px;font-weight:700;color:#333;
  text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff}}

</style>"""


@st.cache_data
def subway_map() -> dict:
    """지도에 깔 호선 선로 [{line, color, segs}] + 역 [{name, lat, lon}]. 파일을 아직 안 받았으면 빈 값."""
    if not SUBWAY_MAP.exists():
        return {"lines": [], "subway_stations": []}
    m = json.loads(SUBWAY_MAP.read_text(encoding="utf-8"))
    return {"lines": [{"line": k, "color": LINE_COLOR[k], "segs": v} for k, v in sorted(m["lines"].items()) if k in LINE_COLOR],
            "subway_stations": m["stations"]}


def map_data(points, dest, stations, note) -> str:
    # </script> 같은 문자열이 들어와도 스크립트 밖으로 못 나가게
    return json.dumps({"points": points, "dest": dest, "stations": list(stations), "note": note, **subway_map()},
                      ensure_ascii=False).replace("</", "<\\/")


def leaflet_map(points, dest, stations=(), note="", height=760):
    """다방식 지도. points: dict(lat, lon, value, name, tip, state[top|on|off]) — 파란 원(value) + 흰 이름표(name).
    배경은 OpenStreetMap 기본 지도(키 불필요, 한글 지명, 컬러). CARTO는 이제 키가 필요하고 Esri는 서울 데이터가 없어서 이걸 쓴다."""
    data = map_data(points, dest, stations, note)
    components.html(f"""
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable.min.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
{map_css(height)}<div id="m"></div><div class="note" id="note"></div>
<script>
const D={data};
const map=L.map('m',{{zoomControl:false,scrollWheelZoom:true,minZoom:10,maxBounds:[[37.38,126.70],[37.75,127.27]],maxBoundsViscosity:1}});  // 서울만
L.control.zoom({{position:'topright'}}).addTo(map);
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{attribution:'&copy; OpenStreetMap',maxZoom:19}}).addTo(map);
for(const l of D.lines) L.polyline(l.segs,{{color:'#fff',weight:8,opacity:.9,interactive:false}}).addTo(map);  // 흰 테두리 먼저
for(const l of D.lines) L.polyline(l.segs,{{color:l.color,weight:5,opacity:.95}}).bindTooltip(l.line+'호선',{{sticky:true}}).addTo(map);
const sl=L.layerGroup();  // 역 점 + 이름은 확대했을 때만 (네이버처럼, 다 펴면 글자가 겹친다)
const busy=new Set([...D.stations.map(x=>x.name),D.dest&&D.dest.stn&&D.dest.stn.name]);  // 파란 이름표가 이미 있는 역
for(const s of D.subway_stations){{
  if(busy.has(s.name+'역')) continue;
  L.circleMarker([s.lat,s.lon],{{radius:4,color:'#555',weight:1.5,fillColor:'#fff',fillOpacity:1,interactive:false}}).addTo(sl);
  L.marker([s.lat,s.lon],{{interactive:false,icon:L.divIcon({{className:'',iconSize:[0,0],html:`<div class="sl">${{s.name}}</div>`}})}}).addTo(sl);
}}
const showSl=()=>map.getZoom()>=13?sl.addTo(map):sl.remove();
const pts=[];
for(const s of D.stations){{
  L.marker([s.lat,s.lon],{{icon:L.divIcon({{className:'',iconSize:[120,18],iconAnchor:[9,9],html:`<div class="stn"><i></i>${{s.name}}</div>`}})}}).addTo(map);
}}
for(const p of D.points){{
  const ic=L.divIcon({{className:'',iconSize:[260,44],iconAnchor:[22,22],html:`<div class="mk ${{p.state}}"><span class="c">${{p.value}}</span><span class="n">${{p.name}}</span></div>`}});
  L.marker([p.lat,p.lon],{{icon:ic,zIndexOffset:p.state==='top'?1000:p.state==='on'?500:0}})
    .bindTooltip(p.tip,{{direction:'top',offset:[0,-24]}}).addTo(map);
  if(p.state!=='off') pts.push([p.lat,p.lon]);
}}
if(D.dest){{
  const pin=`<svg width="30" height="40" viewBox="0 0 30 40"><path d="M15 39C6 26 1 20 1 14A14 14 0 0 1 29 14C29 20 24 26 15 39Z" fill="#FF4B4B"/><circle cx="15" cy="14" r="5.5" fill="#fff"/></svg>`;
  L.marker([D.dest.lat,D.dest.lon],{{icon:L.divIcon({{className:'',iconSize:[220,42],iconAnchor:[15,39],html:`<div class="dst">${{pin}}<b>${{D.dest.label}}</b></div>`}}),zIndexOffset:2000}}).addTo(map);
  pts.push([D.dest.lat,D.dest.lon]);
  const s=D.dest.stn;
  if(s){{
    const walk=D.dest.path||[[D.dest.lat,D.dest.lon],[s.lat,s.lon]];  // 가는 길: 호선(색 실선)과 다르게 검은 동그라미 점선
    L.polyline(walk,{{color:'#fff',weight:10,dashArray:'0.1 11',lineCap:'round',interactive:false}}).addTo(map);
    L.polyline(walk,{{color:'#222',weight:6,dashArray:'0.1 11',lineCap:'round'}}).bindTooltip(`${{s.name}}까지 걷는 길`,{{sticky:true}}).addTo(map);
    if(!D.stations.some(x=>x.name===s.name)) L.marker([s.lat,s.lon],{{icon:L.divIcon({{className:'',iconSize:[120,18],iconAnchor:[9,9],html:`<div class="stn"><i></i>${{s.name}}</div>`}})}}).addTo(map);
    pts.push([s.lat,s.lon]);
  }}
}}
if(pts.length) map.fitBounds(pts,{{padding:[60,60],maxZoom:14}}); else map.setView([37.56,126.99],11);
showSl(); map.on('zoomend',showSl);  // 첫 화면이 잡힌 뒤에 건다 (그 전엔 렌더러가 없어 점 추가가 터진다)
if(D.note) document.getElementById('note').innerHTML=D.note; else document.getElementById('note').remove();
</script>""", height=height)


def img_b64(img: Image.Image, width=900) -> str:
    im = img.copy()
    im.thumbnail((width, width))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def sync_from_rail():
    ss.page = ss.nav_pg = ss.rail_pg


def sync_from_nav():
    if ss.nav_pg is None:  # 이미 고른 항목을 다시 누르면 선택이 풀리는 것을 막음
        ss.nav_pg = ss.page
    ss.page = ss.rail_pg = ss.nav_pg


price, subway, fov_model, transit = load()
ready = REGIONS.exists() and STATIONS.exists() and DEALS.exists() and subway
ss.rail_pg = ss.nav_pg = ss.page

# ── 헤더: 로고 · 검색(+ 가까운 역) · 내비 ──
with st.container(key="hd", horizontal=True):
    st.html('<div class="db-logo">kkc</div>')
    with st.container(key="search", horizontal=True):
        st.html('<span class="db-ico">🔍</span>')
        place = st.text_input("직장·학교", value="경희대학교", label_visibility="collapsed", key="place",
                              placeholder="직장·학교 입력 후 Enter")
        hint = st.empty()
    with st.container(key="nav"):
        st.segmented_control("화면", PAGES, key="nav_pg", on_change=sync_from_nav, label_visibility="collapsed")

rail_col, body = st.columns([1, 17], gap=None)
with rail_col, st.container(key="rail"):
    st.radio("화면", PAGES, key="rail_pg", on_change=sync_from_rail, label_visibility="collapsed",
             format_func=lambda p: f"{PAGE_ICON[p]} {p}")
page = ss.page

# 공통 조건 (필터 줄에서 바꾼다. 값은 session_state에 남아 화면을 옮겨도 유지)
level = ss.get("level", "구")
room = ss.get("room", "원룸")
kind = ss.get("kind", "월세")
max_commute = ss.get("commute", 45)
w = ss.get("w", 65)
if kind == "월세":
    max_deposit = ss.get("dep_m", 2000)
    max_rent = ss.get(f"rent_{room}", 60 if room == "원룸" else 90)
else:
    max_deposit = ss.get(f"dep_j_{room}", 25000 if room == "원룸" else 35000)
    max_rent = float("inf")


with body, st.container(key="fb", horizontal=True):
    if page == "동네 찾기":
        with st.container(key="on_level"), st.popover(f"{level} 단위"):
            level = st.radio("보기 단위", ["구", "동"], key="level", horizontal=True,
                             help="동: 같은 구 안에서도 동마다 다른 가격·통학 시간을 따로 봐요")
        with st.container(key="on_kind"), st.popover(f"{kind}"):
            kind = st.radio("계약", ["월세", "전세"], key="kind", horizontal=True)
        with st.popover(f"{room}"):
            room = st.radio("방", list(ROOMS), key="room", horizontal=True, format_func=lambda k: f"{k} ({ROOMS[k]})")
        if kind == "월세":
            with st.popover(f"보증금 ~{won(max_deposit)}"):
                max_deposit = st.slider("보증금 최대(만원)", 100, 10000, 2000, step=100, key="dep_m")
            with st.popover(f"월세 ~{max_rent:.0f}만"):
                max_rent = st.slider("월세 최대(만원)", 20, 200, 60 if room == "원룸" else 90, step=5, key=f"rent_{room}")
        else:
            with st.popover(f"전세금 ~{won(max_deposit)}"):
                max_deposit = st.slider("전세금 최대(만원)", 5000, 80000, 25000 if room == "원룸" else 35000, step=1000,
                                        key=f"dep_j_{room}", help="10000만원 = 1억")
        with st.container(key="on_commute"), st.popover(f"통학 {max_commute}분 이내"):
            max_commute = st.slider("통학·통근 최대(분)", 10, 90, 45, step=5, key="commute")
        with st.popover("추가필터"):
            w = st.slider("가까운 곳을 얼마나 중요하게?", 0, 100, 65, format="%d%%", key="w",
                          help="0% = 월세가 싼 곳만, 100% = 가까운 곳만")
            st.caption("← 월세가 싼 곳 · 가까운 곳 →")
    elif page == "매물 비교":
        with st.container(key="on_mine"), st.popover("내 후보 매물"):
            st.caption("매물 이름 · 보증금/월세")
            for _, li in ss.listings_df.iterrows():
                st.write(f"{li['name']} · {li['deposit']:,.0f}/{li['rent']:,.0f}")
        with st.popover(f"{room}"):
            room = st.radio("방", list(ROOMS), key="room", horizontal=True, format_func=lambda k: f"{k} ({ROOMS[k]})")
        with st.container(key="on_gap"), st.popover("시세 대비"):
            st.caption("같은 구·동·면적·층·건축년도·건물용도의 AI 예측 시세와 이 매물의 환산 월세"
                       "(월세 + 보증금×5.5%÷12)를 비교해요.")
        with st.popover(f"통학 {max_commute}분"):
            max_commute = st.slider("통학·통근 최대(분)", 10, 90, 45, step=5, key="commute")
        with st.popover("＋ 매물 추가"), st.container(key="editor"):
            st.caption("행을 추가·수정하면 바로 반영돼요. 동을 비우면 예측이 부정확할 수 있어요.")
            edited = st.data_editor(
                ss.listings_df, num_rows="dynamic", width=940, hide_index=True,
                column_config={"name": "매물 이름", "gu": st.column_config.SelectboxColumn(
                                   "구", options=list(SAMPLE.region) if not ready else list(pd.read_csv(REGIONS).region)),
                               "dong": "동", "area_m2": st.column_config.NumberColumn("면적(㎡)"), "floor": "층",
                               "build_year": "건축년도", "btype": st.column_config.SelectboxColumn("건물용도", options=BTYPES),
                               "deposit": st.column_config.NumberColumn("보증금(만원)"),
                               "rent": st.column_config.NumberColumn("월세(만원)")})
            ss.listings_df = edited.reset_index(drop=True)
    else:
        st.html('<div style="height:36px"></div>')
weights = {"rent": (1 - w / 100) * 0.85, "commute": w / 100 * 0.85, "amenity": 0.15}

# ── 계산 (원래 로직 그대로) ──
dest, station, walk = None, None, 0.0
if ready:
    regions, coords = pd.read_csv(REGIONS), pd.read_csv(STATIONS, index_col=0)
    hit = geocode(place.strip()) if place.strip() else None
    if hit:
        lat, lon, addr = hit
        station, walk = walk_station(lat, lon, coords)
    elif place.strip().removesuffix("역") in coords.index:  # 검색 실패(오프라인 등) 시 역 이름으로라도
        station, walk = place.strip().removesuffix("역"), 0.0
        lat, lon = coords.lat[station], coords.lon[station]
    else:
        body.warning("장소를 찾지 못했어요. 다른 이름이나 가까운 지하철역 이름으로 입력해 보세요.")
        st.stop()
    dest = {"lat": lat, "lon": lon, "label": esc(place.strip()),
            "stn": {"lat": float(coords.lat[station]), "lon": float(coords.lon[station]), "name": f"{station}역"},
            "path": walk_path(lat, lon, float(coords.lat[station]), float(coords.lon[station]))}
    hint.html(f'<span class="db-hint">· {station}역 도보 {walk:.0f}분{" ⚠️" if walk > 25 else ""}</span>')
    regions = with_bus(regions, regions.region.map(region_commute(subway, station)) + walk,  # 구 출발점 = 대표역
                       bus_commute(lat, lon, station, walk, regions[["lat", "lon", "station"]].assign(walk_min=0.0)))
    deals = pd.read_csv(DEALS).query("room == @room and kind == @kind").drop(columns=["room", "kind"])
    regions = regions.dropna(subset=["commute_min"]).merge(deals.rename(columns={"gu": "region"}), on="region")
else:
    regions = SAMPLE
    hint.html('<span class="db-hint">· 예시 데이터</span>')

def rank_view(df):
    """순위의 '가격' = 고른 방·계약의 실제 중앙값을 환산월세로 (전세는 월세 0이라 보증금 이자만). AI 예측값은 표에 따로."""
    df = df.assign(model_rent=df.pred_rent, pred_rent=converted_rent(df.deposit_med, df.rent_med))
    ok = df[(df.deposit_med <= max_deposit) & (df.rent_med <= max_rent)]
    return df, score_regions(ok, budget=float("inf"), max_commute=max_commute, weights=weights)


regions, ranked = rank_view(regions)
by_dong = page == "동네 찾기" and level == "동" and ready and DONGS.exists() and DEALS_DONG.exists()
if by_dong:  # 동 단위: 동 위치 → 가까운 역(도보) → 지하철 → 목적지 역(도보). '동네 찾기' 화면에서만 바꾼다.
    dongs = pd.read_csv(DONGS)
    dongs = with_bus(dongs, dongs.station.map(subway.times_from(station)) + dongs.walk_min + walk,
                     bus_commute(lat, lon, station, walk, dongs[["lat", "lon", "station", "walk_min"]]))
    dstats = pd.read_csv(DEALS_DONG).query("room == @room and kind == @kind and deals >= @MIN_DEALS")
    dongs = dongs.dropna(subset=["commute_min"]).merge(dstats.drop(columns=["room", "kind"]), on=["gu", "dong"])
    regions, ranked = rank_view(dongs.assign(region=dongs.gu + " " + dongs.dong))
in_budget = regions[(regions.deposit_med <= max_deposit) & (regions.rent_med <= max_rent)]
tradeoffs = tradeoff_lines(tradeoff_suggestions(in_budget, budget=float("inf"), max_commute=max_commute))
relax = commute_relax(in_budget, max_commute) if ranked.empty else None


def svg_img(svg: str, cls: str = "", style: str = "") -> str:
    """st.html은 인라인 <svg>를 지우므로 data URI 이미지로 넣는다."""
    svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    return f'<img class="{cls}" style="{style}" src="data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}">'


def room_svg(i: int) -> str:
    """사진이 없는 매물용 방 일러스트 (다방 썸네일 자리)."""
    wall = ["#E9E4DC", "#DCE3EA", "#E6E1DA"][i % 3]
    return svg_img('<svg viewBox="0 0 140 140" width="140" height="140">'
            f'<rect width="140" height="140" fill="{wall}"/>'
            '<rect x="14" y="12" width="112" height="100" fill="#F6F2EC"/>'
            '<polygon points="0,0 14,12 14,112 0,126" fill="#D9D2C8"/>'
            '<polygon points="140,0 126,12 126,112 140,126" fill="#D9D2C8"/>'
            '<polygon points="0,126 14,112 126,112 140,126 140,140 0,140" fill="#C9B49A"/>'
            '<rect x="50" y="40" width="40" height="30" fill="#BFE0F5" stroke="#fff" stroke-width="3"/>'
            '<rect x="26" y="80" width="40" height="20" rx="2" fill="#B89C82"/></svg>', "db-room")


@st.cache_data
def photo_src(file: str) -> str:
    return "data:image/webp;base64," + base64.b64encode((PHOTOS / file).read_bytes()).decode()


def room_thumb(i: int) -> str:
    """매물 썸네일: 받아 둔 Unsplash 예시 사진 + 작가 크레딧(Unsplash 가이드라인), 없으면 일러스트."""
    m = PHOTOS / "_pool/manifest.json"
    p = pick_photo(json.loads(m.read_text(encoding="utf-8")), room, i) if m.exists() else None
    if p is None:
        return room_svg(i)
    return (f'<div class="db-room"><img class="db-pic" src="{photo_src(p["file"])}">'
            f'<a class="db-cr" href="{esc(p["author_url"])}" target="_blank">© {esc(p["author"])} / Unsplash</a></div>')


def sample_room(crop: float = 1.0) -> str:
    """광각으로 찍은 듯한 예시 방 그림. crop<1이면 가운데를 잘라 표준 렌즈 시야처럼 보여 준다."""
    w, h = 600 * crop, 400 * crop
    return svg_img(f'<svg viewBox="{(600 - w) / 2:.1f} {(400 - h) / 2:.1f} {w:.1f} {h:.1f}" preserveAspectRatio="xMidYMid slice" '
            'width="600" height="400">'
            '<rect width="600" height="400" fill="#F6F2EC"/>'
            '<polygon points="0,0 600,0 455,80 145,80" fill="#E8E3DC"/>'
            '<polygon points="0,0 145,80 145,300 0,400" fill="#DCD6CE"/>'
            '<polygon points="600,0 455,80 455,300 600,400" fill="#D4CEC6"/>'
            '<polygon points="0,400 145,300 455,300 600,400" fill="#C9B49A"/>'
            '<rect x="250" y="110" width="100" height="74" fill="#CFE5F3" stroke="#fff" stroke-width="7"/>'
            '<rect x="190" y="220" width="90" height="46" rx="6" fill="#B89C82"/>'
            '<rect x="420" y="140" width="28" height="130" fill="#C9AE92"/>'
            '<rect x="0" y="240" width="50" height="80" fill="#DCCDB8"/></svg>', style="width:100%;height:100%;object-fit:cover")


def compare_listings():
    """매물 비교 계산(원래 로직): 환산월세·통학·편의시설로 순위, 가격 모델이 있으면 시세 대비."""
    listings = ss.listings_df.dropna(subset=["gu", "deposit", "rent"]).fillna({"dong": ""})
    if listings.empty:
        return listings, pd.DataFrame()
    by_gu = regions.set_index("region")
    cand = pd.DataFrame({
        "region": listings.name, "pred_rent": converted_rent(listings.deposit, listings.rent).round(1),
        "commute_min": listings.gu.map(by_gu.commute_min), "amenity_cnt": listings.gu.map(by_gu.amenity_cnt),
    })
    if price:
        cand["시세 대비(만원)"] = price_gap(price["model"], listings.assign(contract_ym=price["latest_ym"])).round(1).values
    picked = score_regions(cand, converted_rent(max_deposit, max_rent), max_commute)
    return listings.drop_duplicates("name").set_index("name"), picked


SRC = ("보증금·월세: 서울특별시 전월세가 정보(서울 열린데이터광장, 아파트 제외) 최근 12개월 실제 계약 — 방은 면적으로 구분"
       "(원룸 33㎡ 이하, 투룸 33~66㎡), 보증금은 가운데 50% 범위, 월세는 중앙값. 통학: 구 단위는 구 대표역, 동 단위는 "
       "동 위치에서 가장 가까운 역까지 도보 + 서울교통공사 역간 소요시간(2024-08) + 환승 + 목적지 도보 (1호선 코레일 구간·9호선은 "
       "OpenStreetMap 노선으로 보강, 역간 시간은 거리로 추정). 버스를 섞는 게 더 빠르면 그 시간: 서울시 버스노선별 정류소(2026-11-02) + "
       "TOPIS 구간 평균 운행시간·운행횟수(2026-11-09~13 평일 7~8시, 대기 = 배차 간격의 절반). 동 위치·편의시설: "
       f"소상공인시장진흥공단 상가정보(2026-09, 동 위치 = 그 동 상가 좌표 중앙값). 동 단위는 최근 12개월 거래 {MIN_DEALS}건 "
       "이상인 동만. 위치 검색·지도: OpenStreetMap.")
MODEL_NOTE = (f"AI 예측 시세: Gradient Boosting 환산월세 예측(계약 {price['period'].replace('~', '–')}, 최근 계약 20% 검증 "
              f"MAE {price['mae']:.1f}만 원) · 싼 편/비싼 편 = 같은 계약·방 종류 실제 계약의 예측 오차 하위·상위 10% 밖"
              " · 참고 정보" if price else "가격 모델이 없어 시세 비교는 생략했어요.")


def gap_tag(gap: float, li) -> tuple[str, str]:
    """매물(li)의 계약·방 종류에 맞는 오차 범위로 시세 대비 배지. 예전 빌드(밴드 없음)는 ±MAE."""
    return gap_label(gap, band_for(price.get("band", {}), li.rent, li.area_m2, price["mae"]))

# ── 1. 동네 찾기 ──
if page == "동네 찾기":
    col_list, col_map = body.columns([400, 960], gap=None)
    with col_list, st.container(key="list"):
        keys = ["추천", "월세", "보증금", "거리"] if kind == "월세" else ["추천", "전세금", "거리"]
        with st.container(key="sort"):  # 계약 종류마다 위젯 키를 따로 둬서 전세↔월세 전환 때 없는 항목이 남지 않게
            sort_key = st.segmented_control("정렬 항목", keys, default="추천", key=f"sort_{kind}", required=True,
                                            label_visibility="collapsed")
        with st.container(key="sortdir"):
            _, up, down = SORTS[sort_key]
            desc = st.segmented_control("정렬 방향", [False, True], default=False, key=f"desc_{kind}_{sort_key}", required=True,
                                        label_visibility="collapsed",
                                        format_func=lambda d: f"↓ {down}" if d else f"↑ {up}")
        shown = ranked.assign(rank=ranked.index + 1)
        if by_dong:  # 동 단위는 추천 상위 안에서만 다시 정렬한다
            shown = shown.head(DONG_SHOWN)
        shown, sort_label = sort_regions(shown, sort_key, desc)
        st.html(f'<div class="db-sortby">추천 동네 {len(shown)}곳{f" (상위 {DONG_SHOWN}개 동)" if by_dong else ""}'
                f' · 정렬 기준: <b>{sort_label}</b></div>')
        if tradeoffs:
            st.html('<div class="db-tip">💡 ' + "<br>💡 ".join(esc(t) for t in tradeoffs)
                    + f'<br><small>{room} {kind} 실거래 중앙값의 환산월세 기준 · 가장 싼 동네끼리 비교</small></div>')
        if ranked.empty:
            st.html(f'<div class="db-tip">💡 통학 {relax["extra_minutes"]}분만 늘리면 {relax["count"]}곳이 나와요. '
                    f'위 필터에서 통학을 {max_commute + relax["extra_minutes"]}분으로 바꿔 보세요.</div>' if relax else
                    '<div class="db-note">조건에 맞는 동네가 없어요. 위 필터에서 보증금·월세·통학 시간을 늘려 보세요.</div>')
        with st.container(height=690, border=False):
            if not shown.empty:  # st.html("")은 오류라 빈 목록이면 건너뜀
                st.html("".join(
                f'<div class="db-it"><div class="db-sq">'
                + (f'{r.rent_med:.0f}<small>만 원/월</small>' if r.rent_med else f'{won(r.deposit_med)}<small>전세금</small>')
                + '</div><div>'
                f'<div class="db-pr">{f"월세 {r.deposit_med:,.0f}/{r.rent_med:.0f}" if r.rent_med else "전세 " + won(r.deposit_med)}</div>'
                f'<div class="db-l1">{esc(r.region)}{" · " + esc(r.station) + "역 근처" if "station" in r else ""}'
                f'{f" (도보 {r.walk_min:.0f}분)" if "walk_min" in r else ""}'
                f' · {"보증금" if r.rent_med else "전세금"} {deposit_text(r)}</div>'
                f'<div class="db-l1">{room} · 통학 {r.commute_min:.0f}분{" (버스 포함)" if r.get("bus") else ""}'
                f'{f" · 거래 {r.deals:,.0f}건" if "deals" in r else ""}</div>'
                f'<div class="db-l2">편의점·마트·병원 {r.amenity_cnt:,.0f}곳</div>'
                f'<span class="db-tg">추천 {r["rank"]}위</span></div></div>'
                for _, r in shown.iterrows()))
    with col_map:
        if dest:
            rank_of = {g: i for i, g in enumerate(ranked.region)}
            who = esc(place.strip()) or "직장·학교"
            pts = []
            for _, r in (ranked.head(DONG_SHOWN) if by_dong else regions).iterrows():
                k = rank_of.get(r.region)
                pts.append({"lat": r.lat, "lon": r.lon, "name": r.dong if by_dong else r.region,
                            "value": f"{r.rent_med:.0f}만" if r.rent_med else won(r.deposit_med),
                            "state": "off" if k is None else "top" if k < 3 else "on",
                            "tip": f"<b>{f'추천 {k + 1}위 ' if k is not None else '조건 밖 · '}{r.region}</b><br>{room} "
                                   f"{price_text(r)}<br>{'🚌 버스 포함' if r.get('bus') else '🚇'} {who}까지 {r.commute_min:.0f}분 "
                                   f"({r.station}역) · 거래 {r.deals:,.0f}건"})
            leaflet_map(pts, dest, note="원 안의 숫자 = " + ("원룸 " if room == "원룸" else "투룸 ")
                        + ("월세 중앙값" if kind == "월세" else "전세금 중앙값")
                        + (f" · 추천 상위 {DONG_SHOWN}개 동" if by_dong else " · 회색 = 조건 밖") + " · 마커에 마우스를 올리면 자세히")
        else:
            st.info("실제 데이터를 빌드하면 여기에 서울 지도가 보여요 (scripts/build_data.py).")
    with body, st.container(key="foot"):
        with st.expander("전체 순위 표 보기"):
            st.dataframe(
                ranked.assign(순위=ranked.index + 1, deposit=[deposit_text(r) for _, r in ranked.iterrows()])[
                    ["순위", "region"] + (["station"] if "station" in ranked else [])
                    + ["commute_min", "deposit"] + (["rent_med"] if kind == "월세" else [])
                    + [c for c in ["deals", "model_rent"] if c in ranked] + ["amenity_cnt"]],
                hide_index=True, width="stretch",
                column_config={"region": "지역", "station": "대표역", "commute_min": st.column_config.NumberColumn("통학(분)"),
                               "deposit": "보증금(보통 범위)" if kind == "월세" else "전세금(보통 범위)",
                               "rent_med": st.column_config.NumberColumn("월세(만원)", format="%d"),
                               "deals": st.column_config.NumberColumn("거래 수", help="최근 12개월, 이 방·계약 종류"),
                               "model_rent": st.column_config.NumberColumn("AI 예측 시세(20㎡ 환산월세)", format="%.1f"),
                               "amenity_cnt": st.column_config.NumberColumn("편의점·마트·병원 수")})
        st.caption((SRC + " " + MODEL_NOTE).replace("~", r"\~"))

# ── 2. 매물 비교: 카드 목록 + 지도 ──
elif page == "매물 비교":
    info, picked = compare_listings()
    col_list, col_map = body.columns([400, 960], gap=None)
    with col_list, st.container(key="list"):
        with st.container(key="sort2"):
            order = st.segmented_control("정렬", ["전체", "종합순", "싼 순"], default="전체", key="order2", required=True,
                                         label_visibility="collapsed",
                                         format_func=lambda o: f"내 후보 매물 {len(picked)}" if o == "전체" else o) or "전체"
        rows = picked.assign(rank=picked.index + 1) if not picked.empty else picked
        if order == "싼 순" and price and not rows.empty:
            rows = rows.sort_values("시세 대비(만원)")
        cards = ""
        for _, r in rows.iterrows():
            li = info.loc[r.region]
            tags = f'<span class="db-tg">종합 {r["rank"]}위</span>'
            pred = ""
            if price:
                text, tone = gap_tag(r["시세 대비(만원)"], li)
                tags += f'<span class="db-tg {tone}">{text}</span>'
                pred = (f'<div class="db-l2">AI 예측 {r.pred_rent - r["시세 대비(만원)"]:.1f}만 / '
                        f'환산 {r.pred_rent:.1f}만</div>')
            cards += (f'<div class="db-it">{room_thumb(r.name)}<div>'
                      f'<div class="db-pr">월세 {li.deposit:,.0f}/{li.rent:,.0f}</div>'
                      f'<div class="db-l1">{room} · {li.area_m2:g}㎡ · {li.floor:g}층 · {li.build_year:g}년</div>'
                      f'<div class="db-l1">{esc(li.gu)} {esc(li.dong)} · 통학 {r.commute_min:.0f}분</div>'
                      f'{pred}{tags}</div></div>')
        if len(picked) < len(info):
            cards += (f'<div class="db-note">예산·통학 조건을 넘는 매물 {len(info) - len(picked)}개는 제외됐어요 '
                      '(예산 = 보증금·월세 최대를 환산월세로).</div>')
        with st.container(height=690, border=False):
            st.html(cards + f'<div class="db-note">{MODEL_NOTE}</div>')
    with col_map:
        if dest and not picked.empty:
            by_gu = regions.set_index("region")
            pts, stns, seen = [], [], {}
            for _, r in picked.iterrows():
                li = info.loc[r.region]
                g = by_gu.loc[li.gu]
                n = seen[li.gu] = seen.get(li.gu, -1) + 1  # 같은 구 매물은 대표역 주변에 살짝 벌려 놓는다
                pts.append({"lat": g.lat - 0.004 - 0.0035 * n, "lon": g.lon + 0.004 + 0.002 * n,
                            "value": f"{r.name + 1}위", "name": f"{li.deposit:,.0f}/{li.rent:,.0f}",
                            "state": "top" if r.name == 0 else "on",
                            "tip": f"<b>{esc(r.region)}</b> · {esc(li.gu)} {esc(li.dong)}<br>환산 월세 {r.pred_rent:.1f}만"})
                if li.gu not in [s["gu"] for s in stns]:
                    stns.append({"gu": li.gu, "lat": g.lat, "lon": g.lon, "name": f"{g.station}역"})
            leaflet_map(pts, dest, stations=stns, note="매물 위치는 주소가 아니라 구 대표역 근처에 표시돼요")
        elif not dest:
            st.info("실제 데이터를 빌드하면 여기에 지도가 보여요.")

# ── 3. 사진 화각: 다방 상세 팝업처럼 사진 2장 + 매물 정보 + 화각 분석 ──
else:
    info, picked = compare_listings()
    with body, st.container(key="dim"), st.container(key="modal"):
        photo = st.container()  # 사진 2장은 맨 위, 업로드·매물 선택은 맨 아래 회색 바
        info_box = st.container()
        with st.container(key="tools", horizontal=True, vertical_alignment="center"):
            st.html('<span class="db-tl">어느 매물의 사진인가요?</span>')
            names = list(info.index) if not picked.empty else []
            pick = (st.selectbox("매물", names, key="pick3", label_visibility="collapsed", width=200)
                    if names else None)
            up = st.file_uploader("매물 사진 올리기", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        est, sample = None, False
        if up:
            img = Image.open(up).convert("RGB")
            est = predict_fov(fov_model, img) if fov_model else None
            if est is None:
                with info_box, st.container(key="pick"):
                    est = st.number_input("추정 화각(°) — 화각 모델(models/fov.pt) 학습 전 임시 입력", 40.0, 130.0, 108.0)
            photo.html(f'<div class="db-ph2"><div><img src="{img_b64(img)}"><span>올라온 사진</span></div>'
                       f'<div><img src="{img_b64(restore_standard_view(img, est))}"><span class="o">표준 렌즈(1x) 시야</span></div></div>')
        else:  # 업로드 전에는 예시 방 그림으로 화면 구성을 보여 준다 (오른쪽은 실제 crop_ratio로 자른 것)
            est, sample = 108.0, True
            photo.html(f'<div class="db-ph2"><div>{sample_room()}<span>예시 사진 (광각 108°)</span></div>'
                       f'<div>{sample_room(crop_ratio(est))}<span class="o">표준 렌즈(1x) 시야</span></div></div>')
        left = '<div><h2>매물을 골라 주세요</h2><div class="s">매물 비교 화면에서 매물을 추가할 수 있어요.</div></div>'
        if pick is not None:
            li, r = info.loc[pick], picked.set_index("region").loc[pick]
            rows_html = ""
            if price:
                text, tone = gap_tag(r["시세 대비(만원)"], li)
                rows_html += (f'<tr><td>AI 예측 시세</td><td><b>{r.pred_rent - r["시세 대비(만원)"]:.1f}만</b> (환산 '
                              f'{r.pred_rent:.1f}만) <span class="db-tg {tone}">{text}</span></td></tr>')
            rows_html += (f'<tr><td>{esc(place.strip()) or "직장·학교"}까지</td><td>{r.commute_min:.0f}분</td></tr>'
                          f'<tr><td>편의시설</td><td>{r.amenity_cnt:,.0f}곳 ({esc(li.gu)})</td></tr>')
            left = (f'<div><h2>월세 {li.deposit:,.0f}/{li.rent:,.0f}</h2>'
                    f'<div class="s">{esc(li.gu)} {esc(li.dong)} · {li.area_m2:g}㎡ · {li.floor:g}층 · {li.build_year:g}년 · '
                    f'{esc(li.btype)}</div><table class="db-tbl">{rows_html}</table></div>')
        if est is not None:
            tone = "warn" if crop_ratio(est) < 0.9 else "good"
            right = (f'<div class="db-fv"><h3>사진 화각 분석</h3><div class="d"><b style="color:{"#B26A00" if tone == "warn" else "#12A150"}">'
                     f'{est:.0f}°</b><span class="db-tg {tone}" style="font-size:13px;margin:0">{wide_angle_flag(est)}</span></div>'
                     f'<p>표준 렌즈(약 {STD_FOV_DEG:.0f}°)보다 넓게 찍히면 방이 실제보다 넓어 보일 수 있어요. 오른쪽은 1x로 찍었을 때 '
                     f'보이는 범위(가로·세로 {crop_ratio(est) * 100:.0f}%)예요. 참고 정보이며 허위 매물 판정이 아니에요.'
                     + (' <b>지금은 예시예요 — 아래에서 매물 사진을 올려 보세요.</b>' if sample else '') + '</p></div>')
        else:
            right = ('<div class="db-fv"><h3>사진 화각 분석</h3><p>사진을 올리면 화각을 추정하고, 사람 눈에 가까운 표준 렌즈(1x) '
                     '시야로 바꿔 보여드려요. 참고 정보이며 허위 매물 판정이 아니에요.</p></div>')
        info_box.html(f'<div class="db-md">{left}{right}</div>')
