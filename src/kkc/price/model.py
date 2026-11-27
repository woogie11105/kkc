"""Gradient Boosting 회귀로 월세 예측 + 시세 대비 비교.

데이터: 국토교통부 전월세 실거래가 (자치구/법정동, 전용면적, 층, 건축년도, 보증금, 월세 ...).
보증금이 매물마다 달라서 '환산 월세' = 월세 + 보증금 × 전환율/12 로 맞춘 뒤 학습한다.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from kkc.data import col, read_table

CONVERSION_RATE = 0.055  # 전월세 전환율(연). TODO: 한국부동산원 최신 값으로 교체

FEATURES = ["gu", "dong", "area_m2", "floor", "build_year", "contract_ym", "btype"]
CATEGORICAL = ["gu", "dong", "btype"]
BTYPES = ["단독다가구", "연립다세대", "오피스텔"]  # 서울시 전월세가 '건물용도' (아파트 제외)


def load_rent(path) -> pd.DataFrame:
    """전월세 실거래 CSV → 학습용 표(서울만). 두 형식을 받는다:
    - 국토부 실거래가 공개시스템(rt.molit.go.kr): 시군구, 전용면적, 계약년월, 월세금 ...
    - 서울 열린데이터광장 '서울특별시 부동산 전월세가 정보': 자치구명, 법정동명, 임대면적, 계약일, 임대료 ...
      이 파일은 아파트가 섞여 있어서 자취방 시세에 맞게 아파트는 뺀다.
    전세(월세 0)도 환산월세로 같이 쓴다. 층·건축년도 빈 값(단독다가구는 층 없음)은 GBR이 그대로 처리."""
    raw = read_table(path, must="시군구")
    if "자치구명" in raw.columns:
        raw = raw[raw[col(raw, "건물용도")] != "아파트"]
        btype = raw[col(raw, "건물용도")]
        gu, dong = raw["자치구명"], raw["법정동명"]
        area, ym, rent = raw[col(raw, "임대면적")], raw[col(raw, "계약일")] // 100, raw[col(raw, "임대료")]
        seoul = pd.Series(True, index=raw.index)
    else:
        sgg = raw[col(raw, "시군구")].astype(str).str.split()  # "서울특별시 관악구 신림동"
        gu, dong, seoul = sgg.str[1], sgg.str[2], sgg.str[0].str.startswith("서울")
        btype = None  # 국토부 파일은 건물 유형별로 따로 받아서 컬럼 없음 → 결측
        area, ym, rent = raw[col(raw, "전용면적")], raw[col(raw, "계약년월")], raw[col(raw, "월세금")]
    df = pd.DataFrame({
        "gu": gu, "dong": dong, "area_m2": area, "floor": raw[col(raw, "층")],
        "build_year": raw[col(raw, "건축년도")], "contract_ym": ym,
        "deposit": raw[col(raw, "보증금")], "rent": rent, "btype": btype,
    })
    return df[seoul].dropna(subset=["gu", "dong", "area_m2", "contract_ym", "deposit", "rent"]).reset_index(drop=True)


def region_rents(model, df: pd.DataFrame, area_m2: float = 20.0, by: tuple = ("gu",)) -> pd.Series:
    """지역별 '표준 원룸'(전용 area_m2, 최근 계약월, 그 지역의 층·건축년도, 전체 최빈 건물용도) 예측 환산월세(만원).
    by=("gu",): 자치구별(동은 그 구의 대표 동), by=("gu", "dong"): 동별."""
    agg = {"floor": ("floor", "median"), "build_year": ("build_year", "median")}
    if "dong" not in by:
        agg["dong"] = ("dong", lambda s: s.mode()[0])
    rows = df.groupby(list(by)).agg(**agg).reset_index()
    # 건물용도는 구마다 다르게 두면 오피스텔 많은 구만 비싸 보여서, 전체에서 가장 흔한 용도 하나로 통일
    rows["btype"] = df.btype.mode()[0] if df.btype.notna().any() else None
    rows["area_m2"] = area_m2
    rows["contract_ym"] = df.contract_ym.max()
    return pd.Series(model.predict(_prep(rows)), index=rows.set_index(list(by)).index).round(1)


# shortcut: 방 개수 컬럼이 없어서 면적으로 구분(원룸 ≤ 33㎡=10평, 투룸 33~66㎡). 방 수 데이터가 생기면 그걸로.
ROOMS = {"원룸": "~33㎡", "투룸": "33~66㎡"}
ROOM_BINS = [0, 33, 66]


def region_deal_stats(df: pd.DataFrame, months: int = 12, by: tuple = ("gu",)) -> pd.DataFrame:
    """지역(by: 구 또는 구·동) × 방(원룸/투룸) × 계약(월세/전세)별 보증금·월세 통계(만원), 최근 months개월 실제 계약.
    예측값이 아니라 실제 계약 통계라서 화면에 '보통 이 정도'로 보여주기 좋다."""
    since = int((pd.Period(str(int(df.contract_ym.max())), "M") - (months - 1)).strftime("%Y%m"))
    d = df[df.contract_ym >= since].assign(
        room=lambda x: pd.cut(x.area_m2, ROOM_BINS, labels=list(ROOMS)),
        kind=lambda x: x.rent.gt(0).map({True: "월세", False: "전세"})).dropna(subset=["room"])
    # 보증금은 1000만·500만 같은 딱 떨어지는 값에 몰려서 중앙값이 거의 모든 구에서 1000이 된다 → 가운데 50% 범위도 같이.
    # nearest: 6,793만 같은 보간값 대신 실제 계약에 있는 값으로
    q = lambda p: (lambda s: s.quantile(p, interpolation="nearest"))  # noqa: E731
    return d.groupby([*by, "room", "kind"], observed=True).agg(
        deposit_med=("deposit", "median"), rent_med=("rent", "median"), deals=("rent", "size"),
        deposit_lo=("deposit", q(0.25)), deposit_hi=("deposit", q(0.75)))


def converted_rent(deposit_manwon: pd.Series, rent_manwon: pd.Series) -> pd.Series:
    return rent_manwon + deposit_manwon * CONVERSION_RATE / 12


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def error_band(df: pd.DataFrame, residual: pd.Series, lo: float = 0.1, hi: float = 0.9) -> dict:
    """검증 오차(실제 - 예측)의 하위·상위 분위수를 계약(월세/전세)×방(원룸/투룸)별로. 키 "월세 원룸" 등 + "전체".
    오차 크기가 방마다 크게 달라서(원룸 MAE ≈10만, 투룸 ≈18만) 하나의 기준으로 '싸다/비싸다'를 정하면 안 된다."""
    q = lambda s: (round(float(s.quantile(lo)), 1), round(float(s.quantile(hi)), 1))  # noqa: E731
    res = pd.Series(residual.values, index=df.index)
    kind = df.rent.gt(0).map({True: "월세", False: "전세"})
    room = pd.cut(df.area_m2, ROOM_BINS, labels=list(ROOMS))
    band = {f"{k} {r}": q(s) for (k, r), s in res.groupby([kind, room], observed=True)}
    band["전체"] = q(res)
    return band


def band_for(band: dict, rent: float, area_m2: float, fallback: float = 0.0) -> tuple[float, float]:
    """매물에 맞는 오차 범위. 방 구분 밖(66㎡ 초과)이면 '전체', 밴드가 없는 예전 빌드면 ±fallback(MAE)."""
    room = next((r for r, top in zip(ROOMS, ROOM_BINS[1:]) if area_m2 <= top), None)
    key = f"{'월세' if rent > 0 else '전세'} {room}"
    return tuple(band.get(key) or band.get("전체") or (-fallback, fallback))


def train(df: pd.DataFrame, seed: int = 42):
    """df: FEATURES + deposit, rent 컬럼. 반환: (model, 검증 MAE[만원], 오차 범위 error_band).
    검증 = 계약월이 가장 최근인 20% (앱은 '지금 시세'를 묻기 때문. 무작위 분할은 같은 건물 반복 계약이
    양쪽에 들어가 MAE가 부풀려짐). 검증 후 전체 데이터로 다시 학습해서 반환."""
    df = df.sort_values("contract_ym", kind="stable")
    y = converted_rent(df["deposit"], df["rent"])
    X = _prep(df)
    # HGB 범주 최대 255개라 거래 많은 동 255곳만 쓰고 나머지는 결측(=구 정보로 예측). 예측 때 모르는 동도 결측 처리됨.
    top = X["dong"].value_counts().index[:255]
    X["dong"] = X["dong"].where(X["dong"].isin(top))
    cut = int(len(X) * 0.8)
    # 서울시 전월세가 64만 건, 최근 3개월 검증: 기본 설정 MAE 16.9 → 이 설정 15.8만원 (2026-11-18 실험)
    model = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=seed,
                                          max_iter=2000, max_leaf_nodes=127)
    pred = model.fit(X.iloc[:cut], y.iloc[:cut]).predict(X.iloc[cut:])
    mae = mean_absolute_error(y.iloc[cut:], pred)
    band = error_band(df.iloc[cut:], y.iloc[cut:] - pred)
    return model.fit(X, y), mae, band


def price_gap(model, listing: pd.DataFrame) -> pd.Series:
    """매물 환산월세 - 예측 시세. 음수면 시세보다 싸다."""
    pred = model.predict(_prep(listing))
    return converted_rent(listing["deposit"], listing["rent"]) - pred
