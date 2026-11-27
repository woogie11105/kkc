"""공공데이터 로더 · 모델 학습 파이프라인 테스트 (작은 가짜 CSV로)."""
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kkc.commute.subway import GU_STATION, load_seoul_metro, load_station_coords, nearest_station, region_commute
from kkc.data import read_table
from kkc.price.model import load_rent, region_deal_stats, region_rents, train
from kkc.ranking.rank import dong_table
from kkc.vision.fov import exif_fov, predict_fov, train_fov


def test_read_table_skips_notice_and_cp949(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("□ 공지 문구\n\n시군구,보증금(만원)\n서울특별시 관악구 신림동,\"1,000\"\n", encoding="cp949")
    df = read_table(p, must="시군구")
    assert list(df.columns) == ["시군구", "보증금(만원)"]
    assert df.iloc[0, 1] == 1000


def test_load_rent(tmp_path):
    p = tmp_path / "rent.csv"
    p.write_text(
        "시군구,번지,전월세구분,전용면적(㎡),계약년월,계약일,보증금(만원),월세금(만원),층,건축년도\n"
        "서울특별시 관악구 신림동,1,월세,20.5,202609,1,\"1,000\",50,3,2015\n"
        "서울특별시 강남구 역삼동,2,전세,25,202609,2,\"20,000\",0,5,2010\n"
        "경기도 수원시 장안구 정자동,3,월세,20,202609,3,500,40,2,2000\n",
        encoding="utf-8")
    df = load_rent(p)
    assert list(df.gu) == ["관악구", "강남구"]  # 서울만
    assert df.iloc[0][["dong", "area_m2", "deposit", "rent"]].tolist() == ["신림동", 20.5, 1000, 50]


def _fake_rents(n=400):
    import numpy as np
    rng = np.random.default_rng(0)
    gu = rng.choice(["강남구", "관악구"], n)
    area = rng.uniform(15, 40, n)
    rent = np.where(gu == "강남구", 90, 50) + area + rng.normal(0, 2, n)
    return pd.DataFrame({"gu": gu, "dong": gu + "동", "area_m2": area, "floor": 3, "build_year": 2010,
                         "contract_ym": 202609, "deposit": 0, "rent": rent, "btype": "연립다세대"})


def test_region_rents_orders_by_price():
    df = _fake_rents()
    model, mae, _ = train(df)
    assert mae < 5
    r = region_rents(model, df, area_m2=20)
    assert r["강남구"] > r["관악구"] + 30
    by_dong = region_rents(model, df, area_m2=20, by=("gu", "dong"))
    assert by_dong[("강남구", "강남구동")] > by_dong[("관악구", "관악구동")] + 30


def test_region_rents_same_btype_for_all_regions():
    df = _fake_rents()
    gangnam = df.index[df.gu == "강남구"]
    df["btype"] = "단독다가구"
    df.loc[gangnam[:150], "btype"] = "오피스텔"  # 강남구 최빈 = 오피스텔, 전체 최빈 = 단독다가구
    df["rent"] = df.rent + (df.btype == "오피스텔") * 40
    model, _, _ = train(df)
    r = region_rents(model, df, area_m2=20)
    assert r["강남구"] - r["관악구"] < 55  # 강남 최빈(오피스텔)로 예측했다면 +40이 더 붙음


def test_load_seoul_metro(tmp_path):
    d = tmp_path / "dist.csv"
    d.write_text("연번,호선,역명,운행시간(분),역간거리(km),호선별누계(km)\n"
                 "1,2,신림,0,0,0\n2,2,서울대입구,2:00,1,1\n3,2,사당,1:30,1,2\n"
                 "4,4,사당(대림),0,0,0\n5,4,총신대입구(이수),2,1,1\n", encoding="cp949")
    t = tmp_path / "transfer.csv"
    t.write_text("연번,호선,환승역명,환승노선,환승거리,환승소요시간\n1,2호선,사당,4호선,200,3:00\n", encoding="cp949")
    g = load_seoul_metro(d, t)
    times = g.times_from("신림")
    assert times["사당"] == pytest.approx(3.5)
    assert times["총신대입구"] == pytest.approx(3.5 + 3 + 2)
    assert GU_STATION["관악구"] == "서울대입구"
    commute = region_commute(g, "신림")
    assert commute["관악구"] == pytest.approx(2)
    assert pd.isna(commute["강남구"])  # 그래프에 없는 역 → NaN


def test_dong_table(tmp_path):
    p = tmp_path / "stores.csv"
    p.write_text("상호명,상권업종소분류명,시군구명,법정동명,경도,위도\n"
                 "A,편의점,관악구,신림동,126.92,37.48\nB,편의점,관악구,신림동,126.94,37.50\nC,카페,관악구,신림동,126.93,37.49\n"
                 "D,슈퍼마켓,강남구,역삼동,127.03,37.50\nE,내과의원,강남구,역삼동,127.04,37.50\n", encoding="cp949")
    t = dong_table(p)
    assert t.amenity_cnt.to_dict() == {("강남구", "역삼동"): 2, ("관악구", "신림동"): 2}  # 카페 제외
    assert t.loc[("관악구", "신림동"), ["lat", "lon"]].tolist() == [37.49, 126.93]  # 상가 좌표 중앙값
    assert t.groupby("gu").amenity_cnt.sum().to_dict() == {"강남구": 2, "관악구": 2}  # 구별 개수


def _photo(path, focal_35mm, color):
    img = Image.new("RGB", (64, 48), color)
    exif = Image.Exif()
    exif.get_ifd(0x8769)[41989] = focal_35mm  # FocalLengthIn35mmFilm
    img.save(path, exif=exif)


def test_exif_fov(tmp_path):
    _photo(tmp_path / "a.jpg", 13, "red")
    Image.new("RGB", (8, 8)).save(tmp_path / "noexif.jpg")
    assert exif_fov(tmp_path / "a.jpg") == pytest.approx(108.3, abs=0.2)
    assert exif_fov(tmp_path / "noexif.jpg") is None


def test_train_and_predict_fov_smoke(tmp_path):
    for i in range(4):
        _photo(tmp_path / f"w{i}.jpg", 13, "red")
        _photo(tmp_path / f"s{i}.jpg", 26, "blue")
    out = tmp_path / "fov.pt"
    model, mae = train_fov(tmp_path, epochs=1, out=out, pretrained=False)
    assert out.exists() and mae >= 0
    assert isinstance(predict_fov(model, Image.new("RGB", (64, 48))), float)  # 1 epoch라 값 자체는 의미 없음


def test_load_rent_seoul_format(tmp_path):
    p = tmp_path / "seoul.csv"
    p.write_text(
        "접수년도,자치구명,법정동명,층,계약일,전월세구분,임대면적,보증금(만원),임대료(만원),건축년도,건물용도\n"
        '"2025",관악구,신림동,3,"20260202",월세,20.5,"1000","50","2015",연립다세대\n'
        '"2025",강남구,역삼동,,"20251103",전세,30,"20000","0",,단독다가구\n'
        '"2025",강남구,역삼동,9,"20251103",월세,84,"50000","200","2010",아파트\n',
        encoding="cp949")
    df = load_rent(p)
    assert list(df.gu) == ["관악구", "강남구"]  # 아파트 제외, 층·건축년도 빈 값은 유지
    assert df.contract_ym.tolist() == [202602, 202511]
    assert df.btype.tolist() == ["연립다세대", "단독다가구"]
    assert df.iloc[0][["dong", "area_m2", "deposit", "rent"]].tolist() == ["신림동", 20.5, 1000, 50]


def test_subway_branch_and_alias(tmp_path):
    d = tmp_path / "dist.csv"
    d.write_text("연번,호선,역명,소요시간\n1,2,신도림,00:00\n2,2,성수,05:00\n3,2,시청,05:00\n"
                 "4,2,용답,03:00\n5,2,신설동,01:30\n6,2,도림천,01:30\n"
                 "7,4,사당,00:00\n8,4,총신대입구,02:00\n9,7,이수,00:00\n10,7,내방,02:00\n", encoding="cp949")
    t = tmp_path / "transfer.csv"
    t.write_text("연번,호선,환승역명,환승노선,환승거리,환승소요시간\n1,4,총신대입구,7호선,200,03:00\n"
                 "2,7,이수,4호선,200,03:00\n3,4,사당,공항철도,100,01:00\n", encoding="cp949")
    g = load_seoul_metro(d, t)
    assert g.times_from("성수")["용답"] == pytest.approx(3)  # 시청이 아니라 성수에서 갈라짐
    assert g.times_from("신도림")["도림천"] == pytest.approx(1.5)
    assert g.times_from("사당")["내방"] == pytest.approx(2 + 3 + 2)  # 이수 = 총신대입구, 공항철도 환승은 무시돼야 안 터짐


def test_station_coords_and_nearest(tmp_path):
    p = tmp_path / "coords.csv"
    p.write_text("연번,호선,고유역번호(외부역코드),역명,위도,경도\n1,1,150,서울,37.55,126.97\n"
                 "2,1,124,청량리,37.580,127.046\n3,5,2534,청량리,37.582,127.048\n", encoding="cp949")
    c = load_station_coords(p)
    assert set(c.index) == {"서울역", "청량리"}  # 서울 → 서울역 (역간 파일 이름과 맞춤)
    assert c.loc["청량리", "lat"] == pytest.approx(37.581)
    st, walk = nearest_station(37.5971, 127.0520, c)  # 경희대
    assert st == "청량리" and 33 < walk < 38  # 직선 ~1.85km × 1.3 / 67m


def test_region_deal_stats():
    df = pd.DataFrame({"gu": ["관악구"] * 5 + ["강남구"],
                       "area_m2": [20, 25, 50, 20, 90, 20], "deposit": [500, 1500, 9000, 20000, 5000, 1000],
                       "rent": [40, 60, 150, 0, 100, 80],
                       "contract_ym": [202602, 202503, 202602, 202602, 202602, 202502]})
    s = region_deal_stats(df)
    # 중앙값, 월세, 건수, 25%, 75% — 분위수는 보간 없이 실제 값
    assert s.loc[("관악구", "원룸", "월세")].tolist() == [1000, 50, 2, 500, 1500]
    assert s.loc[("관악구", "투룸", "월세"), "rent_med"] == 150  # 50㎡
    assert s.loc[("관악구", "원룸", "전세"), "deposit_med"] == 20000  # 월세 0 = 전세
    assert len(s) == 3  # 90㎡(투룸 초과)·12개월 전 강남구(202502) 제외
    by_dong = region_deal_stats(df.assign(dong=["신림동", "봉천동", "신림동", "신림동", "신림동", "역삼동"]),
                                by=("gu", "dong"))
    assert by_dong.loc[("관악구", "신림동", "원룸", "월세"), "deals"] == 1  # 봉천동 거래는 따로


def test_error_band_by_kind_and_room():
    from kkc.price.model import band_for, error_band
    df = pd.DataFrame({"area_m2": [20] * 10 + [50] * 10 + [20] * 10, "rent": [50] * 20 + [0] * 10})
    res = pd.Series(list(range(-5, 5)) + list(range(-50, 50, 10)) + [0] * 10, dtype=float)
    band = error_band(df, res)
    assert band["월세 원룸"][0] < 0 < band["월세 원룸"][1] and band["월세 원룸"][1] < 6
    assert band["월세 투룸"][1] > 20  # 투룸은 오차가 더 크다
    assert band_for(band, rent=50, area_m2=20) == band["월세 원룸"]
    assert band_for(band, rent=0, area_m2=20) == band["전세 원룸"]
    assert band_for(band, rent=50, area_m2=100) == band["전체"]  # 66㎡ 넘으면 전체 기준
    assert band_for({}, rent=50, area_m2=20, fallback=15.2) == (-15.2, 15.2)  # 예전 빌드(밴드 없음)는 ±MAE
