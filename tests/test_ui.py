import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kkc.ui import gap_label


def test_gap_label_only_outside_error_band():
    # 밴드 = 같은 계약·방 실제 계약의 예측 오차 하위/상위 10% (-14.2 ~ +15.3)
    assert gap_label(-20.04, (-14.2, 15.3)) == ("시세보다 20.0만 원 싼 편이에요", "good")
    assert gap_label(18.0, (-14.2, 15.3)) == ("시세보다 18.0만 원 비싼 편이에요", "bad")
    assert gap_label(-10.0, (-14.2, 15.3)) == ("시세 범위 안이에요", "neutral")  # 예전 기준(±0.5)이면 '싸요'였음
    assert gap_label(9.0, (-14.2, 15.3)) == ("시세 범위 안이에요", "neutral")


def test_tradeoff_lines_skip_same_region():
    from kkc.ui import tradeoff_lines
    s = [{"extra_minutes": 10, "monthly_saving": 10.0, "region": "C"},
         {"extra_minutes": 20, "monthly_saving": 10.0, "region": "C"},
         {"extra_minutes": 30, "monthly_saving": 14.5, "region": "D"}]
    assert tradeoff_lines(s) == ["통학 10분만 양보하면 C에서 월 10.0만 원 아낄 수 있어요",
                                 "통학 30분만 양보하면 D에서 월 14.5만 원 아낄 수 있어요"]
    assert tradeoff_lines([]) == []


def test_sort_regions_both_directions_and_label():
    import pandas as pd
    from kkc.ui import sort_regions
    df = pd.DataFrame({"region": ["A", "B", "C"], "rent_med": [50, 40, 60], "deposit_med": [500, 1000, 300],
                       "commute_min": [30, 20, 40], "rank": [1, 2, 3]})
    out, label = sort_regions(df, "월세")
    assert list(out.region) == ["B", "A", "C"] and label == "월세 낮은순"
    out, label = sort_regions(df, "월세", desc=True)
    assert list(out.region) == ["C", "A", "B"] and label == "월세 높은순"
    out, label = sort_regions(df, "거리", desc=True)
    assert list(out.region) == ["C", "A", "B"] and label == "거리 먼순"
    out, label = sort_regions(df, "보증금")
    assert list(out.region) == ["C", "A", "B"] and label == "보증금 낮은순"
    out, label = sort_regions(df, "추천")
    assert list(out.region) == ["A", "B", "C"] and label == "추천 높은순"


def test_pick_photo_by_room_cycles_and_falls_back():
    from kkc.ui import pick_photo
    m = {"main": [{"id": "a"}, {"id": "b"}], "living": [{"id": "c"}]}
    assert pick_photo(m, "원룸", 0)["id"] == "a"
    assert pick_photo(m, "원룸", 3)["id"] == "b"  # 사진보다 매물이 많으면 돌려 쓴다
    assert pick_photo(m, "투룸", 1)["id"] == "c"
    assert pick_photo({}, "원룸", 0) is None  # 사진을 안 받았으면 일러스트로
