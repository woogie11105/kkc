# Graph Report - 내프로젝트  (2026-11-29)

## Corpus Check
- 28 files · ~16,320 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 4 file(s) not represented in the graph (top: (none) 3, .toml 1)

## Summary
- 326 nodes · 633 edges · 21 communities (10 shown, 11 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 46 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `08945a63`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_core.py
- AI 자취방 추천 서비스 — 제안서 초안 (팀 KKC)
- pandas
- SubwayGraph
- pandas
- KKC · AI 자취방 추천 서비스 — Claude 작업 규칙
- .claude/CLAUDE.md
- install-plugins.sh script
- streamlit_app.py
- ui.py
- test_core.py
- unsplash_listings.py

## God Nodes (most connected - your core abstractions)
1. `SubwayGraph` - 19 edges
2. `load_seoul_metro()` - 13 edges
3. `read_table()` - 13 edges
4. `add_osm_routes()` - 12 edges
5. `TransitGraph` - 12 edges
6. `small_city()` - 10 edges
7. `test_subway_transfer_cannot_shortcut_through_bus_stop()` - 10 edges
8. `col()` - 10 edges
9. `_km()` - 9 edges
10. `train()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `다음 할 일` --references--> `SubwayGraph`  [INFERRED]
  README.md → src/kkc/commute/subway.py
- `다음 할 일` --references--> `train()`  [INFERRED]
  README.md → src/kkc/price/model.py
- `rank_view()` --calls--> `converted_rent()`  [INFERRED]
  app/streamlit_app.py → src/kkc/price/model.py
- `room_thumb()` --calls--> `pick_photo()`  [INFERRED]
  app/streamlit_app.py → src/kkc/ui.py
- `gap_tag()` --calls--> `band_for()`  [INFERRED]
  app/streamlit_app.py → src/kkc/price/model.py

## Import Cycles
- None detected.

## Communities (21 total, 11 thin omitted)

### Community 0 - "test_core.py"
Cohesion: 0.10
Nodes (16): osm_subway_map(), build_model(), crop_ratio(), exif_fov(), fov_from_focal_35mm(), load_fov_model(), predict_fov(), restore_standard_view() (+8 more)

### Community 1 - "AI 자취방 추천 서비스 — 제안서 초안 (팀 KKC)"
Cohesion: 0.11
Nodes (18): 1. 프로젝트 요약, 2. 목적 및 필요성, 3. 프로젝트 개요, 4. 기대효과 및 활용 방안, 5. 구현 계획 (초안), AI 자취방 추천 서비스 — 제안서 초안 (팀 KKC), 경쟁 서비스 비교, 기획 의도 (+10 more)

### Community 2 - "pandas"
Cohesion: 0.06
Nodes (28): _line(), load_seoul_metro(), load_station_coords(), _minutes(), nearest_station(), nearest_station_by_walk(), region_commute(), _station() (+20 more)

### Community 3 - "SubwayGraph"
Cohesion: 0.10
Nodes (14): add_osm_routes(), match(), _km(), SubwayGraph, TransitGraph, walk_minutes(), test_add_osm_routes_fills_missing_stations_by_position(), test_add_osm_routes_matches_parenthesized_name() (+6 more)

### Community 4 - "pandas"
Cohesion: 0.09
Nodes (20): compare_listings(), Claude Code 플러그인, KKC · AI 자취방 추천 서비스, 구조, 다음 할 일, 빠른 시작, 실제 데이터 넣기, band_for() (+12 more)

### Community 5 - "KKC · AI 자취방 추천 서비스 — Claude 작업 규칙"
Cohesion: 0.33
Nodes (5): graphify, KKC · AI 자취방 추천 서비스 — Claude 작업 규칙, 규칙, 이 프로젝트, 적용된 플러그인 (project scope, `.claude/settings.json`)

### Community 15 - "streamlit_app.py"
Cohesion: 0.05
Nodes (23): bus_commute(), deposit_text(), esc(), geocode(), img_b64(), leaflet_map(), load(), map_css() (+15 more)

### Community 16 - "ui.py"
Cohesion: 0.15
Nodes (9): gap_tag(), gap_label(), pick_photo(), sort_regions(), tradeoff_lines(), test_gap_label_only_outside_error_band(), test_pick_photo_by_room_cycles_and_falls_back(), test_sort_regions_both_directions_and_label() (+1 more)

### Community 17 - "test_core.py"
Cohesion: 0.15
Nodes (8): rank_view(), commute_relax(), _minmax(), score_regions(), tradeoff_suggestions(), train_ltr(), test_commute_relax_finds_smallest_extra(), test_ranking_and_tradeoff()

### Community 20 - "unsplash_listings.py"
Cohesion: 0.14
Nodes (10): fetch(), main(), api_get(), build_listings(), download_file(), fill_pools(), load_manifest(), main() (+2 more)

## Knowledge Gaps
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SubwayGraph` connect `SubwayGraph` to `pandas`, `pandas`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `SubwayGraph` (e.g. with `다음 할 일` and `TransitGraph`) actually correct?**
  _`SubwayGraph` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_core.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10344827586206896 - nodes in this community are weakly interconnected._
- **Why does `TransitGraph` connect `SubwayGraph` to `pandas`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Should `AI 자취방 추천 서비스 — 제안서 초안 (팀 KKC)` be split into smaller, more focused modules?**
  _Cohesion score 0.10526315789473684 - nodes in this community are weakly interconnected._
- **Why does `다음 할 일` connect `pandas` to `SubwayGraph`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Should `pandas` be split into smaller, more focused modules?**
  _Cohesion score 0.05952380952380952 - nodes in this community are weakly interconnected._