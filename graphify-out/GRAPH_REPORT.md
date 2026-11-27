# Graph Report - kkc  (2026-11-27)

## Corpus Check
- 27 files · ~14,497 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 4 file(s) not represented in the graph (top: (none) 3, .toml 1)

## Summary
- 300 nodes · 560 edges · 20 communities (9 shown, 11 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 46 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `5887d335`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_core.py
- AI 자취방 추천 서비스 — 제안서 초안 (팀 KKC)
- SubwayGraph
- test_pipeline.py
- pandas
- KKC · AI 자취방 추천 서비스 — Claude 작업 규칙
- .claude/CLAUDE.md
- install-plugins.sh script
- streamlit_app.py
- ui.py
- test_core.py
- build_data.py

## God Nodes (most connected - your core abstractions)
1. `SubwayGraph` - 17 edges
2. `read_table()` - 13 edges
3. `load_seoul_metro()` - 13 edges
4. `add_osm_routes()` - 12 edges
5. `TransitGraph` - 11 edges
6. `test_subway_transfer_cannot_shortcut_through_bus_stop()` - 10 edges
7. `col()` - 10 edges
8. `small_city()` - 9 edges
9. `_km()` - 9 edges
10. `train()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `다음 할 일` --references--> `SubwayGraph`  [INFERRED]
  README.md → src/kkc/commute/subway.py
- `다음 할 일` --references--> `train()`  [INFERRED]
  README.md → src/kkc/price/model.py
- `load()` --calls--> `load_fov_model()`  [INFERRED]
  app/streamlit_app.py → src/kkc/vision/fov.py
- `walk_station()` --calls--> `nearest_station_by_walk()`  [INFERRED]
  app/streamlit_app.py → src/kkc/commute/subway.py
- `room_thumb()` --calls--> `pick_photo()`  [INFERRED]
  app/streamlit_app.py → src/kkc/ui.py

## Import Cycles
- None detected.

## Communities (20 total, 11 thin omitted)

### Community 0 - "test_core.py"
Cohesion: 0.13
Nodes (13): build_model(), crop_ratio(), exif_fov(), fov_from_focal_35mm(), load_fov_model(), predict_fov(), restore_standard_view(), _to_tensor() (+5 more)

### Community 1 - "AI 자취방 추천 서비스 — 제안서 초안 (팀 KKC)"
Cohesion: 0.11
Nodes (18): 1. 프로젝트 요약, 2. 목적 및 필요성, 3. 프로젝트 개요, 4. 기대효과 및 활용 방안, 5. 구현 계획 (초안), AI 자취방 추천 서비스 — 제안서 초안 (팀 KKC), 경쟁 서비스 비교, 기획 의도 (+10 more)

### Community 2 - "SubwayGraph"
Cohesion: 0.06
Nodes (26): add_osm_routes(), match(), _km(), _line(), load_seoul_metro(), _minutes(), nearest_station(), nearest_station_by_walk() (+18 more)

### Community 3 - "test_pipeline.py"
Cohesion: 0.13
Nodes (13): load_station_coords(), col(), read_table(), load_rent(), dong_table(), _photo(), test_dong_table(), test_exif_fov() (+5 more)

### Community 4 - "pandas"
Cohesion: 0.09
Nodes (21): compare_listings(), rank_view(), Claude Code 플러그인, KKC · AI 자취방 추천 서비스, 구조, 다음 할 일, 빠른 시작, 실제 데이터 넣기 (+13 more)

### Community 5 - "KKC · AI 자취방 추천 서비스 — Claude 작업 규칙"
Cohesion: 0.33
Nodes (5): graphify, KKC · AI 자취방 추천 서비스 — Claude 작업 규칙, 규칙, 이 프로젝트, 적용된 플러그인 (project scope, `.claude/settings.json`)

### Community 15 - "streamlit_app.py"
Cohesion: 0.06
Nodes (21): bus_commute(), deposit_text(), esc(), geocode(), img_b64(), leaflet_map(), load(), map_css() (+13 more)

### Community 16 - "ui.py"
Cohesion: 0.15
Nodes (9): gap_tag(), gap_label(), pick_photo(), sort_regions(), tradeoff_lines(), test_gap_label_only_outside_error_band(), test_pick_photo_by_room_cycles_and_falls_back(), test_sort_regions_both_directions_and_label() (+1 more)

### Community 17 - "test_core.py"
Cohesion: 0.11
Nodes (11): osm_subway_map(), commute_relax(), _minmax(), score_regions(), tradeoff_suggestions(), train_ltr(), test_commute_relax_finds_smallest_extra(), test_nearest_station_by_walk_uses_route_distance_and_falls_back() (+3 more)

## Knowledge Gaps
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SubwayGraph` connect `SubwayGraph` to `pandas`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `SubwayGraph` (e.g. with `다음 할 일` and `TransitGraph`) actually correct?**
  _`SubwayGraph` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_core.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12648221343873517 - nodes in this community are weakly interconnected._
- **Why does `다음 할 일` connect `pandas` to `SubwayGraph`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `load_seoul_metro()` (e.g. with `_minutes()` and `.add_segment()`) actually correct?**
  _`load_seoul_metro()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `AI 자취방 추천 서비스 — 제안서 초안 (팀 KKC)` be split into smaller, more focused modules?**
  _Cohesion score 0.10526315789473684 - nodes in this community are weakly interconnected._
- **Why does `test_subway_transfer_cannot_shortcut_through_bus_stop()` connect `SubwayGraph` to `pandas`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._