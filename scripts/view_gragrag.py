"""Render finance GraphRAG exports into human-friendly artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Iterable, Optional
import json, math
from collections import defaultdict, deque


logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Any:
    """Read JSON from ``path`` if it exists, returning ``None`` when missing."""

    if not path.exists():
        logger.warning("Skipping missing export: %s", path)
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        msg = f"Failed to parse JSON export at {path}: {exc}"
        raise ValueError(msg) from exc


def _load_parquet(path: Path) -> Any:
    if not path.exists():
        logger.warning("Skipping missing export: %s", path)
        return None

    return pd.read_parquet(path)

def _stringify(value: Any) -> str:
    """Convert arbitrary values into readable strings for markdown tables."""

    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (int, bool)):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)

######################################################################
# Json utils
######################################################################

def _json_safe(obj):
    """파이썬 기본형/리스트/딕트/None/문자열만 남도록 재귀 변환"""
    # NumPy 스칼라
    if isinstance(obj, np.generic):
        return obj.item()
    # NumPy 배열
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # pandas 시계열
    if isinstance(obj, (pd.Timestamp, pd.NaT.__class__)):
        return None if pd.isna(obj) else obj.isoformat()
    # pandas Timedelta
    if isinstance(obj, pd.Timedelta):
        return None if pd.isna(obj) else obj.isoformat()
    # pandas NA/NaN
    if obj is pd.NA or (isinstance(obj, float) and np.isnan(obj)):
        return None
    # dict/list/tuple 재귀 처리
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj  # 파이썬 기본형 등


def _df_json_safe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 자주 문제되는 컬럼들: ndarray -> list
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    return df


def _truncate_text(value: Any, limit: int) -> Any:
    if not isinstance(value, str):
        return value
    if limit <= 0 or len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _format_mapping(mapping: dict[str, Any], indent: int = 0) -> list[str]:
    lines: list[str] = []
    padding = "  " * indent
    for key in sorted(mapping):
        value = mapping[key]
        if isinstance(value, dict):
            lines.append(f"{padding}- **{key}**:")
            lines.extend(_format_mapping(value, indent + 1))
        else:
            lines.append(f"{padding}- **{key}**: {_stringify(value)}")
    return lines


def _markdown_table(
    records: list[dict[str, Any]],
    *,
    max_rows: int,
    text_limit: int,
) -> list[str]:
    if not records:
        return ["_(no data)_"]

    columns: list[str] = []
    for record in records:
        for key in record.keys():
            if key not in columns:
                columns.append(key)

    header = " | ".join(columns)
    separator = " | ".join(["---"] * len(columns))
    lines = [header, separator]

    for record in records[:max_rows]:
        row = []
        for column in columns:
            value = _truncate_text(record.get(column), text_limit)
            row.append(_stringify(value).replace("\n", "<br>"))
        lines.append(" | ".join(row))

    remaining = len(records) - max_rows
    if remaining > 0:
        lines.append(f"\n_… {remaining} more rows omitted_")

    return lines


def _build_context_section(
    context: dict[str, Any],
    *,
    max_rows: int,
    text_limit: int,
) -> list[str]:
    lines: list[str] = []
    for key, value in sorted(context.items()):
        title = key.replace("_", " ").title()
        lines.append(f"### {title}")
        if isinstance(value, list) and value and isinstance(value[0], dict):
            lines.extend(
                _markdown_table(value, max_rows=max_rows, text_limit=text_limit)
            )
        elif isinstance(value, list):
            for item in value[:max_rows]:
                lines.append(f"- {_stringify(item)}")
            if len(value) > max_rows:
                lines.append(f"\n_… {len(value) - max_rows} more items omitted_")
        elif isinstance(value, dict):
            lines.extend(_format_mapping(value))
        else:
            lines.append(_stringify(value))
        lines.append("")
    return lines


def _build_markdown(
    local_search: dict[str, Any] | None,
    community_exports: dict[str, Any] | None,
    graph_exports: dict[str, Any] | None,
    *,
    max_rows: int,
    text_limit: int,
) -> str:
    lines: list[str] = ["# GraphRAG Workspace Summary", ""]

    if not local_search:
        lines.append("_(local search export not found)_")
        return "\n".join(lines)

    query = local_search.get("query")
    if query:
        lines.extend(["## Query", f"> {query}", ""])

    response = local_search.get("response")
    if response:
        lines.extend(["## Response", "```", str(response), "```", ""])

    dataset = local_search.get("dataset")
    if isinstance(dataset, dict):
        lines.append("## Dataset")
        lines.extend(_format_mapping(dataset))
        lines.append("")

    metrics = local_search.get("response_metrics") or {}
    if metrics:
        lines.append("## Response Metrics")
        lines.extend(_format_mapping(metrics))
        lines.append("")

    retrieved_vertices = local_search.get("retrieved_vertices") or []
    if retrieved_vertices:
        lines.append("## Retrieved Vertices")
        for vertex in retrieved_vertices:
            lines.append(f"- {vertex}")
        lines.append("")

    vertex_metrics = local_search.get("retrieved_vertex_metrics") or {}
    if vertex_metrics:
        lines.append("## Retrieved Vertex Metrics")
        lines.extend(_format_mapping(vertex_metrics))
        lines.append("")

    graph_summary = local_search.get("graph_summary") or {}
    if graph_summary:
        lines.append("## Graph Summary")
        lines.extend(_format_mapping(graph_summary))
        lines.append("")

    context = local_search.get("retrieved_context")
    if isinstance(context, dict) and context:
        lines.append("## Retrieved Context")
        lines.extend(
            _build_context_section(context, max_rows=max_rows, text_limit=text_limit)
        )

    if community_exports:
        retrieved_reports = community_exports.get("retrieved_reports") or []
        community_reports = community_exports.get("community_reports") or []
        if retrieved_reports or community_reports:
            lines.append("## Community Reports")
        if retrieved_reports:
            lines.append("### Retrieved Reports")
            lines.extend(
                _markdown_table(
                    retrieved_reports,
                    max_rows=max_rows,
                    text_limit=text_limit,
                )
            )
            lines.append("")
        if community_reports:
            lines.append("### Generated Community Reports")
            lines.extend(
                _markdown_table(
                    community_reports,
                    max_rows=max_rows,
                    text_limit=text_limit,
                )
            )
            lines.append("")

    if graph_exports:
        entity_count = len(graph_exports.get("entities") or [])
        relationship_count = len(graph_exports.get("relationships") or [])
        community_count = len(graph_exports.get("communities") or [])
        lines.append("## Graph Artifacts")
        lines.append(
            f"- **Entities**: {entity_count}\n"
            f"- **Relationships**: {relationship_count}\n"
            f"- **Communities**: {community_count}"
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_graph_image(graph_exports: dict[str, Any] | None, image_path: Path) -> bool:
    if not graph_exports:
        return False

    entities = graph_exports.get("entities") or []
    relationships = graph_exports.get("relationships") or []
    if not entities or not relationships:
        return False

    try:  # pragma: no cover - optional dependency in runtime environment
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        logger.warning("Cannot render graph image because matplotlib/networkx is missing: %s", exc)
        return False

    graph = nx.Graph()
    labels: dict[str, str] = {}

    for entity in entities:
        node_id = (
            entity.get("id")
            or entity.get("entity_id")
            or entity.get("entity")
            or entity.get("name")
            or entity.get("title")
        )
        if node_id is None:
            continue
        node_key = str(node_id)
        label = (
            entity.get("title")
            or entity.get("name")
            or entity.get("entity")
            or node_key
        )
        graph.add_node(node_key)
        labels[node_key] = str(label)

    for relationship in relationships:
        source = (
            relationship.get("source")
            or relationship.get("src")
            or relationship.get("head")
        )
        target = (
            relationship.get("target")
            or relationship.get("dst")
            or relationship.get("tail")
        )
        if source is None or target is None:
            continue
        src = str(source)
        dst = str(target)
        weight = relationship.get("weight")
        try:
            weight_value = float(weight) if weight is not None else 1.0
        except (TypeError, ValueError):
            weight_value = 1.0
        graph.add_edge(src, dst, weight=weight_value)

    if graph.number_of_edges() == 0:
        return False

    degrees = [graph.degree(node) for node in graph.nodes]
    max_degree = max(degrees) if degrees else 1

    pos = None
    try:
        pos = nx.spring_layout(graph, seed=42)
    except Exception:  # pragma: no cover - fallback for unexpected layout errors
        pos = nx.random_layout(graph, seed=42)

    widths = [0.4 + (edge_data.get("weight", 1.0) / max_degree) for _, _, edge_data in graph.edges(data=True)]
    node_sizes = [300 + (deg / max_degree) * 700 for deg in degrees]

    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 8))
    nx.draw_networkx(
        graph,
        pos=pos,
        labels=labels,
        node_size=node_sizes,
        width=widths,
        node_color="#1f77b4",
        edge_color="#636363",
        font_size=8,
    )
    plt.axis("off")
    fig.tight_layout()
    fig.savefig(image_path, dpi=200)
    plt.close(fig)
    logger.info("Saved graph visualisation to %s", image_path)
    return True



def generate_graph_markdown_with_nhops(
    entities, relationships,
    max_hops=2,
    start_ids=None,                 # 예: ["ent_1", "ent_7"]; None이면 모든 엔티티를 시작점으로
    max_paths_per_source=50,        # 시작 노드당 n-hop 경로 최대 개수
    bidirectional=False,            # True면 target->source도 탐색에 포함(무향처럼)
):
    # --- 엔티티 사전 ---
    id2title = {}
    id2desc  = {}
    for _, row in entities.iterrows():
        _id = str(row["id"])
        id2title[_id] = str(row.get("title", _id))
        id2desc[_id]  = str(row.get("description", "")) if row.get("description", "") is not None else ""

    # --- 엣지/인접 리스트 ---
    # edge_map[(u,v)] = {"description": ..., "weight": ...}
    edge_map = {}
    adj = defaultdict(list)
    for _, r in relationships.iterrows():
        u = str(r["source"]); v = str(r["target"])
        desc = str(r.get("description", "")) if r.get("description", "") is not None else ""
        w = r.get("weight", None)
        edge_map[(u, v)] = {"description": desc, "weight": w}
        adj[u].append(v)
        if bidirectional:
            edge_map[(v, u)] = {"description": desc, "weight": w}
            adj[v].append(u)

    # --- 시작점 결정 ---
    if start_ids is None:
        starts = [str(x) for x in entities["id"].tolist()]
    else:
        starts = [str(x) for x in start_ids if str(x) in id2title]

    lines = []
    lines.append("# Text–Relationship–Text Graph Summary\n")

    # 1) 1-hop 요약(직접 연결)
    lines.append("## 1-hop Relationships\n")
    for _, r in relationships.iterrows():
        u = str(r["source"]); v = str(r["target"])
        rel_desc = str(r.get("description", "")) if r.get("description", "") is not None else ""
        u_t = id2title.get(u, u); v_t = id2title.get(v, v)
        u_d = id2desc.get(u, ""); v_d = id2desc.get(v, "")
        lines.append(f"### {u_t} → {v_t}")
        lines.append(f"**Relationship:** {rel_desc}")
        if u_d: lines.append(f"- **{u_t} 설명:** {u_d}")
        if v_d: lines.append(f"- **{v_t} 설명:** {v_d}")
        lines.append("---")

    # 2) n-hop 경로
    if max_hops >= 2:
        lines.append(f"\n## n-hop Paths (up to {max_hops} hops)\n")
        for s in starts:
            s_title = id2title.get(s, s)
            lines.append(f"### Start: {s_title} (`{s}`)")
            found = 0

            # BFS로 simple paths 생성 (사이클 방지)
            # 큐 원소: (path_nodes, path_edges)
            #   path_nodes: [u, ..., v]
            #   path_edges: [(u,u1), (u1,u2), ...]
            q = deque()
            q.append(([s], []))

            # 시작점의 1-hop부터 확장
            while q and found < max_paths_per_source:
                nodes_path, edges_path = q.popleft()
                last = nodes_path[-1]

                # 현재 경로 길이가 hop 수로 이미 max면 더 확장 불가
                if len(edges_path) >= max_hops:
                    continue

                for nxt in adj.get(last, []):
                    if nxt in nodes_path:  # simple path 보장
                        continue
                    new_nodes = nodes_path + [nxt]
                    new_edges = edges_path + [(last, nxt)]

                    # 경로 2개 이상의 노드가 되면 하나의 유효 path로 기록
                    if len(new_edges) >= 1:
                        # 마크다운 한 줄로 경로 표현
                        segs = []
                        for (a, b) in new_edges:
                            ad = edge_map.get((a, b), {}).get("description", "")
                            a_t = id2title.get(a, a)
                            b_t = id2title.get(b, b)
                            segs.append(f"{a_t} --({ad})--> {b_t}")
                        lines.append(f"- " + " -- ".join(segs))
                        found += 1
                        if found >= max_paths_per_source:
                            break

                    # 더 확장 가능하면 큐에 넣기
                    q.append((new_nodes, new_edges))

            if found == 0:
                lines.append("- (no paths)")
            lines.append("")  # 빈 줄

    return "\n".join(lines)



def generate_relationship_markdown(entities, relationships):
    # id → (title, description) 매핑 사전
    entity_info = {
        row["id"]: (row["title"], row.get("description", ""))
        for _, row in entities.iterrows()
    }

    lines = []
    lines.append("# Text–Relationship–Text Graph Summary\n")

    for _, rel in relationships.iterrows():
        src = rel["source"]
        tgt = rel["target"]
        rel_desc = rel.get("description", "")

        # 노드 정보 가져오기
        src_title, src_desc = entity_info.get(src, (src, ""))
        tgt_title, tgt_desc = entity_info.get(tgt, (tgt, ""))

        lines.append(f"## {src_title} → {tgt_title}")
        lines.append("")
        lines.append(f"**Relationship:** {rel_desc}")
        lines.append("")
        if src_desc:
            lines.append(f"- **{src_title} 설명:** {src_desc}")
        if tgt_desc:
            lines.append(f"- **{tgt_title} 설명:** {tgt_desc}")
        lines.append("\n---\n")

    return "\n".join(lines)



def save_graph_html(
    nodes_df, edges_df, out_path="graph.html",
    node_id_col="id",      # 노드 고유 ID 컬럼 (예: "id")
    node_label_col="title",# 노드 라벨로 보여줄 컬럼 (예: "title")
    node_group_col=None,   # 노드 색상 그룹핑용 컬럼 (없으면 자동)
    edge_source_col="source",
    edge_target_col="target",
    edge_weight_col="weight",   # 엣지 가중치(없어도 동작)
    edge_desc_col="description" # 엣지 설명(툴팁)
):
    nodes = nodes_df.copy()
    edges = edges_df.copy()

    # 기본 컬럼 존재 보정
    for col in [node_id_col]:
        if col not in nodes.columns:
            raise ValueError(f"nodes_df에 '{col}' 컬럼이 없습니다.")
    for col in [edge_source_col, edge_target_col]:
        if col not in edges.columns:
            raise ValueError(f"edges_df에 '{col}' 컬럼이 없습니다.")

    # 라벨/그룹 기본값
    if node_label_col not in nodes.columns:
        node_label_col = node_id_col
    if node_group_col and node_group_col not in nodes.columns:
        node_group_col = None

    # 엣지 부가정보 컬럼 유무 체크
    has_weight = edge_weight_col in edges.columns
    has_desc = edge_desc_col in edges.columns

    # 레코드로 변환
    node_records = nodes.to_dict(orient="records")
    edge_records = edges.to_dict(orient="records")

    # D3에서 쓸 형태로 필드 정리
    id_set = set()
    for n in node_records:
        n["_id"] = str(n.get(node_id_col))
        id_set.add(n["_id"])
        n["_label"] = str(n.get(node_label_col, n["_id"]))
        n["_group"] = str(n.get(node_group_col, "default")) if node_group_col else "default"

    cleaned_links = []
    for e in edge_records:
        s = str(e.get(edge_source_col))
        t = str(e.get(edge_target_col))
        if s in id_set and t in id_set:
            e["_source"] = s
            e["_target"] = t
            e["_weight"] = float(e.get(edge_weight_col, 1.0)) if has_weight else 1.0
            e["_desc"] = str(e.get(edge_desc_col, "")) if has_desc else ""
            cleaned_links.append(e)

    data = {
        "nodes": node_records,
        "links": cleaned_links,
        "fieldMap": {
            "nodeId": node_id_col,
            "nodeLabel": node_label_col,
            "nodeGroup": node_group_col,
            "edgeSource": edge_source_col,
            "edgeTarget": edge_target_col,
            "edgeWeight": edge_weight_col if has_weight else None,
            "edgeDesc": edge_desc_col if has_desc else None,
        }
    }

    safe_data = _json_safe(data)

    # HTML 템플릿 (D3 v7, 확대/이동, 검색, 라벨 토글, 가중치에 따른 링크 굵기/길이)
    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Network Graph</title>
<style>
  body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }}
  header {{ padding: 10px 14px; border-bottom: 1px solid #eee; position: sticky; top: 0; background: #fff; z-index: 2; }}
  .controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  .controls input[type="text"] {{ padding: 6px 10px; border: 1px solid #ddd; border-radius: 8px; min-width: 240px; }}
  .controls label {{ display: inline-flex; gap: 6px; align-items: center; }}
  #graph {{ width: 100vw; height: calc(100vh - 56px); }}
  .node circle {{ stroke: #333; stroke-width: 0.5px; }}
  .node text {{ font-size: 11px; pointer-events: none; opacity: 0.9; }}
  .link {{ stroke: #999; stroke-opacity: 0.6; }}
  .highlight circle {{ stroke: #000; stroke-width: 2px; }}
  .tooltip {{
    position: absolute; pointer-events: none; background: rgba(0,0,0,0.78); color: #fff;
    padding: 8px 10px; border-radius: 8px; font-size: 12px; line-height: 1.25; z-index: 3;
  }}
</style>
</head>
<body>
<header>
  <div class="controls">
    <input id="search" type="text" placeholder="노드 검색 (id / 라벨)"/>
    <label><input id="toggle-labels" type="checkbox" checked/> 라벨 표시</label>
    <label>링크 길이 <input id="linkDist" type="range" min="40" max="240" value="120"/></label>
    <label>링크 강도 <input id="linkStr" type="range" min="10" max="100" value="50"/></label>
    <span id="stats"></span>
  </div>
</header>
<div id="graph"></div>

<script id="graph-data" type="application/json">{json.dumps(safe_data, ensure_ascii=False)}</script>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>
(() => {{
  const cfg = JSON.parse(document.getElementById("graph-data").textContent);
  const nodes = cfg.nodes;
  const links = cfg.links;

  // 색상 스케일 (그룹마다)
  const groups = [...new Set(nodes.map(d => d._group))];
  const color = d3.scaleOrdinal().domain(groups).range(d3.schemeTableau10);

  // 통계
  document.getElementById("stats").textContent = `노드: ${{nodes.length}} · 엣지: ${{links.length}} · 그룹: ${{groups.length}}`;

  const container = document.getElementById("graph");
  const width = container.clientWidth;
  const height = container.clientHeight;

  const svg = d3.select("#graph").append("svg")
    .attr("width", width)
    .attr("height", height);

  const g = svg.append("g");

  const zoom = d3.zoom().scaleExtent([0.1, 5]).on("zoom", (event) => {{
    g.attr("transform", event.transform);
  }});
  svg.call(zoom);

  // 링크와 노드
  const link = g.append("g")
      .attr("stroke-linecap", "round")
    .selectAll("line")
    .data(links)
    .join("line")
      .attr("class", "link")
      .attr("stroke-width", d => 0.5 + Math.sqrt(d._weight || 1));

  const node = g.append("g")
    .selectAll("g")
    .data(nodes)
    .join("g")
      .attr("class", "node")
      .call(drag(simulation));

  node.append("circle")
      .attr("r", 6)
      .attr("fill", d => color(d._group || "default"));

  const labels = node.append("text")
      .attr("x", 9)
      .attr("y", 3)
      .text(d => d._label);

  // 툴팁
  const tooltip = d3.select("body").append("div").attr("class", "tooltip").style("opacity", 0);

  node.on("mouseover", (event, d) => {{
    tooltip.style("opacity", 1)
      .html(`<b>${{d._label}}</b><br/><small>id: ${{d._id}} | group: ${{d._group}}</small>`);
  }}).on("mousemove", (event) => {{
    tooltip.style("left", (event.pageX + 10) + "px")
           .style("top", (event.pageY + 10) + "px");
  }}).on("mouseout", () => tooltip.style("opacity", 0));

  link.on("mouseover", (event, d) => {{
    const desc = d._desc ? `<br/>${{d._desc}}` : "";
    tooltip.style("opacity", 1)
      .html(`🔗 <b>${{d._source}}</b> → <b>${{d._target}}</b><br/>w=${{d._weight}}${{desc}}`);
  }}).on("mousemove", (event) => {{
    tooltip.style("left", (event.pageX + 10) + "px")
           .style("top", (event.pageY + 10) + "px");
  }}).on("mouseout", () => tooltip.style("opacity", 0));

  // 시뮬레이션
  const linkDistInput = document.getElementById("linkDist");
  const linkStrInput = document.getElementById("linkStr");

  const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links)
      .id(d => d._id)
      .distance(d => (+linkDistInput.value) / Math.sqrt(d._weight || 1))
      .strength(() => (+linkStrInput.value)/100))
    .force("charge", d3.forceManyBody().strength(-80))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(16));

  simulation.on("tick", () => {{
    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);

    node.attr("transform", d => `translate(${{d.x}}, ${{d.y}})`);
  }});

  linkDistInput.addEventListener("input", () => simulation.alpha(0.5).restart());
  linkStrInput.addEventListener("input", () => simulation.alpha(0.5).restart());

  // 라벨 토글
  const toggle = document.getElementById("toggle-labels");
  const updateLabelsVisibility = () => labels.style("display", toggle.checked ? null : "none");
  toggle.addEventListener("change", updateLabelsVisibility);
  updateLabelsVisibility();

  // 드래그
  function drag(sim) {{
    function dragstarted(event, d) {{
      if (!event.active) sim.alphaTarget(0.3).restart();
      d.fx = d.x; d.fy = d.y;
    }}
    function dragged(event, d) {{
      d.fx = event.x; d.fy = event.y;
    }}
    function dragended(event, d) {{
      if (!event.active) sim.alphaTarget(0);
      d.fx = null; d.fy = null;
    }}
    return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended);
  }}

  // 검색/하이라이트
  const search = document.getElementById("search");
  search.addEventListener("input", () => {{
    const q = search.value.trim().toLowerCase();
    node.classed("highlight", d =>
      !q ? false : (d._id.toLowerCase().includes(q) || (d._label||"").toLowerCase().includes(q))
    );
  }});
}})();
</script>
</body>
</html>
"""
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"Saved: {Path(out_path).resolve()}")

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        required=True,
        help="Workspace directory containing an exports/ folder produced by run_finance_graphrag-v2.py.",
    )
    parser.add_argument(
        "--community-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--document-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--entity-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--relationship-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--textunit-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--markdown-path",
        type=Path,
        default=None,
        help="Optional path for the rendered markdown summary (defaults to exports/graphrag_summary.md).",
    )
    parser.add_argument(
        "--graph-image-path",
        type=Path,
        default=None,
        help="Optional path for the rendered graph PNG (defaults to exports/graphrag_graph.png).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=15,
        help="Maximum number of rows to include per table in the markdown output.",
    )
    parser.add_argument(
        "--text-limit",
        type=int,
        default=280,
        help="Maximum number of characters per text field in the markdown tables.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. INFO, DEBUG).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    workspace_dir = args.workspace_dir.resolve()
    output_dir = workspace_dir / "output"
    if not output_dir.exists():
        raise FileNotFoundError(
            f"Workspace output directory not found: {output_dir}. "
            "Run run_finance_graphrag-v2.py before rendering the view."
        )

    community_path = args.community_path or (output_dir / "comunities.parquet")
    document_path = args.document_path or (output_dir / "documents.parquet")
    entity_path = args.entity_path or (output_dir / "entities.parquet")
    relationship_path = args.relationship_path or (output_dir / "relationships.parquet")
    textunit_path = args.textunit_path or (output_dir / "test_units.parquet")

    markdown_path = args.markdown_path or (output_dir / "graphrag_summary.md")
    image_path = args.graph_image_path or (output_dir / "graphrag_graph.png")

    communities = _load_parquet(community_path)
    documents = _load_parquet(document_path)
    entities = _load_parquet(entity_path)
    relationships = _load_parquet(relationship_path)
    test_units = _load_parquet(textunit_path)

    print('='*50)
    print(f'\t communities')
    # print(communities.info())
    print('='*50)
    print(f'\t documents')
    print(documents.info())
    print('='*50)
    print(f'\t entities')
    print(entities.info())
    print('='*50)
    print(f'\t relationships')
    print(relationships.info())
    print('='*50)
    print(f'\t test_units')
    # print(test_units.info())
    print('='*50)


    # Output 1 : HTML for graph visualization

    save_graph_html(entities, relationships, out_path= output_dir / "view-graph.html")

    # 실제 실행
    # markdown_output = generate_relationship_markdown(entities, relationships)
    # (output_dir / "view-graph.md").write_text(markdown_output, encoding='utf-8')
    markdown_output_nhop = generate_graph_markdown_with_nhops(
        entities, relationships,
        max_hops=3,                       # 최대 3-hop
        start_ids=None,                   # 모든 엔티티를 시작점으로
        max_paths_per_source=30,          # 시작점당 30개 경로 제한
        bidirectional=False               # True면 무향처럼 확장
    )
    (output_dir / "view-graph-nhops.md").write_text(markdown_output_nhop, encoding='utf-8')

    # save_graph_html(nodes_df, edges_df,
    #                 node_id_col="id", node_label_col="title",
    #                 edge_source_col="source", edge_target_col="target",
    #                 edge_weight_col="weight", edge_desc_col="description",
    #                 out_path="graph.html")





    # Using documents and entit
    # local_search_exports = _load_json(output_dir / "local_search_output.json")
    # community_exports = _load_json(output_dir / "community_summaries.json")
    # graph_exports = _load_json(output_dir / "graph_data.json")

    # markdown = _build_markdown(
    #     local_search_exports,
    #     community_exports,
    #     graph_exports,
    #     max_rows=max(args.max_rows, 1),
    #     text_limit=max(args.text_limit, 0),
    # )

    # markdown_path.parent.mkdir(parents=True, exist_ok=True)
    # markdown_path.write_text(markdown, encoding="utf-8")
    # logger.info("Saved markdown summary to %s", markdown_path)

    # if _render_graph_image(graph_exports, image_path):
    #     logger.info("Graph image generated at %s", image_path)
    # else:
    #     logger.info("Graph image skipped (insufficient data or optional dependencies missing).")


if __name__ == "__main__":
    main()
