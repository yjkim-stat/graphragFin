"""Render finance GraphRAG exports into human-friendly artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

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


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        required=True,
        help="Workspace directory containing an exports/ folder produced by run_finance_graphrag-v2.py.",
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
    exports_dir = workspace_dir / "exports"
    if not exports_dir.exists():
        raise FileNotFoundError(
            f"Workspace exports directory not found: {exports_dir}. "
            "Run run_finance_graphrag-v2.py before rendering the view."
        )

    markdown_path = args.markdown_path or (exports_dir / "graphrag_summary.md")
    image_path = args.graph_image_path or (exports_dir / "graphrag_graph.png")

    local_search_exports = _load_json(exports_dir / "local_search_output.json")
    community_exports = _load_json(exports_dir / "community_summaries.json")
    graph_exports = _load_json(exports_dir / "graph_data.json")

    markdown = _build_markdown(
        local_search_exports,
        community_exports,
        graph_exports,
        max_rows=max(args.max_rows, 1),
        text_limit=max(args.text_limit, 0),
    )

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    logger.info("Saved markdown summary to %s", markdown_path)

    if _render_graph_image(graph_exports, image_path):
        logger.info("Graph image generated at %s", image_path)
    else:
        logger.info("Graph image skipped (insufficient data or optional dependencies missing).")


if __name__ == "__main__":
    main()
