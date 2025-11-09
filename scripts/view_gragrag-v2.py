#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a knowledge graph from Parquet files, write a Markdown report,
and export an interactive HTML graph (via igraph + pyvis).

Inputs (in --in-dir):
- entities.parquet (required)
- relationships.parquet (required)
- text_units.parquet (optional)
- communities.parquet (required for community summary)
- community_reports.parquet (optional)
- documents.parquet (optional)

Outputs:
- knowledge_graph_report.md (Markdown)
- graph.pickle (igraph pickle)
- graph.html (optional if --html is specified)

Usage:
    pip install -r requirements.txt
    python build_kg_and_report.py --in-dir /path/to/parquets --out knowledge_graph_report.md --html graph.html
"""

import os
import io
import argparse
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import networkx as nx
from igraph import Graph
from script_utils.utils_view import igraph_to_html

# ---- Utility functions (same as previous script) ----

def read_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except Exception as e:
            raise RuntimeError(
                f"Failed to read {path}. Install pyarrow or fastparquet. Example: pip install pyarrow"
            ) from e

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(".", "_")
        .str.lower()
    )
    return df

def first_present(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def guess_entity_schema(df: pd.DataFrame):
    if df.empty:
        return None, None, None
    cols = list(df.columns)
    id_col = first_present(cols, ["entity_id", "id", "node_id", "eid"])
    name_col = first_present(cols, ["name", "title", "label", "content"])
    type_col = first_present(cols, ["type", "category", "entity_type", "label_type"])
    return id_col, name_col, type_col

def guess_relation_schema(df: pd.DataFrame):
    if df.empty:
        return None, None, None
    cols = list(df.columns)
    src_col = first_present(cols, ["source", "src", "head", "subject", "from", "source_id"])
    dst_col = first_present(cols, ["target", "dst", "tail", "object", "to", "target_id"])
    type_col = first_present(cols, ["relation", "predicate", "edge_type", "type", "relation_type"])
    return src_col, dst_col, type_col

def coerce_str(x):
    try:
        return str(x)
    except Exception:
        return x

def build_graph(entities: pd.DataFrame, relationships: pd.DataFrame):
    ent_df = normalize_columns(entities)
    rel_df = normalize_columns(relationships)

    e_id, e_name, e_type = guess_entity_schema(ent_df)
    r_src, r_dst, r_type = guess_relation_schema(rel_df)

    if e_id is None:
        if not ent_df.empty:
            ent_df = ent_df.reset_index().rename(columns={"index": "entity_id"})
            e_id = "entity_id"

    node_df = pd.DataFrame()
    if not ent_df.empty and e_id in ent_df.columns:
        node_df = pd.DataFrame({"id": ent_df[e_id].map(coerce_str)})
        node_df["name"] = ent_df[e_name].astype(str) if e_name in ent_df.columns else node_df["id"]
        node_df["type"] = ent_df[e_type].astype(str) if e_type in ent_df.columns else "Entity"
        for c in ent_df.columns:
            if c not in [e_id, e_name, e_type]:
                node_df[f"attr_{c}"] = ent_df[c]

    edge_df = pd.DataFrame()
    if not rel_df.empty and r_src in rel_df.columns and r_dst in rel_df.columns:
        edge_df = pd.DataFrame({
            "source": rel_df[r_src].map(coerce_str),
            "target": rel_df[r_dst].map(coerce_str),
        })
        edge_df["relation"] = rel_df[r_type].astype(str) if r_type in rel_df.columns else "related_to"
        for c in rel_df.columns:
            if c not in [r_src, r_dst, r_type]:
                edge_df[f"attr_{c}"] = rel_df[c]

    G = nx.DiGraph()
    if not node_df.empty:
        for _, row in node_df.iterrows():
            G.add_node(row["id"], **{k: row[k] for k in row.index if k != "id"})
    if not edge_df.empty:
        for _, row in edge_df.iterrows():
            G.add_edge(row["source"], row["target"], **{k: row[k] for k in row.index if k not in ["source", "target"]})

    if not edge_df.empty:
        for nid in pd.unique(pd.concat([edge_df["source"], edge_df["target"]], ignore_index=True)):
            if nid not in G:
                G.add_node(nid, name=str(nid), type="Entity")

    return G, node_df, edge_df

def summarize_graph(G: nx.DiGraph, node_df: pd.DataFrame, edge_df: pd.DataFrame) -> dict:
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    deg = dict(G.degree())
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())

    def top_k(d, k=10):
        return sorted(d.items(), key=lambda x: (-x[1], x[0]))[:k]

    id_to_name = {n: (G.nodes[n].get("name") or str(n)) for n in G.nodes}

    def prettify(pairs):
        return [(nid, id_to_name.get(nid, str(nid)), val) for nid, val in pairs]

    rel_counts = {}
    for _, _, data in G.edges(data=True):
        rel = data.get("relation", "related_to")
        rel_counts[rel] = rel_counts.get(rel, 0) + 1
    rel_counts_sorted = sorted(rel_counts.items(), key=lambda x: (-x[1], x[0]))

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "top_degree": prettify(top_k(deg)),
        "top_in": prettify(top_k(in_deg)),
        "top_out": prettify(top_k(out_deg)),
        "relation_counts": rel_counts_sorted,
        "sample_edges": list(G.edges(data=True))[:15],
    }

def summarize_communities(comm_df: pd.DataFrame, comm_reports_df: pd.DataFrame, G: nx.DiGraph):
    comm_df = normalize_columns(comm_df)
    comm_reports_df = normalize_columns(comm_reports_df)

    if comm_df.empty:
        return {"error": "communities table empty or missing"}

    comm_col = first_present(comm_df.columns, ["community", "community_id", "cluster", "group"])
    ent_col = first_present(comm_df.columns, ["entity_id", "id", "node_id", "eid"])
    label_col = first_present(comm_df.columns, ["label", "name", "title", "community_label"])
    size_col = first_present(comm_df.columns, ["size", "community_size", "member_count"])

    if ent_col is None:
        ent_col = first_present(comm_df.columns, ["node", "member", "source"])

    if comm_col is None:
        raise ValueError("Could not infer a community ID column from communities.parquet")

    if ent_col and ent_col in comm_df.columns:
        pairs = comm_df[[comm_col, ent_col]].dropna().copy()
        pairs[ent_col] = pairs[ent_col].map(str)
    else:
        pairs = pd.DataFrame(columns=[comm_col, "entity_placeholder"])

    report_text_col = first_present(comm_reports_df.columns, ["report", "summary", "description", "text", "content"])
    report_comm_col = first_present(comm_reports_df.columns, ["community", "community_id", "cluster", "group"])

    report_map = {}
    if not comm_reports_df.empty and report_text_col and report_comm_col:
        for _, r in comm_reports_df.dropna(subset=[report_text_col, report_comm_col]).iterrows():
            report_map.setdefault(r[report_comm_col], []).append(str(r[report_text_col]))

    result = {}
    for cid, g in comm_df.groupby(comm_col):
        members = []
        if ent_col and ent_col in g.columns:
            members = sorted(set(map(str, g[ent_col].dropna().tolist())))

        sub_nodes = [m for m in members if m in G]
        subG = G.subgraph(sub_nodes).copy() if sub_nodes else nx.DiGraph()
        deg_local = dict(subG.degree()) if sub_nodes else {}
        top_local = sorted(deg_local.items(), key=lambda x: (-x[1], x[0]))[:10]
        id_to_name = {n: (G.nodes[n].get("name") or str(n)) for n in sub_nodes}
        top_local_pretty = [(nid, id_to_name.get(nid, str(nid)), val) for nid, val in top_local]

        label_val = None
        if label_col and label_col in g.columns:
            label_val = g[label_col].mode().iloc[0] if not g[label_col].dropna().empty else None
        size_val = None
        if size_col and size_col in g.columns:
            size_val = int(pd.to_numeric(g[size_col], errors="coerce").dropna().median()) if not g[size_col].dropna().empty else None
        member_count = len(members)

        excerpts = report_map.get(cid, [])

        result[cid] = {
            "label": label_val,
            "declared_size": size_val,
            "member_count": member_count,
            "top_local": top_local_pretty,
            "excerpts": excerpts[:3],
        }

    return result

def to_md_table(rows, headers):
    if not rows:
        return "_No data_\\n"
    s = io.StringIO()
    s.write("| " + " | ".join(headers) + " |\\n")
    s.write("| " + " | ".join(["---"] * len(headers)) + " |\\n")
    for r in rows:
        s.write("| " + " | ".join(map(lambda x: str(x) if x is not None else "", r)) + " |\\n")
    return s.getvalue()

def build_markdown_report(G, node_df, edge_df, graph_summary, community_summary, misc_context) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("# Knowledge Graph Report")
    lines.append("")
    lines.append(f"_Generated: {now}_")
    lines.append("")

    lines.append("## Data Overview")
    overview_rows = [
        ("Nodes", graph_summary["n_nodes"]),
        ("Edges", graph_summary["n_edges"]),
        ("Node attributes", len(node_df.columns) - 1 if not node_df.empty else 0),
        ("Edge attributes", len(edge_df.columns) - 2 if not edge_df.empty else 0),
    ]
    lines.append(to_md_table(overview_rows, ["Metric", "Value"]))

    lines.append("## Graph Structure")
    lines.append("### Top Nodes by Degree")
    lines.append(to_md_table([(nid, name, val) for nid, name, val in graph_summary["top_degree"]],
                             ["Node ID", "Name", "Degree"]))
    lines.append("### Top Nodes by In-Degree")
    lines.append(to_md_table([(nid, name, val) for nid, name, val in graph_summary["top_in"]],
                             ["Node ID", "Name", "In-Degree"]))
    lines.append("### Top Nodes by Out-Degree")
    lines.append(to_md_table([(nid, name, val) for nid, name, val in graph_summary["top_out"]],
                             ["Node ID", "Name", "Out-Degree"]))

    lines.append("### Relation Type Distribution")
    lines.append(to_md_table([(rel, cnt) for rel, cnt in graph_summary["relation_counts"]],
                             ["Relation", "Count"]))

    lines.append("### Sample Edges (Triples)")
    triple_rows = []
    for u, v, data in graph_summary["sample_edges"]:
        rel = data.get("relation", "related_to")
        u_name = G.nodes[u].get("name") if G.has_node(u) else str(u)
        v_name = G.nodes[v].get("name") if G.has_node(v) else str(v)
        triple_rows.append((u, u_name, rel, v, v_name))
    lines.append(to_md_table(triple_rows, ["Source ID", "Source Name", "Relation", "Target ID", "Target Name"]))

    lines.append("## Communities")
    if "error" in community_summary:
        lines.append(f"_Community summary unavailable: {community_summary['error']}_")
    else:
        for cid, info in list(community_summary.items())[:50]:
            lines.append(f"### Community `{cid}`")
            meta_rows = [
                ("Label", info.get("label")),
                ("Declared Size", info.get("declared_size")),
                ("Members (count)", info.get("member_count")),
            ]
            lines.append(to_md_table(meta_rows, ["Field", "Value"]))
            lines.append("#### Top Members by Local Degree")
            lines.append(to_md_table([(nid, name, deg) for nid, name, deg in info.get("top_local", [])],
                                     ["Node ID", "Name", "Local Degree"]))
            excerpts = info.get("excerpts", [])
            if excerpts:
                lines.append("#### Excerpts")
                for t in excerpts:
                    lines.append(f"- {t}")
            lines.append("")

    if misc_context:
        lines.append("## Additional Context")
        for title, bullets in misc_context.items():
            lines.append(f"### {title}")
            for b in bullets[:10]:
                lines.append(f"- {b}")
            lines.append("")

    return "\\n".join(lines)

def extract_misc_context(text_units_df: pd.DataFrame, documents_df: pd.DataFrame):
    context = {}
    docs = normalize_columns(documents_df)
    cand_title = first_present(docs.columns, ["title", "name", "doc_title"])
    if not docs.empty and cand_title:
        titles = list(map(str, docs[cand_title].dropna().unique().tolist()))[:20]
        if titles:
            context["Document Titles"] = titles

    tus = normalize_columns(text_units_df)
    cand_text = first_present(tus.columns, ["text", "content", "chunk", "snippet"])
    if not tus.empty and cand_text:
        snippets = [s.strip() for s in tus[cand_text].dropna().astype(str).tolist() if len(str(s)) < 220][:20]
        if snippets:
            context["Sample Text Units"] = snippets

    return context

def networkx_to_igraph(G_nx: nx.DiGraph) -> Graph:
    """Convert NetworkX DiGraph to igraph Graph with attributes."""
    # Mapping nodes to indices
    nodes = list(G_nx.nodes())
    idx_map = {n: i for i, n in enumerate(nodes)}
    g = Graph(directed=True)

    # Add vertices with attributes
    g.add_vertices(len(nodes))
    # set a stable 'name' attribute for each vertex (original node id)
    g.vs["name"] = [str(n) for n in nodes]

    # propagate common attributes
    # flatten keys across all nodes
    node_attr_keys = set()
    for n, data in G_nx.nodes(data=True):
        node_attr_keys.update(list(data.keys()))
    for key in node_attr_keys:
        g.vs[key] = [G_nx.nodes[n].get(key) for n in nodes]

    # Add edges
    edges = [(idx_map[u], idx_map[v]) for u, v in G_nx.edges()]
    g.add_edges(edges)

    # edge attributes
    edge_attr_keys = set()
    for u, v, data in G_nx.edges(data=True):
        edge_attr_keys.update(list(data.keys()))
    for key in edge_attr_keys:
        g.es[key] = [G_nx.edges[u, v].get(key) for u, v in G_nx.edges()]

    return g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=".", help="Directory containing parquet files")
    ap.add_argument("--out", default="knowledge_graph_report.md", help="Markdown output path")
    ap.add_argument("--pickle", default="graph.pickle", help="igraph pickle output path")
    ap.add_argument("--html", default=None, help="If set, also render interactive HTML to this path")
    ap.add_argument("--no-physics", action="store_true", help="Disable physics in HTML layout")
    args = ap.parse_args()

    def p(name): return os.path.join(args.in_dir, name)

    entities = read_parquet(p("entities.parquet"))
    relationships = read_parquet(p("relationships.parquet"))
    text_units = read_parquet(p("text_units.parquet")) if os.path.exists(p("text_units.parquet")) else pd.DataFrame()
    communities = read_parquet(p("communities.parquet")) if os.path.exists(p("communities.parquet")) else pd.DataFrame()
    community_reports = read_parquet(p("community_reports.parquet")) if os.path.exists(p("community_reports.parquet")) else pd.DataFrame()
    documents = read_parquet(p("documents.parquet")) if os.path.exists(p("documents.parquet")) else pd.DataFrame()

    # G_nx, node_df, edge_df = build_graph(entities, relationships)
    # graph_summary = summarize_graph(G_nx, node_df, edge_df)
    
    # community_summary = summarize_communities(communities, community_reports, G_nx)
    # misc_context = extract_misc_context(text_units, documents)

    # md = build_markdown_report(G_nx, node_df, edge_df, graph_summary, community_summary, misc_context)
    # with open(args.out, "w", encoding="utf-8") as f:
    #     f.write(md)
    # print(f"[OK] Markdown written to: {args.out}")

    # # Export igraph pickle
    # g = networkx_to_igraph(G_nx)
    # g.write_pickle(args.pickle)
    # print(f"[OK] igraph pickle written to: {args.pickle}")

    # # Optional HTML
    # if args.html:
    #     # Inline the HTML generation using the same logic as igraph_to_html.py to avoid import path issues
    #     igraph_to_html(args.pickle, args.html, physics=not args.no_physics)
    #     print(f"[OK] HTML written to: {args.html}")

    # Additional context
    misc_context = {}
    docs = normalize_columns(documents)
    title_col = first_present(docs.columns, ["title", "name", "doc_title"])
    if not docs.empty and title_col:
        misc_context["Document Titles"] = list(map(str, docs[title_col].dropna().unique().tolist()))[:20]
    tus = normalize_columns(text_units)
    text_col = first_present(tus.columns, ["text", "content", "chunk", "snippet"])
    if not tus.empty and text_col:
        misc_context["Sample Text Units"] = [s.strip() for s in tus[text_col].dropna().astype(str).tolist() if len(str(s)) < 220][:20]

    # Communities
    community_summary = summarize_communities(communities, community_reports, G) if not communities.empty else {"error": "communities table empty or missing"}

    html_path = None
    if args.html:
        g = networkx_to_igraph(G)
        g.write_pickle(args.pickle)
        html_path = igraph_to_html(args.pickle, args.html, physics=not args.no_physics)

    md = build_markdown_report(G, node_df, edge_df, graph_summary, community_summary, misc_context, html_path=html_path)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
        
if __name__ == "__main__":
    main()
