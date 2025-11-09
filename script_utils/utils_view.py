# igraph_to_html.py
from igraph import Graph
from pyvis.network import Network

def igraph_to_html(pickle_path: str, output_html: str = "graph.html", physics: bool = True):
    g: Graph = Graph.Read_Pickle(pickle_path)

    net = Network(height="800px", width="100%", directed=g.is_directed(), notebook=False)
    # Choose physics solver
    net.force_atlas_2based() if physics else net.barnes_hut()

    node_attr_keys = g.vs.attributes()
    edge_attr_keys = g.es.attributes()

    # Nodes
    for v in g.vs:
        nid = v.index
        # Prefer content, else name, else id
        if "content" in node_attr_keys and "content" in v.attributes():
            label = str(v["content"])
        elif "name" in node_attr_keys and "name" in v.attributes():
            label = str(v["name"])
        elif "label" in node_attr_keys and "label" in v.attributes():
            label = str(v["label"])
        else:
            label = str(nid)

        # Tooltip HTML
        # title_lines = [f"id: {nid}", f"label: {label}"]
        title_lines = []
        for k in node_attr_keys:
            if k in v.attributes():
                try:
                    val = v[k]
                except Exception:
                    val = ""
                title_lines.append(f"{k}: {val}")
        title_html = "\n".join(title_lines)
        print('viewing node ', nid)
        net.add_node(nid, label=label, title=title_html)

    # Edges
    for e in g.es:
        src, dst = e.tuple
        # print('viewing edge ', src, dst)
        
        title_lines = [f"{src} → {dst}"]
        for k in edge_attr_keys:
            if k in e.attributes():
                try:
                    val = e[k]
                except Exception:
                    val = ""
                title_lines.append(f"{k}: {val}")
        title_html = "\n".join(title_lines)

        width = None
        if 'scaled_weight' in edge_attr_keys:
            # For GraphRAG 
            width = float(e['scaled_weight'])
        else:
            if "weight" in edge_attr_keys and "weight" in e.attributes():
                try:
                    w = float(e["weight"])
                    width = max(1, min(10, abs(w)))
                except Exception:
                    pass

        net.add_edge(src, dst, title=title_html, width=width)

    net.set_options("""
    {
      "nodes": { "shape": "dot", "size": 10 },
      "edges": { "arrows": { "to": { "enabled": true } }, "smooth": false },
      "interaction": { "tooltipDelay": 200, "hideEdgesOnDrag": true },
      "physics": {
        "enabled": true,
        "stabilization": { "enabled": true, "iterations": 200 },
        "barnesHut": { "gravitationalConstant": -8000, "springLength": 120 }
      }
    }
    """)

    net.write_html(output_html)
    print(f"Saved HTML to: {output_html}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle_path")
    ap.add_argument("--out", default="graph.html")
    ap.add_argument("--no-physics", action="store_true")
    args = ap.parse_args()
    igraph_to_html(args.pickle_path, args.out, physics=not args.no_physics)
