# igraph_to_html.py
from igraph import Graph
from pyvis.network import Network

def igraph_to_html(pickle_path: str, output_html: str = "graph.html", physics: bool = True):
    g: Graph = Graph.Read_Pickle(pickle_path)

    net = Network(height="800px", width="100%", directed=g.is_directed(), notebook=False)
    net.force_atlas_2based() if physics else net.barnes_hut()

    node_attr_keys = g.vs.attributes()
    edge_attr_keys = g.es.attributes()

    # 노드 추가
    for v in g.vs:
        nid = v.index
        # 라벨은 name 속성이 있으면 name, 없으면 id
        # label = str(v["name"]) if "name" in node_attr_keys and "name" in v.attributes() else str(nid)
        label = str(v["content"]) if "content" in node_attr_keys and "content" in v.attributes() else str(nid)
        # 툴팁용 HTML
        title_lines = [f"<b>id</b>: {nid}", f"<b>label</b>: {label}"]
        for k in node_attr_keys:
            if k in v.attributes():
                title_lines.append(f"<b>{k}</b>: {v[k]}")
        title_html = "<br>".join(title_lines)

        net.add_node(nid, label=label, title=title_html)

    # 엣지 추가
    for e in g.es:
        src, dst = e.tuple
        # 엣지 툴팁
        title_lines = [f"<b>{src}</b> → <b>{dst}</b>"]
        for k in edge_attr_keys:
            if k in e.attributes():
                title_lines.append(f"<b>{k}</b>: {e[k]}")
        title_html = "<br>".join(title_lines)

        # 가중치/두께 반영(있을 때만)
        width = None
        if "weight" in edge_attr_keys and "weight" in e.attributes():
            try:
                w = float(e["weight"])
                width = max(1, min(10, abs(w)))  # 1~10 사이로 스케일
            except Exception:
                pass

        net.add_edge(src, dst, title=title_html, width=width)

    # HTML 저장
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
    # 사용 예시
    working_dir = 'TODO'
    
    igraph_to_html(f"{working_dir}/graph.pickle", "hippo-graph.html", physics=True)
