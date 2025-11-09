import math
import argparse
from tqdm import tqdm
from statistics import quantiles
from typing import Iterable, Tuple, Set

from igraph import Graph
from pyvis.network import Network

def shorten(s: str, n: int) -> str:
    try:
        return textwrap.shorten(s, width=n, placeholder="…")
    except Exception:
        return s if len(s) <= n else s[:n-1] + "…"
        

def auto_repulsion_params(g, scale: float = 1.0):
    """
    연결된 노드 간 간격을 최대로 벌리도록 튜닝된 버전.
    - spring_length를 node_distance보다 확실히 크게
    - 허브(상위 분위수 차수)일수록 더 벌어지게 가중
    - 중앙 당김↓, 스프링 강도↓(특히 조밀/대규모 그래프)
    """
    n = max(1, g.vcount())
    m = g.ecount()

    # 통계
    degs = g.degree(mode="ALL")
    avg_deg = (2*m)/n if n > 0 else 0.0
    max_deg = max(degs) if degs else 0
    # 상위 90 분위수(허브 영향 측정)
    if degs:
        sd = sorted(degs)
        p90 = sd[int(0.9*(len(sd)-1))]
    else:
        p90 = 0
    density = m / (n*(n-1)/2) if n > 1 else 0.0

    # 기본 최소 거리(노드-노드 반발). 겹침 방지 목적으로 다소 크게.
    node_distance = (
        200
        * (1.0 + 0.9*math.log1p(n))               # 그래프 규모↑
        * (1.0 + 0.7/(math.sqrt(avg_deg+1.0)))    # 희소할수록 ↑
        * (1.0 + 0.6*(1.0 - min(density, 1.0)))   # 밀도 낮을수록 ↑
        * scale
    )

    # 연결된 노드 간 기준 길이(스프링 길이): node_distance보다 확실히 크게.
    # 허브가 강하게 당겨 뭉치는 현상을 억제하기 위해 p90/max_deg 효과 반영.
    hub_factor = 1.0 + 0.4 * math.log1p(max(p90, 1))   # 허브가 클수록 더 길게
    sparse_factor = 1.0 + 0.5/(avg_deg + 1.0)          # 평균 차수 낮으면 더 길게
    spring_length = node_distance * (1.25 * hub_factor * sparse_factor)

    # 중앙 당김은 낮게(모여들지 않도록). 큰 그래프일수록 아주 조금만 증가.
    central_gravity = max(0.03, min(0.12, 0.06 + 0.02*math.log1p(n) - 0.02*math.log1p(avg_deg+1)))

    # 스프링 강도는 약하게(당기는 힘을 줄여 간격 유지). 대규모/고밀도일수록 더 약하게.
    spring_strength = 0.06 / (1.0 + math.log1p(n))     # 기본 약화
    spring_strength *= 1.0 / (1.0 + avg_deg/6.0)       # 조밀할수록 더 약하게
    spring_strength = max(0.015, min(0.05, spring_strength))

    # 큰 그래프일수록 감쇠↑(진동 억제). 길이를 키웠으니 약간 더 높임.
    damping = 0.90 if n < 400 else 0.92

    # 안전하게 정수화/클램프
    node_distance = int(node_distance)
    spring_length = int(max(spring_length, node_distance + 40))  # 최소 간격 보장

    return dict(
        node_distance=node_distance,
        central_gravity=central_gravity,
        spring_length=spring_length,
        spring_strength=spring_strength,
        damping=damping,
    )


def igraph_to_html(pickle_path: str, output_html: str = "graph.html", physics: bool = True, target_keyword=None):
    g: Graph = Graph.Read_Pickle(pickle_path)

    net = Network(height="800px", width="100%", directed=g.is_directed(), notebook=False)
    # net.force_atlas_2based() if physics else net.barnes_hut()
    if physics:
        params = auto_repulsion_params(g, scale=1.0)  # 필요하면 0.8~1.5 정도로 조정
        net.repulsion(**params)
        # net.repulsion(
        #     # node_distance=220, central_gravity=0.15, spring_length=140,
        #     node_distance=500, central_gravity=0.15, spring_length=140,
        #     spring_strength=0.05, damping=0.9
        #     )
    else:
        net.barnes_hut(gravity=-8000, central_gravity=0.15, spring_length=120, damping=0.9)

    node_attr_keys = g.vs.attributes()
    edge_attr_keys = g.es.attributes()

    kept = set()

    # --- 노드 루프 들어가기 전 ---
    all_deg = g.degree(mode="ALL")
    if not all_deg:
        all_deg = [0] * g.vcount()

    # 아웃라이어 완화용 상한(p95); 실패 시 max로 대체
    try:
        p95 = quantiles(all_deg, n=100)[94]  # 95번째 분위수
        cap = max(1, int(p95))
    except Exception:
        cap = max(1, max(all_deg))

    size_min, size_max = 8, 30  # 원하는 크기 범위
    log_cap = math.log1p(cap)

    # 노드 추가
    for v in g.vs:
        nid = v.index
        # 라벨은 name 속성이 있으면 name, 없으면 id
        # label = str(v["name"]) if "name" in node_attr_keys and "name" in v.attributes() else str(nid)
        label = str(v["content"]) if "content" in node_attr_keys and "content" in v.attributes() else str(nid)

        label_src = (
            str(v["content"]) if "content" in node_attr_keys and "content" in v.attributes()
            else str(nid)
        )
        label = shorten(label_src, 15)

        # --- 노드 추가 루프 내부 ---
        raw_deg = all_deg[nid] or 0
        deg_capped = min(raw_deg, cap)

        # 로그 정규화 (0~1)
        norm = math.log1p(deg_capped) / log_cap if log_cap > 0 else 0.0

        # # 크기 매핑
        size = int(size_min + (size_max - size_min) * norm)

        # # degree 기반 크기 스케일 (겹침 감소 + 중요 노드 강조)
        # deg = g.degree(nid, mode="ALL") or 0
        # size = 1 + min(4, deg)

        # 툴팁용 HTML
        # title_lines = [f"id: {nid}", f"label: {label}"]
        title_lines = [f"label: {label}", f'degree : {raw_deg}']
        title_lines = []
        for k in node_attr_keys:
            if k in v.attributes():
                if k not in ['label', 'title', 'body', 'hash_id']:
                    title_lines.append(f"{k}: {v[k]}")
        # title_lines.append(f'body:{label_src}')
        title_html = "\n".join(title_lines)

        # net.add_node(nid, label=label, title=title_html)
        # keyword 필터 판단
        include = True
        if target_keyword is not None:
            include = (target_keyword in label) or (target_keyword in title_html) or (target_keyword in label_src)

        if include:
            net.add_node(
                nid,
                label=label,
                title=title_html,
                value=size,
            )
            kept.add(nid)


    # 엣지 weight 분포 계산
    weights = []
    if "weight" in edge_attr_keys:
        for e in g.es:
            if "weight" in e.attributes():
                try:
                    weights.append(float(e["weight"]))
                except:
                    pass

    if len(weights) > 0:
        min_w = min(weights)
        max_w = max(weights)
    else:
        min_w = max_w = None


    # 엣지 추가
    for e in tqdm(g.es, desc='Adding edges...'):
        src, dst = e.tuple

        # 남아있는 노드 간 연결만 추가
        if target_keyword is not None:
            if (src not in kept) or (dst not in kept):
                continue

        # 엣지 툴팁
        # title_lines = [f"{src} → {dst}"]
        title_lines = []
        for k in edge_attr_keys:
            if k in e.attributes():
                title_lines.append(f"{k}: {e[k]}")
        title_html = "\n".join(title_lines)

        # 가중치/두께 반영(있을 때만)
        width = None
        if "weight" in edge_attr_keys and "weight" in e.attributes():
            try:
                w = float(e["weight"])
                if min_w is not None and max_w is not None and max_w > min_w:
                    # (w - min) / (max - min) → 0~1
                    norm = (w - min_w) / (max_w - min_w)
                    width = 0.1 + 0.9 * norm              # 1~10
                else:
                    width = 0.1  # 분포 없을 때 기본값
            except:
                width = 0.1
        if (target_keyword is not None):
            if (target_keyword in label) or (target_keyword in title_html):
                net.add_edge(src, dst, title=title_html, width=width)
        else:        
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


def _save_pickle(graph: Graph, out_path: str):
    # igraph 버전에 따라 write_pickle 유무가 달라서 안전하게 처리
    try:
        graph.write_pickle(out_path)   # igraph ≥ 0.10
    except AttributeError:
        try:
            Graph.Write_Pickle(graph, out_path)  # 구버전 호환
        except Exception as e:
            # 최후의 수단: 파이썬 pickle (대부분 동작함)
            import pickle
            with open(out_path, "wb") as f:
                pickle.dump(graph, f)

def extract_keyword_subgraph(
    pickle_path: str,
    keyword: str,
    out_pickle: str = "subgraph.pkl",
    hops: int = 0,
    mode: str = "ALL",              # "ALL" | "IN" | "OUT"
    case_sensitive: bool = False,
    attr_keys: Tuple[str, ...] = ("content", "name"),
    keep_isolates: bool = True,     # 고립 노드 제거 여부
) -> Graph:
    """
    keyword를 중심으로 서브그래프를 추출하여 pickle로 저장하고 Graph를 반환.
    - hops=0: 매칭 노드 유도 서브그래프
    - hops>=1: 매칭 노드의 k-hop 이웃을 합친 서브그래프
    """
    g: Graph = Graph.Read_Pickle(pickle_path)

    def match_text(s: str) -> bool:
        if s is None:
            return False
        return (keyword in s) if case_sensitive else (keyword.lower() in s.lower())

    # 1) 키워드 매칭 노드 찾기
    matches: Set[int] = set()
    for v in g.vs:
        for k in attr_keys:
            if k in v.attributes():
                if match_text(str(v[k])):
                    matches.add(v.index)
                    break  # 하나라도 매칭되면 해당 노드 채택

    if not matches:
        # 매칭이 없으면 빈 유도 서브그래프를 반환 (혹은 예외를 던지도록 바꿀 수 있음)
        sub = g.induced_subgraph([])
        _save_pickle(sub, out_pickle)
        return sub

    # 2) k-hop 이웃 확장
    if hops > 0:
        mode_map = {"ALL": "ALL", "IN": "IN", "OUT": "OUT"}
        _mode = mode_map.get(mode.upper(), "ALL")
        keep: Set[int] = set()
        for nid in matches:
            # order=hops: k-hop까지의 neighborhood (자기 자신 포함)
            nbhd = g.neighborhood(vertices=nid, order=hops, mode=_mode)
            keep.update(nbhd)
    else:
        keep = set(matches)

    # 3) 유도 서브그래프 생성
    keep_list = sorted(keep)
    try:
        sub = g.induced_subgraph(keep_list)  # igraph ≥ 0.10
    except TypeError:
        sub = g.subgraph(keep_list)          # 구버전 호환

    # 4) 고립 노드 제거 옵션
    if not keep_isolates and sub.vcount() > 0:
        # degree 0 노드 제외
        degs = sub.degree(mode="ALL")
        keep2 = [i for i, d in enumerate(degs) if d > 0]
        try:
            sub = sub.induced_subgraph(keep2)
        except TypeError:
            sub = sub.subgraph(keep2)

    # 5) 저장
    _save_pickle(sub, out_pickle)
    return sub

# (선택) 바로 HTML로 뿌리고 싶다면: 기존 렌더 함수를 활용
def keyword_subgraph_to_html(
    pickle_path: str,
    keyword: str,
    out_pickle: str = "subgraph.pkl",
    out_html: str = "subgraph.html",
    hops: int = 0,
    mode: str = "ALL",
    **html_kwargs
):
    sub = extract_keyword_subgraph(
        pickle_path, keyword, out_pickle=out_pickle, hops=hops, mode=mode
    )
    # 서브그래프 pickle을 바로 시각화 (필터는 이미 완료되었으니 target_keyword=None)
    # igraph_to_html는 당신이 가진 기존 함수 사용
    igraph_to_html(out_pickle, output_html=out_html, physics=True, target_keyword=None, **html_kwargs)




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=".", help="Directory containing parquet files")
    ap.add_argument("--out", default="knowledge_graph_report.md", help="Markdown output path")
    args = ap.parse_args()

    # 사용 예시
    # igraph_to_html(f"{args.in_dir}/graph.pickle", f'{args.out}.html', physics=True)

    keyword = 'copper'
    keyword_subgraph_to_html(f"{args.in_dir}/graph.pickle", keyword=keyword, hops=1,
                            out_pickle=f"{args.in_dir}/graph-keyword{keyword}.pickle", out_html=f'{args.out}-keyword{keyword}.html')

    keyword = 'future'
    keyword_subgraph_to_html(f"{args.in_dir}/graph.pickle", keyword=keyword, hops=0,
                            out_pickle=f"{args.in_dir}/graph-keyword{keyword}.pickle", out_html=f'{args.out}-keyword{keyword}.html')
