"""
RRF 混合融合 — Reciprocal Rank Fusion

策略:
  - 向量检索权重: 1.0 (语义匹配)
  - BM25 检索权重: 0.8 (关键词精确匹配)
  - 图谱检索权重: 1.2 (结构化推理，天然高精度)
"""

from __future__ import annotations

from utils.log import logger


def fuse(
    vec_results: list[dict],
    bm25_results: list[dict],
    graph_results: list[dict],
    top_k: int = 6,
) -> tuple[list[dict], str]:
    """RRF 混合融合：多路检索结果按倒数排名加权合并

    Args:
        vec_results: 向量语义检索结果
        bm25_results: BM25 关键词检索结果
        graph_results: 图谱检索结果
        top_k: 返回 Top-K

    Returns:
        (fused_list, debug_info) 二元组
    """
    weight_vec = 1.0
    weight_bm25 = 0.8
    weight_graph = 1.2

    k = 60  # RRF 平滑常数

    scores: dict[str, tuple[dict, float]] = {}

    def _key(doc: dict) -> str:
        return doc.get("content", "")[:120]

    # 向量结果 (rank starts at 1)
    for rank, doc in enumerate(vec_results, start=1):
        key = _key(doc)
        rrf = weight_vec / (k + rank)
        if key in scores:
            scores[key] = (doc, scores[key][1] + rrf)
        else:
            scores[key] = (doc, rrf)

    # BM25 结果
    for rank, doc in enumerate(bm25_results, start=1):
        key = _key(doc)
        rrf = weight_bm25 / (k + rank)
        if key in scores:
            scores[key] = (doc, scores[key][1] + rrf)
        else:
            scores[key] = (doc, rrf)

    # 图谱结果
    for rank, doc in enumerate(graph_results, start=1):
        key = _key(doc)
        rrf = weight_graph / (k + rank)
        if key in scores:
            scores[key] = (doc, scores[key][1] + rrf)
        else:
            scores[key] = (doc, rrf)

    # Sort by fused score descending
    ranked = sorted(scores.values(), key=lambda x: x[1], reverse=True)
    fused = [doc for doc, _ in ranked[:top_k]]

    debug_info = f"RRF fused: {len(vec_results)} vec + {len(bm25_results)} BM25 + {len(graph_results)} graph → {len(fused)}"
    logger.info(debug_info)
    return fused, debug_info
