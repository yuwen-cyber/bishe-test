#!/usr/bin/env python3
"""混合检索对比评测（创新点二：SQL 预筛 + 向量召回）。

在同一查询集上对比三种检索方式（同口径、离线直连 MySQL + Milvus，不经过后端 HTTP）：

  - keyword : 纯 SQL 全文检索（MySQL MATCH ... AGAINST，ngram）+ 情绪/时间过滤
  - vector  : 纯向量语义召回（bge 编码 -> Milvus ANN）+ 标量过滤
  - hybrid  : SQL 全文召回 与 向量召回 两路候选 RRF 融合排序（SQL+向量混合）

融合说明：
- SQL 全文相关性分数与向量余弦相似度尺度不同、不可直接相加，故采用 RRF
  （Reciprocal Rank Fusion，RRF 分 = Σ 1/(k+rank)，k=60），只依赖两路排名做融合。
- 本脚本为离线评测实现，检索融合策略与生产接口一致；不修改后端现有路由与业务逻辑。

指标口径与 run_eval.py 一致：Recall@1/5/10、MRR、NDCG@10、平均耗时。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\run_hybrid_eval.py
    .venv\\Scripts\\python.exe experiments\\run_hybrid_eval.py --query-set experiments/output/query_set_db.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app import embeddings, emotion, milvus_client  # noqa: E402
from app.database import SessionLocal  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("hybrid_eval")

CAND_N = 50          # 每路召回候选数
TOP_K = 10           # 评估截断
RRF_K = 60           # RRF 平滑常数
# 混合路权重：hybrid 等权；hybrid_weighted 给向量路更高权重（向量为主、SQL 辅助过滤/补召回）
CONFIGS = ["keyword", "vector", "hybrid", "hybrid_weighted"]
RRF_WEIGHTS = {
    "hybrid": (1.0, 1.0),
    "hybrid_weighted": (1.0, 2.0),
}

_BASE_SQL = (
    "SELECT s.id AS segment_id, s.asr_text AS text, s.emotion AS emotion, "
    "s.start_time AS start_time, c.video_object_key AS video_key, "
    "c.audio_object_key AS audio_key, "
    "MATCH(s.asr_text) AGAINST (:q IN NATURAL LANGUAGE MODE) AS score "
    "FROM segments s JOIN conversations c ON s.conversation_id = c.id "
)


def _sql_literal(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "''") + "'"


def _dcg(rels: list[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def _ndcg_at_k(seg_ids: list[int], relevant: set[int], k: int) -> float:
    rels = [1.0 if sid in relevant else 0.0 for sid in seg_ids[:k]]
    dcg = _dcg(rels)
    ideal = [1.0] * min(len(relevant), k)
    idcg = _dcg(ideal)
    return dcg / idcg if idcg > 0.0 else 0.0


def _recall_at_k(seg_ids: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for sid in seg_ids[:k] if sid in relevant) / len(relevant)


def _precision_at_k(seg_ids: list[int], relevant: set[int], k: int) -> float:
    """Precision@K = top-K 内命中真值数 / top-K 实际返回条数。"""
    top = seg_ids[:k]
    if not top:
        return 0.0
    return sum(1 for sid in top if sid in relevant) / len(top)


def _f1_at_k(seg_ids: list[int], relevant: set[int], k: int) -> float:
    p = _precision_at_k(seg_ids, relevant, k)
    r = _recall_at_k(seg_ids, relevant, k)
    if p + r == 0.0:
        return 0.0
    return 2 * p * r / (p + r)


def _mrr(seg_ids: list[int], relevant: set[int]) -> float:
    for rank, sid in enumerate(seg_ids, start=1):
        if sid in relevant:
            return 1.0 / rank
    return 0.0


def _percentile(values: list[float], q: float) -> float:
    """线性插值分位数，口径与 numpy.percentile 默认一致。"""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _infer_emotion(q: str, emotions: list[str]) -> list[str]:
    emo = list(emotions or [])
    if not emo:
        inferred = emotion.infer_emotion_from_lexicon(q)
        if inferred:
            emo = [inferred]
    return emo


def sql_recall(db, q: str, emo_list: list[str], cand_n: int) -> list[int]:
    """纯 SQL 全文召回，返回按相关度排序的 segment_id 列表。"""
    clauses: list[str] = []
    params: dict = {"q": q}
    if emo_list:
        clauses.append("s.emotion IN (" + ",".join(_sql_literal(e) for e in emo_list) + ")")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = text(
        _BASE_SQL + where + " ORDER BY score DESC, s.is_demo ASC LIMIT :n"
    )
    params["n"] = cand_n
    rows = db.execute(sql, params).mappings().all()
    return [int(r["segment_id"]) for r in rows]


def vector_recall(q: str, emo_list: list[str], cand_n: int) -> list[int]:
    """纯向量召回，返回按余弦相似度排序的 segment_id 列表。"""
    vec = embeddings.embed_text(q)
    hits = milvus_client.search_vectors(vec, top_k=cand_n, emotions=emo_list)
    return [int(h["segment_id"]) for h in hits]


def rrf_fuse(keyword_ids: list[int], vector_ids: list[int], rrf_k: int,
             w_kw: float = 1.0, w_vec: float = 1.0) -> list[int]:
    """RRF 融合两路排名，返回融合后的 segment_id 排序。

    加权 RRF：score = w/(k+rank)，权重越大该路排名越受重视。
    """
    scores: dict[int, float] = {}
    for rank, sid in enumerate(keyword_ids, start=1):
        scores[sid] = scores.get(sid, 0.0) + w_kw / (rrf_k + rank)
    for rank, sid in enumerate(vector_ids, start=1):
        scores[sid] = scores.get(sid, 0.0) + w_vec / (rrf_k + rank)
    return sorted(scores, key=lambda s: -scores[s])


def main() -> int:
    parser = argparse.ArgumentParser(description="混合检索（SQL+向量）对比评测")
    parser.add_argument("--query-set", default="experiments/output/query_set_db.json")
    parser.add_argument("--output-dir", default="experiments/output")
    parser.add_argument("--cand-n", type=int, default=CAND_N)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--warmup-queries", type=int, default=1,
                        help="预热查询数：先用前 N 条跑一遍 SQL 与向量两路，排除冷启动，不计入统计")
    args = parser.parse_args()

    top_k = args.top_k
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = json.loads(Path(args.query_set).read_text(encoding="utf-8"))
    embeddings.warmup()

    db = SessionLocal()
    detail_rows: list[dict] = []
    agg: dict[str, dict[str, list[float]]] = {
        c: {"r1": [], "r5": [], "r10": [], "p1": [], "p5": [], "p10": [],
            "f1": [], "f5": [], "f10": [], "mrr": [], "ndcg": [], "time": []}
        for c in CONFIGS
    }

    # ---- 预热：排除首次连接/首次编码的冷启动抖动，不计入统计 ----
    if args.warmup_queries > 0:
        for q in queries[: args.warmup_queries]:
            qtext = q.get("query", "")
            emo_list = _infer_emotion(qtext, q.get("emotions") or [])
            sql_recall(db, qtext, emo_list, args.cand_n)
            vector_recall(qtext, emo_list, args.cand_n)
        print(f"[info] 预热完成 {args.warmup_queries} 条查询，不计入统计")

    for q in queries:
        qid = q.get("query_id")
        qtext = q.get("query", "")
        relevant = set(int(x) for x in q.get("relevant_segment_ids", []))
        emo_list = _infer_emotion(qtext, q.get("emotions") or [])

        # 两路召回（计时）
        t0 = time.perf_counter()
        kw_ids = sql_recall(db, qtext, emo_list, args.cand_n)
        t_sql = time.perf_counter() - t0

        t1 = time.perf_counter()
        vec_ids = vector_recall(qtext, emo_list, args.cand_n)
        t_vec = time.perf_counter() - t1

        hybrid_ids = rrf_fuse(kw_ids, vec_ids, RRF_K)
        hybrid_w_ids = rrf_fuse(kw_ids, vec_ids, RRF_K, *RRF_WEIGHTS["hybrid_weighted"])

        results = {
            "keyword": (kw_ids, t_sql * 1000),
            "vector": (vec_ids, t_vec * 1000),
            "hybrid": (hybrid_ids, (t_sql + t_vec) * 1000),
            "hybrid_weighted": (hybrid_w_ids, (t_sql + t_vec) * 1000),
        }

        for name, (seg_ids, elapsed) in results.items():
            r1 = _recall_at_k(seg_ids, relevant, 1)
            r5 = _recall_at_k(seg_ids, relevant, 5)
            r10 = _recall_at_k(seg_ids, relevant, 10)
            p1 = _precision_at_k(seg_ids, relevant, 1)
            p5 = _precision_at_k(seg_ids, relevant, 5)
            p10 = _precision_at_k(seg_ids, relevant, 10)
            f1 = _f1_at_k(seg_ids, relevant, 1)
            f5 = _f1_at_k(seg_ids, relevant, 5)
            f10 = _f1_at_k(seg_ids, relevant, 10)
            mrr = _mrr(seg_ids, relevant)
            ndcg = _ndcg_at_k(seg_ids, relevant, top_k)

            agg[name]["r1"].append(r1)
            agg[name]["r5"].append(r5)
            agg[name]["r10"].append(r10)
            agg[name]["p1"].append(p1)
            agg[name]["p5"].append(p5)
            agg[name]["p10"].append(p10)
            agg[name]["f1"].append(f1)
            agg[name]["f5"].append(f5)
            agg[name]["f10"].append(f10)
            agg[name]["mrr"].append(mrr)
            agg[name]["ndcg"].append(ndcg)
            agg[name]["time"].append(elapsed)

            detail_rows.append({
                "query_id": qid,
                "query": qtext,
                "config": name,
                "q_emotion": emo_list[0] if emo_list else "",
                "relevant_ids": "|".join(str(x) for x in sorted(relevant)),
                "retrieved_segment_ids": "|".join(str(x) for x in seg_ids[:top_k]),
                "hit_segment_ids": "|".join(str(x) for x in seg_ids if x in relevant),
                "recall@1": round(r1, 4),
                "recall@5": round(r5, 4),
                "recall@10": round(r10, 4),
                "precision@1": round(p1, 4),
                "precision@5": round(p5, 4),
                "precision@10": round(p10, 4),
                "f1@1": round(f1, 4),
                "f1@5": round(f5, 4),
                "f1@10": round(f10, 4),
                "mrr": round(mrr, 4),
                "ndcg@10": round(ndcg, 4),
                "time_ms": round(elapsed, 2),
            })

    db.close()

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    metrics_rows = []
    for name in CONFIGS:
        a = agg[name]
        metrics_rows.append({
            "config": name,
            "recall@1": round(mean(a["r1"]), 4),
            "recall@5": round(mean(a["r5"]), 4),
            "recall@10": round(mean(a["r10"]), 4),
            "precision@1": round(mean(a["p1"]), 4),
            "precision@5": round(mean(a["p5"]), 4),
            "precision@10": round(mean(a["p10"]), 4),
            "f1@1": round(mean(a["f1"]), 4),
            "f1@5": round(mean(a["f5"]), 4),
            "f1@10": round(mean(a["f10"]), 4),
            "mrr": round(mean(a["mrr"]), 4),
            "ndcg@10": round(mean(a["ndcg"]), 4),
            "avg_time_ms": round(mean(a["time"]), 2),
            "p50_time_ms": round(_percentile(a["time"], 0.50), 2),
            "p95_time_ms": round(_percentile(a["time"], 0.95), 2),
            "max_time_ms": round(max(a["time"]) if a["time"] else 0.0, 2),
            "num_queries": len(queries),
        })

    _write_csv(out_dir / "metrics_hybrid.csv", metrics_rows)
    _write_csv(out_dir / "per_query_hybrid.csv", detail_rows)

    print(f"\n=== 混合检索对比（{len(queries)} 查询，候选 {args.cand_n}，top{top_k}，RRF k={RRF_K}）===")
    for r in metrics_rows:
        print(f"{r['config']:10s} R@1={r['recall@1']:.4f} R@5={r['recall@5']:.4f} "
              f"R@10={r['recall@10']:.4f} P@5={r['precision@5']:.4f} F1@5={r['f1@5']:.4f} "
              f"MRR={r['mrr']:.4f} NDCG@10={r['ndcg@10']:.4f} "
              f"耗时={r['avg_time_ms']}ms P95={r['p95_time_ms']}ms")
    print(f"\n汇总 -> {out_dir / 'metrics_hybrid.csv'}")
    print(f"明细 -> {out_dir / 'per_query_hybrid.csv'}")
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
