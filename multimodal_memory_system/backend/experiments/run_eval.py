#!/usr/bin/env python3
"""独立检索评测脚本：调用本地 FastAPI，计算 Recall@K / MRR / NDCG@10。

用法：
    python experiments/run_eval.py --base-url http://localhost:8000 --query-set experiments/query_set.json

说明：
- 不 import app 内部模块，全部走 HTTP API。
- 四条检索配置：
    A. baseline           GET /api/baseline/search          （MySQL 全文检索）
    B. advanced_plain     GET /api/advanced/search expand=none
    C. advanced_lexicon   GET /api/advanced/search expand=lexicon
    D. advanced_llm       GET /api/advanced/search expand=llm
- 输出到 experiments/output/：
    metrics.csv           每个配置的汇总指标
    per_query_detail.csv  每条查询 x 每个配置的命中详情

注意：
- 当前后端两个检索接口最多返回 top 3，所以 Recall@5 / Recall@10 / NDCG@10
  实际最多只能计入 3 条命中。
- advanced_search 目前尚未实现 expand 参数，B/C/D 三者的结果会相同（expand 被后端忽略）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests

# (配置名, 接口路径, 额外 query 参数)
CONFIGS: list[tuple[str, str, dict[str, str]]] = [
    ("baseline", "/api/baseline/search", {}),
    ("advanced_plain", "/api/advanced/search", {"expand": "none"}),
    ("advanced_lexicon", "/api/advanced/search", {"expand": "lexicon"}),
    ("advanced_llm", "/api/advanced/search", {"expand": "llm"}),
]

K_VALUES = (1, 5, 10)


def _dcg(rels: list[float]) -> float:
    """二进制相关度下的 DCG，discount 采用 log2(i+2)（i 从 0 开始）。"""
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
    hits = sum(1 for sid in seg_ids[:k] if sid in relevant)
    return hits / len(relevant)


def _precision_at_k(seg_ids: list[int], relevant: set[int], k: int) -> float:
    """Precision@K = top-K 内命中真值数 / top-K 实际返回条数。

    后端最多返回 top3，返回不足 K 时按实际条数做分母（标准 Precision@K 口径）。
    """
    top = seg_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for sid in top if sid in relevant)
    return hits / len(top)


def _f1_at_k(seg_ids: list[int], relevant: set[int], k: int) -> float:
    """F1@K = 2·P·R / (P+R)，P、R 同取 K；P+R 为 0 时记 0。"""
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


def _search(base_url: str, path: str, params: dict[str, str]) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _fmt_float(v: Optional[float]) -> str:
    return "" if v is None else f"{v:.6f}"


def _percentile(values: list[float], q: float) -> float:
    """线性插值分位数，口径与 numpy.percentile 默认一致（避免额外依赖 numpy）。"""
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


def main() -> int:
    parser = argparse.ArgumentParser(description="检索实验评测脚本")
    parser.add_argument("--base-url", default="http://localhost:8000", help="后端 base URL")
    parser.add_argument("--query-set", default="experiments/query_set.json", help="query set JSON 路径")
    parser.add_argument("--output-dir", default="experiments/output", help="输出目录")
    parser.add_argument("--warmup-queries", type=int, default=1,
                        help="预热查询数：先用前 N 条查询跑一遍全部配置，排除冷启动抖动，不计入统计")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    query_set_path = Path(args.query_set).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with query_set_path.open("r", encoding="utf-8") as f:
        queries = json.load(f)

    detail_rows: list[dict[str, Any]] = []
    config_metrics: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in CONFIGS}

    # ---- 预热：排除首次请求的冷启动抖动（连接建立、模型/索引首次加载）----
    if args.warmup_queries > 0:
        for q in queries[: args.warmup_queries]:
            query_text = q.get("query", "")
            for name, path, extra in CONFIGS:
                params: dict[str, str] = {"q": query_text}
                emotions = q.get("emotions") or []
                if emotions:
                    params["emotions"] = ",".join(emotions)
                params.update(extra)
                try:
                    _search(base_url, path, params)
                except Exception as exc:
                    print(f"[warn] warmup config={name}: {exc}", file=sys.stderr)
        print(f"[info] 预热完成 {args.warmup_queries} 条 × {len(CONFIGS)} 配置，不计入统计")

    for q in queries:
        qid = q.get("query_id")
        query_text = q.get("query", "")
        relevant = set(int(x) for x in q.get("relevant_segment_ids", []))

        for name, path, extra in CONFIGS:
            params: dict[str, str] = {"q": query_text}
            emotions = q.get("emotions") or []
            if emotions:
                params["emotions"] = ",".join(emotions)
            params.update(extra)

            try:
                t0 = time.perf_counter()
                payload = _search(base_url, path, params)
                client_ms = (time.perf_counter() - t0) * 1000
            except Exception as exc:  # 单条失败不中断整体评测
                print(f"[warn] query_id={qid} config={name}: {exc}", file=sys.stderr)
                continue

            results = payload.get("results", [])
            seg_ids = [int(r["segment_id"]) for r in results]
            scores = [r.get("score") for r in results]
            time_ms = float(payload.get("retrieval_time_ms", 0.0))

            recalls = {k: _recall_at_k(seg_ids, relevant, k) for k in K_VALUES}
            precisions = {k: _precision_at_k(seg_ids, relevant, k) for k in K_VALUES}
            f1s = {k: _f1_at_k(seg_ids, relevant, k) for k in K_VALUES}
            mrr = _mrr(seg_ids, relevant)
            ndcg10 = _ndcg_at_k(seg_ids, relevant, 10)

            first_hit_rank = 0
            for rank, sid in enumerate(seg_ids, start=1):
                if sid in relevant:
                    first_hit_rank = rank
                    break
            hit_ids = [sid for sid in seg_ids if sid in relevant]

            detail_rows.append({
                "query_id": qid,
                "query": query_text,
                "config": name,
                "relevant_ids": "|".join(str(x) for x in sorted(relevant)),
                "n_rel": len(relevant),
                "total_returned": len(seg_ids),
                "recall@1": recalls[1],
                "recall@5": recalls[5],
                "recall@10": recalls[10],
                "precision@1": precisions[1],
                "precision@5": precisions[5],
                "precision@10": precisions[10],
                "f1@1": f1s[1],
                "f1@5": f1s[5],
                "f1@10": f1s[10],
                "mrr": mrr,
                "ndcg@10": ndcg10,
                "retrieval_time_ms": time_ms,
                "client_latency_ms": client_ms,
                "first_hit_rank": first_hit_rank,
                "retrieved_segment_ids": "|".join(str(x) for x in seg_ids[:10]),
                "retrieved_scores": "|".join(_fmt_float(s) for s in scores[:10]),
                "hit_segment_ids": "|".join(str(x) for x in hit_ids),
            })

            config_metrics[name].append({
                "recalls": recalls,
                "precisions": precisions,
                "f1s": f1s,
                "mrr": mrr,
                "ndcg10": ndcg10,
                "time_ms": time_ms,
                "client_ms": client_ms,
            })

    # ---- metrics.csv ----
    metrics_path = output_dir / "metrics.csv"
    metrics_header = [
        "config",
        "recall@1", "recall@5", "recall@10",
        "precision@1", "precision@5", "precision@10",
        "f1@1", "f1@5", "f1@10",
        "mrr", "ndcg@10",
        "avg_time_ms", "client_p50_ms", "client_p95_ms", "client_max_ms",
        "num_queries",
    ]
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(metrics_header)
        for name, _, _ in CONFIGS:
            items = config_metrics[name]
            if not items:
                w.writerow([name] + [""] * (len(metrics_header) - 2) + [0])
                continue
            n = len(items)
            clients = [it["client_ms"] for it in items]
            w.writerow([
                name,
                round(sum(it["recalls"][1] for it in items) / n, 6),
                round(sum(it["recalls"][5] for it in items) / n, 6),
                round(sum(it["recalls"][10] for it in items) / n, 6),
                round(sum(it["precisions"][1] for it in items) / n, 6),
                round(sum(it["precisions"][5] for it in items) / n, 6),
                round(sum(it["precisions"][10] for it in items) / n, 6),
                round(sum(it["f1s"][1] for it in items) / n, 6),
                round(sum(it["f1s"][5] for it in items) / n, 6),
                round(sum(it["f1s"][10] for it in items) / n, 6),
                round(sum(it["mrr"] for it in items) / n, 6),
                round(sum(it["ndcg10"] for it in items) / n, 6),
                round(sum(it["time_ms"] for it in items) / n, 6),
                round(_percentile(clients, 0.50), 3),
                round(_percentile(clients, 0.95), 3),
                round(max(clients), 3),
                n,
            ])

    # ---- per_query_detail.csv ----
    detail_path = output_dir / "per_query_detail.csv"
    detail_header = [
        "query_id", "query", "config", "relevant_ids", "n_rel", "total_returned",
        "recall@1", "recall@5", "recall@10",
        "precision@1", "precision@5", "precision@10",
        "f1@1", "f1@5", "f1@10",
        "mrr", "ndcg@10",
        "retrieval_time_ms", "client_latency_ms", "first_hit_rank",
        "retrieved_segment_ids", "retrieved_scores", "hit_segment_ids",
    ]
    with detail_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(detail_header)
        for row in detail_rows:
            w.writerow([
                row["query_id"], row["query"], row["config"],
                row["relevant_ids"], row["n_rel"], row["total_returned"],
                round(row["recall@1"], 6), round(row["recall@5"], 6), round(row["recall@10"], 6),
                round(row["precision@1"], 6), round(row["precision@5"], 6), round(row["precision@10"], 6),
                round(row["f1@1"], 6), round(row["f1@5"], 6), round(row["f1@10"], 6),
                round(row["mrr"], 6), round(row["ndcg@10"], 6),
                round(row["retrieval_time_ms"], 6), round(row["client_latency_ms"], 3),
                row["first_hit_rank"],
                row["retrieved_segment_ids"], row["retrieved_scores"], row["hit_segment_ids"],
            ])

    print(f"metrics -> {metrics_path}")
    print(f"detail  -> {detail_path}")
    print(f"evaluated {len(queries)} queries x {len(CONFIGS)} configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())