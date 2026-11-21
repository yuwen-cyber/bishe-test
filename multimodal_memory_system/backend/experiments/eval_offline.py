#!/usr/bin/env python3
"""检索层离线评测：绕过 HTTP 接口的 top3 截断，直接查 Milvus top-N 计算指标。

为什么需要它：
    后端检索接口为前端展示只返回前 3 条，且带 _MIN_SCORE 过滤，导致
    Recall@5/10 与 Recall@3 同值，无法反映真实召回能力；同时接口不支持
    "只在正式 700 条库内检索"，无法把噪声干扰与算法能力分开考察。

做法：
    - 复用 app 的文本编码器与 Milvus 集合，直接取 top-N；
    - 两个检索范围：
        clean = segment_id <= 700（仅正式 700 条，等价扩充前的实验环境）
        full  = 全库 2003 条（正式 700 + 噪声 1300）
    - 指标：Recall@1/5/10/20、Precision@5、MRR、NDCG@10，并按查询类型分组。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\eval_offline.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
logging.disable(logging.WARNING)

from app import embeddings, milvus_client  # noqa: E402

QUERY_SET = BACKEND_DIR / "experiments" / "output" / "query_set_db.json"
OUTPUT_DIR = BACKEND_DIR / "experiments" / "output"
K_LIST = (1, 5, 10, 20)


def recall_at_k(ids: list[int], rel: set[int], k: int) -> float:
    if not rel:
        return 0.0
    return sum(1 for x in ids[:k] if x in rel) / len(rel)


def precision_at_k(ids: list[int], rel: set[int], k: int) -> float:
    top = ids[:k]
    if not top:
        return 0.0
    return sum(1 for x in top if x in rel) / len(top)


def mrr(ids: list[int], rel: set[int]) -> float:
    for i, x in enumerate(ids, start=1):
        if x in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ids: list[int], rel: set[int], k: int) -> float:
    dcg = sum((1.0 if x in rel else 0.0) / math.log2(i + 2) for i, x in enumerate(ids[:k]))
    ideal = [1.0] * min(len(rel), k)
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    queries = json.loads(QUERY_SET.read_text(encoding="utf-8"))
    col = milvus_client._get_collection()
    col.load()

    scopes = {"clean": "segment_id <= 700", "full": None}
    detail: list[dict] = []

    for scope, expr in scopes.items():
        for q in queries:
            rel = set(int(x) for x in q.get("relevant_segment_ids") or [])
            vec = embeddings.embed_text(q["query"])
            hits = col.search(
                data=[vec],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": max(64, args.top)}},
                limit=args.top,
                expr=expr,
                output_fields=["segment_id"],
            )
            ids = [int(h.entity.get("segment_id")) for h in hits[0]]
            detail.append(
                {
                    "scope": scope,
                    "query_id": q["query_id"],
                    "type": q.get("type", ""),
                    "query": q["query"],
                    "n_rel": len(rel),
                    "best_rank": next((i for i, x in enumerate(ids, 1) if x in rel), 0),
                    **{f"recall@{k}": recall_at_k(ids, rel, k) for k in K_LIST},
                    "precision@5": precision_at_k(ids, rel, 5),
                    "mrr": mrr(ids, rel),
                    "ndcg@10": ndcg_at_k(ids, rel, 10),
                    "top_ids": "|".join(str(x) for x in ids[:10]),
                    "rel_ids": "|".join(str(x) for x in sorted(rel)),
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "metrics_offline_detail.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
        w.writeheader()
        w.writerows(detail)

    agg = defaultdict(list)
    for r in detail:
        agg[(r["scope"], "全部")].append(r)
        agg[(r["scope"], r["type"])].append(r)

    lines = ["| 检索范围 | 查询类型 | 条数 | R@1 | R@5 | R@10 | R@20 | P@5 | MRR | NDCG@10 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for (scope, qtype), rows in sorted(agg.items()):
        n = len(rows)
        row = [scope, qtype, str(n)]
        for k in K_LIST:
            row.append(f"{sum(r[f'recall@{k}'] for r in rows)/n:.3f}")
        row.append(f"{sum(r['precision@5'] for r in rows)/n:.3f}")
        row.append(f"{sum(r['mrr'] for r in rows)/n:.3f}")
        row.append(f"{sum(r['ndcg@10'] for r in rows)/n:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    md = "\n".join(lines)
    (OUTPUT_DIR / "metrics_offline.md").write_text(
        "# 检索层离线评测（top%d，绕过接口 top3 截断）\n\n%s\n" % (args.top, md), encoding="utf-8"
    )
    print(md)
    print(f"\n明细 -> {OUTPUT_DIR / 'metrics_offline_detail.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
