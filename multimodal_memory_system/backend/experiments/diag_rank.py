#!/usr/bin/env python3
"""诊断检索真实排名：直接查 Milvus top-K，看标注真值实际排在第几名。

目的：
    HTTP 接口只返回 top3，Recall@10 的天花板被人为压低。本脚本绕过接口限制，
    对每条查询取 top 50/100，统计真值排名分布，判断指标低是"接口截断"还是"检索失效"。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\diag_rank.py --top 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
logging.disable(logging.WARNING)

from app import embeddings, milvus_client  # noqa: E402

QUERY_SET = BACKEND_DIR / "experiments" / "output" / "query_set_db.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    queries = json.loads(QUERY_SET.read_text(encoding="utf-8"))
    rows = []
    for q in queries:
        rel = set(int(x) for x in q.get("relevant_segment_ids") or [])
        if not rel:
            continue
        vec = embeddings.embed_text(q["query"])
        hits = milvus_client.search_vectors(vec, top_k=args.top)
        ids = [int(h["segment_id"]) for h in hits]
        ranks = [ids.index(x) + 1 for x in rel if x in ids]
        best = min(ranks) if ranks else 0
        rows.append((q["query_id"], q["type"], q["query"], len(rel), best, len(ranks)))

    print(f"检索深度 top{args.top}，共 {len(rows)} 条带真值的查询")
    print("type | 查询 | 真值数 | 最优排名 | 命中的真值数")
    for r in rows:
        print(f"{r[1]} | {r[2][:24]} | {r[3]} | {r[4] or '未命中'} | {r[5]}/{r[3]}")

    print("\n按类型汇总（真值最优排名中位数 / 平均）：")
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[r[1]].append(r[4] or (args.top + 100))
    for t, rs in agg.items():
        rs_sorted = sorted(rs)
        med = rs_sorted[len(rs_sorted) // 2]
        print(f"  {t}: n={len(rs)} 中位排名={med} 平均={sum(rs)/len(rs):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
