"""混合检索 RRF 权重敏感性扫描：证明 w_vec:w_sql = 2:1 的选择有依据，而非随意设定。

对 (w_sql, w_vec) 网格逐组计算 R@10 / MRR / NDCG@10，输出 CSV 与最优组合。
用法（backend 目录）：.venv\\Scripts\\python.exe experiments\\run_weight_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_hybrid_eval as H  # noqa: E402

GRID = [(w_kw, w_vec)
        for w_kw in (0.5, 1.0, 1.5, 2.0, 3.0)
        for w_vec in (0.5, 1.0, 1.5, 2.0, 3.0)]


def main() -> int:
    queries = json.loads(Path("experiments/output/query_set_db.json").read_text(encoding="utf-8"))
    H.embeddings.warmup()
    db = H.SessionLocal()

    # 两路召回结果缓存一次，避免网格重复检索
    cache = []
    for q in queries:
        qtext = q.get("query", "")
        emo = H._infer_emotion(qtext, q.get("emotions") or [])
        relevant = set(int(x) for x in q.get("relevant_segment_ids", []))
        kw = H.sql_recall(db, qtext, emo, H.CAND_N)
        vec = H.vector_recall(qtext, emo, H.CAND_N)
        cache.append((kw, vec, relevant))
    db.close()

    rows = []
    for w_kw, w_vec in GRID:
        r10 = mrr = ndcg = 0.0
        for kw, vec, rel in cache:
            ids = H.rrf_fuse(kw, vec, H.RRF_K, w_kw, w_vec)
            r10 += H._recall_at_k(ids, rel, 10)
            mrr += H._mrr(ids, rel)
            ndcg += H._ndcg_at_k(ids, rel, 10)
        n = len(cache)
        rows.append({
            "w_sql": w_kw, "w_vec": w_vec, "ratio": round(w_vec / w_kw, 2),
            "recall@10": round(r10 / n, 4), "mrr": round(mrr / n, 4),
            "ndcg@10": round(ndcg / n, 4),
        })

    out = Path("experiments/output/metrics_weight_sweep.csv")
    import csv
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    best_r10 = max(rows, key=lambda r: (r["recall@10"], r["mrr"]))
    best_mrr = max(rows, key=lambda r: (r["mrr"], r["recall@10"]))
    cur = next(r for r in rows if r["w_sql"] == 1.0 and r["w_vec"] == 2.0)

    print(f"\n=== 权重敏感性扫描（{len(cache)} 查询，RRF k={H.RRF_K}）===")
    print(f"{'w_sql':>6}{'w_vec':>7}{'R@10':>9}{'MRR':>9}{'NDCG@10':>10}")
    for r in rows:
        mark = "  <- 当前生产配置" if (r["w_sql"], r["w_vec"]) == (1.0, 2.0) else ""
        print(f"{r['w_sql']:>6.1f}{r['w_vec']:>7.1f}{r['recall@10']:>9.4f}"
              f"{r['mrr']:>9.4f}{r['ndcg@10']:>10.4f}{mark}")
    print(f"\n当前 1:2 配置   R@10={cur['recall@10']:.4f} MRR={cur['mrr']:.4f} NDCG@10={cur['ndcg@10']:.4f}")
    print(f"R@10 最优       w_sql={best_r10['w_sql']} w_vec={best_r10['w_vec']} "
          f"R@10={best_r10['recall@10']:.4f}")
    print(f"MRR  最优       w_sql={best_mrr['w_sql']} w_vec={best_mrr['w_vec']} "
          f"MRR={best_mrr['mrr']:.4f}")
    print(f"\n输出 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
