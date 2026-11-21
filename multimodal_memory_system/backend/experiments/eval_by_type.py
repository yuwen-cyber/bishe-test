#!/usr/bin/env python3
"""实验 3/4 配套：按查询类型分组的指标拆解。

导师文档要求区分「单条件结构化精确」「关键词全文检索」「语义查询」等场景，
因此把已有逐条明细按 `type` 分组重算平均指标，避免只看总平均掩盖分组差异。

输入（已有产物，不改动业务代码）：
    experiments/output/query_set_db.json
    experiments/output/per_query_detail.csv   四模式明细（含 recall/precision/f1/mrr/ndcg）
    experiments/output/per_query_hybrid.csv   混合检索明细

输出：
    experiments/output/metrics_by_type.csv    配置 × 查询类型 的分组指标
    experiments/output/metrics_by_type.md     论文可用的分组对照表

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\eval_by_type.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

METRICS = ["recall@1", "recall@5", "recall@10", "precision@5", "f1@5", "mrr", "ndcg@10"]

TYPE_ORDER = ["结构化精确查询", "关键词查询", "语义查询", "跨模态查询"]


def _load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="按查询类型分组统计指标")
    parser.add_argument("--query-set", default="experiments/output/query_set_db.json")
    parser.add_argument("--detail", default="experiments/output/per_query_detail.csv")
    parser.add_argument("--hybrid", default="experiments/output/per_query_hybrid.csv")
    parser.add_argument("--output-dir", default="experiments/output")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    queries = json.loads(Path(args.query_set).read_text(encoding="utf-8"))
    qtype = {int(q["query_id"]): q.get("type", "") for q in queries}

    rows: list[dict] = []
    for r in _load(Path(args.detail)) + _load(Path(args.hybrid)):
        qid = int(r["query_id"])
        rows.append({**r, "type": qtype.get(qid, "未标注")})

    # 汇总：(配置, 类型) -> 指标列表
    bucket: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["config"], r["type"])
        for m in METRICS:
            try:
                bucket[key][m].append(float(r[m]))
            except (KeyError, ValueError):
                continue

    out_rows: list[dict] = []
    configs = sorted({c for c, _ in bucket})
    for cfg in configs:
        for t in TYPE_ORDER + [x for x in {r["type"] for r in rows} if x not in TYPE_ORDER]:
            if (cfg, t) not in bucket:
                continue
            b = bucket[(cfg, t)]
            n = len(b[METRICS[0]]) if b[METRICS[0]] else 0
            rec = {"config": cfg, "type": t, "n": n}
            for m in METRICS:
                vals = b[m]
                rec[m] = round(sum(vals) / len(vals), 4) if vals else ""
            out_rows.append(rec)

    csv_path = out_dir / "metrics_by_type.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "type", "n"] + METRICS)
        w.writeheader()
        w.writerows(out_rows)

    # ---- Markdown 对照表（按类型分块，便于直接放进论文）----
    md: list[str] = ["# 分组指标：配置 × 查询类型\n"]
    md.append("说明：同一 50 条查询按类型分组后重算平均，用于观察不同检索方式在各类场景下的能力差异。")
    md.append("四类场景各 12 / 12 / 18 / 8 条。\n")
    for t in TYPE_ORDER:
        sub = [r for r in out_rows if r["type"] == t]
        if not sub:
            continue
        md.append(f"## {t}（{sub[0]['n']} 条）\n")
        md.append("| 配置 | R@1 | R@5 | R@10 | P@5 | F1@5 | MRR | NDCG@10 |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in sub:
            md.append(
                f"| {r['config']} | {r['recall@1']} | {r['recall@5']} | {r['recall@10']} | "
                f"{r['precision@5']} | {r['f1@5']} | {r['mrr']} | {r['ndcg@10']} |"
            )
        md.append("")
    md_path = out_dir / "metrics_by_type.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(f"配置数 {len(configs)}，分组行数 {len(out_rows)}")
    for r in out_rows:
        if r["config"] in ("baseline", "advanced_plain", "vector", "hybrid_weighted"):
            print(f"{r['config']:18s} {r['type']:8s} n={r['n']:2d} "
                  f"R@5={r['recall@5']} R@10={r['recall@10']} MRR={r['mrr']}")
    print(f"\n分组表 -> {csv_path}")
    print(f"表格文档 -> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
