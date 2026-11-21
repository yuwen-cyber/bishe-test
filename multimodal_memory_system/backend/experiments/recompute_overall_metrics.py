#!/usr/bin/env python3
"""总体指标口径修正：剔除"不可判定"的结构化查询后重算。

背景：
原 50 条查询集里有 12 条「结构化精确查询」，形如「找一下 user003 悲伤的对话」。
当前数据是 700 条单片段会话，**没有说话人（user）维度**，这 12 条在原理上不可能命中，
把它们计入总体平均会系统性压低所有配置的指标（12/50 = 24%）。

本脚本给出两套口径，供论文如实报告：
  A. 原口径（全部 50 条）——保留，便于复现历史数字
  B. 有效口径（剔除 12 条不可判定查询，保留 38 条）——建议作为主表

指标取各查询的算术平均（与 run_eval.py / run_hybrid_eval.py 的汇总方式一致）。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\recompute_overall_metrics.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

INVALID_TYPE = "结构化精确查询"
METRICS = ["recall@1", "recall@5", "recall@10", "precision@5", "precision@10", "f1@10", "mrr", "ndcg@10"]


def _load_detail(path: Path, source: str) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_source"] = source
    return rows


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="剔除不可判定查询后重算总体指标")
    parser.add_argument("--query-set", default="experiments/output/query_set_db.json")
    parser.add_argument("--output-dir", default="experiments/output")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    query_types = {int(q["query_id"]): q.get("type") for q in json.loads(Path(args.query_set).read_text(encoding="utf-8"))}
    invalid_ids = {qid for qid, t in query_types.items() if t == INVALID_TYPE}

    details = _load_detail(out_dir / "per_query_detail.csv", "四模式HTTP") + \
              _load_detail(out_dir / "per_query_hybrid.csv", "混合离线")

    configs: dict[tuple[str, str], list[dict]] = {}
    for r in details:
        configs.setdefault((r["_source"], r["config"]), []).append(r)

    rows_out: list[dict] = []
    for (source, config), rs in sorted(configs.items()):
        valid = [r for r in rs if int(r["query_id"]) not in invalid_ids]
        for scope, subset in (("原口径(50条)", rs), ("有效口径(38条)", valid)):
            row = {"source": source, "config": config, "scope": scope, "num_queries": len(subset)}
            for m in METRICS:
                row[m] = _mean(subset, m)
            rows_out.append(row)

    csv_path = out_dir / "metrics_overall_adjusted.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    md: list[str] = ["# 总体指标口径修正（剔除不可判定查询）\n"]
    md.append("## 为什么改口径\n")
    md.append(f"- 原查询集 50 条中有 {len(invalid_ids)} 条「结构化精确查询」，形如「找一下 user003 悲伤的对话」。")
    md.append("- 当前数据集为 700 条单片段会话，**没有说话人（user）维度**，"
              "这类查询在原理上不可能命中，若计入总体平均会把所有配置的指标系统性拉低"
              f"（占 {len(invalid_ids)}/50 = {len(invalid_ids)/50:.0%}）。")
    md.append("- 处理方式：不删查询、不改数据，只给出两套口径并明确标注；"
              "结构化查询能力另用 20 条「可判定组合条件查询」单独评测（见 `structured_eval.md`）。\n")
    md.append("## 两套口径对比\n")
    md.append("| 数据源 | 配置 | 口径 | 查询数 | R@1 | R@5 | R@10 | P@5 | P@10 | F1@10 | MRR | NDCG@10 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows_out:
        md.append(f"| {r['source']} | {r['config']} | {r['scope']} | {r['num_queries']} | "
                  f"{r['recall@1']} | {r['recall@5']} | {r['recall@10']} | {r['precision@5']} | "
                  f"{r['precision@10']} | {r['f1@10']} | {r['mrr']} | {r['ndcg@10']} |")
    md.append("")
    md.append("## 写论文时的建议说法\n")
    md.append("- 主表用**有效口径（38 条）**，并在表注写明：剔除了 12 条因缺少说话人维度而不可判定的查询。")
    md.append("- 若需要与历史跑分对齐，附录给出原口径（50 条），并说明其偏低是口径所致、非检索能力下降。")
    md.append("- 结构化精确查询的实测结果单独见 `structured_eval.md`（20 条组合条件查询，"
              "精确过滤集合级 P/R/F1 = 1.0）。")
    (out_dir / "metrics_overall_adjusted.md").write_text("\n".join(md), encoding="utf-8")

    print(f"已输出 -> {csv_path}")
    print(f"       -> {out_dir / 'metrics_overall_adjusted.md'}")
    for r in rows_out:
        print(f"{r['source']:9s} {r['config']:16s} {r['scope']:12s} n={r['num_queries']:2d} "
              f"R@5={r['recall@5']:.4f} R@10={r['recall@10']:.4f} MRR={r['mrr']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
