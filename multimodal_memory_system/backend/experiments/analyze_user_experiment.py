#!/usr/bin/env python3
"""用户记忆体验实验 · 统计分析脚本（创新点五配套）

读取实验台导出的 user_experiment.csv（长表：一个被试 × 一个条件一行），
计算三种条件在各量表维度上的描述统计，并做配对比较：
  - semantic（语义检索） vs keyword（关键词检索）
  - semantic（语义检索） vs no_memory（无记忆）
  - keyword（关键词检索） vs no_memory（无记忆）

统计方法：
  - 先对配对差值做 Shapiro 正态检验；
    p > 0.05 视为近似正态，用配对样本 t 检验；否则用 Wilcoxon 符号秩检验。
  - 效应量报告配对 Cohen's d = mean(差值) / std(差值, ddof=1)。

用法（在 backend 目录下）：
    .venv\\Scripts\\python.exe experiments\\analyze_user_experiment.py --csv experiments\\output\\user_experiment.csv

输出（写到 --output-dir，默认 experiments/output）：
    user_exp_condition_summary.csv  各条件 × 各维度 均值/标准差/N
    user_exp_pairwise.csv           配对检验明细（均值差、p 值、方法、Cohen's d）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# 维度 -> 该维度包含的题目字段
DIMENSION_ITEMS: dict[str, list[str]] = {
    "memory": ["M1", "M2"],
    "understand": ["U1", "U2"],
    "trust": ["T1", "T2"],
    "companion": ["C1", "C2"],
    "overall": ["O1"],
}

CONDITION_LABEL: dict[str, str] = {
    "no_memory": "无记忆",
    "keyword": "关键词检索",
    "semantic": "语义检索",
}

# 需要做的配对比较（前者 - 后者；正值表示前者得分更高）
PAIRS: list[tuple[str, str]] = [
    ("semantic", "keyword"),
    ("semantic", "no_memory"),
    ("keyword", "no_memory"),
]


def build_dimension_scores(df: pd.DataFrame) -> pd.DataFrame:
    """在长表上追加各维度均分列。"""
    out = df.copy()
    for dim, items in DIMENSION_ITEMS.items():
        out[dim] = out[items].mean(axis=1)
    return out


def condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for cond, label in CONDITION_LABEL.items():
        sub = df[df["condition"] == cond]
        for dim in DIMENSION_ITEMS:
            vals = sub[dim].dropna()
            rows.append(
                {
                    "condition": cond,
                    "condition_label": label,
                    "dimension": dim,
                    "n": int(len(vals)),
                    "mean": round(float(vals.mean()), 3) if len(vals) else np.nan,
                    "std": round(float(vals.std(ddof=1)), 3) if len(vals) > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def paired_compare(df: pd.DataFrame, a: str, b: str, dim: str) -> dict:
    """取同一被试在 a、b 两条件下的维度得分做配对比较。"""
    da = df[df["condition"] == a].set_index("subject_id")[dim]
    db = df[df["condition"] == b].set_index("subject_id")[dim]
    common = da.index.intersection(db.index)
    a_vals = da.loc[common].astype(float)
    b_vals = db.loc[common].astype(float)
    diff = (a_vals - b_vals).to_numpy()
    n = len(common)

    result: dict = {
        "pair": f"{a} - {b}",
        "pair_label": f"{CONDITION_LABEL[a]} - {CONDITION_LABEL[b]}",
        "dimension": dim,
        "n": n,
        "mean_diff": round(float(np.mean(diff)), 3) if n else np.nan,
        "cohens_d": np.nan,
        "method": "",
        "statistic": np.nan,
        "p_value": np.nan,
    }

    if n < 2:
        result["method"] = "样本不足"
        return result

    # 配对 Cohen's d
    std_diff = np.std(diff, ddof=1)
    if std_diff > 0:
        result["cohens_d"] = round(float(np.mean(diff) / std_diff), 3)

    # 正态性检验（差值）
    try:
        _, shapiro_p = stats.shapiro(diff)
    except Exception:
        shapiro_p = 0.0

    if shapiro_p > 0.05:
        try:
            t_stat, p_val = stats.ttest_rel(a_vals, b_vals)
            result["method"] = "配对t检验"
            result["statistic"] = round(float(t_stat), 4)
            result["p_value"] = round(float(p_val), 4)
        except Exception as e:
            result["method"] = f"t检验失败:{e}"
    else:
        try:
            w_stat, p_val = stats.wilcoxon(a_vals, b_vals)
            result["method"] = "Wilcoxon符号秩"
            result["statistic"] = round(float(w_stat), 4)
            result["p_value"] = round(float(p_val), 4)
        except Exception as e:
            result["method"] = f"Wilcoxon失败:{e}"

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="用户记忆体验实验统计分析")
    parser.add_argument("--csv", default="experiments/output/user_experiment.csv", help="实验台导出的 CSV 路径")
    parser.add_argument("--output-dir", default="experiments/output", help="结果输出目录")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"找不到 CSV 文件: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = ["subject_id", "condition"] + [it for items in DIMENSION_ITEMS.values() for it in items]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV 缺少字段: {missing}")

    df = build_dimension_scores(df)

    summary = condition_summary(df)

    pairwise_rows: list[dict] = []
    for a, b in PAIRS:
        for dim in DIMENSION_ITEMS:
            pairwise_rows.append(paired_compare(df, a, b, dim))
    pairwise = pd.DataFrame(pairwise_rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "user_exp_condition_summary.csv"
    pairwise_path = out_dir / "user_exp_pairwise.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pairwise.to_csv(pairwise_path, index=False, encoding="utf-8-sig")

    # 控制台摘要
    print(f"被试数: {df['subject_id'].nunique()}；记录数: {len(df)}")
    print("\n=== 各条件 × 维度 均值（标准差） ===")
    pivot = summary.pivot_table(index="condition_label", columns="dimension", values="mean")
    print(pivot.round(3).to_string())

    print("\n=== 配对检验（overall 总体满意度） ===")
    ov = pairwise[pairwise["dimension"] == "overall"]
    print(ov[["pair_label", "n", "mean_diff", "cohens_d", "method", "p_value"]].to_string(index=False))

    print(f"\n汇总表 -> {summary_path}")
    print(f"配对检验 -> {pairwise_path}")
    print("\n说明: p<0.05 视为差异显著；Cohen's d 参考标准 |d|≥0.2 小效应、≥0.5 中效应、≥0.8 大效应。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
