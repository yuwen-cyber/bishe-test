#!/usr/bin/env python3
"""实验 1：标注一致性（Cohen's Kappa）与模型-人一致率计算。

用途：
  1) 两名标注者的标注文件 -> 算 Cohen's Kappa、观察一致率 Po、期望一致率 Pe、分歧清单；
  2) 只有一名标注者时 -> **不算 Kappa**（会明确提示需要第二人），
     仅输出标签分布与「与原始标签的一致率」，供流程验证用。

注意口径：
- Kappa 度量的是**标注者之间**的一致性，不能拿单人数据凑；
- 「模型-人一致率」只是参考指标，不等于 Kappa，论文里必须分开写。

用法（backend 目录）：
  两人都有：
    .venv\\Scripts\\python.exe experiments\\compute_kappa.py \\
        --a experiments/annotation/annotations_张三.csv \\
        --b experiments/annotation/annotations_李四.csv
  只有一人（流程验证）：
    .venv\\Scripts\\python.exe experiments\\compute_kappa.py \\
        --a experiments/annotation/annotations_张三.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

LABEL_ORDER = ["喜悦", "惊讶", "愤怒", "恐惧", "悲伤", "中性", "厌恶"]


def _load(path: Path) -> dict[int, str]:
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {int(r["segment_id"]): r["label"].strip() for r in rows if r.get("label")}


def cohen_kappa(a: dict[int, str], b: dict[int, str]) -> dict:
    """返回 Kappa 及其组成项（仅对两人都标过的条目计算）。"""
    common = sorted(set(a) & set(b))
    n = len(common)
    if n == 0:
        return {"n": 0, "po": 0.0, "pe": 0.0, "kappa": float("nan"), "labels": []}

    po = sum(1 for sid in common if a[sid] == b[sid]) / n
    labels = sorted({a[sid] for sid in common} | {b[sid] for sid in common},
                    key=lambda x: LABEL_ORDER.index(x) if x in LABEL_ORDER else 99)
    pa = Counter(a[sid] for sid in common)
    pb = Counter(b[sid] for sid in common)
    pe = sum((pa[l] / n) * (pb[l] / n) for l in labels)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else float("nan")
    return {"n": n, "po": po, "pe": pe, "kappa": kappa, "labels": labels}


def _interpret(k: float) -> str:
    """Landis & Koch (1977) 常用判读标准。"""
    if k != k:  # NaN
        return "无法计算"
    if k < 0:
        return "差（低于随机水平）"
    if k <= 0.20:
        return "轻微一致"
    if k <= 0.40:
        return "一般一致"
    if k <= 0.60:
        return "中等一致"
    if k <= 0.80:
        return "较强一致"
    return "几乎完全一致"


def main() -> int:
    parser = argparse.ArgumentParser(description="标注一致性 / 模型-人一致率")
    parser.add_argument("--a", required=True, help="标注者 A 的 CSV")
    parser.add_argument("--b", help="标注者 B 的 CSV（缺省则只做单人流程验证）")
    parser.add_argument("--key", default="experiments/annotation/answer_key.csv",
                        help="原始情绪标签（模型-人一致率用；不存在则跳过该项）")
    parser.add_argument("--output-dir", default="experiments/output")
    args = parser.parse_args()

    pa = Path(args.a)
    if not pa.exists():
        print(f"[错误] 找不到 {pa}")
        return 1
    a = _load(pa)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    key: dict[int, str] = {}
    kp = Path(args.key)
    if kp.exists():
        with kp.open(encoding="utf-8-sig") as f:
            key = {int(r["segment_id"]): r["original_emotion"].strip()
                   for r in csv.DictReader(f) if r.get("original_emotion")}

    md: list[str] = ["# 情绪标注一致性结果\n"]
    md.append(f"- 标注者 A：`{pa.name}`，有效标注 {len(a)} 条")

    rows_out: list[dict] = []

    if args.b:
        pb = Path(args.b)
        if not pb.exists():
            print(f"[错误] 找不到 {pb}")
            return 1
        b = _load(pb)
        md.append(f"- 标注者 B：`{pb.name}`，有效标注 {len(b)} 条")

        res = cohen_kappa(a, b)
        md.append(f"- 共同标注条目：{res['n']} 条（Kappa 仅在这部分上计算）\n")
        md.append("## Cohen's Kappa\n")
        md.append("| 指标 | 数值 | 含义 |")
        md.append("|---|---|---|")
        md.append(f"| 观察一致率 Po | {res['po']:.4f} | 两人标签相同的比例 |")
        md.append(f"| 期望一致率 Pe | {res['pe']:.4f} | 按各自标签分布随机配对时的期望 |")
        md.append(f"| **Cohen's Kappa** | **{res['kappa']:.4f}** | {_interpret(res['kappa'])} |")
        md.append("")

        labels = res["labels"]
        md.append("## 混淆矩阵（行=A，列=B）\n")
        md.append("| A \\ B | " + " | ".join(labels) + " |")
        md.append("|---|" + "---|" * len(labels))
        for la in labels:
            counts = [sum(1 for sid in set(a) & set(b) if a[sid] == la and b[sid] == lb)
                      for lb in labels]
            md.append(f"| {la} | " + " | ".join(str(x) for x in counts) + " |")
        md.append("")

        disagreements = [(sid, a[sid], b[sid]) for sid in sorted(set(a) & set(b)) if a[sid] != b[sid]]
        md.append(f"## 分歧条目（{len(disagreements)} 条）\n")
        if disagreements:
            md.append("| segment_id | A | B |")
            md.append("|---|---|---|")
            for sid, la, lb in disagreements:
                md.append(f"| {sid} | {la} | {lb} |")
        else:
            md.append("无分歧。")
        md.append("")

        rows_out.append({
            "metric": "cohen_kappa",
            "n": res["n"],
            "po": round(res["po"], 4),
            "pe": round(res["pe"], 4),
            "kappa": round(res["kappa"], 4) if res["kappa"] == res["kappa"] else "",
            "interpretation": _interpret(res["kappa"]),
        })
    else:
        md.append("- 标注者 B：**未提供**\n")
        md.append("## 状态：Kappa 未计算\n")
        md.append("Cohen's Kappa 需要两名**独立**标注者的结果；当前只有一份标注文件，"
                  "因此不输出 Kappa。补齐第二名标注者后重跑本脚本即可得到结果。\n")
        rows_out.append({
            "metric": "cohen_kappa",
            "n": 0, "po": "", "pe": "", "kappa": "",
            "interpretation": "仅有 1 名标注者，Kappa 无法计算（待第二标注者）",
        })

    # 标签分布
    md.append("## 标注者 A 的标签分布\n")
    md.append("| 标签 | 条数 | 占比 |")
    md.append("|---|---|---|")
    dist = Counter(a.values())
    for lb in LABEL_ORDER:
        if lb in dist:
            md.append(f"| {lb} | {dist[lb]} | {dist[lb] / len(a):.1%} |")
    md.append("")

    # 模型-人一致率（参考指标，不等于 Kappa）
    if key:
        common = sorted(set(a) & set(key))
        if common:
            agree = sum(1 for sid in common if a[sid] == key[sid])
            rate = agree / len(common)
            md.append("## 与原始标签的一致率（参考指标）\n")
            md.append(f"- 共同条目 {len(common)} 条，一致 {agree} 条，一致率 **{rate:.2%}**")
            md.append("- 该指标衡量「人工标注 vs 库里原始情绪标签」，"
                      "**不是**标注者间一致性，论文中须与 Kappa 分开表述。")
            per_class = []
            for lb in LABEL_ORDER:
                sel = [sid for sid in common if key[sid] == lb]
                if sel:
                    hit = sum(1 for sid in sel if a[sid] == lb)
                    per_class.append((lb, len(sel), hit / len(sel)))
            if per_class:
                md.append("")
                md.append("| 原始标签 | 条数 | 人工判为该标签的比例 |")
                md.append("|---|---|---|")
                for lb, cnt, r in per_class:
                    md.append(f"| {lb} | {cnt} | {r:.1%} |")
            rows_out.append({
                "metric": "human_vs_original_agreement",
                "n": len(common),
                "po": round(rate, 4),
                "pe": "",
                "kappa": "",
                "interpretation": "参考指标（非 Kappa）",
            })

    with (out_dir / "metrics_annotation_agreement.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "n", "po", "pe", "kappa", "interpretation"])
        writer.writeheader()
        writer.writerows(rows_out)

    md_path = out_dir / "annotation_agreement.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print("\n".join(md[:12]))
    print(f"...\n输出 -> {md_path}、{out_dir / 'metrics_annotation_agreement.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
