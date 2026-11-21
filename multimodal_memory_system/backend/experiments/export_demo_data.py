#!/usr/bin/env python3
"""把 experiments/output 下的指标 CSV 汇总导出为前端演示面板用的 JSON。

目的：前端「演示面板」不需要连后端、也不需要读 CSV，直接 fetch 一个静态 JSON 即可。
数据全部来自已跑完的实验产物，不做任何二次计算或修饰（数值原样搬运）。

输出：frontend/public/demo-data.json

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\export_demo_data.py
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

FILES = {
    "fourModes": ("metrics.csv", "四模式 HTTP 端到端指标"),
    "hybrid": ("metrics_hybrid.csv", "混合检索离线指标"),
    "modalAblation": ("metrics_ablation.csv", "模态消融（7 组）"),
    "fusionAblation": ("metrics_fusion_ablation.csv", "融合策略消融"),
    "scale": ("metrics_scale.csv", "三档规模性能"),
    "structured": ("metrics_structured.csv", "结构化组合条件查询"),
    "byType": ("metrics_by_type.csv", "按查询类型分组"),
    "cases": ("case_study_cross_literal.csv", "跨字面语义召回的案例"),
    "overallAdjusted": ("metrics_overall_adjusted.csv", "总体指标两套口径"),
}


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out: list[dict] = []
    for r in rows:
        conv: dict = {}
        for k, v in r.items():
            if v is None or v == "":
                conv[k] = v
                continue
            try:
                conv[k] = int(v)
            except ValueError:
                try:
                    conv[k] = float(v)
                except ValueError:
                    conv[k] = v
        out.append(conv)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="导出演示面板数据")
    parser.add_argument("--source-dir", default="experiments/output")
    parser.add_argument("--out", default="../frontend/public/demo-data.json")
    args = parser.parse_args()

    src = Path(args.source_dir)
    payload: dict = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "project": "面向桌面情感陪伴机器人的多模态对话记忆与语义检索系统",
            "data_scale": "2003 条片段（正式 700 + 噪声 1300 + 用户自录 3）/ 50 条查询（157 个真值项）",
            "notes": [
                "四模式接口最多返回 top3，@5/@10 的 Precision 分母实际只到 3。",
                "结构化精确查询原设计含用户ID维度，当前数据集无说话人字段，"
                "已改为 20 条可判定的组合条件查询，说话人维度写入局限。",
                "总体指标给出「原口径 50 条」与「有效口径 38 条」两套，"
                "后者剔除 12 条因缺说话人维度而不可判定的查询。",
            ],
        },
        "sources": {},
    }

    for key, (filename, desc) in FILES.items():
        rows = _read(src / filename)
        payload[key] = rows
        payload["sources"][key] = {"file": filename, "desc": desc, "rows": len(rows)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已导出 -> {out_path.resolve()}")
    for key, info in payload["sources"].items():
        print(f"  {key:16s} {info['rows']:3d} 行  {info['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
