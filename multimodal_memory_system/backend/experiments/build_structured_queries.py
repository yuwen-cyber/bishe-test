#!/usr/bin/env python3
"""实验 3 配套：构建"结构化组合条件查询集"（不改动任何业务代码）。

背景（务必写进论文）：
导师文档要求的结构化查询是「用户ID + 情绪 + 时间范围」。但当前数据集是 700 条
单片段会话，没有说话人（user）维度，时间戳也集中在 2026-11-11 的 13~15 点，
因此"user003 的悲伤片段"这类查询在本数据上无法判定真值。
若强行按答案倒推添加 user 字段，等于用答案造标签（数据泄漏），不做。

本脚本改为构建当前数据可判定的组合条件查询：
    单条件   ：情绪等值 / 小时区间 / 时长阈值
    多条件   ：情绪 + 小时、情绪 + 关键词、情绪 + 时长 等 2~3 个条件组合

真值口径：**由 SQL 条件机械判定**（满足条件的片段集合即为相关片段），
不是人工标注，也不含任何人工挑选，因此可复现、无泄漏。

输出：experiments/output/query_set_structured.json
用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\build_structured_queries.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.database import SessionLocal  # noqa: E402

MONTH = "11 月 11 日"  # 全部数据落在这一天，查询文案里如实体现
QUERY_DATE = "2026-11-11"  # 与文案一致：小时类查询必须锁到当天，避免跨日片段混入

# 单条件：情绪等值 / 小时区间 / 时长阈值
SINGLE_SPECS: list[dict] = [
    {"emotion": "喜悦", "text": "找情绪是喜悦的片段"},
    {"emotion": "惊讶", "text": "找情绪是惊讶的片段"},
    {"emotion": "愤怒", "text": "找情绪是愤怒的片段"},
    {"emotion": "恐惧", "text": "找情绪是恐惧的片段"},
    {"emotion": "悲伤", "text": "找情绪是悲伤的片段"},
    {"emotion": "厌恶", "text": "找情绪是厌恶的片段"},
    {"emotion": "中性", "text": "找情绪是中性的片段"},
    {"hour": 13, "text": f"找 {MONTH} 13 点这一小时里的片段"},
    {"hour": 15, "text": f"找 {MONTH} 15 点这一小时里的片段"},
    {"min_duration_ms": 20000, "text": "找时长在 20 秒以上的片段"},
]

# 多条件（2~3 个条件组合）
MULTI_SPECS: list[dict] = [
    {"emotion": "喜悦", "hour": 14, "text": "找情绪是喜悦、且在 14 点的片段"},
    {"emotion": "悲伤", "hour": 13, "text": "找情绪是悲伤、且在 13 点的片段"},
    {"emotion": "恐惧", "hour": 15, "text": "找情绪是恐惧、且在 15 点的片段"},
    {"emotion": "愤怒", "hour": 14, "text": "找情绪是愤怒、且在 14 点的片段"},
    {"emotion": "惊讶", "hour": 13, "text": "找情绪是惊讶、且在 13 点的片段"},
    {"emotion": "恐惧", "keyword": "手机", "text": "找情绪是恐惧、且提到「手机」的片段"},
    {"emotion": "悲伤", "keyword": "累", "text": "找情绪是悲伤、且提到「累」的片段"},
    {"emotion": "喜悦", "keyword": "菜", "text": "找情绪是喜悦、且提到「菜」的片段"},
    {"emotion": "厌恶", "hour": 15, "text": "找情绪是厌恶、且在 15 点的片段"},
    {"emotion": "中性", "hour": 14, "text": "找情绪是中性、且在 14 点的片段"},
]


def build_predicate(spec: dict) -> tuple[str, dict]:
    clauses: list[str] = []
    params: dict = {}
    if spec.get("emotion"):
        clauses.append("s.emotion = :emotion")
        params["emotion"] = spec["emotion"]
    if spec.get("hour") is not None:
        # 必须同时锁日期：库里另有 11-15、11-16 的离群片段，
        # 只按 HOUR() 过滤会把它们混进来，与"11 月 11 日某点"的语义不符。
        clauses.append("DATE(s.start_time) = :qdate")
        params["qdate"] = QUERY_DATE
        clauses.append("HOUR(s.start_time) = :hour")
        params["hour"] = int(spec["hour"])
    if spec.get("keyword"):
        clauses.append("s.asr_text LIKE :kw")
        params["kw"] = "%" + spec["keyword"] + "%"
    if spec.get("min_duration_ms"):
        clauses.append("c.duration_ms >= :dur")
        params["dur"] = int(spec["min_duration_ms"])
    if not clauses:
        raise ValueError(f"条件为空：{spec}")
    return " AND ".join(clauses), params


def main() -> int:
    parser = argparse.ArgumentParser(description="构建结构化组合条件查询集")
    parser.add_argument("--output", default="experiments/output/query_set_structured.json")
    args = parser.parse_args()

    db = SessionLocal()
    out: list[dict] = []
    empty: list[str] = []

    for group, specs in [("单条件查询", SINGLE_SPECS), ("多条件组合查询", MULTI_SPECS)]:
        for spec in specs:
            pred, params = build_predicate(spec)
            sql = text(
                "SELECT s.id FROM segments s JOIN conversations c ON s.conversation_id = c.id "
                "WHERE " + pred + " ORDER BY s.start_time"
            )
            ids = [int(r[0]) for r in db.execute(sql, params).fetchall()]
            if not ids:
                empty.append(spec["text"])
            out.append({
                "query_id": len(out) + 1,
                "group": group,
                "query": spec["text"],
                "spec": spec,
                "n_relevant": len(ids),
                "relevant_segment_ids": ids,
            })

    db.close()

    if empty:
        print("[错误] 以下查询真值为空，请调整条件：")
        for e in empty:
            print("  -", e)
        return 1

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"共 {len(out)} 条查询，真值由 SQL 条件机械判定：")
    for r in out:
        print(f"  #{r['query_id']:2d} [{r['group']}] {r['query']} -> {r['n_relevant']} 条")
    print(f"\n输出 -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
