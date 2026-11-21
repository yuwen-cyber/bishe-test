#!/usr/bin/env python3
"""实验 4 配套：跨字面语义召回案例分析生成器。

从已有评测产物里挑出"关键词/SQL 基线全错过、但向量语义检索命中"的查询，
生成可直接放入论文的案例分析文档（含逐条证据），用于支撑
"语义检索能跨字面表达召回"这一论点。

输入（均为已有产物，不改动任何业务代码）：
    experiments/output/query_set_db.json     查询真值与类型
    experiments/output/per_query_detail.csv  四模式（baseline/advanced_plain…）逐条明细
    experiments/output/per_query_hybrid.csv  混合评测（keyword/vector/hybrid…）逐条明细

输出：
    experiments/output/case_study_cross_literal.md   论文可用的案例段落
    experiments/output/case_study_cross_literal.csv  案例证据表

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\pick_case_study.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _ids(field: str) -> list[int]:
    return [int(x) for x in field.split("|") if x.strip()]


def _fetch_segment_text(ids: list[int]) -> dict[int, str]:
    """只读取片段原文用于案例举证；失败时返回空表，不影响主流程。"""
    if not ids:
        return {}
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from sqlalchemy import text
        from app.database import SessionLocal

        db = SessionLocal()
        try:
            sql = text(
                "SELECT id, asr_text FROM segments WHERE id IN ("
                + ",".join(str(int(i)) for i in sorted(set(ids)))
                + ")"
            )
            rows = db.execute(sql).fetchall()
            return {int(r[0]): (r[1] or "") for r in rows}
        finally:
            db.close()
    except Exception as exc:  # 离线环境/数据库不可用时降级
        print(f"[warn] 读取片段原文失败（跳过原文列）：{exc}")
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="跨字面语义召回案例生成")
    parser.add_argument("--query-set", default="experiments/output/query_set_db.json")
    parser.add_argument("--detail", default="experiments/output/per_query_detail.csv")
    parser.add_argument("--hybrid", default="experiments/output/per_query_hybrid.csv")
    parser.add_argument("--output-dir", default="experiments/output")
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--no-db-text", action="store_true",
                        help="不连 MySQL 取片段原文，只输出 ID")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    queries = json.loads(Path(args.query_set).read_text(encoding="utf-8"))
    qmeta = {int(q["query_id"]): q for q in queries}

    detail = _load_csv(Path(args.detail))
    hybrid = _load_csv(Path(args.hybrid))

    def index(rows: list[dict]) -> dict[tuple[int, str], dict]:
        return {(int(r["query_id"]), r["config"]): r for r in rows}

    d_idx = index(detail)
    h_idx = index(hybrid)

    cases: list[dict] = []
    for qid in sorted(qmeta):
        v = h_idx.get((qid, "vector"))
        k = h_idx.get((qid, "keyword"))
        b = d_idx.get((qid, "baseline"))
        a = d_idx.get((qid, "advanced_plain"))
        if not (v and k and b):
            continue

        vec_hits = _ids(v["hit_segment_ids"])
        kw_hits = _ids(k["hit_segment_ids"])
        base_hits = _ids(b["hit_segment_ids"])
        adv_hits = _ids(a["hit_segment_ids"]) if a else []

        # 语义检索命中，而关键词与 SQL 基线都未命中 —— 典型的跨字面召回
        if vec_hits and not kw_hits and not base_hits:
            cases.append({
                "query_id": qid,
                "type": qmeta[qid]["type"],
                "query": qmeta[qid]["query"],
                "relevant_ids": _ids(v["relevant_ids"]),
                "vector_hits": vec_hits,
                "keyword_hits": kw_hits,
                "baseline_hits": base_hits,
                "advanced_hits": adv_hits,
                "vector_r1": v["recall@1"],
                "vector_r5": v["recall@5"],
                "vector_top_ids": _ids(v["retrieved_segment_ids"])[:3],
            })

    # 优先挑语义查询与跨模态查询，它们最能体现"说法不同但意思相同"
    priority = {"语义查询": 0, "跨模态查询": 1, "关键词查询": 2, "结构化精确查询": 3}
    cases.sort(key=lambda c: (priority.get(c["type"], 9), c["query_id"]))
    picked = cases[: args.max_cases]

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 证据表 CSV ----
    csv_path = out_dir / "case_study_cross_literal.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "query_id", "type", "query", "relevant_ids", "vector_hits",
                "keyword_hits", "baseline_hits", "advanced_hits",
                "vector_recall@1", "vector_recall@5",
            ],
        )
        w.writeheader()
        for c in picked:
            w.writerow({
                "query_id": c["query_id"],
                "type": c["type"],
                "query": c["query"],
                "relevant_ids": "|".join(map(str, c["relevant_ids"])),
                "vector_hits": "|".join(map(str, c["vector_hits"])) or "无",
                "keyword_hits": "|".join(map(str, c["keyword_hits"])) or "无",
                "baseline_hits": "|".join(map(str, c["baseline_hits"])) or "无",
                "advanced_hits": "|".join(map(str, c["advanced_hits"])) or "无",
                "vector_recall@1": c["vector_r1"],
                "vector_recall@5": c["vector_r5"],
            })

    # ---- 论文用 Markdown ----
    texts: dict[int, str] = {}
    if not args.no_db_text:
        all_ids = [sid for c in picked for sid in c["vector_hits"]]
        texts = _fetch_segment_text(all_ids)

    md_path = out_dir / "case_study_cross_literal.md"
    lines: list[str] = []
    lines.append("# 案例分析：语义检索的跨字面表达能力\n")
    lines.append(
        "统计口径：在同一 50 条查询集上，筛选「纯关键词（MySQL 全文）与基线 SQL 均未命中真值、"
        "而纯向量语义检索命中」的查询，作为语义检索跨字面召回的证据。"
        f"共筛出 {len(cases)} 条，下表按查询类型优先级取前 {len(picked)} 条。\n"
    )
    lines.append("| # | 查询类型 | 用户查询（字面表达） | 真值片段 | 语义检索命中 | 关键词命中 | 基线命中 |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, c in enumerate(picked, start=1):
        lines.append(
            f"| {i} | {c['type']} | {c['query']} | "
            f"{', '.join(map(str, c['relevant_ids']))} | "
            f"{', '.join(map(str, c['vector_hits']))} | 无 | 无 |"
        )
    lines.append("")

    if texts:
        lines.append("## 命中片段原文（举证）\n")
        for i, c in enumerate(picked, start=1):
            lines.append(f"**案例 {i}｜查询：{c['query']}**\n")
            for sid in c["vector_hits"]:
                origin = texts.get(sid, "（未取到原文）")
                lines.append(f"- `#{sid}`：{origin}")
            lines.append("")

    lines.append("## 结论\n")
    lines.append(
        "- 这些查询的字面表述（如口语化情绪描述）与片段转写文本用词并不重合，"
        "全文检索依赖词面匹配因而全部落空；向量检索把查询与片段映射到同一语义空间后仍能召回。"
    )
    lines.append(
        "- 这说明纯关键词方案的能力边界在「词面不一致」，而语义检索补的正是这块长尾，"
        "也是后续引入 SQL+向量混合检索（提升召回广度）的动机。\n"
    )
    lines.append("> 注意：以上命中均以 `output/query_set_db.json` 中的真值 `relevant_segment_ids` 判定，")
    lines.append("> 指标只统计 top-10 内命中，与 `metrics_hybrid.csv` 口径一致。\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"筛出跨字面案例 {len(cases)} 条，输出前 {len(picked)} 条")
    for c in picked:
        print(f"  #{c['query_id']} [{c['type']}] {c['query']} -> 命中 {c['vector_hits']}")
    print(f"\n案例文档 -> {md_path}")
    print(f"证据表   -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
