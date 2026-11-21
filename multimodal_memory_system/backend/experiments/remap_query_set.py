#!/usr/bin/env python3
"""用灌库映射表把 query_set.json 的原始编号回填为数据库 segment_id。

背景：
- 数据集原始编号 0~700（segments_dataset.json 的 segment_id）；
- 灌入 MySQL 后 segment.id 是数据库自增编号，两者不一致；
- run_eval.py 用后端检索返回的 segment_id 跟 relevant_segment_ids 匹配，
  因此评测前必须把查询集真值里的原始编号替换成数据库编号。

用法（在 backend 目录下）：
    .venv\\Scripts\\python.exe experiments\\remap_query_set.py

输入：
    experiments/output/seed_mapping_real.json  （seed_real_data.py 产出）
    experiments/query_set.json
输出：
    experiments/output/query_set_db.json      （回填后的评测查询集）

说明：
- 映射不到的原始编号会被记到缺失清单并从真值中剔除，避免把未入库片段误当成相关项；
- 原始 query_set.json 不会被修改。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENTS_DIR / "output"
MAPPING_PATH = OUTPUT_DIR / "seed_mapping_real.json"
QUERYSET_PATH = EXPERIMENTS_DIR / "query_set.json"
OUTPUT_PATH = OUTPUT_DIR / "query_set_db.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="回填查询集真值编号到数据库 segment_id")
    parser.add_argument("--mapping", default=str(MAPPING_PATH))
    parser.add_argument("--query-set", default=str(QUERYSET_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    orig_to_db: dict[int, int] = {}
    for m in mapping:
        db_seg = m.get("db_segment_id")
        if db_seg is not None:
            orig_to_db[int(m["orig_segment_id"])] = int(db_seg)

    queries = json.loads(Path(args.query_set).read_text(encoding="utf-8"))

    missing: dict[str, list[int]] = {}
    out_queries: list[dict] = []
    n_rel_total = 0
    n_mapped_total = 0

    for q in queries:
        orig_ids = [int(x) for x in q.get("relevant_segment_ids", [])]
        mapped: list[int] = []
        miss: list[int] = []
        for oid in orig_ids:
            if oid in orig_to_db:
                mapped.append(orig_to_db[oid])
            else:
                miss.append(oid)
        # 去重保序
        seen = set()
        mapped_unique = [x for x in mapped if not (x in seen or seen.add(x))]

        n_rel_total += len(orig_ids)
        n_mapped_total += len(mapped_unique)
        if miss:
            missing[str(q.get("query_id"))] = miss

        new_q = dict(q)
        new_q["relevant_segment_ids"] = mapped_unique
        out_queries.append(new_q)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(out_queries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"映射表条目数: {len(orig_to_db)}")
    print(f"查询条数: {len(out_queries)}；相关项合计 {n_rel_total}，成功回填 {n_mapped_total}")
    if missing:
        print(f"\n[警告] 有 {len(missing)} 条查询包含未映射到数据库的原始编号（已剔除）：")
        for qid, miss in missing.items():
            print(f"   query_id={qid} 缺失原始编号={miss}")
    else:
        print("全部相关编号均已成功回填。")
    print(f"\n回填后的查询集 -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
