#!/usr/bin/env python3
"""实验 3 补充：结构化查询的**接口层**验证与延迟测试。

为什么单独做这一份：
- 离线脚本（run_structured_eval.py）测的是 SQL 层耗时；
- 论文里"基准方案的能力展示"需要证明**系统接口**真的能执行这类条件查询。
- 现有 /api/baseline/search、/api/advanced/search 都强制要求自然语言查询词 q，
  且向量接口带 0.5 相似度阈值，无法表达"只按条件查"。为此新增独立路由
  app/routers/structured.py（GET /api/structured/search，不改原有两个接口）。

本脚本用 FastAPI TestClient 直接挂载该路由（不加载 ASR/Embedding 等模型，
路由本身只查 MySQL，因此这里的延迟即为接口+SQL 的真实开销），检查：
  1) 返回结果与 SQL 真值集合是否**完全一致**（正确性）
  2) 客户端实测延迟 P50 / P95 / max（接口层）

注意：正式部署的后端需重启才会加载该路由（main.py 已注册两行）。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\run_structured_http_eval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.routers import structured  # noqa: E402

DAY = datetime(2026, 11, 11)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _params_from_spec(spec: dict) -> dict:
    params: dict = {"limit": 300}
    if spec.get("emotion"):
        params["emotion"] = spec["emotion"]
    if spec.get("hour") is not None:
        start = DAY + timedelta(hours=int(spec["hour"]))
        end = start + timedelta(hours=1) - timedelta(seconds=1)
        params["start"] = start.strftime("%Y-%m-%d %H:%M:%S")
        params["end"] = end.strftime("%Y-%m-%d %H:%M:%S")
    if spec.get("keyword"):
        params["keyword"] = spec["keyword"]
    if spec.get("min_duration_ms"):
        params["min_duration_ms"] = int(spec["min_duration_ms"])
    return params


def main() -> int:
    parser = argparse.ArgumentParser(description="结构化查询接口层验证")
    parser.add_argument("--query-set", default="experiments/output/query_set_structured.json")
    parser.add_argument("--output-dir", default="experiments/output")
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    queries = json.loads(Path(args.query_set).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI()
    app.include_router(structured.router, prefix="/api")
    client = TestClient(app)

    # 预热：排除首个请求的建连/导入开销
    for _ in range(args.warmup):
        client.get("/api/structured/search", params={"emotion": "喜悦", "limit": 5})

    rows: list[dict] = []
    latencies: list[float] = []
    for q in queries:
        spec = q["spec"]
        truth = set(int(x) for x in q["relevant_segment_ids"])
        params = _params_from_spec(spec)

        t0 = time.perf_counter()
        resp = client.get("/api/structured/search", params=params)
        ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code != 200:
            print(f"[失败] #{q['query_id']} HTTP {resp.status_code}: {resp.text[:120]}")
            return 1

        body = resp.json()
        returned = [int(it["segment_id"]) for it in body["results"]]
        hits = sum(1 for sid in returned if sid in truth)
        p = hits / len(returned) if returned else 0.0
        r = hits / len(truth) if truth else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        exact = set(returned) == truth

        latencies.append(ms)
        rows.append({
            "query_id": q["query_id"],
            "group": q["group"],
            "query": q["query"],
            "params": json.dumps(params, ensure_ascii=False),
            "n_relevant": len(truth),
            "api_total": body.get("total"),
            "n_returned": len(returned),
            "n_hits": hits,
            "precision_set": round(p, 4),
            "recall_set": round(r, 4),
            "f1_set": round(f, 4),
            "set_exact_match": exact,
            "client_latency_ms": round(ms, 2),
            "server_retrieval_ms": round(float(body.get("retrieval_time_ms", 0.0)), 2),
        })

    with (out_dir / "per_query_structured_http.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_exact = sum(1 for r in rows if r["set_exact_match"])
    p95 = _percentile(latencies, 0.95)
    summary = {
        "num_queries": len(rows),
        "exact_match": n_exact,
        "mean_precision": round(sum(r["precision_set"] for r in rows) / len(rows), 4),
        "mean_recall": round(sum(r["recall_set"] for r in rows) / len(rows), 4),
        "mean_f1": round(sum(r["f1_set"] for r in rows) / len(rows), 4),
        "client_p50_ms": round(_percentile(latencies, 0.5), 2),
        "client_p95_ms": round(p95, 2),
        "client_max_ms": round(max(latencies), 2),
    }
    with (out_dir / "metrics_structured_http.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    md = [
        "# 结构化查询：接口层验证（实验 3 补充）\n",
        "## 目的",
        "证明**系统接口**（`GET /api/structured/search`）能真正执行"
        "「情绪 / 时间范围 / 关键词 / 时长」任意组合的条件查询，"
        "并给出接口层客户端实测延迟。",
        "原 `/api/advanced/search` 强制要求自然语言查询词且带 0.5 相似度阈值，"
        "无法表达「只按条件查」，故新增独立路由（不改原接口）。\n",
        "## 结果",
        f"- 查询数：{summary['num_queries']} 条",
        f"- 返回集合与 SQL 真值**完全一致**：{n_exact}/{summary['num_queries']} 条",
        f"- 集合级 Precision / Recall / F1：{summary['mean_precision']} / "
        f"{summary['mean_recall']} / {summary['mean_f1']}",
        f"- 客户端延迟 P50 / P95 / max：{summary['client_p50_ms']} / "
        f"{summary['client_p95_ms']} / {summary['client_max_ms']} ms",
        "",
        "## 口径与说明",
        "- 本测试用 FastAPI TestClient 直接挂载该路由，**不加载 ASR/Embedding 模型**"
        "（该路由只查 MySQL，不含推理），因此延迟反映的是接口 + SQL 的真实开销，"
        "与 `/advanced/search` 的向量检索延迟不可直接比较。",
        "- 正式运行的后端需重启以加载该路由；重启后可用 "
        "`curl \"http://localhost:8000/api/structured/search?emotion=悲伤&limit=5\"` 自查。",
        "- 指标为 1.0 属构造必然（真值即条件结果），此处价值在于"
        "**接口可用性 + 结果一致性 + 接口延迟**三项证据，不构成算法比较。",
    ]
    (out_dir / "structured_http_eval.md").write_text("\n".join(md), encoding="utf-8")

    print(f"接口层验证：{n_exact}/{len(rows)} 条集合完全一致；"
          f"P/R/F1={summary['mean_precision']}/{summary['mean_recall']}/{summary['mean_f1']}；"
          f"客户端 P50={summary['client_p50_ms']}ms P95={summary['client_p95_ms']}ms")
    print(f"输出 -> {out_dir / 'metrics_structured_http.csv'}、{out_dir / 'structured_http_eval.md'}")
    return 0 if n_exact == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
