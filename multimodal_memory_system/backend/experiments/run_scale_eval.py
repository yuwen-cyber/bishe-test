#!/usr/bin/env python3
"""系统性能与可扩展性测试（简化版，实验 #6）。

按文档建议，在 100 / 500 / 1000 三个数据规模档位上测量端到端查询延迟分解与资源占用。

设计原则（不修改业务代码、不污染生产库）：
- 生产 Milvus 集合 `segments` 只读：仅 query 取出文本向量与元数据映射，不做任何写入/删除。
- 每个档位在独立的临时集合 `segments_scale_<n>` 上建 HNSW 索引并检索；测完自动 drop。
- 1000 档真实样本不足时，按文档允许的方式“复制扩充”：循环复用已有向量补齐到 1000，
  用于观察索引规模增长对检索延迟的趋势，论文需如实说明为复制模拟、非真实新增样本。

延迟分解（每段单独计时，单位 ms）：
  1. embed  查询编码（bge 文本向量化）
  2. search Milvus ANN 向量检索
  3. fetch 结果元数据组装（本地映射取文本/情绪，模拟回查数据库与结果拼装）
  4. total  端到端 = embed + search + fetch

资源占用：
  - cpu_delta_s   本档进程累计 CPU 时间增量（秒，time.process_time）
  - mem_mb        本档结束时进程工作集内存（MB）
  - mem_peak_mb   进程工作集峰值（MB）
  - api_calls     外部大模型/API 调用次数（本项目本地编码器 + 词典降级，记 0）

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\run_scale_eval.py
    .venv\\Scripts\\python.exe experiments\\run_scale_eval.py --levels 100,500,1000 --top-k 10
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from pymilvus import (  # noqa: E402
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from app import embeddings, milvus_client  # noqa: E402
from app.config import settings  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("scale_eval")

TOP_K = 10
TEMP_PREFIX = "segments_scale_"


# ---------------- 资源占用（Windows 标准库，无 psutil 依赖） ----------------

def _mem_usage_mb() -> tuple[float, float]:
    """返回 (当前工作集 MB, 峰值工作集 MB)；非 Windows 兜底返回 0。"""
    if sys.platform != "win32":
        return 0.0, 0.0
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32.dll")
    psapi = ctypes.WinDLL("psapi.dll")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD,
    ]

    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    h = k32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    if not ok:
        return 0.0, 0.0
    return pmc.WorkingSetSize / 1024 / 1024, pmc.PeakWorkingSetSize / 1024 / 1024


# ---------------- 读取生产语料（只读） ----------------

def load_corpus() -> tuple[list[int], list[list[float]], dict[int, dict]]:
    col = milvus_client._get_collection()
    col.load()
    rows = col.query(
        expr="id >= 0",
        output_fields=["segment_id", "text", "emotion", "embedding"],
        limit=16384,
    )
    if not rows:
        raise RuntimeError("生产集合为空，无法做规模测试。")
    seg_ids, vectors, meta = [], [], {}
    for r in rows:
        sid = int(r["segment_id"])
        seg_ids.append(sid)
        vectors.append(r["embedding"])
        meta[sid] = {"text": r.get("text") or "", "emotion": r.get("emotion") or ""}
    return seg_ids, vectors, meta


def build_level(seg_ids: list[int], vectors: list[list[float]], n: int) -> tuple[list[int], list[list[float]]]:
    """取前 n 条；不足 n 时循环复制补齐（确定性，无随机）。"""
    actual = len(seg_ids)
    idx = [i % actual for i in range(n)]
    ids = [seg_ids[i] for i in idx]
    vecs = [vectors[i] for i in idx]
    return ids, vecs


# ---------------- 临时集合构建 / 清理 ----------------

def _connect() -> None:
    connections.connect(alias="default", host=settings.milvus_host, port=settings.milvus_port)


def create_temp_collection(name: str, dim: int, ids: list[int], vecs: list[list[float]]) -> Collection:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="segment_id", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="scale benchmark temp collection")
    col = Collection(name, schema=schema)
    col.insert([ids, vecs])
    col.flush()
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }
    col.create_index("embedding", index_params)
    col.load()
    return col


def drop_temp(name: str) -> None:
    try:
        _connect()
        if utility.has_collection(name):
            utility.drop_collection(name)
    except Exception as e:  # pragma: no cover
        logger.warning("清理临时集合 %s 失败: %s", name, e)


# ---------------- 统计辅助 ----------------

def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    arr = np.sort(np.asarray(xs, dtype=np.float64))
    return float(np.percentile(arr, q * 100))


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="三档规模性能与资源占用测试")
    parser.add_argument("--levels", default="100,500,1000", help="逗号分隔的规模档位")
    parser.add_argument("--query-set", default="experiments/output/query_set_db.json")
    parser.add_argument("--output-dir", default="experiments/output")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = json.loads(Path(args.query_set).read_text(encoding="utf-8"))
    embeddings.warmup()

    seg_ids, vectors, meta = load_corpus()
    dim = len(vectors[0])
    logger.info("生产语料 %d 条，向量维度 %d", len(seg_ids), dim)

    # 先把所有查询文本编码好，避免编码耗时在档位间反复抖动；编码段仍单独计时
    qtexts = [q["query"] for q in queries]

    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    for level in levels:
        name = f"{TEMP_PREFIX}{level}"
        ids, vecs = build_level(seg_ids, vectors, level)
        col: Collection | None = None

        cpu_base = time.process_time()
        _, mem_peak_base = _mem_usage_mb()

        try:
            col = create_temp_collection(name, dim, ids, vecs)

            # 预热：前 3 条查询走一遍完整流程，排除冷加载/缓存抖动，不计入统计
            for qtext in qtexts[:3]:
                v = embeddings.embed_text(qtext)
                col.search(data=[v], anns_field="embedding", limit=args.top_k,
                           param={"metric_type": "COSINE", "params": {"ef": 64}},
                           output_fields=["segment_id"])

            embed_ms, search_ms, fetch_ms, total_ms = [], [], [], []

            for q in queries:
                qtext = q["query"]

                t0 = time.perf_counter()
                q_vec = embeddings.embed_text(qtext)
                t1 = time.perf_counter()

                hits = col.search(
                    data=[q_vec],
                    anns_field="embedding",
                    limit=args.top_k,
                    param={"metric_type": "COSINE", "params": {"ef": 64}},
                    output_fields=["segment_id"],
                )
                t2 = time.perf_counter()

                # 结果组装：取 top-K segment_id，从本地映射取文本/情绪（模拟回查拼装）
                top_ids: list[int] = []
                for batch in hits:
                    for h in batch:
                        sid = int(h.entity.get("segment_id"))
                        top_ids.append(sid)
                        _ = meta.get(sid, {})
                t3 = time.perf_counter()

                e = (t1 - t0) * 1000
                s = (t2 - t1) * 1000
                f = (t3 - t2) * 1000
                embed_ms.append(e)
                search_ms.append(s)
                fetch_ms.append(f)
                total_ms.append(e + s + f)

                detail_rows.append({
                    "level": level,
                    "query_id": q.get("query_id"),
                    "query": qtext,
                    "embed_ms": round(e, 3),
                    "search_ms": round(s, 3),
                    "fetch_ms": round(f, 3),
                    "total_ms": round(e + s + f, 3),
                    "top_ids": "|".join(str(x) for x in top_ids),
                })

            cpu_delta = time.process_time() - cpu_base
            mem_now, mem_peak = _mem_usage_mb()

            summary_rows.append({
                "level": level,
                "num_queries": len(queries),
                "embed_avg_ms": round(_mean(embed_ms), 3),
                "embed_p95_ms": round(_pct(embed_ms, 0.95), 3),
                "search_avg_ms": round(_mean(search_ms), 3),
                "search_p95_ms": round(_pct(search_ms, 0.95), 3),
                "fetch_avg_ms": round(_mean(fetch_ms), 3),
                "total_avg_ms": round(_mean(total_ms), 3),
                "total_p50_ms": round(_pct(total_ms, 0.5), 3),
                "total_p95_ms": round(_pct(total_ms, 0.95), 3),
                "cpu_delta_s": round(cpu_delta, 3),
                "mem_mb": round(mem_now, 1),
                "mem_peak_mb": round(max(mem_peak_base, mem_peak), 1),
                "api_calls": 0,
            })
            logger.info("档位 %d 完成：端到端平均 %.2fms / P95 %.2fms",
                        level, _mean(total_ms), _pct(total_ms, 0.95))
        finally:
            if col is not None:
                try:
                    col.drop()
                except Exception:
                    pass
            drop_temp(name)

    _write_csv(out_dir / "metrics_scale.csv", summary_rows)
    _write_csv(out_dir / "per_query_scale.csv", detail_rows)

    print("\n=== 三档规模性能测试 ===")
    for r in summary_rows:
        print(f"N={r['level']:<5} 编码={r['embed_avg_ms']:.2f}ms 检索={r['search_avg_ms']:.2f}ms "
              f"端到端均值={r['total_avg_ms']:.2f}ms P95={r['total_p95_ms']:.2f}ms "
              f"CPU={r['cpu_delta_s']}s 内存峰值={r['mem_peak_mb']}MB")
    print(f"\n汇总 -> {out_dir / 'metrics_scale.csv'}")
    print(f"明细 -> {out_dir / 'per_query_scale.csv'}")
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
