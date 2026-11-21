#!/usr/bin/env python3
"""核对重建结果：实体数、segment_id 缺口、四路向量维度与范数。只读，不写库。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from app import milvus_client  # noqa: E402

mapping = json.loads(Path("experiments/output/seed_mapping_real.json").read_text(encoding="utf-8"))
known = {int(x["db_segment_id"]) for x in mapping}

col = milvus_client._get_collection()
col.load()
col.flush()
print("num_entities =", col.num_entities)

rows = col.query(
    expr="id >= 0",
    output_fields=["segment_id", "emotion", "embedding", "audio_embedding",
                   "image_embedding", "emotion_embedding"],
    limit=16384,
)
print("query 返回行数 =", len(rows))

ids = [int(r["segment_id"]) for r in rows]
id_set = set(ids)
missing = sorted(known - id_set)
extra = sorted(id_set - known)
print(f"真值 {len(known)}，已入库 {len(id_set)}，缺失 {len(missing)}，多余 {len(extra)}")
print("缺失 segment_id:", missing)
print("多余 segment_id:", extra)

# 维度与范数抽查（全量统计）
def stats(key: str, expect_dim: int | None = None) -> None:
    dims, norms, empty = {}, [], 0
    for r in rows:
        v = r.get(key)
        if v is None or len(v) == 0:
            empty += 1
            continue
        v = np.asarray(v, dtype=np.float32)
        dims[len(v)] = dims.get(len(v), 0) + 1
        norms.append(float(np.linalg.norm(v)))
    norms = np.array(norms) if norms else np.array([0.0])
    print(f"{key:18s} 空向量={empty} 维度分布={dims} 范数 min/mean/max="
          f"{norms.min():.4f}/{norms.mean():.4f}/{norms.max():.4f}")

stats("embedding", 512)
stats("audio_embedding", 768)
stats("image_embedding", 512)
stats("emotion_embedding", 64)

# 情绪分布
from collections import Counter
emo = Counter(r.get("emotion") or "" for r in rows)
print("库内情绪分布:", dict(emo))
