#!/usr/bin/env python3
"""模态消融评测（路线 B 配套）。

从 Milvus 拉取全量片段的四路真向量，在 50 条文本查询真值集上对比不同模态组合的检索效果。

背景与口径（务必在论文局限中说明）：
- 查询侧只有文本；音频(Wav2Vec2)、图像(CLIP) 向量与文本向量原本不在同一语义空间，
  也没有训练过跨模态对齐层。本脚本用库内“同一片段自身的文本向量”作为监督目标，
  以岭回归学一个线性投影，把音频/图像向量映射到文本向量空间（轻量对齐，非深度跨模态预训练）。
- 文本向量(bge)、音频向量、图像向量入库前均已 L2 归一化；投影后再次归一化，余弦即点积。

对比配置（加权余弦融合，权重见 CONFIGS）：
  text_only            仅文本
  audio_only           仅音频（线性投影到文本空间）
  image_only           仅图像（线性投影到文本空间）
  text_audio           文本 + 音频
  text_image           文本 + 图像
  text_audio_image     文本 + 音频 + 图像
  full_plus_emotion    文本 + 音频 + 图像 + 情绪标签一致性

另出附表：各模态“留一自检索”同情绪命中率（衡量编码器在本数据集上的可分性，不接文本查询真值）。

用法（backend 目录）：
    .venv\\Scripts\\python.exe experiments\\run_ablation.py
    .venv\\Scripts\\python.exe experiments\\run_ablation.py --lam 10 --top-k 10
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

# 限制数值库线程数，降低内存副本（与重建脚本一致）
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import embeddings, milvus_client  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("ablation")

K_VALUES = (1, 5, 10)
TOP_K = 10

EMOTION_LABELS = ["喜悦", "愤怒", "悲伤", "恐惧", "惊讶", "厌恶", "中性"]

# 查询文本 -> 情绪标签：先匹配标签名，再匹配触发词
EMOTION_TRIGGERS: list[tuple[tuple[str, ...], str]] = [
    (("喜悦", "开心", "高兴", "兴奋", "满足", "愉快", "幸福", "快乐"), "喜悦"),
    (("愤怒", "生气", "恼火", "气愤", "怒", "不爽", "火大"), "愤怒"),
    (("悲伤", "难过", "伤心", "失落", "沮丧", "委屈", "想哭", "低落", "郁闷"), "悲伤"),
    (("恐惧", "害怕", "紧张", "担心", "不安", "心慌", "焦虑"), "恐惧"),
    (("惊讶", "吃惊", "意外", "震惊"), "惊讶"),
    (("厌恶", "讨厌", "反感", "恶心", "嫌弃"), "厌恶"),
    (("中性", "平静", "平常", "一般"), "中性"),
]

CONFIGS: dict[str, dict[str, float]] = {
    "text_only": {"text": 1.0},
    "audio_only": {"audio": 1.0},
    "image_only": {"image": 1.0},
    "text_audio": {"text": 0.6, "audio": 0.4},
    "text_image": {"text": 0.6, "image": 0.4},
    "text_audio_image": {"text": 0.5, "audio": 0.25, "image": 0.25},
    "full_plus_emotion": {"text": 0.5, "audio": 0.2, "image": 0.2, "emotion": 0.1},
}


# ---------- 指标（口径与 run_eval.py 一致） ----------

def _dcg(rels: list[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def _ndcg_at_k(seg_ids: list[int], relevant: set[int], k: int) -> float:
    rels = [1.0 if sid in relevant else 0.0 for sid in seg_ids[:k]]
    dcg = _dcg(rels)
    ideal = [1.0] * min(len(relevant), k)
    idcg = _dcg(ideal)
    return dcg / idcg if idcg > 0.0 else 0.0


def _recall_at_k(seg_ids: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for sid in seg_ids[:k] if sid in relevant)
    return hits / len(relevant)


def _mrr(seg_ids: list[int], relevant: set[int]) -> float:
    for rank, sid in enumerate(seg_ids, start=1):
        if sid in relevant:
            return 1.0 / rank
    return 0.0


def detect_emotion(query: str) -> str | None:
    for label in EMOTION_LABELS:
        if label in query:
            return label
    for triggers, label in EMOTION_TRIGGERS:
        if any(t in query for t in triggers):
            return label
    return None


# ---------- 线性投影对齐 ----------

def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def fit_projection(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """岭回归：x(N,d) -> y(N,k)，返回增广权重 W(d+1,k)。"""
    n = x.shape[0]
    xb = np.hstack([x, np.ones((n, 1), dtype=x.dtype)])
    d = xb.shape[1]
    a = xb.T @ xb + lam * np.eye(d, dtype=x.dtype)
    b = xb.T @ y
    return np.linalg.solve(a, b)


def apply_projection(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xb = np.hstack([x, np.ones((x.shape[0], 1), dtype=x.dtype)])
    p = xb @ w
    return _normalize(p)


def load_corpus() -> dict:
    col = milvus_client._get_collection()
    col.load()
    rows = col.query(
        expr="id >= 0",
        output_fields=[
            "segment_id", "emotion", "embedding",
            "audio_embedding", "image_embedding", "emotion_embedding",
        ],
        limit=16384,
    )
    if not rows:
        raise RuntimeError("Milvus 集合为空，无法做消融评测，请先重建向量。")

    seg_ids, emotions = [], []
    text_v, audio_v, image_v, emo_v = [], [], [], []
    for r in rows:
        seg_ids.append(int(r["segment_id"]))
        emotions.append(r.get("emotion") or "")
        text_v.append(r["embedding"])
        audio_v.append(r["audio_embedding"])
        image_v.append(r["image_embedding"])
        emo_v.append(r["emotion_embedding"])

    return {
        "seg_ids": np.array(seg_ids, dtype=np.int64),
        "emotions": np.array(emotions),
        "text": _normalize(np.asarray(text_v, dtype=np.float32)),
        "audio": _normalize(np.asarray(audio_v, dtype=np.float32)),
        "image": _normalize(np.asarray(image_v, dtype=np.float32)),
        "emo": np.asarray(emo_v, dtype=np.float32),
    }


def score_candidates(
    q_text: np.ndarray,
    weights: dict[str, float],
    q_emotion: str | None,
    corpus: dict,
    audio_proj: np.ndarray,
    image_proj: np.ndarray,
) -> np.ndarray:
    n = corpus["text"].shape[0]
    scores = np.zeros(n, dtype=np.float64)
    if weights.get("text"):
        scores += weights["text"] * (corpus["text"] @ q_text)
    if weights.get("audio"):
        scores += weights["audio"] * (audio_proj @ q_text)
    if weights.get("image"):
        scores += weights["image"] * (image_proj @ q_text)
    if weights.get("emotion") and q_emotion is not None:
        match = (corpus["emotions"] == q_emotion).astype(np.float64)
        scores += weights["emotion"] * match
    return scores


def self_retrieval_hit(matrix: np.ndarray, emotions: np.ndarray, k: int) -> dict[str, float]:
    """留一自检索：每条作查询，找除自身外最近邻，统计同情绪命中率。"""
    sim = matrix @ matrix.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)
    n = matrix.shape[0]
    hit1 = hitk = 0.0
    for i in range(n):
        neigh = order[i, :k]
        neigh_emo = emotions[neigh]
        if neigh_emo[0] == emotions[i]:
            hit1 += 1.0
        if np.any(neigh_emo == emotions[i]):
            hitk += 1.0
    return {"hit@1": hit1 / n, f"hit@{k}": hitk / n}


def main() -> int:
    parser = argparse.ArgumentParser(description="模态消融评测")
    parser.add_argument("--query-set", default="experiments/output/query_set_db.json")
    parser.add_argument("--output-dir", default="experiments/output")
    parser.add_argument("--lam", type=float, default=10.0, help="岭回归正则系数")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    top_k = args.top_k
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = json.loads(Path(args.query_set).read_text(encoding="utf-8"))
    corpus = load_corpus()
    n = corpus["text"].shape[0]
    logger.info("库内片段 %d 条", n)

    # 线性投影对齐：音频/图像 -> 文本空间
    w_audio = fit_projection(corpus["audio"], corpus["text"], args.lam)
    w_image = fit_projection(corpus["image"], corpus["text"], args.lam)
    audio_proj = apply_projection(corpus["audio"], w_audio)
    image_proj = apply_projection(corpus["image"], w_image)

    embeddings.warmup()

    # 逐查询逐配置评测
    detail_rows: list[dict] = []
    agg: dict[str, dict[str, list[float]]] = {
        name: {"r1": [], "r5": [], "r10": [], "mrr": [], "ndcg10": [], "time": []}
        for name in CONFIGS
    }

    for q in queries:
        qid = q.get("query_id")
        qtext = q["query"]
        relevant = set(int(x) for x in q.get("relevant_segment_ids", []))
        q_emotion = detect_emotion(qtext)

        t0 = time.perf_counter()
        q_vec = np.asarray(embeddings.embed_text(qtext), dtype=np.float32)
        q_vec = q_vec / max(float(np.linalg.norm(q_vec)), 1e-12)
        t_embed = time.perf_counter() - t0

        for name, weights in CONFIGS.items():
            t1 = time.perf_counter()
            scores = score_candidates(q_vec, weights, q_emotion, corpus, audio_proj, image_proj)
            order = np.argsort(-scores)[:top_k]
            seg_ids = [int(corpus["seg_ids"][i]) for i in order]
            elapsed = (time.perf_counter() - t1) * 1000 + t_embed * 1000

            r1 = _recall_at_k(seg_ids, relevant, 1)
            r5 = _recall_at_k(seg_ids, relevant, 5)
            r10 = _recall_at_k(seg_ids, relevant, 10)
            mrr = _mrr(seg_ids, relevant)
            ndcg = _ndcg_at_k(seg_ids, relevant, 10)

            agg[name]["r1"].append(r1)
            agg[name]["r5"].append(r5)
            agg[name]["r10"].append(r10)
            agg[name]["mrr"].append(mrr)
            agg[name]["ndcg10"].append(ndcg)
            agg[name]["time"].append(elapsed)

            detail_rows.append({
                "query_id": qid,
                "query": qtext,
                "config": name,
                "q_emotion": q_emotion or "",
                "relevant_ids": "|".join(str(x) for x in sorted(relevant)),
                "retrieved_segment_ids": "|".join(str(x) for x in seg_ids),
                "hit_segment_ids": "|".join(str(x) for x in seg_ids if x in relevant),
                "recall@1": round(r1, 4),
                "recall@5": round(r5, 4),
                "recall@10": round(r10, 4),
                "mrr": round(mrr, 4),
                "ndcg@10": round(ndcg, 4),
            })

    # 汇总表
    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    metrics_rows = []
    for name in CONFIGS:
        a = agg[name]
        metrics_rows.append({
            "config": name,
            "recall@1": round(mean(a["r1"]), 4),
            "recall@5": round(mean(a["r5"]), 4),
            "recall@10": round(mean(a["r10"]), 4),
            "mrr": round(mean(a["mrr"]), 4),
            "ndcg@10": round(mean(a["ndcg10"]), 4),
            "avg_time_ms": round(mean(a["time"]), 2),
            "num_queries": len(queries),
        })

    metrics_path = out_dir / "metrics_ablation.csv"
    detail_path = out_dir / "per_query_ablation.csv"
    _write_csv(metrics_path, metrics_rows)
    _write_csv(detail_path, detail_rows)

    # 附表：单模态留一自检索同情绪命中率
    self_rows = []
    for modal, mat in [("text", corpus["text"]), ("audio", audio_proj),
                       ("image", image_proj), ("emotion", corpus["emo"])]:
        h = self_retrieval_hit(mat, corpus["emotions"], 5)
        self_rows.append({"modality": modal, "hit@1": round(h["hit@1"], 4),
                          "hit@5": round(h["hit@5"], 4)})
    self_path = out_dir / "ablation_self_retrieval.csv"
    _write_csv(self_path, self_rows)

    print(f"\n=== 模态消融汇总（{len(queries)} 查询，top{top_k}，岭回归 λ={args.lam}）===")
    for r in metrics_rows:
        print(f"{r['config']:32s} R@1={r['recall@1']:.4f} R@5={r['recall@5']:.4f} "
              f"R@10={r['recall@10']:.4f} MRR={r['mrr']:.4f} NDCG@10={r['ndcg@10']:.4f}")
    print(f"\n明细 -> {detail_path}\n汇总 -> {metrics_path}\n自检索附表 -> {self_path}")
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
