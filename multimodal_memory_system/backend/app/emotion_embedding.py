from __future__ import annotations

import numpy as np

# 情绪标签集合（与 emotion.py 保持一致）
EMOTIONS = ["喜悦", "愤怒", "悲伤", "恐惧", "惊讶", "厌恶", "中性"]
_DIM = 64

# 固定 seed 生成 7x64 确定性查找表：每个标签对应一行，不训练
_rng = np.random.default_rng(42)
_TABLE = _rng.standard_normal((len(EMOTIONS), _DIM)).astype(np.float32)


def embed_emotion(label: str) -> list[float]:
    """返回情绪标签对应的 64 维向量；未知标签返回全零向量。"""
    if label in EMOTIONS:
        return _TABLE[EMOTIONS.index(label)].tolist()
    return [0.0] * _DIM