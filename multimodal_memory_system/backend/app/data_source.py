"""数据来源判定：区分 正式数据 / 噪声数据 / 用户自录。

只做展示用标记，不改动任何业务字段：
    - 正式数据：id 1~700（最早灌入、来自真实录音转写）
    - 噪声数据：id 命中 experiments/output/batch_import_mapping.json 的 1300 个
    - 用户自录：is_demo=False 且不在噪声映射内
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_MAPPING_PATH = (
    Path(__file__).resolve().parent.parent
    / "experiments" / "output" / "batch_import_mapping.json"
)


@lru_cache(maxsize=1)
def _noise_ids() -> frozenset[int]:
    try:
        data = json.loads(_MAPPING_PATH.read_text(encoding="utf-8"))
        return frozenset(
            int(x["db_segment_id"]) for x in data if x.get("db_segment_id") is not None
        )
    except Exception as e:  # pragma: no cover
        logger.warning("无法读取噪声映射 %s: %s", _MAPPING_PATH, e)
        return frozenset()


def data_source(segment_id: int, is_demo: bool | None = None) -> str:
    """返回 formal / noise / user / unknown。"""
    sid = int(segment_id)
    if sid in _noise_ids():
        return "noise"
    if is_demo is False:
        return "user"
    if 1 <= sid <= 700:
        return "formal"
    return "unknown"
