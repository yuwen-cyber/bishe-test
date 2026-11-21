from __future__ import annotations

import hashlib
import logging

import numpy as np

from .config import settings

logger = logging.getLogger(__name__)
_model = None


def _load_model():
    global _model
    if _model is None:
        import os

        from sentence_transformers import SentenceTransformer

        model_name = "BAAI/bge-small-zh-v1.5"
        # 优先离线加载本地缓存，避免 HuggingFace 镜像限流(HTTP 429)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            _model = SentenceTransformer(model_name)
        except Exception as e:  # 本地无缓存时回退到在线下载
            logger.warning("Offline load failed (%s), falling back to online download", e)
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            _model = SentenceTransformer(model_name)
    return _model


def _mock_embed(text: str, dim: int) -> list:
    """无模型时的确定性降级向量化：基于字符 n-gram 哈希，归一化到单位向量。"""
    vec = np.zeros(dim, dtype=np.float32)
    for gram_size in (1, 2):
        for i in range(len(text) - gram_size + 1):
            h = int(hashlib.md5(text[i : i + gram_size].encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        vec = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
    else:
        vec = vec / norm
    return vec.tolist()


def embed_dim() -> int:
    if settings.use_real_embedding:
        return int(_load_model().get_sentence_embedding_dimension())
    return settings.embedding_dim


def embed_text(text: str | list) -> list | list[list]:
    single = isinstance(text, str)
    texts = [text] if single else text
    if settings.use_real_embedding:
        model = _load_model()
        vecs = [v.tolist() for v in model.encode(texts, normalize_embeddings=True)]
    else:
        vecs = [_mock_embed(t, settings.embedding_dim) for t in texts]
    return vecs[0] if single else vecs


def warmup() -> None:
    if settings.use_real_embedding:
        try:
            _load_model()
            logger.info("Embedding model loaded: BAAI/bge-small-zh-v1.5")
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to load embedding model, fallback to mock: %s", e)
    else:
        logger.info("Embedding uses mock encoder (dim=%s)", settings.embedding_dim)