from __future__ import annotations

import logging
import os
import subprocess
import tempfile

from .config import settings

logger = logging.getLogger(__name__)
_nli_model = None
_er_model = None

# 候选情绪标签
EMOTIONS = ["喜悦", "愤怒", "悲伤", "恐惧", "惊讶", "厌恶", "中性"]

# 中文情感词典：显式情感词 → 情绪，保证“悲伤/难过/生气”这类词不被弱模型误判
_LEXICON = {
    "喜悦": ["开心", "高兴", "快乐", "喜悦", "幸福", "兴奋", "愉悦", "欢喜", "开心快乐", "太好了", "哈哈", "笑得"],
    "愤怒": ["生气", "愤怒", "气死", "恼火", "发火", "气炸", "气人", "想骂", "怒了", "很生气"],
    "悲伤": ["难过", "悲伤", "伤心", "想哭", "哭", "悲哀", "沮丧", "低落", "掉眼泪", "难受", "心酸", "委屈", "忧郁"],
    "恐惧": ["害怕", "恐惧", "吓", "紧张", "焦虑", "担心", "恐慌", "畏惧"],
    "惊讶": ["惊讶", "吃惊", "没想到", "居然", "竟然", "震惊", "意外"],
    "厌恶": ["恶心", "讨厌", "厌恶", "反感", "嫌弃"],
}

# emotion2vec 标签（英文 token）→ 中文情绪
_ER_LABEL_MAP = {
    "angry": "愤怒",
    "happy": "喜悦",
    "neutral": "中性",
    "sad": "悲伤",
    "surprised": "惊讶",
    "disgusted": "厌恶",
    "fearful": "恐惧",
    "other": "中性",
    "<unk>": "中性",
}


def _load_nli_model():
    global _nli_model
    if _nli_model is None:
        from transformers import pipeline

        # 多语言 NLI 模型做 zero-shot 分类，返回落在 EMOTIONS 上的概率分布
        _nli_model = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            device=-1,
        )
    return _nli_model


def _load_er_model():
    global _er_model
    if _er_model is None:
        from funasr import AutoModel

        # 语音情绪识别（emotion2vec），直接从音频语气/语调识别情绪
        _er_model = AutoModel(
            model="iic/emotion2vec_plus_large",
            device="cpu",
            disable_update=True,
        )
    return _er_model


def _to_wav_16k(audio_bytes: bytes) -> str:
    """把 webm 转成 16k 单声道 wav，供 emotion2vec 识别。"""
    fd, webm_path = tempfile.mkstemp(suffix=".webm")
    wav_path = webm_path[:-5] + ".wav"
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        subprocess.run(
            ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True,
            capture_output=True,
        )
        return wav_path
    finally:
        if os.path.exists(webm_path):
            os.remove(webm_path)


def _count_word(text: str, word: str) -> int:
    """统计情感词出现次数，含简单否定（不/没/别 紧邻则不计）。"""
    cnt = 0
    idx = text.find(word)
    while idx != -1:
        prev = text[idx - 1] if idx > 0 else ""
        if prev not in ("不", "没", "别"):
            cnt += 1
        idx = text.find(word, idx + len(word))
    return cnt


def _nli_distribution(text: str) -> dict[str, float]:
    model = _load_nli_model()
    out = model(text, EMOTIONS)
    return {out["labels"][i]: float(out["scores"][i]) for i in range(len(out["labels"]))}


def _text_distribution(text: str) -> dict[str, float]:
    hits = {e: sum(_count_word(text, w) for w in words) for e, words in _LEXICON.items()}
    nli = _nli_distribution(text)
    total = sum(hits.values())
    if total == 0:
        # 无语义词：交给 NLI
        return nli
    lex = {e: hits.get(e, 0) / total for e in EMOTIONS}
    # 词典主导（0.85）+ NLI 微调（0.15）
    return {e: 0.85 * lex.get(e, 0.0) + 0.15 * nli.get(e, 0.0) for e in EMOTIONS}


def _audio_distribution(audio_bytes: bytes | None) -> dict[str, float] | None:
    if not audio_bytes:
        return None
    try:
        model = _load_er_model()
        wav_path = _to_wav_16k(audio_bytes)
        try:
            out = model.generate(input=wav_path, granularity="utterance")
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        if not out:
            return None
        item = out[0]
        labels = item.get("labels")
        scores = item.get("scores")
        if not labels or not scores:
            return None
        # emotion2vec 返回完整标签集与逐标签概率，取 argmax（概率最大）才是识别结果
        top_idx = max(range(len(scores)), key=lambda i: float(scores[i]))
        top_score = float(scores[top_idx])
        top_label = str(labels[top_idx]).strip()
        eng = top_label.split("/")[-1].strip().lower()
        mapped = _ER_LABEL_MAP.get(eng, "中性")
        dist = {e: (1.0 - top_score) / (len(EMOTIONS) - 1) for e in EMOTIONS}
        dist[mapped] = top_score
        return dist
    except Exception as e:  # pragma: no cover
        logger.warning("Audio emotion failed, fallback to text-only: %s", e)
        return None


def predict(text: str, audio_bytes: bytes | None = None) -> tuple[str, float]:
    if not settings.use_real_emotion:
        # 占位实现：基于文本哈希确定性返回标签，保证同一文本结果稳定
        idx = sum(map(ord, text)) % len(EMOTIONS)
        return EMOTIONS[idx], 0.8

    text_dist = _text_distribution(text)
    audio_dist = _audio_distribution(audio_bytes)

    if audio_dist is None:
        fused = text_dist
    else:
        # 文本 0.55 + 语音 0.45 加权融合
        fused = {e: 0.55 * text_dist.get(e, 0.0) + 0.45 * audio_dist.get(e, 0.0) for e in EMOTIONS}

    label = max(fused, key=fused.get)
    return label, float(fused[label])


def infer_emotion_from_lexicon(text: str) -> str | None:
    """仅用中文情感词典把查询词映射到情绪标签（不做模型推理，开销极小）。

    用于检索时的情绪一致过滤：命中词典返回对应标签（如「开心」→喜悦、「焦虑」→恐惧），
    未命中返回 None。
    """
    for label, words in _LEXICON.items():
        if any(_count_word(text, w) > 0 for w in words):
            return label
    return None


def warmup() -> None:
    if settings.use_real_emotion:
        try:
            _load_nli_model()
            logger.info("Emotion NLI model loaded")
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to load emotion NLI model: %s", e)
        try:
            _load_er_model()
            logger.info("Emotion2Vec (audio) model loaded")
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to load emotion2vec, audio emotion disabled: %s", e)
    else:
        logger.info("Emotion analysis uses mock predictor")