import logging
import os
import subprocess
import tempfile

from .config import settings

logger = logging.getLogger(__name__)
_model = None

_MOCK_TEXTS = [
    "今天天气很好，我们一起讨论了下周的出行计划。",
    "用户询问了多模态记忆系统的设计思路，我们讨论得很充分。",
    "会议纪要中提到需要优化检索结果的排序策略。",
    "我们聊到了向量检索与全文检索在中文场景下的对比。",
    "这段对话讨论了情绪分析模型在不同数据集上的准确率问题。",
]


def _load_model():
    global _model
    if _model is None:
        from funasr import AutoModel

        # FunASR Paraformer 中文模型：直接输出简体中文，附带 VAD 去静音 + 标点
        _model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            device="cpu",
            disable_update=True,
        )
    return _model


def _to_wav_16k(audio_bytes: bytes) -> str:
    """用 ffmpeg 把 webm 转成 16k 单声道 wav，供 FunASR 识别。"""
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


def transcribe(audio_bytes: bytes) -> str:
    if settings.use_real_asr:
        model = _load_model()
        wav_path = _to_wav_16k(audio_bytes)
        try:
            result = model.generate(input=wav_path, batch_size_s=300)
            text = result[0]["text"] if result else ""
            return (text or "").strip()
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
    # 占位实现：根据音频大小确定性挑选一句，便于演示检索效果
    return _MOCK_TEXTS[len(audio_bytes) % len(_MOCK_TEXTS)]


def warmup() -> None:
    if settings.use_real_asr:
        try:
            _load_model()
            logger.info("FunASR model loaded (paraformer-zh + fsmn-vad + ct-punc)")
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to load FunASR, fallback to mock: %s", e)
    else:
        logger.info("ASR uses mock transcriber")