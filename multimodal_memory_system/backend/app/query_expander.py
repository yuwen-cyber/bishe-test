"""
查询扩展模块（用于消融实验）
三种模式：
  - "none"    ：不扩展，直接用原查询
  - "lexicon" ：基于中文情感词典的规则扩展
  - "llm"     ：调用 DeepSeek API 做语义扩展（失败自动降级为 lexicon）

接入方式：把本文件复制到 backend/app/query_expander.py
"""
from __future__ import annotations

import logging
import re

import requests

from .config import settings

logger = logging.getLogger(__name__)

# 去掉空白与标点，只保留中文/字母/数字
_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def is_low_information(q: str) -> bool:
    """查询信息量是否过低（去标点后不足 2 个字符，如「了」「的」）。

    MySQL ngram 分词的最小单元是 2 字，单字查询不会命中任何 token；
    bge 对这类查询也编码不出有效语义，向量近邻基本是随机的。
    因此这类查询不应走向量路，SQL 路也要退化为 LIKE 包含匹配。
    """
    core = _PUNCT_RE.sub("", q or "")
    return len(core) < 2

# 内存缓存：避免重复调 API 或重复跑规则
_expand_cache: dict[str, str] = {}


# ============ 情感词典（lexicon 模式用） ============

# 每类情感的扩展词表
EMOTION_LEXICON: dict[str, list[str]] = {
    "压力": [
        "工作压力", "焦虑", "疲惫", "低落", "烦躁", "郁闷",
        "压力大", "扛不住", "太累了", "喘不过气", "身心俱疲",
    ],
    "悲伤": [
        "难过", "伤心", "失落", "沮丧", "委屈", "想哭",
        "不开心", "低沉", "心痛", "伤感", "绝望",
    ],
    "喜悦": [
        "开心", "高兴", "兴奋", "满足", "愉快", "幸福",
        "激动", "舒心", "快乐", "欣慰",
    ],
    "愤怒": [
        "生气", "恼火", "气愤", "烦躁", "怒了", "火大",
        "不爽", "气愤",
    ],
    "恐惧": [
        "害怕", "紧张", "担心", "不安", "心慌", "恐惧",
        "焦虑", "惶恐",
    ],
    "厌恶": [
        "讨厌", "反感", "恶心", "嫌弃", "不舒服",
    ],
}

# 触发词表：查询里命中这些词就扩展对应情感类别
_TRIGGER_TO_CATEGORY: list[tuple[tuple[str, ...], str]] = [
    (("压力", "累", "烦", "忙", "加班", "焦虑", "扛不住", "绷不住", "窒息"), "压力"),
    (("难过", "伤心", "失落", "沮丧", "委屈", "想哭", "不开心", "低落", "抑郁", "心情不好", "郁闷"), "悲伤"),
    (("开心", "高兴", "兴奋", "满足", "愉快", "幸福", "爽", "快乐"), "喜悦"),
    (("生气", "恼火", "气愤", "怒", "不爽", "火大"), "愤怒"),
    (("害怕", "紧张", "担心", "不安", "心慌", "恐惧"), "恐惧"),
    (("讨厌", "反感", "恶心", "嫌弃"), "厌恶"),
]


def _lexicon_expand(q: str) -> str:
    """基于情感词典的规则扩展：命中哪类情感，就把该类扩展词拼到原查询后面。"""
    extra_terms: list[str] = []
    for triggers, category in _TRIGGER_TO_CATEGORY:
        if any(t in q for t in triggers):
            extra_terms.extend(EMOTION_LEXICON[category])

    if not extra_terms:
        return q

    # 去重，且不要把原查询里已有的词再拼一遍
    seen: set[str] = set()
    unique_extra: list[str] = []
    for term in extra_terms:
        if term not in seen and term not in q:
            seen.add(term)
            unique_extra.append(term)

    if not unique_extra:
        return q
    return q + " " + " ".join(unique_extra)


# ============ LLM 扩展（DeepSeek API） ============

_LLM_SYSTEM_PROMPT = (
    "你是检索查询改写助手。用户原始查询通常很短、很口语化。"
    "请从情感和语义角度，把用户查询扩展为 3~5 个相关表达或同义词，"
    "目标是帮助向量检索召回字面不同但语义相近的对话片段。"
    "只返回扩展词，用中文逗号分隔，不要解释，不要复述原查询。"
)


def _llm_expand(q: str) -> str:
    """调用 DeepSeek API 扩展查询。失败时自动降级为词典扩展。"""
    if not settings.deepseek_api_key:
        logger.warning("DeepSeek API key 未配置，LLM 扩展降级为词典扩展")
        return _lexicon_expand(q)

    try:
        resp = requests.post(
            f"{settings.deepseek_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.deepseek_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": f"原始查询：{q}"},
                ],
                "temperature": 0.3,
                "max_tokens": 100,
            },
            timeout=10,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        # 把"压力，焦虑，疲惫"拆成 ["压力","焦虑","疲惫"]
        extra_words = [
            w.strip()
            for w in content.replace("，", ",").split(",")
            if w.strip()
        ]
        # 过滤掉和原查询重复的词
        extra_words = [w for w in extra_words if w not in q]

        if not extra_words:
            return q

        result = q + " " + " ".join(extra_words)
        logger.info("[LLM-EXPAND] %s  ==>  %s", q, result)
        return result

    except Exception as e:
        logger.warning("LLM 扩展失败 (%s)，降级为词典扩展", e)
        return _lexicon_expand(q)


# ============ 对外主接口 ============

def expand_query(q: str, mode: str = "none") -> str:
    """
    扩展查询。

    参数：
        q:    原始查询
        mode: "none" | "lexicon" | "llm"

    返回：
        扩展后的查询字符串（用于后续 embed_text）
    """
    if not q or not q.strip():
        return q

    if mode == "none":
        return q

    cache_key = f"{mode}::{q}"
    if cache_key in _expand_cache:
        return _expand_cache[cache_key]

    if mode == "lexicon":
        result = _lexicon_expand(q)
    elif mode == "llm":
        result = _llm_expand(q)
    else:
        logger.warning("未知扩展模式 %r，按 none 处理", mode)
        result = q

    _expand_cache[cache_key] = result
    return result


def expand_query_batch(queries: list[str], mode: str = "none") -> list[str]:
    """批量扩展。"""
    return [expand_query(q, mode) for q in queries]


def clear_cache() -> None:
    """清空扩展缓存（测试或换数据集时调用）。"""
    _expand_cache.clear()