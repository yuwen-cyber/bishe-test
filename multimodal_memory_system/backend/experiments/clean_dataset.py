"""
把大模型聊天框里复制出来的 JSON 片段清洗成合法的 segments_dataset.json
用法：python clean_dataset.py
"""
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent
RAW_FILE = BASE_DIR / "raw_segments.txt"
OUT_FILE = BASE_DIR / "segments_dataset.json"

def main():
    text = RAW_FILE.read_text(encoding="utf-8")

    # 1) 去掉可能的 markdown 代码块包裹
    text = re.sub(r"^```(json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())

    # 2) 提取所有 {...} JSON 对象
    #    用正则匹配每个独立的 {...} 块
    objects = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                chunk = text[start:i+1]
                try:
                    obj = json.loads(chunk)
                    objects.append(obj)
                except json.JSONDecodeError as e:
                    print(f"[WARN] 跳过一个无法解析的片段: {e}")
                start = None

    # 3) 按 segment_id 排序并去重（统一转成 int，防止大模型把 id 写成字符串）
    def _sid(obj):
        try:
            return int(obj.get("segment_id", 0))
        except (TypeError, ValueError):
            return 0

    objects.sort(key=_sid)
    seen_ids = set()
    unique = []
    for obj in objects:
        sid = _sid(obj)
        if sid in seen_ids:
            continue
        seen_ids.add(sid)
        obj["segment_id"] = sid  # 统一回写成 int
        unique.append(obj)

    # 4) 写出
    OUT_FILE.write_text(
        json.dumps(unique, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"✅ 清洗完成，共 {len(unique)} 条片段")
    print(f"   输出文件: {OUT_FILE}")
    print(f"   segment_id 范围: {unique[0]['segment_id']} ~ {unique[-1]['segment_id']}")

    # 5) 简单统计
    emotions = {}
    scenarios = {}
    for obj in unique:
        emotions[obj.get("emotion", "未知")] = emotions.get(obj.get("emotion", "未知"), 0) + 1
        scenarios[obj.get("scenario", "未知")] = scenarios.get(obj.get("scenario", "未知"), 0) + 1
    print(f"   情绪分布: {emotions}")
    print(f"   场景分布: {scenarios}")

if __name__ == "__main__":
    main()
