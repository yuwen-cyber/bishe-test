"""
把话术2生成的查询真值集清洗成 query_set.json
用法：python clean_queries.py
"""
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent
RAW_FILE = BASE_DIR / "raw_queries.txt"
OUT_FILE = BASE_DIR / "query_set.json"  # 直接覆盖 experiments/ 下已有的 query_set.json

def main():
    text = RAW_FILE.read_text(encoding="utf-8")
    text = re.sub(r"^```(json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())

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
                    print(f"[WARN] 跳过: {e}")
                start = None

    def _qid(obj):
        try:
            return int(obj.get("query_id", 0))
        except (TypeError, ValueError):
            return 0

    objects.sort(key=_qid)
    seen = set()
    unique = []
    for obj in objects:
        qid = _qid(obj)
        if qid in seen:
            continue
        seen.add(qid)
        obj["query_id"] = qid
        unique.append(obj)

    OUT_FILE.write_text(
        json.dumps(unique, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"✅ 清洗完成，共 {len(unique)} 条查询")
    print(f"   输出文件: {OUT_FILE}")
    types = {}
    for q in unique:
        t = q.get("type", "未知")
        types[t] = types.get(t, 0) + 1
    print(f"   类型分布: {types}")

if __name__ == "__main__":
    main()
