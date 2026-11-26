"""验证 structured/search 情绪筛选修复：复数 emotions 现在应生效，且原单数写法仍可用。"""
import json
import requests

BASE = "http://127.0.0.1:8000/api"


def check(**params):
    r = requests.get(f"{BASE}/structured/search", params=params, timeout=60)
    items = r.json().get("results", []) if r.ok else []
    return {"status": r.status_code, "n": len(items),
            "emotions": sorted({i["emotion"] for i in items}),
            "all_match": all(i["emotion"] == "喜悦" for i in items) if items else None}


print("复数 emotions=喜悦 :", json.dumps(check(emotions="喜悦", limit=10), ensure_ascii=False))
print("单数 emotion=喜悦  :", json.dumps(check(emotion="喜悦", limit=10), ensure_ascii=False))
print("两者同时传        :", json.dumps(check(emotion="喜悦", emotions="喜悦", limit=10), ensure_ascii=False))
print("无筛选            :", json.dumps(check(limit=10), ensure_ascii=False))
