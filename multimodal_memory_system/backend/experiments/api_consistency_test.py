"""核对四个检索接口的返回字段一致性与结构化接口的参数名。"""
import json
import pathlib
import requests

BASE = "http://127.0.0.1:8000/api"
r = {}


def get(ep, **params):
    resp = requests.get(f"{BASE}/{ep}", params=params, timeout=60)
    body = resp.json() if resp.ok else resp.text
    return resp.status_code, body


# 1) structured 用单数 emotion
st, b = get("structured/search", emotion="喜悦", limit=5)
r["structured_emotion_singular"] = {
    "status": st, "n": len(b.get("results", [])),
    "emotions": sorted({i["emotion"] for i in b.get("results", [])}),
}

# 2) structured 用复数 emotions（应被忽略）
st, b = get("structured/search", emotions="喜悦", limit=5)
r["structured_emotions_plural"] = {
    "status": st, "n": len(b.get("results", [])),
    "emotions": sorted({i["emotion"] for i in b.get("results", [])}),
}

# 3) 各接口返回字段
for ep in ("baseline/search", "advanced/search", "hybrid/search"):
    st, b = get(ep, q="压力很大")
    item = (b.get("results") or [{}])[0]
    r[f"fields::{ep}"] = {"status": st, "keys": sorted(item.keys()),
                          "has_data_source": "data_source" in item,
                          "data_source": item.get("data_source")}
st, b = get("structured/search", emotion="中性", limit=3)
item = (b.get("results") or [{}])[0]
r["fields::structured/search"] = {"status": st, "keys": sorted(item.keys()),
                                   "has_data_source": "data_source" in item,
                                   "data_source": item.get("data_source")}

print(json.dumps(r, ensure_ascii=False, indent=1))
pathlib.Path("experiments/output/api_consistency_result.json").write_text(
    json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
