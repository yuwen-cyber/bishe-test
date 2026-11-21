from pathlib import Path

BASE = Path(__file__).resolve().parent
HYBRID = BASE / "app" / "routers" / "hybrid.py"

t = HYBRID.read_text(encoding="utf-8")
t = t.replace(
    "from .. import embeddings, emotion, milvus_client, query_expander, schemas, storage",
    "from .. import (data_source, embeddings, emotion, milvus_client,\n"
    "                query_expander, schemas, storage)",
)
HYBRID.write_text(t, encoding="utf-8")
print("hybrid.py 已补 import，data_source 次数:", t.count("data_source"))
