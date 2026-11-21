import importlib.util as u
import sys

pkgs = ["fastapi", "uvicorn", "torch", "transformers", "sentence_transformers",
        "funasr", "pymilvus", "minio", "pymysql", "numpy"]
print("解释器:", sys.executable)
for p in pkgs:
    print(f"  {p}: {'OK' if u.find_spec(p) else '缺'}")
