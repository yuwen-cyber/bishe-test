import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import asr, audio_encoder, embeddings, emotion, image_encoder, milvus_client
from .database import init_db

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时完成一次：建表/索引 + 模型加载 + Milvus 集合就绪
    init_db()
    embeddings.warmup()
    asr.warmup()
    emotion.warmup()
    audio_encoder.warmup()
    image_encoder.warmup()
    milvus_client.warmup()
    yield


app = FastAPI(title="Multimodal Memory System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from .routers import hybrid, search, structured, upload  # noqa: E402

app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(hybrid.router, prefix="/api", tags=["hybrid"])
app.include_router(structured.router, prefix="/api", tags=["structured"])


@app.get("/api/health")
def health():
    return {"status": "ok"}