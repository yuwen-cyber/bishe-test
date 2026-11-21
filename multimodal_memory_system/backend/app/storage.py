from __future__ import annotations

import io
from datetime import timedelta

from minio import Minio

from .config import settings

_client: Minio | None = None


def get_client() -> Minio:
    global _client
    if _client is None:
        _client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        _ensure_bucket(_client)
    return _client


def _ensure_bucket(client: Minio) -> None:
    if not client.bucket_exists(settings.minio_bucket):
        client.make_bucket(settings.minio_bucket)


def upload_bytes(data: bytes, object_key: str, content_type: str = "application/octet-stream") -> str:
    client = get_client()
    client.put_object(
        settings.minio_bucket,
        object_key,
        io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )
    return object_key


def presigned_url(object_key: str, expires_minutes: int = 60) -> str:
    client = get_client()
    return client.presigned_get_object(
        settings.minio_bucket,
        object_key,
        expires=timedelta(minutes=expires_minutes),
    )