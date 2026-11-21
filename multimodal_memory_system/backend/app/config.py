import os

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # MySQL
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "root"
    mysql_db: str = "multimodal"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "multimodal"
    minio_secure: bool = False

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "segments"

    # 模型开关（默认关闭真实模型，使用占位实现保证可直接运行）
    use_real_asr: bool = False
    use_real_emotion: bool = False
    use_real_embedding: bool = False
    use_real_audio: bool = False
    use_real_image: bool = False
    embedding_dim: int = 512

    # HuggingFace 模型下载镜像（国内加速），通过环境变量传给 huggingface_hub
    hf_endpoint: str = "https://hf-mirror.com"

    # DeepSeek LLM 查询扩展（消融实验用）
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    @property
    def database_url(self) -> URL:
        # 使用 URL.create 以便正确处理密码中的特殊字符（如 @ : / 等）
        return URL.create(
            drivername="mysql+pymysql",
            username=self.mysql_user,
            password=self.mysql_password,
            host=self.mysql_host,
            port=self.mysql_port,
            database=self.mysql_db,
            query={"charset": "utf8mb4"},
        )


settings = Settings()
os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")