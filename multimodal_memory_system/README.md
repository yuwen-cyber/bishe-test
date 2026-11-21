# 多模态记忆系统（Multimodal Memory System）

一个面向「对话录制 + 多模态检索」的毕业设计项目。系统将视频/音频存入 MinIO，语音转写（ASR）后做情绪分析与文本向量化，元数据写入 MySQL、向量写入 Milvus，并提供「MySQL 全文检索」「Milvus 向量检索」「SQL + 向量混合检索」三套方案的实时对比，以及集中展示实验结果的静态演示面板。

## 技术栈

| 层 | 技术 |
|---|---|
| 前端 | Vue 3 + Vite + TypeScript + Element Plus + ECharts，`MediaRecorder` 录制、Axios `FormData` 上传 |
| 后端 | FastAPI + SQLAlchemy + PyMySQL + MinIO SDK + Milvus SDK |
| ASR | FunASR `paraformer-zh`（中文，直接输出简体，支持真实模型 / 占位切换） |
| 情绪分析 | transformers `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` zero-shot 七分类（支持真实模型 / 占位切换） |
| 向量化 | `BAAI/bge-small-zh-v1.5`（支持真实模型 / 占位切换） |
| 多模态向量 | 音频 `facebook/wav2vec2-base-960h`（768 维）、图像 `openai/clip-vit-base-patch32`（512 维）、情绪固定查表（64 维） |
| 基础设施 | 本机 MySQL 8.0、MinIO、Milvus Standalone（v2.4.6）+ 独立 etcd、Redis（可选） |

## 目录结构

```
multimodal_memory_system/
├── docker-compose.yml          # MySQL / MinIO / Milvus 编排
├── backend/
│   ├── requirements.txt
│   ├── .env.example            # 环境变量示例
│   ├── app/
│   │   ├── main.py             # 应用入口（启动时一次性加载模型）
│   │   ├── config.py           # 配置
│   │   ├── database.py         # 数据库连接 + FULLTEXT(ngram) 索引初始化
│   │   ├── models.py           # conversations / segments
│   │   ├── schemas.py          # 响应模型
│   │   ├── storage.py          # MinIO 存储
│   │   ├── asr.py              # FunASR 中文 ASR
│   │   ├── emotion.py          # 情绪分析
│   │   ├── embeddings.py       # 文本向量化
│   │   ├── audio_encoder.py    # 音频向量化（Wav2Vec2，768 维）
│   │   ├── image_encoder.py    # 视频关键帧图像向量化（CLIP，512 维）
│   │   ├── emotion_embedding.py# 情绪标签向量化（固定查表，64 维）
│   │   ├── milvus_client.py    # Milvus 集合 / 索引 / ANN
│   │   ├── query_expander.py   # 查询扩展（none / lexicon / llm，消融实验用）
│   │   └── routers/
│   │       ├── upload.py       # POST /api/upload
│   │       ├── search.py       # 基准 / 高性能两个检索接口
│   │       └── hybrid.py       # 混合检索接口（SQL + 向量加权 RRF 融合）
│   └── experiments/            # 评测脚本（run_eval / seed / rebuild_milvus / export_demo_data 等）
└── frontend/
    ├── public/demo-data.json        # 演示面板静态数据（由 export_demo_data.py 生成）
    └── src/
        ├── views/Home.vue           # 对话录制页
        ├── views/SearchCompare.vue  # 检索对比页
        ├── views/DemoPanel.vue      # 演示面板（4 页签 / 8 图表）
        ├── components/ChartBox.vue  # ECharts 容器（统一 init / dispose / resize）
        ├── components/ResultCard.vue
        └── _archive/                # 已下线页面的归档（.bak 不参与编译）
```

## 环境要求

- Docker / Docker Compose（运行 etcd、MinIO、Milvus）
- **Python 3.10+（推荐 3.12）**，Node.js 18+
- 浏览器需在 `localhost`（或 HTTPS）下才能调用摄像头与麦克风

> 注意：如本机默认 `python` 是 Python 3.9，请改用 Python 3.10+，否则部分依赖（如 `greenlet`）无预编译 wheel 会报 `Microsoft Visual C++ 14.0 is required`。

## 快速开始

### 方式一：一键启动（推荐）

双击项目根目录的 `start.bat`（或运行 `powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1`），脚本会自动：

1. 检测并启动 Docker Desktop，拉起 `etcd / minio / milvus`
2. 新窗口启动后端（8000）
3. 新窗口启动前端（5173）
4. 自动打开浏览器

> 后端首次启动会加载 bge + 情绪 + FunASR 模型，约 20~40 秒，稍等片刻接口才就绪。

### 方式二：手动启动

#### 1. 确认本机 MySQL 运行

```powershell
Get-NetTCPConnection -LocalPort 3306 -State Listen
```

未运行则在「服务」（`services.msc`）中启动 MySQL（服务名一般为 `MySQL80`）。数据库 `multimodal` 已建好（root / `Root@123456`）。

#### 2. 启动中间件

先打开 Docker Desktop 至 `Engine running`，然后：

```powershell
docker compose up -d etcd minio milvus
```

> 不要 `docker compose up -d`（全量会再起一个 mysql 抢 3306，与本机 MySQL 冲突）。

#### 3. 启动后端

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

接口文档：http://localhost:8000/docs

#### 4. 启动前端

```powershell
cd frontend
npm run dev
```

> 若 `npm` 提示「不是内部或外部命令」：整关重开 VS Code，或先执行
> `$env:Path = "$env:LOCALAPPDATA\Programs\nodejs\node-v22.12.0-win-x64;$env:Path"`。

访问 http://localhost:5173

### 一键关闭

双击项目根目录 `stop.bat`：一键停掉前端、后端与中间件（Docker）；本机 MySQL 服务保持运行。

## 接口说明

### POST `/api/upload`

- 参数（`multipart/form-data`）：`video`、`audio` 文件，可选 `duration_ms`
- 流程：存入 MinIO → ASR → 情绪分析 → 写 MySQL → 文本/音频/图像/情绪四路向量写 Milvus（音频/图像/情绪向量只存不查）
- 成功返回：`{ conversation_id, segment_id, emotion, asr_text, message: "对话已保存" }`

### GET `/api/baseline/search`

- 参数：`q`（必填）、`emotions`（逗号分隔多选）、`start`、`end`
- 逻辑：MySQL `MATCH...AGAINST`（ngram 解析器）全文检索 + 情绪/时间过滤
- 单字等低信息查询（去标点后不足 2 字）无法被 ngram 切出 token，自动退化为 `LIKE '%q%'` 包含匹配，保证含该字的片段能正常返回

### GET `/api/advanced/search`

- 参数：`q`（必填）、`emotions`（逗号分隔多选）、`start`、`end`、`expand`（`none|lexicon|llm`，默认 `none`）
- 逻辑：先按 `expand` 扩展查询词 → 查询向量化 → Milvus 向量检索 + 标量过滤（情绪/时间）→ 先过滤掉 cosine < 0.5 的低相关命中，再让真实录制（`is_demo=false`）优先（凑不满 3 条则返回更少，不再硬凑）
- 低信息查询（如单个「了」）的向量近邻基本是随机的，接口直接返回空结果，避免无关片段冒充命中
- `expand` 参数见下文「查询扩展模块（消融实验）」

### GET `/api/hybrid/search`

- 参数：`q`（必填）、`emotions`、`start`、`end`
- 逻辑：SQL 全文召回（仅保留 `score > 0`）与向量语义召回（各取 top50）并行 → 加权 RRF 融合（向量路权重 2.0、SQL 路 1.0，K=60）→ 回查 `is_demo` 做真实录制优先 → 取 top3
- RRF 只依赖两路排名、不比较分数尺度，避免全文相关性分与余弦相似度不可直接相加
- 低信息查询只走 SQL 的 `LIKE` 匹配，向量路不参与融合（避免随机近邻被 `is_demo` 顶到首位）

三个检索接口均返回含 `retrieval_time_ms` 字段的结果列表，用于直观对比耗时。

### GET `/api/health`

健康检查，返回 `{ "status": "ok" }`。

## 演示面板（`/demo`）

用于答辩与论文取材：把六项实验的结果图表集中到一个页面，替代散落的截图。

- **纯前端静态页**：数据取自 `frontend/public/demo-data.json`，**不发任何 API 请求**，后端完全不启动也能正常打开
- 4 个页签：检索效果 / 消融实验 / 性能与规模 / 结构化查询与案例，共 8 张 ECharts 图表 + 2 张数据表
- 图表统一由 `src/components/ChartBox.vue` 渲染（负责 init / `setOption` / `dispose` / 窗口自适应），单页不重复写图表生命周期
- 原「用户实验」页已下线，源码归档在 `frontend/src/_archive/ExperimentLab.vue.bak`（`.bak` 不参与编译），顶栏菜单现为：对话录制 / 检索对比 / 演示面板

重新生成数据（先跑完对应评测脚本，再导出）：

```powershell
cd backend
.\.venv\Scripts\python.exe experiments\export_demo_data.py
```

> 导出脚本会把 `experiments/output/*.csv` 汇总写成 `frontend/public/demo-data.json`；刷新页面即可看到新数据。
> 首次拉取前端代码后需执行 `npm install`（已加入 `echarts` 依赖）。

## 查询扩展模块（消融实验）

`app/query_expander.py` 为 `/api/advanced/search` 提供查询扩展能力，用于消融实验对比三种策略：

| `expand` 值 | 含义 | 用在哪组实验 |
|---|---|---|
| `none` | 不扩展，直接用原查询 | 基线组 |
| `lexicon` | 情感词典规则扩展（命中压力/悲伤/喜悦类词后拼上同义词） | 消融对照组 1 |
| `llm` | 调 DeepSeek API 做语义扩展 | 消融对照组 2（创新点四） |

### 配置

在 `backend/.env` 里添加：

```properties
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

- 不填 `DEEPSEEK_API_KEY` 时，`expand=llm` 会自动降级为 `lexicon`，并打印告警日志。
- LLM 调用带 10 秒超时、内存缓存（相同查询只调一次 API），失败时降级返回原查询。
- key 有效但账户未充值时会返回 `402 Payment Required`；充 1 元即可跑完整篇论文的消融实验（一次全套约几分钱）。

### 验证

启动后端后访问（不同 `expand` 返回结果不同）：

```
http://localhost:8000/api/advanced/search?q=有点烦&expand=none
http://localhost:8000/api/advanced/search?q=有点烦&expand=lexicon
http://localhost:8000/api/advanced/search?q=有点烦&expand=llm
```

后端日志会打印每次扩展结果，例如：

```
query expand [mode=llm] '有点烦' -> '有点烦 烦躁, 郁闷, 压力大, 焦虑'
```

### 消融实验对应关系

| 实验组 | 调用方式 | 论文里写 |
|---|---|---|
| 无查询扩展 | `expand=none` | 基线 |
| 词典扩展 | `expand=lexicon` | 证明规则扩展有用 |
| LLM 扩展 | `expand=llm` | 证明大模型扩展更好（创新点四的证据） |

三组跑同一个查询集，用 `backend/experiments/run_eval.py` 自动算 Recall@K / MRR / NDCG，画成对比表。

## 数据模型

**conversations**：`id`、`title`、`video_object_key`、`audio_object_key`、`duration_ms`、`created_at`

**segments**：`id`、`conversation_id`、`start_time`、`end_time`、`asr_text`、`emotion`、`emotion_confidence`、`is_demo`、`created_at`

> `is_demo=true` 表示评测注入数据，检索排序时真实录制（`is_demo=false`）优先；该字段只影响排序，不影响是否参与检索。

`segments.asr_text` 上创建了 `FULLTEXT` 索引（`ngram` 解析器），由后端启动时幂等创建。

**Milvus 集合**（`embedding` 为检索主字段，其余向量只存不查）：

| 字段 | 类型 | 维度 | 说明 |
|---|---|---|---|
| `embedding` | FLOAT_VECTOR | 512 | 文本向量，检索字段（HNSW + COSINE） |
| `audio_embedding` | FLOAT_VECTOR | 768 | 音频向量（Wav2Vec2 mean-pooling），仅存储 |
| `image_embedding` | FLOAT_VECTOR | 512 | 视频中间帧图像向量（CLIP），仅存储 |
| `emotion_embedding` | FLOAT_VECTOR | 64 | 情绪标签向量（固定查表），仅存储 |
| `text` / `emotion` / `start_time` / `video_key` / `audio_key` 等 | 标量 | — | 过滤与回源元数据 |

四个向量字段均建 HNSW + COSINE 索引（M=16，efConstruction=200），检索参数 ef=64。

## 模型配置

各 AI 能力（含多模态向量）均在后端启动时加载一次（避免每次请求重复加载）。真实模型需联网下载，首次启动较慢。

| 能力 | 当前状态 | 开关（`.env`） | 依赖 |
|---|---|---|---|
| ASR 语音转写 | 真实（FunASR `paraformer-zh`，简体） | `USE_REAL_ASR=true` | `funasr` + `kaldi-native-fbank` + 本机 ffmpeg |
| 向量化 | 真实（`BAAI/bge-small-zh-v1.5`，512 维） | `USE_REAL_EMBEDDING=true` | `sentence-transformers` |
| 情绪分析 | 真实（zero-shot 七分类） | `USE_REAL_EMOTION=true` | `transformers` + `torch` |
| 音频向量化 | 占位（默认关闭，md5 确定性向量） | `USE_REAL_AUDIO=false` | `transformers`（Wav2Vec2）+ `soundfile` + ffmpeg |
| 图像向量化 | 占位（默认关闭，md5 确定性向量） | `USE_REAL_IMAGE=false` | `transformers`（CLIP）+ `torch` + ffmpeg |
| 情绪向量化 | 固定查表（`seed=42`，无需模型） | 无开关 | `numpy` |

- 关掉对应开关即回退到轻量占位实现，便于在无 GPU / 无模型环境下运行。
- 依赖已列在 `requirements.txt`；如需换更优模型，改 `.env` 配置并重启后端即可。
- 下载模型的网络可能较慢，必要时可配置 HF 镜像（`HF_ENDPOINT`，见 `.env`）。
- `USE_REAL_AUDIO` / `USE_REAL_IMAGE` 默认关闭，开启后才用真实模型（需联网下载）；情绪向量始终走固定查表，不依赖模型。

## 重建 Milvus 集合

Milvus 集合 schema 变更后（如新增向量字段），旧集合需删除重建：

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.milvus_client           # 1. 删除旧集合
.\.venv\Scripts\python.exe experiments\rebuild_milvus.py  # 2. 从 MySQL + MinIO 回填四路向量
```

> `rebuild_milvus.py` 不会新增或删除 MySQL / MinIO 数据，只重建 Milvus 里的向量。
> 重建前需确认 Milvus 容器已运行：`docker compose up -d etcd minio milvus`。