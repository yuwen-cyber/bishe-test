# 查询扩展模块接入说明（消融实验用）

## 这个模块干嘛的

给你的 `/api/advanced/search` 接口加一个 `expand` 参数，支持三种模式：



| expand 值  | 含义                                  | 用在哪组实验        |
| --------- | ----------------------------------- | ------------- |
| `none`    | 不扩展，直接用原查询                          | 基线组           |
| `lexicon` | 情感词典规则扩展（"有点烦"→自动拼上 "焦虑 / 压力 / 低落"） | 消融对照组 1       |
| `llm`     | 调 DeepSeek API 做语义扩展                | 消融对照组 2（创新点四） |

论文里跑三组对比，证明 "LLM 查询扩展能提升 Recall@K"。



***

## 三步接入

### 第 1 步：把 query\_expander.py 复制到后端



```
把本目录下的 query\_expander.py

复制到 D:\biyesheji\multimodal\_memory\_system\backend\app\query\_expander.py
```

### 第 2 步：修改 app/config.py

在 `Settings` 类里**加三行**（放在 `hf_endpoint` 那一行下面就行）：



```
&#x20;   # DeepSeek API（LLM 查询扩展用，做消融实验时填）

&#x20;   deepseek\_api\_key: str = ""

&#x20;   deepseek\_base\_url: str = "https://api.deepseek.com/v1"
```

> 不填 key 也能跑：
>
> `expand=llm`
>
>  会自动降级成 
>
> `lexicon`
>
>  模式，不会报错。

### 第 3 步：修改 app/routers/search.py

**3.1 顶部 import 加一行**：

找到这一行：



```
from .. import embeddings, milvus\_client, schemas, storage
```

改成：



```
from .. import embeddings, milvus\_client, query\_expander, schemas, storage
```

**3.2 在&#x20;**`advanced_search`**&#x20;函数里加一个参数**：

找到这一段：



```
@router.get("/advanced/search", response\_model=schemas.SearchResponse)

def advanced\_search(

&#x20;   q: str = Query(..., min\_length=1),

&#x20;   emotions: Optional\[str] = Query(None),

&#x20;   start: Optional\[datetime] = Query(None),

&#x20;   end: Optional\[datetime] = Query(None),

):
```

改成（就加一个 `expand` 参数）：



```
@router.get("/advanced/search", response\_model=schemas.SearchResponse)

def advanced\_search(

&#x20;   q: str = Query(..., min\_length=1),

&#x20;   emotions: Optional\[str] = Query(None),

&#x20;   start: Optional\[datetime] = Query(None),

&#x20;   end: Optional\[datetime] = Query(None),

&#x20;   expand: str = Query("none", pattern="^(none|lexicon|llm)\$"),

):
```

**3.3 在&#x20;**`vec = embeddings.embed_text(q)`**&#x20;这一行前面加两行**：

找到：



```
&#x20;   vec = embeddings.embed\_text(q)
```

改成：



```
&#x20;   q = query\_expander.expand\_query(q, mode=expand)

&#x20;   vec = embeddings.embed\_text(q)
```



***

## 怎么验证接入成功

启动后端后，浏览器或 Postman 访问：



```
http://localhost:8000/api/advanced/search?q=有点烦\&expand=none

http://localhost:8000/api/advanced/search?q=有点烦\&expand=lexicon

http://localhost:8000/api/advanced/search?q=有点烦\&expand=llm
```

三次返回的结果应该不一样（扩展后的查询召回的片段更多 / 更准）。后端日志里会打印：



```
\[LLM-EXPAND] 有点烦  ==>  有点烦 烦躁, 郁闷, 压力大, 焦虑
```

看到这行日志就说明 LLM 扩展真的生效了。



***

## 怎么填 DeepSeek API Key



1. 去 [https://platform.deepseek.com/](https://platform.deepseek.com/) 注册

2. 充 1 块钱够你跑完整篇论文的实验（30\~40 条查询，一次几分钱）

3. 创建 API Key，复制下来

4. 在 `backend/.env` 文件里加一行（没有 .env 就新建）：



```
DEEPSEEK\_API\_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

重启后端就生效。



***

## 消融实验怎么跑（对应论文章节）



| 实验组    | 调用方式             | 论文里写               |
| ------ | ---------------- | ------------------ |
| 无查询扩展  | `expand=none`    | 基线                 |
| 词典扩展   | `expand=lexicon` | 证明规则扩展有用           |
| LLM 扩展 | `expand=llm`     | 证明大模型扩展更好（创新点四的证据） |

三组跑同一个查询集，用 run\_eval.py 自动算 Recall@K / MRR / NDCG，画成对比表。