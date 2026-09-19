# Deep-RAG · 深度学术文档知识库

Deep-RAG 是一个以“可回到原始页码的证据”为核心的学术 PDF 研读系统。它将上传、异步解析、向量/关键词双路召回、重排序、严格引文生成与可恢复 SSE 交互连接为一条可部署的工作流。

> 开发环境默认启用 `local_hash` embedding 与 `local` LLM，用于无凭据的完整链路验证；它们不应作为生产语义模型。生产部署必须配置真实 Embedding 服务与所选 LLM 凭据。

## 能力概览

- PDF 经 Celery 异步处理：PyMuPDF 按页抽取、递归语义切片、页码元数据与 `halfvec` embedding 一同写入 PostgreSQL。
- pgvector HNSW 稠密召回与 `rank_bm25` 稀疏召回，通过手写 RRF 融合；可接入 BGE/Cross-Encoder HTTP 重排序服务并过滤低置信片段。
- Redis 语义缓存：查询 embedding 余弦相似度严格大于 `0.95` 时复用已验证答案，并在 SSE 中标记 `cache_hit`。
- 强制引用提示词与服务端引文守卫：回答仅保留具有已检索 `Ref + Page` 对应关系的句子。
- Typed SSE：`thought`、`citation`、`delta`、`error` 四类 JSON 帧写入 Redis，可使用 `Last-Event-ID` 续接断开的浏览器连接。
- Next.js 15 双栏证据工作台：上传与摄取进度、可收起检索过程、可点击页码引文、源片段与文档分布并列呈现。
- 论文发现台：关键词或 DOI 并行查询 OpenAlex 与 Crossref；开放版本可直接打开，学校连接器只跳转官方 Library Search / OpenURL，不接触用户 SSO 凭据。

## 架构

```text
                         ┌──────────────────────────────────────┐
                         │        Next.js 15 / React 19          │
                         │ upload · SSE reader · evidence map    │
                         └───────────────┬──────────────────────┘
                                         │ HTTP / Typed SSE
                         ┌───────────────▼──────────────────────┐
                         │          FastAPI application          │
                         │ documents · retrieval · chat stream   │
                         └───────┬─────────────┬───────────┬────┘
                                 │             │           │
                  upload enqueue│             │           │OpenAI-compatible
                                 ▼             ▼           ▼
                 ┌──────────────────┐  ┌────────────┐  ┌──────────────────┐
                 │ Celery + Redis   │  │ Redis 7.2+ │  │ OpenAI / DeepSeek │
                 │ parse / chunk /  │  │ broker +   │  │ / Claude adapter  │
                 │ embed pipeline   │  │ cache + SSE│  └──────────────────┘
                 └────────┬─────────┘  └────────────┘
                          │
                          ▼
        ┌──────────────────────────────────────────────────────┐
        │ PostgreSQL 16 + pgvector                              │
        │ documents · page chunks · halfvec · HNSW cosine index │
        └──────────────────────────────────────────────────────┘
```

## 快速启动

前提：Docker Desktop / Docker Engine 已运行。

```bash
git clone https://github.com/JaYZHOU96916/deep-rag-engine.git
cd deep-rag-engine
cp .env.example .env
# 为 POSTGRES_PASSWORD 设置一个非示例的强密码；按需填入模型凭据
./start.sh
```

启动完成后访问：

- 前端：`http://localhost:3000`
- FastAPI OpenAPI：`http://localhost:8000/docs`

`start.sh` 会构建五个服务、等待依赖健康、验证 pgvector 扩展和 API 健康状态。它拒绝使用 `.env.example` 中的示例数据库密码。开发环境不需要外部模型凭据；生产环境则必须按下表配置。

## 服务与端点

| 服务 | 职责 | 默认端口 |
| --- | --- | --- |
| `postgres` | PostgreSQL 16、pgvector、HNSW/halfvec 存储 | 5432 |
| `redis` | Celery broker/result、语义缓存、SSE 重放日志 | 6379 |
| `backend` | FastAPI API、Alembic 迁移、流式生成 | 8000 |
| `celery-worker` | PDF 解析、切片与 embedding 摄取任务 | — |
| `frontend` | Next.js 证据工作台 | 3000 |

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/api/v1/documents` | 上传 PDF，立即返回文档 ID 与 Celery task ID |
| `GET` | `/api/v1/documents/{document_id}` | 轮询 `pending → parsing → chunking → embedding → completed` |
| `POST` | `/api/v1/retrieval/search` | 调试混合检索与返回页码级引用 |
| `GET` | `/api/v1/papers/institutions` | 返回已配置的学校图书馆连接器 |
| `POST` | `/api/v1/papers/search` | 关键词或 DOI 论文发现；可附带学校 ID 生成官方访问链接 |
| `POST` | `/api/v1/papers/import` | 仅下载并摄取论文发现结果提供的开放 PDF；拒绝私网地址、非 HTTPS、超限或非 PDF 内容 |
| `POST` | `/api/v1/chat/stream` | 创建 Typed SSE 流，响应头返回 `X-Stream-ID` |
| `GET` | `/api/v1/chat/stream/{stream_id}` | 携带 `Last-Event-ID` 回放未接收帧 |

## 检索、缓存与引用约束

### Hybrid RRF

对问题（可额外拼接 HyDE 假设性文档）同时执行：

1. `halfvec` 上的 pgvector HNSW cosine 距离稠密召回；
2. `rank_bm25` 稀疏召回；
3. 按下式手写融合两个排名列表：

```text
RRF(d) = Σ 1 / (k + rank_i(d)),  k = 60
```

融合前列候选由 `LexicalReranker`（无凭据开发模式）或 `BGEHTTPReranker` 重排序。生产环境设置 `RERANKER_PROVIDER=bge_http` 和 `RERANKER_ENDPOINT`，该端点应提供标准 `POST /rerank`（`query`、`documents`、`results[index,relevance_score]`）接口。

### Semantic cache

每次问题都生成 query embedding，并与 Redis 中未过期的历史 embedding 逐一计算余弦相似度。分数 **严格高于** `SEMANTIC_CACHE_THRESHOLD`（默认 `0.95`）时，系统直接重放已审核答案和引文；否则执行 HyDE、检索、重排和回答审核。缓存默认 TTL 为 24 小时。

### 引用完整性

生成阶段包含三层防护：HyDE 仅用于检索扩展；回答提示词强制每个事实性句子以 `[Ref: ID, Page X]` 收尾；Self-RAG 审核后，服务端只允许与本次候选片段匹配的 ID/页码对进入最终答案。证据不足时返回：`检索到的参考资料中未包含关于 [具体问题] 的确切信息。`

## 论文发现与学校图书馆

论文发现不依赖学校配置：普通关键词通过 OpenAlex 与 Crossref 查询，直接粘贴 DOI 则进行精确解析。结果明确区分三种访问状态：

- `open_access`：显示来自元数据源的开放版本 URL；带有明确 PDF URL 的结果可一键进入现有 Celery 摄取流水线。
- `institution_login`：生成学校官方的 OpenURL 或 Library Search 地址；浏览器在学校页面中自行执行 SSO，Deep-RAG 不代理登录、不保存密码、Cookie 或访问令牌。
- `metadata_only`：仅显示书目信息和来源页，不能暗示全文可用。

默认提供 University of Melbourne 的 Library Search 跳转。其他大学由部署管理员以 JSON 一次性配置；普通用户只需在论文发现台选择学校。若某校提供 OpenURL，可优先设置 `openurl_base_url`；若没有，提供带 `{query}` 占位符的 HTTPS 检索 URL 即可。

```dotenv
INSTITUTION_CONNECTORS_JSON='[{"id":"example-u","name":"Example University","catalog_search_url_template":"https://library.example.edu/search?q={query}","openurl_base_url":"https://resolver.example.edu/openurl?"}]'
```

`INSTITUTION_CONNECTORS_JSON` 是服务端受信配置，系统会拒绝非 HTTPS URL、缺少 `{query}` 的检索模板或重复学校 ID。开放 PDF 导入还会拒绝私网/回环 DNS、重定向链过长、超出上传上限和非 PDF 内容。该阶段不抓取订阅库网页，也不导入受限 PDF；OAI-PMH 元数据收割将在学校提供经过验证的公开 endpoint 后作为独立连接器加入。

## Typed SSE 协议

所有帧均为 UTF-8 JSON，使用标准 SSE 的 `id`、`event`、`data` 字段：

```text
id: 3
event: citation
data: {"document_id":"…","page_number":12,"excerpt":"…","rrf_score":0.032}

id: 4
event: delta
data: {"text":"正文片段","cache_hit":false}
```

- `thought`：安全的检索进度与子查询摘要，不暴露模型内部思维链。
- `citation`：文档名、页码、原始片段与排名分数。
- `delta`：经过引文守卫后的正文流。
- `error`：结构化错误码、消息和可重试标识。

事件先写入 Redis，再交给浏览器；因此客户端断开后可使用最后收到的事件编号续接。

## 环境变量

| 变量 | 说明 |
| --- | --- |
| `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` | PostgreSQL 基础连接信息；生产环境使用密钥管理服务注入密码。 |
| `DATABASE_URL`, `REDIS_URL` | 容器内服务连接串；不要将容器内主机名替换为 `localhost`。 |
| `EMBEDDING_PROVIDER` | `local_hash`（仅开发）或 `openai_compatible`。 |
| `EMBEDDING_MODEL`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY` | 生产 embedding 服务配置；维度必须与数据库的 384 维 schema 匹配。 |
| `LLM_PROVIDER` | `local`、`openai`、`deepseek` 或 `claude`；前端也可在请求级覆盖。 |
| `LLM_MODEL`, `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `CLAUDE_API_KEY` | LLM 路由与凭据。OpenAI / DeepSeek 使用同一兼容客户端；Claude 由统一异步适配层接入。 |
| `RERANKER_PROVIDER`, `RERANKER_ENDPOINT` | 设置为 `bge_http` 并指向受控 BGE/Cross-Encoder 服务以启用生产重排。 |
| `SEMANTIC_CACHE_THRESHOLD`, `SEMANTIC_CACHE_TTL_SECONDS` | 缓存相似度阈值与存活时间。 |
| `CORS_ORIGINS` | JSON 数组格式的允许前端来源列表。 |
| `CROSSREF_MAILTO`, `OPENALEX_API_KEY` | 可选的公共论文元数据服务联系地址与 API Key；开发环境可留空。 |
| `INSTITUTION_CONNECTORS_JSON` | 学校名称、HTTPS 检索模板和可选 OpenURL Resolver 的服务端 JSON 配置。 |

## 验证

```bash
# 单元测试：递归切分、PDF 页码、RRF、语义缓存、提示词、SSE 重放、引文守卫
docker compose --profile application run --rm --no-deps \
  -v "$PWD/backend/tests:/app/tests:ro" backend pytest -q

# 真实服务验证：先确保 backend 和 celery-worker 正在运行
docker compose --profile application run --rm --no-deps \
  -e API_BASE_URL=http://backend:8000 backend python scripts/e2e_ingestion.py
docker compose --profile application run --rm --no-deps backend python scripts/e2e_retrieval.py
docker compose --profile application run --rm --no-deps \
  -e API_BASE_URL=http://backend:8000 backend python scripts/e2e_sse.py
docker compose --profile application run --rm --no-deps \
  -e API_BASE_URL=http://backend:8000 backend python scripts/e2e_paper_discovery.py

# 前端生产构建与类型检查
cd frontend && npm ci && npm run lint && npm run build
```

## 生产部署清单

1. 使用唯一的强 `POSTGRES_PASSWORD`，并让 CI/CD 从受控 secret store 注入所有 API key；`.env` 永不提交。
2. 将 `EMBEDDING_PROVIDER` 切换为受控的兼容 embedding 服务，并固定 `EMBEDDING_DIMENSION=384` 对应的模型版本；如改维度，先创建新迁移与新索引。
3. 部署 BGE/Cross-Encoder 重排端点，设置 `RERANKER_PROVIDER=bge_http`；开发 fallback 不用于生产质量评估。
4. 将 PostgreSQL、Redis 与上传卷置于持久化、备份和受访问控制的存储中；对外层配置 TLS 反向代理与请求大小限制。
5. 根据 PDF 吞吐量横向扩展 `celery-worker`，监控 failed 任务、Redis 内存、HNSW 查询延迟、SSE 客户端断连率与缓存命中率。

## 项目结构

```text
backend/
  app/                 FastAPI、Celery、检索、缓存、SSE、模型路由
  alembic/             PostgreSQL / pgvector schema migration
  tests/               单元测试
  scripts/             摄取、检索、SSE 端到端验证
frontend/
  app/                 Next.js App Router 证据工作台
infra/postgres/init/   pgvector / pgcrypto 初始扩展
docker-compose.yml     五服务部署编排
start.sh               一键构建、启动和健康检查
```
