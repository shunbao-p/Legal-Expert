# Legal-Expert

面向法律咨询场景的 GraphRAG / Hybrid RAG 项目原型。  
项目重点不是“做一个能聊天的网页”，而是围绕法律问答里最关键的几个问题展开工程化设计：

- 如何更完整地召回法规与条文
- 如何让答案尽量基于可验证证据
- 如何在复杂问题中兼顾图关系推理与文本语义检索
- 如何用评测和观测而不是主观感觉来持续优化系统

当前仓库提供了完整的后端 API、前端演示界面、知识入库脚本、批量评测脚本，以及可直接运行的 Milvus 容器依赖。

## 项目亮点

### 1. 面向法律问答的双引擎检索架构

- 基于 `Milvus` 进行向量语义检索
- 基于 `Neo4j` 进行图谱关系检索
- 通过查询路由在 `hybrid_traditional`、`graph_rag`、`combined` 三种策略间动态切换

### 2. 不是只做召回，而是做检索质量控制

- 混合召回：向量检索 + BM25 + 图谱补充检索
- `Cross-Encoder Rerank` 对候选结果进行重排
- 图检索低质量结果抑制
- 图检索空结果自动回退传统检索，减少“查不到就直接失败”的情况

### 3. 面向法律场景的回答可信度设计

- 根据证据强弱进行 `Evidence Gate`
- 在证据不足时做回答降级，避免强行输出确定性结论
- 对答案进行 `claim-level Refine`，提升回答可解释性与可信度

### 4. 可直接演示的完整系统闭环

- 后端：`FastAPI`
- 前端：`React + Vite + TypeScript`
- 支持会话创建、问答调用、证据展示
- 支持 PDF / Word / Excel / 图片等多类型文件解析后参与当前会话问答

### 5. 评测与观测驱动优化

- 仓库内置批量评测脚本 `scripts/eval/run_eval.py`
- 内置 10 / 50 题示例评测集切换脚本 `scripts/eval/switch_eval_dataset.py`
- 支持输出结构化评测结果，便于持续比较命中率、稳定性与时延
- 已接入 `LangSmith` 观测链路，便于定位路由、召回与生成阶段问题

## 系统流程

```text
用户问题
  -> 查询路由 / 意图解析
  -> Hybrid Retrieval / GraphRAG / Combined
  -> Rerank + 质量门控
  -> Evidence Gate + Refine
  -> 生成答案 + 返回证据文档
```

## 适用场景

本项目更适合以下场景：

- 法律咨询类 RAG / GraphRAG 工程验证
- 面向法规与条文可追溯的问答原型
- 检索策略、重排、回答可信度控制相关实验
- AI 应用开发 / RAG 工程方向的项目展示

本项目不是正式法律服务系统，也不构成法律意见。

## 项目结构

```text
.
├── api/                    # FastAPI 接口层
├── rag_modules/            # 检索、路由、图谱、生成核心模块
├── frontend/               # React 演示前端
├── scripts/ingest/         # 法律知识入库脚本
├── scripts/eval/           # 评测脚本
├── data/raw/               # 原始法律文本数据目录
├── data/eval/              # 示例评测数据与结果
├── config.py               # 系统配置
├── main.py                 # 系统装配与交互入口
└── docker-compose.yml      # Milvus 相关容器编排
```

## 技术栈

- Backend: Python, FastAPI
- Frontend: React, TypeScript, Vite
- Retrieval: Milvus, Neo4j, BM25
- Rerank: `BAAI/bge-reranker-v2-m3`
- Embedding: `BAAI/bge-small-zh-v1.5`
- Observability: LangSmith

## 快速开始

### 1. 环境准备

- Python 3.10+
- Node.js 18+
- Docker / Docker Compose
- 本地 Neo4j 实例

默认连接信息见 `.env.example`：

- Neo4j: `bolt://localhost:7687`
- Milvus: `localhost:19530`

### 2. 安装依赖

```bash
pip install -r requirements.txt

cd frontend
npm install
cd ..
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

至少需要确认：

- `MOONSHOT_API_KEY` / `DEEPSEEK_API_KEY`
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `MILVUS_HOST`
- `MILVUS_PORT`

如需启用链路观测，还需要配置：

- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`

### 4. 启动 Milvus 依赖

```bash
docker compose up -d
```

说明：

- 当前 `docker-compose.yml` 主要负责启动 Milvus 相关依赖
- 后端、前端和 Neo4j 需要本地单独启动

### 5. 准备法律知识库数据

项目默认从 `data/raw/Laws` 读取公开法律法规 Markdown 文本，并将其写入 Neo4j 和 Milvus。

若你还没有原始数据，请先准备：

```text
data/raw/Laws/
```

然后执行入库脚本：

```bash
python scripts/ingest/p2_ingest_from_lawrefbook.py \
  --source-root data/raw/Laws \
  --target-laws 120 \
  --target-articles 15000 \
  --target-chunks 50000
```

该脚本会完成：

- 法律 Markdown 文本解析
- Neo4j 图谱写入
- Milvus 向量写入
- 数据来源清单生成到 `data/manifests/legal_sources_manifest.csv`

### 6. 启动后端 API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 7. 启动前端

```bash
cd frontend
npm run dev
```

访问：

- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:8000`

## 评测与回归

### 1. 切换示例评测集

```bash
python scripts/eval/switch_eval_dataset.py status
python scripts/eval/switch_eval_dataset.py 10
python scripts/eval/switch_eval_dataset.py 50
```

说明：

- 仓库当前内置的是便于回归验证的示例评测集
- 评测输出会写入 `data/eval/results/`

### 2. 执行批量评测

```bash
python scripts/eval/run_eval.py --base-url http://127.0.0.1:8000
```

评测结果主要关注：

- 法规命中情况
- 条文命中情况
- 关键词覆盖情况
- 请求失败率
- 时延分布
- 图检索回退情况

## 主要 API

- `GET /health`：查看服务健康状态与 reranker 预热状态
- `POST /chats`：创建问答会话
- `POST /files/upload`：上传会话文件
- `GET /chats/{chat_id}/files`：查询会话文件列表
- `DELETE /chats/{chat_id}/files/{file_id}`：删除会话文件
- `POST /chat`：执行法律问答

## 为什么这个项目值得看

如果你正在看一个法律 RAG 项目，我认为这个仓库最值得看的不是“接了哪个模型”，而是以下几点：

- 它不是只做语义检索，而是把图谱检索和文本检索结合起来
- 它不是只做生成，而是做了回答可信度控制
- 它不是只会演示，而是有批量评测和链路观测能力
- 它不是单点模块堆叠，而是考虑了回退、降级和可用性

## 当前限制

- 当前 `docker-compose.yml` 只覆盖 Milvus 相关依赖，未把整个系统一体化容器编排
- 图片输入本质上通过 OCR 转文本后进入统一 RAG 链路，不属于真正的视觉推理系统
- 音频输入目前仍是占位能力，尚未接入完整 ASR 流程
- 本项目用于技术研究与工程验证，不构成正式法律意见

## 安全与使用说明

- 请不要将 `.env`、私有 API Key 或敏感测试数据提交到仓库
- 若你要对外演示，请先核查模型 Key、数据库连接和本地数据目录是否配置正确
