启动服务
cd /Users/kele/Desktop/ollama/embeddedModel
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000


主要接口
健康检查: GET /health
构建索引: POST /kb/index
追加文档: POST /kb/append
追加网页: POST /kb/ingest-url
相似度查询: POST /kb/query
RAG 对话: POST /chat/rag
示例见 README.md 的 REST API 部分；所有接口支持传 ollama_host 或用环境变量 OLLAMA_HOST。

### 接口说明与用法（简版）

- **POST /kb/index（构建索引）**：扫描 `input_path`（文件或目录），切块并重建索引。
  - 必填：`index_dir`、`input_path`
```bash
curl -X POST http://127.0.0.1:8000/kb/index \
  -H 'Content-Type: application/json' \
  -d '{
    "index_dir": "/Users/kele/Desktop/ollama/embeddedModel/kb",
    "input_path": "/Users/kele/Desktop/ollama/embeddedModel/sample_docs",
    "ollama_host": "http://127.0.0.1:11434"
  }'
```

- **POST /kb/append（追加文档）**：在现有索引上追加新文档。
  - 必填：`index_dir`、`input_path`
```bash
curl -X POST http://127.0.0.1:8000/kb/append \
  -H 'Content-Type: application/json' \
  -d '{
    "index_dir": "/Users/kele/Desktop/ollama/embeddedModel/kb",
    "input_path": "/path/to/new_docs"
  }'
```

- **POST /kb/ingest-url（追加网页）**：抓取网页正文并追加到索引。
  - 必填：`index_dir`、`urls`（数组）
```bash
curl -X POST http://127.0.0.1:8000/kb/ingest-url \
  -H 'Content-Type: application/json' \
  -d '{
    "index_dir": "/Users/kele/Desktop/ollama/embeddedModel/kb",
    "urls": ["https://example.com"]
  }'
```

- **POST /kb/query（相似度查询）**：检索最相关片段，调试命中效果。
  - 必填：`index_dir`、`question`；可选：`top_k`
```bash
curl -X POST http://127.0.0.1:8000/kb/query \
  -H 'Content-Type: application/json' \
  -d '{
    "index_dir": "/Users/kele/Desktop/ollama/embeddedModel/kb",
    "question": "这个项目是做什么的？",
    "top_k": 5
  }'
```

- **POST /chat/rag（RAG 对话）**：先检索片段，再让聊天模型基于参考作答（更口语化、非逐字复述）。
  - 必填：`question`；可选：`index_dir`、`embed_model`、`chat_model`、`top_k`
```bash
curl -X POST http://127.0.0.1:8000/chat/rag \
  -H 'Content-Type: application/json' \
  -d '{
    "embed_model": "nomic-embed-text:latest",
    "chat_model": "llama3.2:latest",
    "question": "根据知识库，如何搭建这套系统？",
    "top_k": 5
  }'
```

- **POST /kb/clear（清空知识库）**：删除 `index.faiss`、`meta.jsonl`、`kb.json` 并重置内存状态。
  - 必填：`index_dir`
```bash
curl -X POST http://127.0.0.1:8000/kb/clear \
  -H 'Content-Type: application/json' \
  -d '{
    "index_dir": "/Users/kele/Desktop/ollama/embeddedModel/kb"
  }'
```

---

## REST API 参考（详细）

### 健康检查
- 方法与路径：`GET /health`
- 请求参数：无
- 响应：
```json
{
  "status": "ok"
}
```

### 构建索引
- 方法与路径：`POST /kb/index`
- 作用：从本地文件或目录读取文本，切块、嵌入并重建知识库索引（会覆盖旧索引）。
- 请求体字段：
  - `index_dir` (string, required): 知识库目录，生成 `index.faiss`、`meta.jsonl`、`kb.json`。
  - `input_path` (string, required): 文件或目录路径。
  - `model` (string, optional, default: `nomic-embed-text:latest`): 嵌入模型。
  - `ollama_host` (string, optional): Ollama 服务地址，如 `http://127.0.0.1:11434`。
  - `include` (string[], optional, default: `["**/*.txt", "**/*.md", "**/*.pdf", "**/*.docx", "**/*.html", "**/*.htm"]`): 当 `input_path` 为目录时的文件匹配模式。
  - `chunk_size` (int, optional, default: 800): 切块大小（字符数）。
  - `chunk_overlap` (int, optional, default: 200): 切块重叠（字符数）。
- 成功响应：
```json
{
  "chunks": 1234,
  "rows": 1234,
  "index_dir": "/absolute/path/to/kb"
}
```

### 追加文档
- 方法与路径：`POST /kb/append`
- 作用：在已有索引上追加新的文件内容（不覆盖已有数据）。
- 请求体字段：同 `/kb/index`，但不会重建，仅追加。
- 成功响应：
```json
{
  "chunks": 100,
  "rows": 100,
  "index_dir": "/absolute/path/to/kb"
}
```

### 追加网页
- 方法与路径：`POST /kb/ingest-url`
- 作用：抓取网页正文、切块并追加到现有索引。
- 请求体字段：
  - `index_dir` (string, required)
  - `urls` (string[], required): 目标 URL 列表。
  - 其他可选字段同上：`model`、`ollama_host`、`chunk_size`、`chunk_overlap`。
- 成功响应：
```json
{
  "chunks": 50,
  "rows": 50,
  "index_dir": "/absolute/path/to/kb"
}
```

### 相似度查询
- 方法与路径：`POST /kb/query`
- 作用：用问题做向量检索，返回最相关文档片段（用于调试与观测命中效果）。
- 请求体字段：
  - `index_dir` (string, required)
  - `question` (string, required)
  - `model` (string, optional, default: `nomic-embed-text:latest`)
  - `ollama_host` (string, optional)
  - `top_k` (int, optional, default: 5)
- 成功响应：
```json
{
  "results": [
    {
      "rank": 1,
      "score": 0.9876,
      "path": "/abs/path/doc.md",
      "chunk_index": 0,
      "text": "片段内容..."
    }
  ]
}
```

### RAG 对话（参考式回答）
- 方法与路径：`POST /chat/rag`
- 作用：先检索相关片段（若传 `index_dir` 且 KB 非空），将片段作为参考，以更口语化的风格作答；当未传 `index_dir` 或 KB 为空时自动走非 RAG 对话。
- 请求体字段：
  - `question` (string, required)
  - `index_dir` (string, optional): 指定则尝试使用 KB 检索；为空或 KB 为空则跳过检索。
  - `embed_model` (string, optional, default: `nomic-embed-text:latest`)
  - `chat_model` (string, optional, default: `llama3.2:latest`)
  - `ollama_host` (string, optional)
  - `top_k` (int, optional, default: 5)
- 成功响应：
```json
{
  "answer": "一句话结论…\n- 要点1\n- 要点2"
}
```

### 清空知识库
- 方法与路径：`POST /kb/clear`
- 作用：删除 `index.faiss`、`meta.jsonl`、`kb.json` 并重置内存状态（不会删除 `index_dir` 目录本身）。
- 请求体字段：
  - `index_dir` (string, required)
  - `ollama_host` (string, optional)
- 成功响应：
```json
{
  "cleared": true,
  "index_dir": "/absolute/path/to/kb"
}
```