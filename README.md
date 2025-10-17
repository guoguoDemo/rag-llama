## Ollama KB Toolkit (nomic-embed-text)

本工具使用 Ollama 的 `nomic-embed-text:latest` 将文档嵌入为向量，使用 FAISS 构建本地知识库，支持查询、RAG 对话，并提供 REST API。

### REST API 服务
- 启动服务：
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
- 健康检查：
```bash
curl http://127.0.0.1:8000/health
```

- 索引（KB 构建）：`POST /kb/index`
```bash
curl -X POST http://127.0.0.1:8000/kb/index \
  -H 'Content-Type: application/json' \
  -d '{
    "index_dir": "/Users/kele/Desktop/ollama/embeddedModel/kb",
    "input_path": "/Users/kele/Desktop/ollama/embeddedModel/sample_docs",
    "model": "nomic-embed-text:latest",
    "ollama_host": "http://127.0.0.1:11434",
    "chunk_size": 800,
    "chunk_overlap": 200
  }'
```

- 追加文档：`POST /kb/append`
```bash
curl -X POST http://127.0.0.1:8000/kb/append \
  -H 'Content-Type: application/json' \
  -d '{
    "index_dir": "/Users/kele/Desktop/ollama/embeddedModel/kb",
    "input_path": "/path/to/new_docs",
    "ollama_host": "http://127.0.0.1:11434"
  }'
```

- 追加网页：`POST /kb/ingest-url`
```bash
curl -X POST http://127.0.0.1:8000/kb/ingest-url \
  -H 'Content-Type: application/json' \
  -d '{
    "index_dir": "/Users/kele/Desktop/ollama/embeddedModel/kb",
    "urls": ["https://example.com"],
    "ollama_host": "http://127.0.0.1:11434"
  }'
```

- 查询：`POST /kb/query`
```bash
curl -X POST http://127.0.0.1:8000/kb/query \
  -H 'Content-Type: application/json' \
  -d '{
    "index_dir": "/Users/kele/Desktop/ollama/embeddedModel/kb",
    "question": "这个项目是做什么的？",
    "top_k": 5,
    "ollama_host": "http://127.0.0.1:11434"
  }'
```

- RAG 对话：`POST /chat/rag`
```bash
curl -X POST http://127.0.0.1:8000/chat/rag \
  -H 'Content-Type: application/json' \
  -d '{
    "index_dir": "/Users/kele/Desktop/ollama/embeddedModel/kb",
    "embed_model": "nomic-embed-text:latest",
    "chat_model": "llama3.2:latest",
    "question": "根据知识库，如何搭建这套系统？",
    "top_k": 5,
    "ollama_host": "http://127.0.0.1:11434"
  }'
```

说明：所有接口支持 `ollama_host` 字段来指定 Ollama 服务地址，或通过环境变量 `OLLAMA_HOST` 统一配置。

---

其余本地 CLI 的使用（索引/追加/查询、RAG 对话）与前文一致。

### 新增支持
- 文件类型：`.txt`、`.md`、`.pdf`、`.docx`、`.html/.htm`
- URL 采集：`ingest-url` 子命令直接抓取网页文本
- RAG 对话：`rag_chat.py` 基于检索的上下文与本地 LLM 对话
- 可配置 Ollama 地址：`--ollama-host` 或 `OLLAMA_HOST`

### 先决条件
- Ollama 安装并运行，且已拉取模型：
```bash
ollama pull nomic-embed-text:latest
ollama pull llama3.2:latest  # 或你喜欢的聊天模型
```
- Python 3.9+

### 安装
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 指定 Ollama 地址
- 通过参数：
```bash
python kb_cli.py query \
  --index-dir /path/to/kb \
  --model nomic-embed-text:latest \
  --ollama-host http://127.0.0.1:11434 \
  --question "什么是余弦相似度？"
```
- 或通过环境变量：
```bash
export OLLAMA_HOST=http://127.0.0.1:11434
python kb_cli.py query --index-dir /path/to/kb --question "..."
```
`rag_chat.py` 同样支持：
```bash
python rag_chat.py \
  --index-dir /path/to/kb \
  --embed-model nomic-embed-text:latest \
  --chat-model llama3.2:latest \
  --ollama-host http://127.0.0.1:11434 \
  --question "根据知识库，如何搭建这套系统？"
```

### 构建/追加/查询（与之前一致）
- 从目录构建索引：
```bash
python kb_cli.py index \
  --input-path /path/to/docs \
  --index-dir /path/to/kb \
  --model nomic-embed-text:latest \
  --chunk-size 800 --chunk-overlap 200
```
- 追加文档：
```bash
python kb_cli.py append \
  --input-path /path/to/new_docs \
  --index-dir /path/to/kb
```
- 追加网页（URL）：
```bash
python kb_cli.py ingest-url \
  --index-dir /path/to/kb \
  --urls https://example.com https://example.org \
  --chunk-size 800 --chunk-overlap 200
```
- 查询：
```bash
python kb_cli.py query \
  --index-dir /path/to/kb \
  --question "什么是余弦相似度？" \
  --top-k 5
```

### RAG 对话（聊天）
```bash
python rag_chat.py \
  --index-dir /path/to/kb \
  --embed-model nomic-embed-text:latest \
  --chat-model llama3.2:latest \
  --question "根据知识库，如何搭建这套系统？" \
  --top-k 5
```
- 可替换 `--chat-model` 为任意可用聊天模型，例如：`llama3.1:8b-instruct`、`qwen2.5:7b-instruct`、`deepseek-r1:1.5b` 等。

### 说明
- 使用 FAISS `IndexFlatIP` + L2 归一化实现余弦相似度。
- 索引持久化文件：`index.faiss`、`meta.jsonl`、`kb.json`。
- 可通过 `--include` 指定自定义匹配模式，例如：`**/*.md **/*.pdf`。
