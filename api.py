#!/usr/bin/env python3
import os
from typing import List, Optional

from fastapi import FastAPI, Body, HTTPException
from fastapi import UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse
from fastapi.responses import RedirectResponse
from fastapi import Response
from pydantic import BaseModel

from kb_cli import KnowledgeBase, configure_ollama_host, chunk_text, l2_normalize, read_any, iter_input_files
from rag_chat import try_load_kb, kb_has_data, retrieve
import faiss
import json
import numpy as np

app = FastAPI(title="Ollama KB API", version="1.0.0")

# Enable CORS for development convenience (adjust origins in production)
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Serve frontend static files from ./frontend under /ui (avoid shadowing API)
frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')
if os.path.isdir(frontend_dir):
	app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="frontend")


class IndexRequest(BaseModel):
	index_dir: str
	input_path: str
	model: str = 'nomic-embed-text:latest'
	ollama_host: Optional[str] = None
	include: List[str] = ['**/*.txt', '**/*.md', '**/*.pdf', '**/*.docx', '**/*.html', '**/*.htm']
	chunk_size: int = 800
	chunk_overlap: int = 200


class AppendRequest(BaseModel):
	index_dir: str
	input_path: str
	model: str = 'nomic-embed-text:latest'
	ollama_host: Optional[str] = None
	include: List[str] = ['**/*.txt', '**/*.md', '**/*.pdf', '**/*.docx', '**/*.html', '**/*.htm']
	chunk_size: int = 800
	chunk_overlap: int = 200


class IngestURLRequest(BaseModel):
	index_dir: str
	urls: List[str]
	model: str = 'nomic-embed-text:latest'
	ollama_host: Optional[str] = None
	chunk_size: int = 800
	chunk_overlap: int = 200


class QueryRequest(BaseModel):
	index_dir: str
	question: str
	model: str = 'nomic-embed-text:latest'
	ollama_host: Optional[str] = None
	top_k: int = 5


class RAGChatRequest(BaseModel):
	index_dir: Optional[str] = None
	question: str
	embed_model: str = 'nomic-embed-text:latest'
	chat_model: str = 'llama3.2:latest'
	ollama_host: Optional[str] = None
	top_k: int = 5


class ClearRequest(BaseModel):
	index_dir: str
	ollama_host: Optional[str] = None


@app.post('/kb/index')
async def kb_index(req: IndexRequest):
	configure_ollama_host(req.ollama_host)
	kb = KnowledgeBase(index_dir=req.index_dir, model_name=req.model)
	n_chunks, n_meta = kb.index_from_path(
		input_path=req.input_path,
		include_patterns=req.include,
		chunk_size=req.chunk_size,
		chunk_overlap=req.chunk_overlap,
	)
	return {"chunks": n_chunks, "rows": n_meta, "index_dir": req.index_dir}


@app.post('/kb/append')
async def kb_append(req: AppendRequest):
	configure_ollama_host(req.ollama_host)
	kb = KnowledgeBase(index_dir=req.index_dir, model_name=req.model)
	n_chunks, n_meta = kb.append_from_path(
		input_path=req.input_path,
		include_patterns=req.include,
		chunk_size=req.chunk_size,
		chunk_overlap=req.chunk_overlap,
	)
	return {"chunks": n_chunks, "rows": n_meta, "index_dir": req.index_dir}


from kb_cli import read_url  # reuse


@app.post('/kb/ingest-url')
async def kb_ingest_url(req: IngestURLRequest):
	configure_ollama_host(req.ollama_host)
	kb = KnowledgeBase(index_dir=req.index_dir, model_name=req.model)
	n_chunks, n_meta = kb.append_from_urls(req.urls, chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
	return {"chunks": n_chunks, "rows": n_meta, "index_dir": req.index_dir}


@app.post('/kb/query')
async def kb_query(req: QueryRequest):
	configure_ollama_host(req.ollama_host)
	kb = KnowledgeBase(index_dir=req.index_dir, model_name=req.model)
	results = kb.query(req.question, top_k=req.top_k)
	return {"results": results}


# RAG endpoints
from rag_chat import chat_with_rag
from rag_chat import build_messages


@app.post('/chat/rag')
async def chat_rag(req: RAGChatRequest):
	configure_ollama_host(req.ollama_host)
	answer = chat_with_rag(
		index_dir=req.index_dir,
		embed_model=req.embed_model,
		chat_model=req.chat_model,
		question=req.question,
		top_k=req.top_k,
	)
	return {"answer": answer}


@app.post('/kb/clear')
async def kb_clear(req: ClearRequest):
	configure_ollama_host(req.ollama_host)
	kb = KnowledgeBase(index_dir=req.index_dir)
	kb.clear()
	return {"cleared": True, "index_dir": req.index_dir}


# Health
@app.get('/health')
async def health():
	return {"status": "ok"}


# To run: uvicorn api:app --host 0.0.0.0 --port 8000

# Redirect root to UI
@app.get('/')
async def root_redirect():
	return RedirectResponse(url='/ui/')


# Handle any CORS preflight to avoid 405 on OPTIONS
@app.options('/{full_path:path}')
async def preflight(full_path: str) -> Response:
	return Response(status_code=204)


# ------------------------------
# File upload → append into KB
# ------------------------------

@app.post('/kb/upload')
async def kb_upload(
	index_dir: str = Form(...),
	model: str = Form('nomic-embed-text:latest'),
	ollama_host: Optional[str] = Form(None),
	chunk_size: int = Form(800),
	chunk_overlap: int = Form(200),
	files: List[UploadFile] = File(...),
):
	configure_ollama_host(ollama_host)
	kb = KnowledgeBase(index_dir=index_dir, model_name=model)
	upload_root = os.path.join(index_dir, '_uploads')
	os.makedirs(upload_root, exist_ok=True)
	saved_paths: List[str] = []
	for uf in files:
		filename = os.path.basename(uf.filename or 'upload')
		target = os.path.join(upload_root, filename)
		with open(target, 'wb') as out:
			data = await uf.read()
			out.write(data)
		saved_paths.append(target)

	# Dedup existing content for these file paths before appending
	try:
		kb._dedup_paths([os.path.abspath(p) for p in saved_paths])  # type: ignore[attr-defined]
	except Exception:
		pass

	include_patterns = [os.path.basename(p) for p in saved_paths]
	n_chunks, n_meta = kb.append_from_path(
		input_path=upload_root,
		include_patterns=include_patterns,
		chunk_size=chunk_size,
		chunk_overlap=chunk_overlap,
	)
	return {"chunks": n_chunks, "rows": n_meta, "index_dir": index_dir, "files": [os.path.basename(p) for p in saved_paths]}


# ------------------------------
# Streaming RAG chat
# ------------------------------

@app.post('/chat/rag/stream')
async def chat_rag_stream(req: RAGChatRequest):
	configure_ollama_host(req.ollama_host)
	kb, _err = try_load_kb(req.index_dir)
	contexts: List[dict] = []
	if kb is not None and kb_has_data(kb):
		try:
			contexts = retrieve(kb, embed_model=req.embed_model, question=req.question, top_k=req.top_k)
		except Exception:
			contexts = []
	messages = build_messages(contexts, user_question=req.question)

	import ollama

	def gen():
		try:
			for chunk in ollama.chat(model=req.chat_model, messages=messages, stream=True, options={"temperature": 0.7, "top_p": 0.9}):
				piece = (chunk.get('message') or {}).get('content', '')
				if piece:
					yield piece
		except Exception as e:
			yield f"\n[stream error] {e}"

	return StreamingResponse(gen(), media_type='text/plain; charset=utf-8')
