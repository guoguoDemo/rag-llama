#!/usr/bin/env python3
import argparse
import json
import os
import sys
import glob
import uuid
from typing import Iterable, List, Dict, Tuple, Optional

import chardet
import faiss
import numpy as np
from tqdm import tqdm

from urllib.parse import quote

try:
	import ollama  # type: ignore
except Exception as exc:  # pragma: no cover
	print("Failed to import 'ollama'. Did you run 'pip install ollama'?", file=sys.stderr)
	raise

# New parsers
try:
	from pdfminer.high_level import extract_text as pdf_extract_text  # type: ignore
except Exception:
	pdf_extract_text = None

try:
	import docx  # type: ignore
except Exception:
	docx = None

try:
	from bs4 import BeautifulSoup  # type: ignore
except Exception:
	BeautifulSoup = None

try:
	import requests  # type: ignore
except Exception:
	requests = None


# ------------------------------
# Utilities
# ------------------------------

def configure_ollama_host(ollama_host: Optional[str]) -> None:
	if ollama_host:
		os.environ['OLLAMA_HOST'] = ollama_host


def read_text_file_auto(path: str) -> str:
	with open(path, 'rb') as f:
		data = f.read()
		result = chardet.detect(data)
		encoding = result.get('encoding') or 'utf-8'
	try:
		return data.decode(encoding, errors='ignore')
	except Exception:
		return data.decode('utf-8', errors='ignore')


def read_pdf_file(path: str) -> str:
	if pdf_extract_text is None:
		raise SystemExit("pdfminer.six is not installed. Install requirements and retry.")
	try:
		return pdf_extract_text(path) or ""
	except Exception as e:
		return f""


def read_docx_file(path: str) -> str:
	if docx is None:
		raise SystemExit("python-docx is not installed. Install requirements and retry.")
	try:
		d = docx.Document(path)
		return "\n".join(p.text for p in d.paragraphs if p.text)
	except Exception:
		return ""


def read_html_file(path: str) -> str:
	if BeautifulSoup is None:
		raise SystemExit("beautifulsoup4/lxml not installed. Install requirements and retry.")
	try:
		with open(path, 'rb') as f:
			content = f.read()
		result = chardet.detect(content)
		encoding = result.get('encoding') or 'utf-8'
		html = content.decode(encoding, errors='ignore')
		soup = BeautifulSoup(html, 'lxml')
		for tag in soup(['script', 'style', 'noscript']):
			tag.decompose()
		text = soup.get_text("\n")
		lines = [line.strip() for line in text.splitlines()]
		return "\n".join(l for l in lines if l)
	except Exception:
		return ""


def read_url(url: str, timeout: int = 20) -> str:
    if requests is None or BeautifulSoup is None:
        raise SystemExit("requests/beautifulsoup4 not installed. Install requirements and retry.")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Cache-Control': 'no-cache',
        }
        r = requests.get(url, timeout=timeout, headers=headers)
        r.raise_for_status()

        # Prefer bytes + chardet for robust decoding over requests' guess
        raw = r.content
        guess = chardet.detect(raw)
        encoding = (guess.get('encoding') or r.encoding or 'utf-8')
        html = raw.decode(encoding, errors='ignore')

        soup = BeautifulSoup(html, 'lxml')
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        text = soup.get_text("\n")
        lines = [line.strip() for line in text.splitlines()]
        cleaned = "\n".join(l for l in lines if l)

        # Heuristic: detect common Chinese site anti-bot pages and drop
        bad_markers = [
            '安全验证', '百度安全验证', '网络不给力', '请稍后重试', '返回首页'
        ]
        if any(m in cleaned for m in bad_markers) and len(cleaned) < 500:
            cleaned = ""

        if cleaned.strip():
            return cleaned

        # Fallback: use Jina Reader proxy extractor if original is blocked
        # Doc: https://github.com/jina-ai/jina-reader (public text extraction proxy)
        try:
            # Jina Reader expects: https://r.jina.ai/{scheme}://{host}/path
            proxy_url = f"https://r.jina.ai/{url}"
            pr = requests.get(proxy_url, timeout=timeout, headers=headers)
            if pr.status_code == 200 and pr.text.strip():
                return pr.text
        except Exception:
            pass

        return ""
    except Exception:
        return ""


def read_any(path: str) -> str:
	ext = os.path.splitext(path)[1].lower()
	if ext in ('.txt', '.md', '.markdown'):
		return read_text_file_auto(path)
	if ext == '.pdf':
		return read_pdf_file(path)
	if ext == '.docx':
		return read_docx_file(path)
	if ext in ('.html', '.htm'):
		return read_html_file(path)
	return read_text_file_auto(path)


def iter_input_files(input_path: str, include_patterns: List[str]) -> Iterable[str]:
	if os.path.isfile(input_path):
		yield input_path
		return
	for pattern in include_patterns:
		pattern_path = os.path.join(input_path, pattern)
		for p in glob.glob(pattern_path, recursive=True):
			if os.path.isfile(p):
				yield p


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
	if chunk_size <= 0:
		return [text]
	if chunk_overlap < 0:
		chunk_overlap = 0
	chunks: List[str] = []
	start = 0
	step = max(1, chunk_size - chunk_overlap)
	while start < len(text):
		end = min(len(text), start + chunk_size)
		chunk = text[start:end]
		if chunk.strip():
			chunks.append(chunk)
		start += step
	return chunks


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
	norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
	return vectors / norms


# ------------------------------
# Knowledge Base Manager
# ------------------------------

class KnowledgeBase:
	def __init__(self, index_dir: str, model_name: str = 'nomic-embed-text:latest') -> None:
		self.index_dir = index_dir
		self.model_name = model_name
		self.index_path = os.path.join(index_dir, 'index.faiss')
		self.meta_path = os.path.join(index_dir, 'meta.jsonl')
		self.cfg_path = os.path.join(index_dir, 'kb.json')
		self.index: Optional[faiss.Index] = None
		self.metadata: List[Dict] = []
		self.dimension: Optional[int] = None

		os.makedirs(self.index_dir, exist_ok=True)
		self._load_if_exists()

	def clear(self) -> None:
		"""Delete index and metadata files and reset in-memory state."""
		# Remove files if present
		for p in (self.index_path, self.meta_path, self.cfg_path):
			try:
				if os.path.exists(p):
					os.remove(p)
			except Exception:
				pass
		# Reset in-memory state
		self.index = None
		self.metadata = []
		self.dimension = None

	def _load_if_exists(self) -> None:
		index_exists = os.path.exists(self.index_path)
		meta_exists = os.path.exists(self.meta_path)
		cfg_exists = os.path.exists(self.cfg_path)
		if index_exists and meta_exists and cfg_exists:
			self.index = faiss.read_index(self.index_path)
			self.metadata = [json.loads(line) for line in open(self.meta_path, 'r', encoding='utf-8')]
			with open(self.cfg_path, 'r', encoding='utf-8') as f:
				cfg = json.load(f)
			self.dimension = int(cfg.get('dimension')) if cfg.get('dimension') is not None else None
			# Ensure we reuse the exact embedding model that built this index
			stored_model = cfg.get('model')
			if isinstance(stored_model, str) and stored_model.strip():
				self.model_name = stored_model.strip()

	def _save(self) -> None:
		assert self.index is not None, "Index is not initialized"
		faiss.write_index(self.index, self.index_path)
		with open(self.meta_path, 'w', encoding='utf-8') as f:
			for row in self.metadata:
				f.write(json.dumps(row, ensure_ascii=False) + "\n")
		with open(self.cfg_path, 'w', encoding='utf-8') as f:
			json.dump({
				'model': self.model_name,
				'dimension': self.dimension,
				'index_type': 'IndexFlatIP',
				'version': 1
			}, f, ensure_ascii=False, indent=2)

	def _rebuild_index_from_metadata(self) -> None:
		"""Rebuild in-memory FAISS index from current self.metadata by re-embedding.
		Expensive but enables deletion/dedup without IDMap support.
		"""
		texts = [row['text'] for row in self.metadata]
		if not texts:
			# Reset to empty state; next add() will re-init
			self.index = None
			self.dimension = None
			return
		emb = self._embed_texts(texts)
		emb = l2_normalize(emb)
		self.dimension = int(emb.shape[1])
		self.index = faiss.IndexFlatIP(self.dimension)
		self.index.add(emb)

	def _dedup_paths(self, paths: List[str]) -> int:
		"""Remove all existing rows whose path is in paths; rebuild index. Return removed count."""
		if not paths:
			return 0
		if not self.metadata:
			return 0
		path_set = set(paths)
		before = len(self.metadata)
		self.metadata = [row for row in self.metadata if row.get('path') not in path_set]
		removed = before - len(self.metadata)
		if removed > 0:
			self._rebuild_index_from_metadata()
			# Persist current state if not immediately saving later
			if self.index is not None:
				self._save()
		return removed

	def _embed_texts(self, texts: List[str]) -> np.ndarray:
		vectors: List[List[float]] = []
		for t in tqdm(texts, desc='Embedding', unit='chunk'):
			resp = ollama.embeddings(model=self.model_name, prompt=t)
			vec = resp.get('embedding')
			if not isinstance(vec, list):
				raise RuntimeError('Unexpected embedding response shape')
			vectors.append(vec)
		arr = np.array(vectors, dtype='float32')
		if self.dimension is None:
			self.dimension = int(arr.shape[1])
		return arr

	def _ensure_index(self) -> None:
		if self.index is None:
			assert self.dimension is not None and self.dimension > 0, 'Unknown embedding dimension'
			self.index = faiss.IndexFlatIP(self.dimension)

	def index_from_path(self, input_path: str, include_patterns: List[str], chunk_size: int, chunk_overlap: int) -> Tuple[int, int]:
		self.metadata = []
		self.index = None
		self.dimension = None

		all_chunks: List[str] = []
		all_meta: List[Dict] = []
		for path in iter_input_files(input_path, include_patterns):
			text = read_any(path)
			chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
			for i, c in enumerate(chunks):
				row = {
					'id': str(uuid.uuid4()),
					'path': os.path.abspath(path),
					'chunk_index': i,
					'text': c
				}
				all_meta.append(row)
				all_chunks.append(c)

		if not all_chunks:
			raise SystemExit('No input files/chunks found to index')

		emb = self._embed_texts(all_chunks)
		emb = l2_normalize(emb)
		self._ensure_index()
		assert self.index is not None
		self.index.add(emb)
		self.metadata = all_meta
		self._save()
		return len(all_chunks), len(all_meta)

	def append_from_path(self, input_path: str, include_patterns: List[str], chunk_size: int, chunk_overlap: int) -> Tuple[int, int]:
		if self.index is None:
			raise SystemExit('KB not initialized. Run index first.')

		# Collect target file absolute paths for dedup
		target_paths: List[str] = []
		all_chunks: List[str] = []
		all_meta: List[Dict] = []
		for path in iter_input_files(input_path, include_patterns):
			target_paths.append(os.path.abspath(path))
			text = read_any(path)
			chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
			for i, c in enumerate(chunks):
				row = {
					'id': str(uuid.uuid4()),
					'path': os.path.abspath(path),
					'chunk_index': i,
					'text': c
				}
				all_meta.append(row)
				all_chunks.append(c)

		# Deduplicate any existing rows for these paths by rebuilding index
		if target_paths:
			self._dedup_paths(target_paths)

		if not all_chunks:
			return 0, 0

		emb = self._embed_texts(all_chunks)
		emb = l2_normalize(emb)
		assert self.index is not None
		self.index.add(emb)
		self.metadata.extend(all_meta)
		self._save()
		return len(all_chunks), len(all_meta)

	def append_from_urls(self, urls: List[str], chunk_size: int, chunk_overlap: int) -> Tuple[int, int]:
		if self.index is None:
			raise SystemExit('KB not initialized. Run index first.')
		all_chunks: List[str] = []
		all_meta: List[Dict] = []
		incoming_paths: List[str] = []
		for u in urls:
			text = read_url(u)
			if not text.strip():
				continue
			chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
			for i, c in enumerate(chunks):
				row = {
					'id': str(uuid.uuid4()),
					'path': u,
					'chunk_index': i,
					'text': c
				}
				all_meta.append(row)
				all_chunks.append(c)
			incoming_paths.append(u)

		# Dedup existing rows for these URLs
		if incoming_paths:
			self._dedup_paths(incoming_paths)

		if not all_chunks:
			return 0, 0
		emb = self._embed_texts(all_chunks)
		emb = l2_normalize(emb)
		assert self.index is not None
		self.index.add(emb)
		self.metadata.extend(all_meta)
		self._save()
		return len(all_chunks), len(all_meta)

	def query(self, question: str, top_k: int = 5) -> List[Dict]:
		if self.index is None:
			raise SystemExit('KB not initialized. Run index first.')
		resp = ollama.embeddings(model=self.model_name, prompt=question)
		q = np.array([resp['embedding']], dtype='float32')
		q = l2_normalize(q)
		assert self.index is not None
		scores, idxs = self.index.search(q, top_k)
		results: List[Dict] = []
		for rank, (score, idx) in enumerate(zip(scores[0].tolist(), idxs[0].tolist())):
			if idx == -1:
				continue
			meta = self.metadata[idx]
			results.append({
				'rank': rank + 1,
				'score': float(score),
				'path': meta['path'],
				'chunk_index': meta['chunk_index'],
				'text': meta['text']
			})
		return results


# ------------------------------
# CLI
# ------------------------------

def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description='Ollama KB Toolkit using nomic-embed-text:latest')
	sub = p.add_subparsers(dest='cmd', required=True)

	common = argparse.ArgumentParser(add_help=False)
	common.add_argument('--index-dir', required=True, help='Directory to store/load the KB index')
	common.add_argument('--model', default='nomic-embed-text:latest', help='Ollama embedding model')
	common.add_argument('--ollama-host', default=None, help='Ollama host, e.g. http://127.0.0.1:11434')
	common.add_argument('--include', nargs='*', default=['**/*.txt', '**/*.md', '**/*.pdf', '**/*.docx', '**/*.html', '**/*.htm'], help='Glob patterns relative to input-path')
	common.add_argument('--chunk-size', type=int, default=800, help='Chunk size (characters)')
	common.add_argument('--chunk-overlap', type=int, default=200, help='Chunk overlap (characters)')

	p_index = sub.add_parser('index', parents=[common], help='Build a new KB index from scratch')
	p_index.add_argument('--input-path', required=True, help='File or directory of documents to index')

	p_append = sub.add_parser('append', parents=[common], help='Append new documents to existing KB')
	p_append.add_argument('--input-path', required=True, help='File or directory of documents to append')

	p_query = sub.add_parser('query', help='Query the KB')
	p_query.add_argument('--index-dir', required=True, help='Directory of the KB index')
	p_query.add_argument('--model', default='nomic-embed-text:latest', help='Ollama embedding model')
	p_query.add_argument('--ollama-host', default=None, help='Ollama host, e.g. http://127.0.0.1:11434')
	p_query.add_argument('--question', required=True, help='Question to search for')
	p_query.add_argument('--top-k', type=int, default=5, help='Top K results')

	p_url = sub.add_parser('ingest-url', help='Append web pages by URL to existing KB')
	p_url.add_argument('--index-dir', required=True, help='Directory of the KB index')
	p_url.add_argument('--model', default='nomic-embed-text:latest', help='Ollama embedding model')
	p_url.add_argument('--ollama-host', default=None, help='Ollama host, e.g. http://127.0.0.1:11434')
	p_url.add_argument('--urls', nargs='+', required=True, help='One or more URLs to ingest')
	p_url.add_argument('--chunk-size', type=int, default=800, help='Chunk size (characters)')
	p_url.add_argument('--chunk-overlap', type=int, default=200, help='Chunk overlap (characters)')

	return p


def cmd_index(args: argparse.Namespace) -> None:
	configure_ollama_host(args.ollama_host)
	kb = KnowledgeBase(index_dir=args.index_dir, model_name=args.model)
	n_chunks, n_meta = kb.index_from_path(
		input_path=args.input_path,
		include_patterns=args.include,
		chunk_size=args.chunk_size,
		chunk_overlap=args.chunk_overlap,
	)
	print(f"Indexed {n_chunks} chunks from {n_meta} metadata rows into '{args.index_dir}'.")


def cmd_append(args: argparse.Namespace) -> None:
	configure_ollama_host(args.ollama_host)
	kb = KnowledgeBase(index_dir=args.index_dir, model_name=args.model)
	n_chunks, n_meta = kb.append_from_path(
		input_path=args.input_path,
		include_patterns=args.include,
		chunk_size=args.chunk_size,
		chunk_overlap=args.chunk_overlap,
	)
	print(f"Appended {n_chunks} chunks from {n_meta} metadata rows into '{args.index_dir}'.")


def cmd_query(args: argparse.Namespace) -> None:
	configure_ollama_host(args.ollama_host)
	kb = KnowledgeBase(index_dir=args.index_dir, model_name=args.model)
	results = kb.query(args.question, top_k=args.top_k)
	for r in results:
		print('=' * 80)
		print(f"Rank: {r['rank']}  Score: {r['score']:.4f}")
		print(f"Path: {r['path']}  Chunk: {r['chunk_index']}")
		print('-' * 80)
		print(r['text'])


def cmd_ingest_url(args: argparse.Namespace) -> None:
	configure_ollama_host(args.ollama_host)
	kb = KnowledgeBase(index_dir=args.index_dir, model_name=args.model)
	n_chunks, n_meta = kb.append_from_urls(args.urls, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
	print(f"Ingested {n_chunks} chunks from {n_meta} URL metadata rows into '{args.index_dir}'.")


def main(argv: Optional[List[str]] = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	if args.cmd == 'index':
		cmd_index(args)
	elif args.cmd == 'append':
		cmd_append(args)
	elif args.cmd == 'query':
		cmd_query(args)
	elif args.cmd == 'ingest-url':
		cmd_ingest_url(args)
	else:
		parser.print_help()
		return 2
	return 0


if __name__ == '__main__':
	sys.exit(main())
