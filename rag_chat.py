#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import List, Dict, Optional, Tuple

import faiss
import numpy as np

try:
	import ollama  # type: ignore
except Exception:
	print("Failed to import 'ollama'. Did you run 'pip install ollama'?", file=sys.stderr)
	raise


def configure_ollama_host(ollama_host: Optional[str]) -> None:
	if ollama_host:
		os.environ['OLLAMA_HOST'] = ollama_host


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
	norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
	return vectors / norms


def load_kb(index_dir: str) -> Dict:
	index_path = os.path.join(index_dir, 'index.faiss')
	meta_path = os.path.join(index_dir, 'meta.jsonl')
	cfg_path = os.path.join(index_dir, 'kb.json')
	if not (os.path.exists(index_path) and os.path.exists(meta_path) and os.path.exists(cfg_path)):
		raise SystemExit(f"KB not found under '{index_dir}'. Run indexing first.")
	index = faiss.read_index(index_path)
	metadata: List[Dict] = [json.loads(line) for line in open(meta_path, 'r', encoding='utf-8')]
	cfg = json.load(open(cfg_path, 'r', encoding='utf-8'))
	return {'index': index, 'metadata': metadata, 'cfg': cfg}


def try_load_kb(index_dir: Optional[str]) -> Tuple[Optional[Dict], Optional[str]]:
	if not index_dir:
		return None, None
	try:
		kb = load_kb(index_dir)
		return kb, None
	except SystemExit as e:
		return None, str(e)


def kb_has_data(kb: Dict) -> bool:
	try:
		index: faiss.Index = kb['index']  # type: ignore
		meta = kb.get('metadata') or []
		ntotal = getattr(index, 'ntotal', 0)
		return bool(meta) and int(ntotal) > 0
	except Exception:
		return False


def retrieve(kb: Dict, embed_model: str, question: str, top_k: int) -> List[Dict]:
	resp = ollama.embeddings(model=embed_model, prompt=question)
	q = np.array([resp['embedding']], dtype='float32')
	q = l2_normalize(q)
	index: faiss.Index = kb['index']
	scores, idxs = index.search(q, top_k)
	results: List[Dict] = []
	metadata: List[Dict] = kb['metadata']
	for rank, (score, idx) in enumerate(zip(scores[0].tolist(), idxs[0].tolist())):
		if idx == -1:
			continue
		meta = metadata[idx]
		results.append({
			'rank': rank + 1,
			'score': float(score),
			'path': meta['path'],
			'chunk_index': meta['chunk_index'],
			'text': meta['text']
		})
	return results


def build_system_prompt() -> str:
	return (
		"你是中文对话助手，语气自然、口语化、简洁，不啰嗦。\n"
		"- 使用知识库片段作为参考进行归纳，不要逐字复述，也不要说“作为模型/基于知识库”这类套话；\n"
		"- 先用1句话直接给结论，再补充最多3条要点（如有必要）；\n"
		"- 无把握就说不确定或不知道，避免硬编；\n"
		"- 可引用极短关键词并用 [文档 N] 标注来源；\n"
		"- 默认不超过120字，除非用户要求详细。\n"
	)



def _truncate(text: str, max_len: int = 400) -> str:
	if len(text) <= max_len:
		return text
	return text[:max_len] + "\u2026"  # ellipsis


def build_messages(contexts: List[Dict], user_question: str) -> List[Dict[str, str]]:
	system_prompt = build_system_prompt()
	if contexts:
		context_block = "\n\n".join(
			f"[文档 {c['rank']} | 分数: {c['score']:.4f} | 来源: {c['path']} | 片段: {c['chunk_index']}]\n{_truncate(c['text'])}"
			for c in contexts
		)
		user_content = (
			f"以下是用于参考的知识库片段（仅作辅助，避免逐字引用）：\n\n{context_block}\n\n"
			f"请像日常对话一样回答：先一句话结论，再补充最多3条要点；没有依据就直说不知道。\n\n问题：{user_question}"
		)
	else:
		user_content = (
			f"当前没有可用的知识库片段。请基于常识与一般经验，"
			f"用口语化、简洁的方式回答我的问题：{user_question}"
		)
	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_content}
	]
	return messages


def chat_with_rag(index_dir: Optional[str], embed_model: str, chat_model: str, question: str, top_k: int) -> str:
	kb, _err = try_load_kb(index_dir)
	contexts: List[Dict] = []
	if kb is not None and kb_has_data(kb):
		try:
			contexts = retrieve(kb, embed_model=embed_model, question=question, top_k=top_k)
		except Exception:
			contexts = []
	messages = build_messages(contexts, user_question=question)
	# 适度提升自然度并控制冗长
	resp = ollama.chat(
		model=chat_model,
		messages=messages,
		options={
			"temperature": 0.7,
			"top_p": 0.9
		}
	)
	return resp.get('message', {}).get('content', '')


def main(argv: Optional[List[str]] = None) -> int:
	p = argparse.ArgumentParser(description='RAG chat using Ollama + FAISS KB')
	p.add_argument('--index-dir', required=False, help='KB directory containing index.faiss/meta.jsonl/kb.json (optional)')
	p.add_argument('--embed-model', default='nomic-embed-text:latest', help='Ollama embedding model')
	p.add_argument('--chat-model', default='llama3.2:latest', help='Ollama chat model')
	p.add_argument('--ollama-host', default=None, help='Ollama host, e.g. http://127.0.0.1:11434')
	p.add_argument('--question', required=True, help='Question to ask')
	p.add_argument('--top-k', type=int, default=5, help='Top K contexts to retrieve')
	args = p.parse_args(argv)

	configure_ollama_host(args.ollama_host)
	answer = chat_with_rag(
		index_dir=args.index_dir,
		embed_model=args.embed_model,
		chat_model=args.chat_model,
		question=args.question,
		top_k=args.top_k,
	)
	print(answer)
	return 0


if __name__ == '__main__':
	sys.exit(main())
