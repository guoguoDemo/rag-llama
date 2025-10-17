function $(id) { return document.getElementById(id); }

function getCommonPayload() {
  const ollama_host = $("ollamaHost").value.trim() || null;
  return { ollama_host };
}

function getIndexDir() {
  return $("indexDir").value.trim();
}

async function apiPost(path, body) {
  const url = path;
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }
  return await resp.json();
}

function toast(el, msg) {
  el.textContent = msg;
}

function parseInclude(str) {
  return str.trim() ? str.trim().split(/\s+/g) : undefined;
}

// KB actions
$("btnIndex").addEventListener('click', async () => {
  const output = $("kbOutput");
  output.textContent = '正在构建索引...';
  try {
    const payload = {
      index_dir: getIndexDir(),
      input_path: $("inputPath").value.trim(),
      model: $("embedModel").value.trim() || 'nomic-embed-text:latest',
      include: parseInclude($("include").value) || ['**/*.txt','**/*.md','**/*.pdf','**/*.docx','**/*.html','**/*.htm'],
      chunk_size: parseInt($("chunkSize").value || '800', 10),
      chunk_overlap: parseInt($("chunkOverlap").value || '200', 10),
      ...getCommonPayload()
    };
    const res = await apiPost('/kb/index', payload);
    output.textContent = `完成：共 ${res.chunks} 块，元数据 ${res.rows} 条。\nindex_dir: ${res.index_dir}`;
  } catch (e) {
    output.textContent = `失败：${e.message}`;
  }
});

$("btnAppend").addEventListener('click', async () => {
  const output = $("kbOutput");
  output.textContent = '正在追加文档...';
  try {
    const payload = {
      index_dir: getIndexDir(),
      input_path: $("inputPath").value.trim(),
      model: $("embedModel").value.trim() || 'nomic-embed-text:latest',
      include: parseInclude($("include").value) || ['**/*.txt','**/*.md','**/*.pdf','**/*.docx','**/*.html','**/*.htm'],
      chunk_size: parseInt($("chunkSize").value || '800', 10),
      chunk_overlap: parseInt($("chunkOverlap").value || '200', 10),
      ...getCommonPayload()
    };
    const res = await apiPost('/kb/append', payload);
    output.textContent = `完成：追加 ${res.chunks} 块，新增元数据 ${res.rows} 条。`;
  } catch (e) {
    output.textContent = `失败：${e.message}`;
  }
});

$("btnIngestUrl").addEventListener('click', async () => {
  const output = $("kbOutput");
  output.textContent = '正在采集网页...';
  try {
    const urls = $("urls").value.trim().split(/\s+/g).filter(Boolean);
    const payload = {
      index_dir: getIndexDir(),
      urls,
      model: $("embedModel").value.trim() || 'nomic-embed-text:latest',
      chunk_size: parseInt($("chunkSize").value || '800', 10),
      chunk_overlap: parseInt($("chunkOverlap").value || '200', 10),
      ...getCommonPayload()
    };
    const res = await apiPost('/kb/ingest-url', payload);
    output.textContent = `完成：采集 ${res.chunks} 块，新增元数据 ${res.rows} 条。`;
  } catch (e) {
    output.textContent = `失败：${e.message}`;
  }
});

$("btnClear").addEventListener('click', async () => {
  const output = $("kbOutput");
  if (!confirm('确定要清空索引吗？该操作不可撤销。')) return;
  output.textContent = '正在清空...';
  try {
    const payload = { index_dir: getIndexDir(), ...getCommonPayload() };
    const res = await apiPost('/kb/clear', payload);
    output.textContent = res.cleared ? `已清空：${res.index_dir}` : '清空失败';
  } catch (e) {
    output.textContent = `失败：${e.message}`;
  }
});

// Query & Chat
$("btnQuery").addEventListener('click', async () => {
  const list = $("results");
  list.innerHTML = '<div class="item">正在检索...</div>';
  try {
    const payload = {
      index_dir: getIndexDir(),
      question: $("question").value.trim(),
      model: $("embedModel").value.trim() || 'nomic-embed-text:latest',
      top_k: parseInt($("topK").value || '5', 10),
      ...getCommonPayload()
    };
    const res = await apiPost('/kb/query', payload);
    const items = (res.results || []).map(r => `
      <div class="item">
        <div class="meta">[文档 ${r.rank}] 分数 ${r.score.toFixed(4)} · 来源 ${r.path} · 片段 ${r.chunk_index}</div>
        <div class="text">${escapeHtml(r.text)}</div>
      </div>
    `).join('');
    list.innerHTML = items || '<div class="item">没有结果</div>';
  } catch (e) {
    list.innerHTML = `<div class="item">失败：${escapeHtml(e.message)}</div>`;
  }
});

$("btnChat").addEventListener('click', async () => {
  const answer = $("answer");
  answer.textContent = '正在生成回答...';
  try {
    const payload = {
      index_dir: getIndexDir() || null,
      embed_model: $("embedModel").value.trim() || 'nomic-embed-text:latest',
      chat_model: $("chatModel").value.trim() || 'llama3.2:latest',
      question: $("question").value.trim(),
      top_k: parseInt($("topK").value || '5', 10),
      ...getCommonPayload()
    };
    const res = await apiPost('/chat/rag', payload);
    answer.textContent = normalizeText(res.answer || '（无内容）');
  } catch (e) {
    answer.textContent = `失败：${e.message}`;
  }
});

function escapeHtml(str) {
  return String(str)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function normalizeText(s) {
  if (!s) return s;
  return s
    // unify line endings
    .replace(/\r\n?/g, '\n')
    // remove trailing spaces before newline
    .replace(/[ \t]+\n/g, '\n')
    // collapse any blank lines to a single newline (no empty paragraph)
    .replace(/\n[ \t]*\n+/g, '\n')
    // collapse multiple spaces
    .replace(/[\u00A0\s]{2,}/g, ' ')
    .trim();
}

// Prefill sensible defaults for local demo
window.addEventListener('DOMContentLoaded', () => {
  if (!$("ollamaHost").value) $("ollamaHost").value = 'http://127.0.0.1:11434';
  if (!$("indexDir").value) $("indexDir").value = '/Users/kele/Desktop/ollama/embeddedModel/kb';
  if (!$("inputPath").value) $("inputPath").value = '/Users/kele/Desktop/ollama/embeddedModel/sample_docs';
  // read URL params
  const params = new URLSearchParams(location.search);
  const oh = params.get('ollama_host');
  const id = params.get('index_dir');
  const ip = params.get('input_path');
  if (oh) $("ollamaHost").value = oh;
  if (id) $("indexDir").value = id;
  if (ip) $("inputPath").value = ip;
});

// Streaming chat
$("btnChatStream").addEventListener('click', async () => {
  const answer = $("answer");
  const hint = $("streamHint");
  answer.textContent = '';
  hint.textContent = '（流式中…）';
  try {
    const payload = {
      index_dir: getIndexDir() || null,
      embed_model: $("embedModel").value,
      chat_model: $("chatModel").value,
      question: $("question").value.trim(),
      top_k: parseInt($("topK").value || '5', 10),
      ...getCommonPayload()
    };
    const resp = await fetch('/chat/rag/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);
    const reader = resp.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let done, value;
    while (true) {
      ({ done, value } = await reader.read());
      if (done) break;
      answer.textContent += decoder.decode(value, { stream: true });
    }
    answer.textContent = normalizeText(answer.textContent);
  } catch (e) {
    answer.textContent = `失败：${e.message}`;
  } finally {
    hint.textContent = '';
    pushHistory($("question").value.trim(), answer.textContent);
  }
});

// Chat history
const history = [];
function pushHistory(q, a) {
  if (!q && !a) return;
  history.push({ q, a, t: new Date().toISOString() });
  renderHistory();
}
function renderHistory() {
  const el = $("history");
  el.innerHTML = history.map(h => `
    <div class="item">
      <div class="meta">Q · ${escapeHtml(h.t)}</div>
      <div class="text">${escapeHtml(h.q)}</div>
      <div class="meta">A</div>
      <div class="text">${escapeHtml(h.a)}</div>
    </div>
  `).join('');
}

// Upload files
$("btnUpload").addEventListener('click', async () => {
  const files = $("fileInput").files;
  const prog = $("uploadProgress");
  if (!files || files.length === 0) { prog.style.display='block'; prog.textContent = '请选择文件'; return; }
  prog.style.display='block'; prog.textContent = '上传中...';
  try {
    const fd = new FormData();
    fd.append('index_dir', getIndexDir());
    fd.append('model', $("embedModel").value);
    fd.append('chunk_size', $("chunkSize").value || '800');
    fd.append('chunk_overlap', $("chunkOverlap").value || '200');
    const host = $("ollamaHost").value.trim(); if (host) fd.append('ollama_host', host);
    Array.from(files).forEach(f => fd.append('files', f));
    const resp = await fetch('/kb/upload', { method: 'POST', body: fd });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    prog.textContent = `完成：索引 ${data.chunks} 块 · ${data.rows} 条 · 文件 ${data.files.join(', ')}`;
  } catch (e) {
    prog.textContent = `失败：${e.message}`;
  }
});

// URL ingestion progress (simple spinner)
const oldIngest = $("btnIngestUrl").onclick;
$("btnIngestUrl").addEventListener('click', () => {
  const out = $("kbOutput");
  const timer = setInterval(() => { out.textContent += '.'; }, 500);
  const stop = () => clearInterval(timer);
  const orig = out.textContent;
  const obs = new MutationObserver(() => { stop(); obs.disconnect(); });
  obs.observe(out, { childList: true, characterData: true, subtree: true });
});

// Config save (localStorage) and multi KB list
$("btnSaveCfg").addEventListener('click', () => {
  const cfg = {
    ollamaHost: $("ollamaHost").value.trim(),
    indexDir: $("indexDir").value.trim(),
    embedModel: $("embedModel").value,
    chatModel: $("chatModel").value,
  };
  localStorage.setItem('cfg', JSON.stringify(cfg));
  // maintain recent KBs
  const set = new Set(JSON.parse(localStorage.getItem('kbList') || '[]'));
  if (cfg.indexDir) set.add(cfg.indexDir);
  localStorage.setItem('kbList', JSON.stringify(Array.from(set)));
  alert('已保存');
});

$("btnManageKb").addEventListener('click', () => {
  const list = JSON.parse(localStorage.getItem('kbList') || '[]');
  const pick = prompt('输入序号选择或直接输入新路径:\n' + list.map((p,i)=>`${i+1}. ${p}`).join('\n'));
  if (!pick) return;
  const num = Number(pick);
  if (Number.isInteger(num) && num >= 1 && num <= list.length) {
    $("indexDir").value = list[num-1];
  } else {
    $("indexDir").value = pick;
  }
});

// Load saved cfg
window.addEventListener('DOMContentLoaded', () => {
  try {
    const s = localStorage.getItem('cfg');
    if (s) {
      const c = JSON.parse(s);
      if (c.ollamaHost) $("ollamaHost").value = c.ollamaHost;
      if (c.indexDir) $("indexDir").value = c.indexDir;
      if (c.embedModel) $("embedModel").value = c.embedModel;
      if (c.chatModel) $("chatModel").value = c.chatModel;
    }
  } catch {}
});

// Theme toggle & i18n (very light)
$("btnTheme").addEventListener('click', () => {
  document.body.classList.toggle('light');
});
$("btnI18n").addEventListener('click', () => {
  alert('轻量示例：当前界面文本固定为中文，如需完整 i18n 我可以替换成字典驱动。');
});

// Export results
$("btnExport").addEventListener('click', () => {
  const data = { history, lastAnswer: $("answer").textContent };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'rag_results.json';
  a.click();
  URL.revokeObjectURL(a.href);
});


