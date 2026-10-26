/* 前端逻辑：三件事
   1. 把问题 POST 到 /chat/stream，用 fetch 读 body 流（SSE），边收边往气泡里追加
   2. 把回答里的 [1][2] 变成可点的引用，点了高亮右栏对应来源
   3. 左栏：文档列表 / 上传 PDF / 模式切换

   没有用 EventSource：EventSource 只能发 GET，没法带 JSON body。
   自己读流的好处是请求体想怎么带就怎么带。
*/

const $ = (id) => document.getElementById(id);

const els = {
  messages: $("messages"),
  input: $("input"),
  send: $("send"),
  composer: $("composer"),
  mode: $("mode"),
  modeHint: $("modeHint"),
  docList: $("docList"),
  docCount: $("docCount"),
  file: $("file"),
  uploadHint: $("uploadHint"),
  sourceList: $("sourceList"),
  srcCount: $("srcCount"),
  health: $("health"),
  clearBtn: $("clearBtn"),
};

const MODE_HINT = {
  hybrid_rerank: "检索先粗筛 20 条，再用交叉编码器精排出 5 条。最准，多花 1~2 秒。",
  hybrid: "向量 + BM25 双路召回，用 RRF 融合名次。中等精度，快。",
  vector: "只用向量相似度。最快，但关键词型提问容易漏。",
};

const STATUS_TEXT = {
  uploaded: "排队中",
  parsing: "解析中",
  embedding: "向量化",
  indexed: "已入库",
  failed: "失败",
};

const WELCOME_HTML = els.messages.innerHTML;

let busy = false;          // 一次只允许一个请求在跑
let sources = [];          // 当前回答对应的来源，供引用点击查

/* ────────────────────────────── 工具 ────────────────────────────── */

function scrollToBottom(force = false) {
  const box = els.messages;
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 120;
  if (force || nearBottom) box.scrollTop = box.scrollHeight;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

/* 极简 markdown → HTML。
   为什么不用 marked：要引 CDN，国内可能白屏；而且要额外接 DOMPurify 防注入。
   这里先 escapeHtml 再拼标签，天然没有 XSS 风险，够用的语法都覆盖了。 */
function renderMarkdown(src) {
  const blocks = [];
  // 先把 ``` 代码块抽走，否则里面的 * # - 会被当成格式
  let text = String(src).replace(/```[a-zA-Z0-9]*\n?([\s\S]*?)(```|$)/g, (_, code) => {
    blocks.push(`<pre class="code"><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });

  let html = escapeHtml(text)
    .replace(/`([^`\n]+)`/g, '<code class="inline">$1</code>')
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");

  const out = [];
  let list = null; // "ul" | "ol"
  const closeList = () => {
    if (list) { out.push(`</${list}>`); list = null; }
  };

  for (const raw of html.split("\n")) {
    const line = raw.trim();
    if (!line) { closeList(); continue; }

    const ph = line.match(/^\u0000(\d+)\u0000$/);
    if (ph) { closeList(); out.push(blocks[Number(ph[1])]); continue; }

    const h = line.match(/^#{1,4}\s+(.*)$/);
    if (h) { closeList(); out.push(`<h3>${h[1]}</h3>`); continue; }

    const ul = line.match(/^[-*+]\s+(.*)$/);
    if (ul) {
      if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; }
      out.push(`<li>${ul[1]}</li>`);
      continue;
    }

    const ol = line.match(/^\d+[.)]\s+(.*)$/);
    if (ol) {
      if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; }
      out.push(`<li>${ol[1]}</li>`);
      continue;
    }

    closeList();
    out.push(`<p>${line}</p>`);
  }
  closeList();
  return out.join("");
}

/* 把 text node 里的 [1] 换成可点 span。
   为什么走 TreeWalker 而不是直接对 innerHTML 做 replace：
   直接 replace 会误伤 HTML 属性里的方括号，还会打乱已经生成的标签。 */
function linkifyCitations(root) {
  const re = /\[(\d{1,2})\]/g;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const targets = [];

  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (re.test(node.nodeValue)) targets.push(node);
    re.lastIndex = 0;
  }

  for (const node of targets) {
    const text = node.nodeValue;
    const frag = document.createDocumentFragment();
    let last = 0;
    let m;
    re.lastIndex = 0;

    while ((m = re.exec(text))) {
      if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
      const span = document.createElement("span");
      span.className = "cite";
      span.dataset.i = m[1];
      span.textContent = m[0];
      frag.appendChild(span);
      last = m.index + m[0].length;
    }
    if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
    node.parentNode.replaceChild(frag, node);
  }
}

/* ────────────────────────────── 对话 ────────────────────────────── */

function appendUser(text) {
  const wrap = document.createElement("div");
  wrap.className = "msg msg-user";
  wrap.innerHTML = '<div class="msg-role">你</div>';
  const bubble = document.createElement("div");
  bubble.className = "bubble-user";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  els.messages.appendChild(wrap);
  scrollToBottom(true);
}

function appendBot() {
  const wrap = document.createElement("div");
  wrap.className = "msg";
  wrap.innerHTML = '<div class="msg-role">助手</div>';

  const bubble = document.createElement("div");
  bubble.className = "bubble-bot";
  bubble.innerHTML = '<span class="caret"></span>';  // 先给个光标，表示已收到

  const meta = document.createElement("div");
  meta.className = "msg-meta";

  wrap.appendChild(bubble);
  wrap.appendChild(meta);
  els.messages.appendChild(wrap);
  scrollToBottom(true);

  return { bubble, meta };
}

function setBusy(v) {
  busy = v;
  els.send.disabled = v;
  els.send.querySelector("span").textContent = v ? "生成中" : "发送";
}

function renderSources(list) {
  sources = list || [];
  els.srcCount.textContent = sources.length;
  els.sourceList.innerHTML = "";

  if (!sources.length) {
    els.sourceList.innerHTML =
      '<p class="empty">这次没有命中任何片段（触发了拒答门槛）。</p>';
    return;
  }

  sources.forEach((s) => {
    const card = document.createElement("div");
    card.className = "src";
    card.dataset.i = String(s.index);
    // 分数可能是 rerank 概率(0~1)，也可能是向量余弦；统一按 0~1 画条
    const pct = Math.max(2, Math.min(100, Math.round((s.score || 0) * 100)));
    const pages = (s.pages || []).map((p) => `第${p}页`).join("、") || "未知页";
    card.innerHTML = `
      <div class="src-head">
        <span class="src-idx">${s.index}</span>
        <span class="src-file" title="${escapeHtml(s.filename)}">${escapeHtml(s.filename)}</span>
        <span class="src-score">${(s.score || 0).toFixed(3)}</span>
      </div>
      <p class="src-text">${escapeHtml(s.preview || "")}</p>
      <div class="src-page doc-meta">${pages}</div>
      <div class="bar"><i style="width:${pct}%"></i></div>
    `;
    els.sourceList.appendChild(card);
  });
}

function highlightSource(i) {
  const cards = els.sourceList.querySelectorAll(".src");
  let target = null;
  cards.forEach((c) => {
    const on = c.dataset.i === String(i);
    c.classList.toggle("active", on);
    if (on) target = c;
  });
  if (target) target.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

async function ask(question) {
  if (busy || !question.trim()) return;

  const welcome = els.messages.querySelector(".welcome");
  if (welcome) welcome.remove();

  setBusy(true);
  appendUser(question);
  const { bubble, meta } = appendBot();

  let acc = "";
  let info = null;
  let refused = false;
  let failed = false;
  const t0 = performance.now();

  const paint = () => {
    bubble.innerHTML = renderMarkdown(acc) + '<span class="caret"></span>';
    linkifyCitations(bubble);
    scrollToBottom();
  };

  const onEvent = (evt) => {
    if (evt.type === "meta") {
      info = evt;
      // 闲聊/身份类走短路，后端 top_score 是 null——不要显示成 top 0.000
      meta.textContent = evt.smalltalk
        ? "闲聊/身份类问题 · 未检索"
        : `检索 ${evt.search_seconds}s · top ${Number(evt.top_score).toFixed(3)} / 门槛 ${evt.threshold}`;
    } else if (evt.type === "sources") {
      renderSources(evt.sources);
    } else if (evt.type === "token") {
      acc += evt.text;
      paint();
    } else if (evt.type === "done") {
      refused = !!evt.refused;
      bubble.innerHTML = renderMarkdown(acc || "（空）");
      linkifyCitations(bubble);
      // refused=true 是检索门槛拦的；这里再兜一层：模型按 prompt 规矩
      // 自己说"无法回答"时后端不知道（refused 仍是 false），前端按文本补标黄。
      if (refused || /无法回答|无法根据现有资料/.test(acc)) bubble.classList.add("refused");
      const e = evt.elapsed || {};
      const total = ((performance.now() - t0) / 1000).toFixed(1);
      meta.textContent = info?.smalltalk
        ? `共 ${total}s · 未检索 · 固定话术`
        : refused
        ? `拒答 · top ${Number(info?.top_score || 0).toFixed(3)} < 门槛 ${info?.threshold} · 未调用模型`
        : `检索 ${e.search_seconds ?? "?"}s · 生成 ${e.generate_seconds ?? "?"}s · 共 ${total}s · 模式 ${info?.mode ?? ""}`;
      scrollToBottom();
    } else if (evt.type === "error") {
      failed = true;
      bubble.classList.add("error");
      bubble.textContent = evt.message || "出错了";
      meta.textContent = "请求失败";
    }
  };

  try {
    const resp = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode: els.mode.value, top_k: 5 }),
    });

    if (!resp.ok || !resp.body) {
      const detail = await resp.text().catch(() => "");
      throw new Error(`HTTP ${resp.status} ${detail.slice(0, 200)}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      // SSE：一个事件以空行结尾。可能一次读到半个包，所以要循环切。
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, cut);
        buf = buf.slice(cut + 2);
        for (const line of chunk.split("\n")) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          try {
            onEvent(JSON.parse(payload));
          } catch {
            /* 收到半个 JSON 就丢掉，下个包会补齐 */
          }
        }
      }
    }

    if (!acc && !failed) {
      bubble.classList.add("error");
      bubble.textContent = "服务端没有返回内容，检查后端日志。";
    }
  } catch (err) {
    bubble.classList.add("error");
    bubble.textContent = `请求失败：${err.message}`;
    meta.textContent = "网络或后端异常";
  } finally {
    setBusy(false);
    els.input.focus();
  }
}

/* ────────────────────────────── 文档 ────────────────────────────── */

function renderDocs(docs) {
  els.docCount.textContent = docs.length;
  if (!docs.length) {
    els.docList.innerHTML = '<p class="empty">还没有文档，先传一个 PDF 试试。</p>';
    return;
  }
  els.docList.innerHTML = docs
    .map((d) => {
      const st = d.status || "uploaded";
      const size = d.file_size ? `${(d.file_size / 1024 / 1024).toFixed(1)} MB` : "";
      const tip = d.error_message ? ` title="${escapeHtml(d.error_message)}"` : "";
      return `<div class="doc"${tip}>
        <span class="dot ${st}"></span>
        <div>
          <div class="doc-name">${escapeHtml(d.filename)}</div>
          <div class="doc-meta">${STATUS_TEXT[st] || st}${size ? " · " + size : ""}</div>
        </div>
      </div>`;
    })
    .join("");
}

async function loadDocs() {
  try {
    const r = await fetch("/documents");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    renderDocs(await r.json());
    return true;
  } catch {
    els.docList.innerHTML = '<p class="empty">文档列表拉取失败（Postgres 没起？）</p>';
    return false;
  }
}

/* 上传后状态是后台异步变的，前端只能轮询。
   30 秒内每 2 秒刷一次，看到全部不是"处理中"就停。 */
function pollDocs(duration = 30000) {
  const pending = new Set(["uploaded", "parsing", "embedding"]);
  const deadline = Date.now() + duration;
  const timer = setInterval(async () => {
    const ok = await loadDocs();
    if (!ok || Date.now() > deadline) {
      clearInterval(timer);
      return;
    }
    const still = els.docList.querySelector(".dot.uploaded, .dot.parsing, .dot.embedding");
    if (!still) clearInterval(timer);
  }, 2000);
}

/* ────────────────────────────── 健康检查 ─────────────────────────── */

async function checkHealth() {
  try {
    const r = await fetch("/health");
    const data = await r.json();
    const bad = (data.components || []).filter((c) => !c.ok).map((c) => c.name);
    if (data.status === "ok") {
      els.health.className = "pill ok";
      els.health.textContent = "全部在线";
    } else {
      els.health.className = "pill warn";
      els.health.textContent = `降级：${bad.join("/") || "未知"}`;
      els.health.title = (data.components || [])
        .filter((c) => !c.ok)
        .map((c) => `${c.name}: ${c.detail}`)
        .join("\n");
    }
  } catch {
    els.health.className = "pill bad";
    els.health.textContent = "后端未响应";
  }
}

/* ────────────────────────────── 事件绑定 ────────────────────────── */

els.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = els.input.value.trim();
  if (!q) return;
  els.input.value = "";
  els.input.style.height = "auto";
  ask(q);
});

els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});

els.input.addEventListener("input", () => {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 140) + "px";
});

els.mode.addEventListener("change", () => {
  els.modeHint.textContent = MODE_HINT[els.mode.value] || "";
});

// 示例问题 & 引用点击：都用事件委托，元素重建后不用重新绑
els.messages.addEventListener("click", (e) => {
  const sample = e.target.closest(".sample");
  if (sample) {
    ask(sample.dataset.q || sample.textContent);
    return;
  }
  const cite = e.target.closest(".cite");
  if (cite) highlightSource(cite.dataset.i);
});

els.sourceList.addEventListener("click", (e) => {
  const card = e.target.closest(".src");
  if (!card) return;
  const on = card.classList.contains("active");
  els.sourceList.querySelectorAll(".src").forEach((c) => c.classList.remove("active"));
  if (!on) card.classList.add("active");
});

els.clearBtn.addEventListener("click", () => {
  if (busy) return;
  els.messages.innerHTML = WELCOME_HTML;
  renderSources([]);
  els.sourceList.innerHTML =
    '<p class="empty">提问后这里显示命中的原文片段，回答里的 [1][2] 可以点。</p>';
});

els.file.addEventListener("change", async () => {
  const f = els.file.files && els.file.files[0];
  if (!f) return;

  els.uploadHint.textContent = `上传中：${f.name}`;
  const fd = new FormData();
  fd.append("file", f);

  try {
    const r = await fetch("/documents/upload", { method: "POST", body: fd });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    els.uploadHint.textContent = `已接收，后台解析入库中：${data.filename}`;
    pollDocs();
  } catch (err) {
    els.uploadHint.textContent = `上传失败：${err.message}`;
  } finally {
    els.file.value = "";
  }
});

/* ────────────────────────────── 启动 ────────────────────────────── */

checkHealth();
loadDocs();
els.input.focus();
setInterval(checkHealth, 30000);
