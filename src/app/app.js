const form = document.querySelector("#chat-form");
const input = document.querySelector("#question");
const messages = document.querySelector("#messages");
const sendButton = document.querySelector("#send-button");
const clearButton = document.querySelector("#clear-chat");

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function addUserMessage(text) {
  const article = document.createElement("article");
  article.className = "message user-message";
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "Bạn";
  const body = document.createElement("div");
  body.className = "message-body";
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = "Bạn";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  body.append(label, bubble);
  article.append(avatar, body);
  messages.append(article);
  scrollToBottom();
}

function addLoadingMessage() {
  const article = document.createElement("article");
  article.className = "message assistant-message";
  article.dataset.loading = "true";
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "AI";
  const body = document.createElement("div");
  body.className = "message-body";
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = "Đang tìm bằng chứng...";
  const bubble = document.createElement("div");
  bubble.className = "bubble typing";
  bubble.setAttribute("aria-label", "Đang xử lý");
  for (let i = 0; i < 3; i += 1) bubble.append(document.createElement("span"));
  body.append(label, bubble);
  article.append(avatar, body);
  messages.append(article);
  scrollToBottom();
  return article;
}

function addAssistantMessage(result) {
  const article = document.createElement("article");
  article.className = "message assistant-message";
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "AI";
  const body = document.createElement("div");
  body.className = "message-body";
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = "Trợ lý";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = result.answer || "Không nhận được câu trả lời.";
  body.append(label, bubble);

  const status = document.createElement("div");
  status.className = `message-status ${result.status || "error"}`;
  status.textContent = result.status === "answered"
    ? "Đã trả lời có citation"
    : result.status === "abstain"
      ? "Chưa đủ bằng chứng"
      : "Có lỗi khi xử lý";
  body.append(status);

  if (Array.isArray(result.sources) && result.sources.length) {
    const sourceList = document.createElement("div");
    sourceList.className = "sources";
    for (const source of result.sources) {
      const link = document.createElement("a");
      link.className = "source-card";
      link.href = `/api/pdf/${encodeURIComponent(source.source)}#page=${source.page}`;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      const title = document.createElement("span");
      title.className = "source-title";
      title.textContent = `${source.source} · trang ${source.page} · ${source.id}`;
      const preview = document.createElement("span");
      preview.className = "source-preview";
      preview.textContent = source.text;
      link.append(title, preview);
      sourceList.append(link);
    }
    body.append(sourceList);
  }

  if (result.details) {
    const details = document.createElement("div");
    details.className = "trace-note";
    const methods = Array.isArray(result.details.methods) ? result.details.methods.join(" → ") : "";
    const duration = Number.isFinite(result.details.elapsed_ms)
      ? `${(result.details.elapsed_ms / 1000).toFixed(1)} giây`
      : "";
    details.textContent = `Tìm ${result.details.search_count ?? 0} lượt${methods ? ` · ${methods}` : ""}${duration ? ` · ${duration}` : ""}`;
    body.append(details);
  }

  article.append(avatar, body);
  messages.append(article);
  scrollToBottom();
}

async function sendQuestion(question) {
  sendButton.disabled = true;
  const loading = addLoadingMessage();
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const result = await response.json();
    loading.remove();
    if (!response.ok) {
      addAssistantMessage({ status: "error", answer: result.error || "Không thể kết nối Agent." });
      return;
    }
    addAssistantMessage(result);
  } catch (error) {
    loading.remove();
    addAssistantMessage({ status: "error", answer: "Không kết nối được backend. Kiểm tra cửa sổ chạy Python." });
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || sendButton.disabled) return;
  addUserMessage(question);
  input.value = "";
  input.style.height = "auto";
  sendQuestion(question);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
});

clearButton.addEventListener("click", () => {
  messages.replaceChildren();
  const article = document.createElement("article");
  article.className = "message assistant-message welcome-message";
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "AI";
  const body = document.createElement("div");
  body.className = "message-body";
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = "Trợ lý";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = "Hội thoại đã được làm mới. Hãy nhập câu hỏi về luận văn.";
  body.append(label, bubble);
  article.append(avatar, body);
  messages.append(article);
  input.focus();
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.textContent;
    form.requestSubmit();
  });
});
