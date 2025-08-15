document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const chat = document.getElementById("chat-messages");

  const appendMessage = (html, cls) => {
    const el = document.createElement("div");
    el.className = cls;
    el.innerHTML = html;
    chat.appendChild(el);
    chat.scrollTop = chat.scrollHeight;
    return el;
  };

  const cleanText = (text) => {
    // Remove asterisks or markdown bullets
    return text
      .replace(/\*\*/g, "")          // remove bold markers
      .replace(/^(\*|-|•)\s+/gm, "") // remove bullet chars
      .trim();
  };

  const markdownToHtml = (text) => {
    const esc = (s) => s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    let t = esc(cleanText(text));

    // simple paragraph formatting
    const lines = t.split(/\r?\n/).filter(line => line.trim() !== "");
    return lines.map(line => `<p>${line}</p>`).join("");
  };

  const showLoading = () => {
const el = document.createElement("div");
  el.className = "bot-message loading";
  el.textContent = "Responding...";
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
  };

  const sendMessage = async () => {
    const prompt = input.value.trim();
    if (!prompt) return;

    appendMessage(markdownToHtml(prompt), "user-message");
    input.value = "";

    const loading = showLoading();

    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt })
      });
      const data = await res.json();

      loading.remove();
      const safeHtml = markdownToHtml(data.response || "No response.");
      appendMessage(safeHtml, "bot-message");
    } catch (err) {
      loading.remove();
      appendMessage("⚠️ Network error. Please try again.", "bot-message");
      console.error(err);
    }
  };

  sendBtn.addEventListener("click", sendMessage);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault(); // prevent newline
      sendMessage();
    }
  });

  input.focus();
});
