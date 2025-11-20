// ----------------------------------------------
// LEVEL 1 – UI UPGRADE + Typing Animation
// ----------------------------------------------

document.addEventListener("DOMContentLoaded", () => {

  console.log("🔥 script.js loaded (with UI Level 1)");

  const input = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const chat = document.getElementById("chat-messages");

  // ---------------- Message Rendering ----------------
  const appendMessage = (html, cls) => {
    const el = document.createElement("div");
    el.className = cls + " fade-in";
    el.innerHTML = html;
    chat.appendChild(el);
    chat.scrollTop = chat.scrollHeight;
    return el;
  };

  // ---------------- Markdown Cleaner ----------------
  const cleanText = (text) => {
    return text
      .replace(/\*\*/g, "")
      .replace(/^(\*|-|•)\s+/gm, "")
      .trim();
  };

  const markdownToHtml = (text) => {
    const esc = (s) =>
      s.replace(/&/g, "&amp;")
       .replace(/</g, "&lt;")
       .replace(/>/g, "&gt;");

    let t = esc(cleanText(text));
    const lines = t.split(/\r?\n/).filter(line => line.trim() !== "");
    return lines.map(line => `<p>${line}</p>`).join("");
  };

  // --------------- Typing Animation ------------------
  const showLoading = () => {
    const el = document.createElement("div");
    el.className = "bot-message loading";
    el.innerHTML = `
      <span class="typing">
        <span class="dot"></span>
        <span class="dot"></span>
        <span class="dot"></span>
      </span>
      <span class="sr-only">AI is thinking</span>
    `;
    chat.appendChild(el);
    chat.scrollTop = chat.scrollHeight;
    return el;
  };

  // ---------------- Send Message ---------------------
  const sendMessage = async () => {
    const prompt = input.value.trim();
    if (!prompt) return;

    console.log("🔥 Sending:", prompt);

    appendMessage(markdownToHtml(prompt), "user-message");
    input.value = "";

  // show loading animation (three bouncing dots)
  const typingEl = showLoading();
  // disable send while awaiting response to prevent duplicate sends
  sendBtn.disabled = true;

    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });

      const data = await res.json();
  // Remove loading animation and show message with slow typing effect
  typingEl.remove();
  sendBtn.disabled = false;
      const safeHtml = markdownToHtml(data.response || "No response");

      // Create bot bubble and type text slowly (strip tags for typing)
      const temp = document.createElement('div');
      temp.innerHTML = safeHtml;
      const plain = temp.textContent || temp.innerText || "";

      const botEl = document.createElement('div');
      botEl.className = 'bot-message bot-typing fade-in';
      chat.appendChild(botEl);

      // slow typing
      let i = 0;
      const speed = 18; // ms per char (adjust for effect)
      const typer = setInterval(() => {
        i += 1;
        botEl.textContent = plain.slice(0, i);
        chat.scrollTop = chat.scrollHeight;
        if (i >= plain.length) {
          clearInterval(typer);
          // after typing, convert plain text back to formatted HTML
          botEl.innerHTML = safeHtml;
          chat.scrollTop = chat.scrollHeight;
        }
      }, speed);

    } catch (err) {
      console.error("🔥 FETCH ERROR:", err);
      typingEl.remove();
      sendBtn.disabled = false;
      appendMessage("⚠️ Network error. Check server logs.", "bot-message");
    }
  };

  // -------------- Event Listeners --------------------
  sendBtn.addEventListener("click", sendMessage);

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  input.focus();

  // --------------- Health Modal Handlers -----------------
  const healthBtn = document.getElementById('health-btn');
  const modal = document.getElementById('health-modal');
  const form = document.getElementById('health-form');
  const submit = document.getElementById('health-submit');

  if (healthBtn && modal && form) {
    console.log("🔥 Health modal initialized");
    // Ensure modal starts hidden using the .hidden class
    modal.classList.add('hidden');
    const closeModal = () => { 
      console.log("🔥 Closing modal");
      modal.classList.add('hidden'); 
    };
    const openModal = () => { 
      console.log("🔥 Opening modal");
      modal.classList.remove('hidden'); 
    };

    // Open modal when Health Plan button is clicked
    healthBtn.addEventListener('click', (e) => {
      e.preventDefault();
      console.log("🔥 Health button clicked");
      openModal();
    });

    // Close modal when close button is clicked (event delegation)
    document.addEventListener('click', (e) => {
      if (e.target && e.target.id === 'health-close') {
        e.preventDefault();
        console.log("🔥 Close button clicked");
        closeModal();
      }
    });

    // Close modal when clicking on modal backdrop
    modal.addEventListener('click', (e) => { 
      if (e.target === modal) {
        console.log("🔥 Modal backdrop clicked");
        closeModal(); 
      }
    });

    form.addEventListener('submit', async (ev) => {
      ev.preventDefault();
      submit.disabled = true;

      const age = parseInt(document.getElementById('age').value || 0, 10);
      const sex = document.getElementById('sex').value || '';
      const weight = parseFloat(document.getElementById('weight').value || 0);
      const weightUnit = document.getElementById('weight-unit').value || 'kg';
      const height = parseFloat(document.getElementById('height').value || 0);
      const heightUnit = document.getElementById('height-unit').value || 'cm';
      const activity = document.getElementById('activity').value || 'moderate';
      const goal = document.getElementById('goal').value || 'maintain';

      // show user message in chat and close modal
      const userText = `Age: ${age}, Weight: ${weight}${weightUnit}, Height: ${height}${heightUnit}, Goal: ${goal}`;
      appendMessage(userText, 'user-message');
      closeModal();

      // show loading using the closure-scoped showLoading
      const loadingEl = showLoading();

      try {
        const payload = { age, sex, weight, weight_unit: weightUnit, height, height_unit: heightUnit, activity, goal };
        const res = await fetch('/health_plan', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
        });
        // handle non-JSON or error responses more gracefully
        let data = null;
        if (!res.ok) {
          const txt = await res.text();
          console.error('Health API returned error:', res.status, txt);
          loadingEl.remove();
          appendMessage(`⚠️ Server error (${res.status}): ${txt}`, 'bot-message');
          submit.disabled = false;
          return;
        }
        try {
          data = await res.json();
        } catch (parseErr) {
          const txt = await res.text();
          console.error('Failed to parse JSON from /health_plan:', parseErr, txt);
          loadingEl.remove();
          appendMessage('⚠️ Invalid response from server.', 'bot-message');
          submit.disabled = false;
          return;
        }
        loadingEl.remove();

        console.log('Health API response:', data);
        const safeHtml = markdownToHtml(data.response || 'No response');
        // show bot response with slow typing like sendMessage
        const temp = document.createElement('div'); temp.innerHTML = safeHtml;
        const plain = temp.textContent || temp.innerText || '';
        const botEl = document.createElement('div'); botEl.className = 'bot-message bot-typing fade-in'; chat.appendChild(botEl);
        let i = 0; const speed = 16;
        const typer = setInterval(() => {
          i += 1; botEl.textContent = plain.slice(0, i); chat.scrollTop = chat.scrollHeight;
          if (i >= plain.length) { clearInterval(typer); botEl.innerHTML = safeHtml; chat.scrollTop = chat.scrollHeight; }
        }, speed);

      } catch (err) {
        console.error('Health submit error', err);
        loadingEl.remove();
        appendMessage('⚠️ Network error. Could not fetch health plan.', 'bot-message');
      } finally {
        submit.disabled = false;
      }
    });
  }

  // --------------- Voice Input / Output (Step 7) -----------------
  const micBtn = document.getElementById('mic-btn');
  const voiceStatus = document.getElementById('voice-status');
  const voiceStatusText = document.getElementById('voice-status-text');
  
  // Initialize Web Speech API (speech recognition)
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition = null;
  let isListening = false;

  if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      console.log("🎤 Speech recognition started");
      isListening = true;
      micBtn.classList.add('active');
      voiceStatus.classList.remove('hidden');
      voiceStatusText.textContent = 'Listening...';
    };

    recognition.onresult = (event) => {
      let interimTranscript = '';
      let finalTranscript = '';
      
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript + ' ';
        } else {
          interimTranscript += transcript;
        }
      }

      if (finalTranscript) {
        input.value = finalTranscript.trim();
        console.log("🎤 Recognized:", finalTranscript);
      }
    };

    recognition.onend = () => {
      console.log("🎤 Speech recognition ended");
      isListening = false;
      micBtn.classList.remove('active');
      voiceStatus.classList.add('hidden');
    };

    recognition.onerror = (event) => {
      console.error("🎤 Speech recognition error:", event.error);
      voiceStatusText.textContent = `Error: ${event.error}`;
    };

    // Mic button click handler
    micBtn.addEventListener('click', () => {
      if (isListening) {
        recognition.stop();
      } else {
        input.value = '';
        recognition.start();
      }
    });
  } else {
    console.warn("⚠️ Speech Recognition API not supported");
    micBtn.disabled = true;
    micBtn.title = "Speech Recognition not supported in your browser";
  }

  // Text-to-Speech function (reads bot responses)
  const speakText = (text) => {
    if ('speechSynthesis' in window) {
      // Cancel any ongoing speech
      speechSynthesis.cancel();
      
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.95; // slightly slower for clarity
      utterance.pitch = 1;
      utterance.volume = 1;

      utterance.onstart = () => console.log("🔊 Speaking...");
      utterance.onend = () => console.log("🔊 Speech finished");
      utterance.onerror = (e) => console.error("🔊 Speech error:", e);

      speechSynthesis.speak(utterance);
    }
  };

  // Hook text-to-speech into bot responses
  // Store the original appendMessage to wrap it
  const originalAppendMessage = appendMessage;
  const wrappedAppendMessage = (html, cls) => {
    const el = originalAppendMessage(html, cls);
    
    // Auto-speak bot messages (but not loading/typing indicators)
    if (cls === 'bot-message' && !el.classList.contains('loading') && !el.classList.contains('bot-typing')) {
      // Extract plain text and speak after a short delay
      const plainText = el.textContent || '';
      if (plainText.trim()) {
        setTimeout(() => speakText(plainText), 500);
      }
    }
    
    return el;
  };

  // Replace appendMessage with the wrapped version in the closure
  window.wrappedAppendMessage = wrappedAppendMessage;
  
  // Update the sendMessage function to use wrapped version for bot responses
  const originalSendMessage = sendMessage;
  const enhancedSendMessage = async () => {
    const prompt = input.value.trim();
    if (!prompt) return;

    console.log("🔥 Sending:", prompt);
    appendMessage(markdownToHtml(prompt), "user-message");
    input.value = "";

    const typingEl = showLoading();
    sendBtn.disabled = true;

    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });

      const data = await res.json();
      typingEl.remove();
      sendBtn.disabled = false;
      const safeHtml = markdownToHtml(data.response || "No response");

      const temp = document.createElement('div');
      temp.innerHTML = safeHtml;
      const plain = temp.textContent || temp.innerText || "";

      const botEl = document.createElement('div');
      botEl.className = 'bot-message bot-typing fade-in';
      chat.appendChild(botEl);

      let i = 0;
      const speed = 18;
      const typer = setInterval(() => {
        i += 1;
        botEl.textContent = plain.slice(0, i);
        chat.scrollTop = chat.scrollHeight;
        if (i >= plain.length) {
          clearInterval(typer);
          botEl.innerHTML = safeHtml;
          chat.scrollTop = chat.scrollHeight;
          // Speak the bot response after typing completes
          speakText(plain);
        }
      }, speed);

    } catch (err) {
      console.error("🔥 FETCH ERROR:", err);
      typingEl.remove();
      sendBtn.disabled = false;
      appendMessage("⚠️ Network error. Check server logs.", "bot-message");
    }
  };

  // Replace sendMessage with enhanced version
  sendBtn.removeEventListener("click", sendMessage);
  sendBtn.addEventListener("click", enhancedSendMessage);

  input.removeEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      enhancedSendMessage();
    }
  });
});
