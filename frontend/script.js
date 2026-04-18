/**
 * script.js — SentiCore Frontend
 * --------------------------------
 * Handles:
 *  - Character count tracking
 *  - API calls to Flask backend (/predict, /accuracy)
 *  - Rendering result card with animated confidence bar
 *  - Example sentence chips
 *  - Error display
 */

// ─── Config ───────────────────────────────────────────────────
const API_BASE = "http://127.0.0.1:5002";   // change if deployed elsewhere

// ─── Example sentences ────────────────────────────────────────
const EXAMPLES = [
  {
    text: "This movie was absolutely breathtaking! The performances were outstanding and the story left me in tears of joy.",
    label: "positive",
  },
  {
    text: "An incredible masterpiece. The director perfectly captured the essence of human emotion.",
    label: "positive",
  },
  {
    text: "Best film I have seen in years. Highly recommend to everyone!",
    label: "positive",
  },
  {
    text: "Terrible acting and a predictable plot. I wasted two hours of my life.",
    label: "negative",
  },
  {
    text: "Extremely disappointing. The special effects were cheap and the dialogue was cringe-worthy.",
    label: "negative",
  },
  {
    text: "I regret buying this ticket. The film was boring, incoherent, and painfully long.",
    label: "negative",
  },
  {
    text: "A hidden gem! The cinematography was stunning and the soundtrack was hauntingly beautiful.",
    label: "positive",
  },
  {
    text: "Worst sequel ever made. It completely destroyed the legacy of the original.",
    label: "negative",
  },
];

// ─── DOM helpers ──────────────────────────────────────────────
const $id  = (id) => document.getElementById(id);
const show = (el) => { el.style.display = ""; };
const hide = (el) => { el.style.display = "none"; };

// ─── On load ──────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  setupCharCount();
  renderExamples();
  fetchAccuracy();
  setupEnterKey();
});

// ─── Character counter ────────────────────────────────────────
function setupCharCount() {
  const textarea = $id("inputText");
  const counter  = $id("charCount");

  textarea.addEventListener("input", () => {
    counter.textContent = textarea.value.length;
  });
}

// ─── Enter-to-submit (Ctrl+Enter) ─────────────────────────────
function setupEnterKey() {
  $id("inputText").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) {
      analyseText();
    }
  });
}

// ─── Fetch model accuracy on load ────────────────────────────
async function fetchAccuracy() {
  try {
    const res  = await fetch(`${API_BASE}/accuracy`);
    const data = await res.json();

    const acc = data.accuracy + "%";
    $id("heroAccuracy").textContent  = acc;
    $id("navAccuracy").textContent   = "Accuracy: " + acc;
  } catch (_) {
    $id("navAccuracy").textContent = "Server offline";
  }
}

// ─── Render example chips ────────────────────────────────────
function renderExamples() {
  const grid = $id("exampleGrid");
  grid.innerHTML = "";

  EXAMPLES.forEach(({ text, label }) => {
    const isPos = label === "positive";
    const chip  = document.createElement("div");
    chip.className = "example-chip";
    chip.title = "Click to analyse this example";

    chip.innerHTML = `
      <div class="example-chip__inner">
        <span class="example-chip__dot example-chip__dot--${isPos ? "pos" : "neg"}"></span>
        <div>
          <div class="example-chip__text">${text}</div>
          <div class="example-chip__tag example-chip__tag--${isPos ? "pos" : "neg"}">
            ${isPos ? "✦ Positive" : "✕ Negative"}
          </div>
        </div>
      </div>
    `;

    chip.addEventListener("click", () => {
      $id("inputText").value = text;
      $id("charCount").textContent = text.length;
      // scroll to analyser
      document.getElementById("analyser").scrollIntoView({ behavior: "smooth" });
      // short delay then auto-analyse for delight
      setTimeout(analyseText, 400);
    });

    grid.appendChild(chip);
  });
}

// ─── Main analysis function ───────────────────────────────────
async function analyseText() {
  const text = $id("inputText").value.trim();

  // Validate
  if (!text) {
    showError("Please enter some text before analysing.");
    return;
  }
  if (text.split(" ").length < 2) {
    showError("Please enter at least a couple of words for an accurate prediction.");
    return;
  }

  // UI: loading state
  hideResults();
  setLoading(true);

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || `Server error ${res.status}`);
    }

    const data = await res.json();
    renderResult(data, text);
  } catch (err) {
    if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
      showError("Cannot reach the Flask server. Make sure you ran  python app.py  first.");
    } else {
      showError("Error: " + err.message);
    }
  } finally {
    setLoading(false);
  }
}

// ─── Render prediction result ─────────────────────────────────
function renderResult({ sentiment, confidence }, originalText) {
  const isPos   = sentiment === "Positive";
  const pct     = Math.round(confidence * 100);
  const barId   = "confidenceBar";
  const pctId   = "confidencePct";
  const card    = $id("resultCard");

  // Header badge
  $id("resultHeader").innerHTML = `
    <i class="fa-solid fa-${isPos ? "face-smile" : "face-frown"}"></i>
    Sentiment Analysis Result
    <span class="badge-${isPos ? "pos" : "neg"}">${sentiment}</span>
  `;

  // Sentiment label
  $id("resultSentiment").innerHTML =
    `<span class="sentiment--${isPos ? "positive" : "negative"}">${sentiment}</span>`;

  // Confidence bar
  const bar = $id(barId);
  bar.className = `confidence-bar confidence-bar--${isPos ? "pos" : "neg"}`;
  bar.style.width = "0%";   // reset for animation

  // Percentage text
  $id(pctId).textContent = pct + "%";
  $id(pctId).style.color = isPos ? "var(--teal)" : "var(--red)";

  // Quoted input snippet
  const snippet = originalText.length > 90
    ? originalText.substring(0, 90) + "…"
    : originalText;
  $id("resultQuote").textContent = `"${snippet}"`;

  show(card);
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });

  // Animate bar after brief paint delay
  requestAnimationFrame(() => {
    setTimeout(() => { bar.style.width = pct + "%"; }, 60);
  });
}

// ─── Show error ───────────────────────────────────────────────
function showError(msg) {
  const el = $id("errorCard");
  $id("errorMsg").textContent = msg;
  show(el);
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ─── Hide all result elements ─────────────────────────────────
function hideResults() {
  hide($id("resultCard"));
  hide($id("errorCard"));
}

// ─── Toggle loading state on button ───────────────────────────
function setLoading(on) {
  const btn     = $id("analyseBtn");
  const btnText = btn.querySelector(".sc-btn__text");
  const loader  = btn.querySelector(".sc-btn__loader");

  btn.disabled = on;

  if (on) {
    hide(btnText);
    show(loader);
  } else {
    show(btnText);
    hide(loader);
  }
}
