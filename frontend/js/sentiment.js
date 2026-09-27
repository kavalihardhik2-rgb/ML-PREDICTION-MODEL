/* ================================================
   sentiment.js — Sentiment Analyzer UI
   ================================================ */

function updateCharCounter(textarea) {
  document.getElementById('char-counter').textContent =
    `${textarea.value.length} / 2000 chars`;
}

function setSentimentPreset(text) {
  const ta = document.getElementById('sentiment_text');
  ta.value = text;
  updateCharCounter(ta);
}

async function predictSentiment() {
  hideError('sentiment-error');
  setLoading('sentiment-btn', true);

  const text = document.getElementById('sentiment_text').value.trim();
  if (!text) {
    showError('sentiment-error', 'sentiment-error-msg', 'Please enter some text to analyze.');
    setLoading('sentiment-btn', false);
    return;
  }

  try {
    const res  = await fetch(`${API_BASE}/api/sentiment/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderSentimentResult(data);
  } catch (e) {
    showError('sentiment-error', 'sentiment-error-msg', e.message || 'Failed to reach backend. Is the server running?');
  } finally {
    setLoading('sentiment-btn', false);
  }
}

const SENTIMENT_GRADIENTS = {
  positive: [
    'linear-gradient(135deg,#22c55e,#06b6d4)',
    'linear-gradient(135deg,#ef4444,#ec4899)',
    'linear-gradient(135deg,#f59e0b,#f97316)',
  ],
  negative: [
    'linear-gradient(135deg,#ef4444,#ec4899)',
    'linear-gradient(135deg,#22c55e,#06b6d4)',
    'linear-gradient(135deg,#f59e0b,#f97316)',
  ],
  neutral: [
    'linear-gradient(135deg,#f59e0b,#f97316)',
    'linear-gradient(135deg,#22c55e,#06b6d4)',
    'linear-gradient(135deg,#ef4444,#ec4899)',
  ],
};

function renderSentimentResult(data) {
  const sentimentClass = `sentiment-${data.sentiment}`;

  document.getElementById('sentiment-emoji').textContent   = data.emoji;
  document.getElementById('sentiment-label').textContent   = data.sentiment.charAt(0).toUpperCase() + data.sentiment.slice(1);
  document.getElementById('sentiment-conf-text').textContent = `Confidence: ${data.confidence.toFixed(1)}%`;

  // Apply sentiment colour class to result hero
  const hero = document.getElementById('sentiment-result-hero');
  hero.className = `result-hero ${sentimentClass}`;

  // Probability bars — sort by probability desc
  const gradients = SENTIMENT_GRADIENTS[data.sentiment] || SENTIMENT_GRADIENTS.positive;
  const sorted = Object.fromEntries(
    Object.entries(data.probabilities).sort((a, b) => b[1] - a[1])
  );
  renderProbBars('sentiment-prob-bars', sorted, gradients);

  // Stats
  const avgWordLen = data.char_count > 0 && data.word_count > 0
    ? (data.char_count / data.word_count).toFixed(1)
    : '0';
  document.getElementById('sentiment-stats').innerHTML = `
    <div class="stat-card">
      <div class="stat-num">${data.word_count}</div>
      <div class="stat-name">Words</div>
    </div>
    <div class="stat-card">
      <div class="stat-num">${data.char_count}</div>
      <div class="stat-name">Characters</div>
    </div>
    <div class="stat-card">
      <div class="stat-num">${avgWordLen}</div>
      <div class="stat-name">Avg Word Len</div>
    </div>
    <div class="stat-card">
      <div class="stat-num">${data.confidence.toFixed(0)}%</div>
      <div class="stat-name">Confidence</div>
    </div>
  `;

  showResult('sentiment-result');
}
