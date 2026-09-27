/* ================================================
   app.js — Router, navigation, API health check
   ================================================ */

const API_BASE = 'http://localhost:8000';

// ── Navigation ──────────────────────────────────
const PAGE_META = {
  overview:  { title: '📊 Overview Dashboard',     sub: 'Select a model to start predicting' },
  iris:      { title: '🌸 Iris Classifier',         sub: 'Random Forest · Multi-class classification' },
  house:     { title: '🏡 House Price Predictor',   sub: 'Gradient Boosting · Regression' },
  churn:     { title: '👥 Customer Churn Predictor',sub: 'Random Forest · Binary classification' },
  sentiment: { title: '💬 Sentiment Analyzer',      sub: 'Naive Bayes + TF-IDF · NLP' },
};

function navigate(page) {
  // Hide all pages
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  // Show target
  const pageEl = document.getElementById(`page-${page}`);
  if (pageEl) pageEl.classList.add('active');

  const navEl = document.getElementById(`nav-${page}`);
  if (navEl) navEl.classList.add('active');

  // Update topbar
  const meta = PAGE_META[page] || {};
  document.getElementById('topbar-title').textContent = meta.title || page;
  document.getElementById('topbar-sub').textContent   = meta.sub   || '';
}

// Wire nav clicks
document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', e => {
    e.preventDefault();
    navigate(item.dataset.page);
  });
});

// ── API Health Check ─────────────────────────────
async function checkApiStatus() {
  const pill = document.getElementById('api-pill');
  const pillText = document.getElementById('api-pill-text');
  const sidebar  = document.getElementById('sidebar-status');
  const sidebarText = document.getElementById('sidebar-status-text');

  try {
    const res = await fetch(`${API_BASE}/`, { signal: AbortSignal.timeout(4000) });
    if (res.ok) {
      pill.className = 'api-status-pill';
      pillText.textContent = 'API Online';
      sidebar.className = 'status-badge';
      sidebarText.textContent = 'API Connected';
    } else {
      throw new Error('Non-OK response');
    }
  } catch {
    pill.className = 'api-status-pill offline';
    pillText.textContent = 'API Offline';
    sidebar.className = 'status-badge offline';
    sidebarText.textContent = 'API Offline';
  }
}

checkApiStatus();
setInterval(checkApiStatus, 15000);

// ── Shared Utilities ─────────────────────────────

/** Update a range hint label */
function updateRangeVal(id, value, unit) {
  const el = document.getElementById(id);
  if (el) el.textContent = `${parseFloat(value).toFixed(unit === 'cm' || unit === '' ? 1 : 0)} ${unit}`.trim();
}

/** Render probability bars inside a container element */
function renderProbBars(containerId, probabilities, gradients) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  const defaultGrad = ['var(--grad-main)', 'linear-gradient(135deg,#06b6d4,#22c55e)', 'linear-gradient(135deg,#f59e0b,#f97316)'];

  container.innerHTML = entries.map(([name, val], i) => `
    <div class="prob-bar-wrapper">
      <div class="prob-bar-header">
        <span class="prob-bar-name">${name}</span>
        <span class="prob-bar-val">${val.toFixed(1)}%</span>
      </div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="background:${(gradients||defaultGrad)[i]||defaultGrad[0]}"
             data-target="${val}"></div>
      </div>
    </div>
  `).join('');

  // Animate after a tick
  setTimeout(() => {
    container.querySelectorAll('.prob-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.target + '%';
    });
  }, 60);
}

/** Render feature importance bars */
function renderImportanceBars(containerId, importance) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const entries = Object.entries(importance).sort((a, b) => b[1] - a[1]).slice(0, 8);
  container.innerHTML = entries.map(([name, val]) => `
    <div class="fi-bar-wrapper">
      <div class="fi-bar-header">
        <span>${name}</span>
        <span>${val.toFixed(1)}%</span>
      </div>
      <div class="fi-bar-track">
        <div class="fi-bar-fill" data-target="${val}"></div>
      </div>
    </div>
  `).join('');
  setTimeout(() => {
    container.querySelectorAll('.fi-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.target + '%';
    });
  }, 60);
}

/** Show/hide loading on button */
function setLoading(btnId, loading) {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  if (loading) {
    btn.dataset.originalText = btn.innerHTML;
    btn.innerHTML = '<div class="spinner"></div> Predicting…';
    btn.disabled = true;
  } else {
    btn.innerHTML = btn.dataset.originalText || 'Predict';
    btn.disabled = false;
  }
}

/** Show error banner */
function showError(bannerId, msgId, msg) {
  document.getElementById(bannerId).classList.add('visible');
  document.getElementById(msgId).textContent = msg;
}

/** Hide error banner */
function hideError(bannerId) {
  document.getElementById(bannerId).classList.remove('visible');
}

/** Show result panel */
function showResult(panelId) {
  document.getElementById(panelId).classList.add('visible');
}

/** Confidence ring SVG */
function buildConfidenceRing(pct, color) {
  const R = 50, CX = 60, CY = 60, STROKE = 8;
  const circ = 2 * Math.PI * R;
  const dash = (pct / 100) * circ;
  return `
    <div class="confidence-ring-wrapper">
      <div class="confidence-ring">
        <svg width="120" height="120" viewBox="0 0 120 120">
          <circle cx="${CX}" cy="${CY}" r="${R}" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="${STROKE}"/>
          <circle cx="${CX}" cy="${CY}" r="${R}" fill="none" stroke="${color}" stroke-width="${STROKE}"
            stroke-dasharray="${dash} ${circ}" stroke-linecap="round"
            style="transition:stroke-dasharray 1s cubic-bezier(0.4,0,0.2,1)"/>
        </svg>
        <div class="confidence-ring-text">
          <span>${pct.toFixed(0)}%</span>
          <span class="confidence-ring-label">conf.</span>
        </div>
      </div>
    </div>`;
}
