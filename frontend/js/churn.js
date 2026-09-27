/* ================================================
   churn.js — Customer Churn Predictor UI
   ================================================ */

function updateMonthlyCharge(val) {
  document.getElementById('mc-val').textContent = `$${parseFloat(val).toFixed(2)}`;
}

function updateTotalCharge(val) {
  document.getElementById('tc-val').textContent = `$${parseFloat(val).toFixed(2)}`;
}

async function predictChurn() {
  hideError('churn-error');
  setLoading('churn-btn', true);

  const payload = {
    tenure:          parseInt(document.getElementById('tenure').value),
    monthly_charges: parseFloat(document.getElementById('monthly_charges').value),
    total_charges:   parseFloat(document.getElementById('total_charges').value),
    num_products:    parseInt(document.getElementById('num_products').value),
    has_internet:    parseInt(document.getElementById('has_internet').value),
    contract_type:   parseInt(document.getElementById('contract_type').value),
    tech_support:    parseInt(document.getElementById('tech_support').value),
    payment_method:  parseInt(document.getElementById('payment_method').value),
  };

  try {
    const res  = await fetch(`${API_BASE}/api/churn/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderChurnResult(data);
  } catch (e) {
    showError('churn-error', 'churn-error-msg', e.message || 'Failed to reach backend. Is the server running?');
  } finally {
    setLoading('churn-btn', false);
  }
}

function renderChurnResult(data) {
  const willChurn = data.prediction === 'Will Churn';
  const emoji = willChurn ? '⚠️' : '✅';
  const color = willChurn ? 'var(--red)' : 'var(--green)';

  document.getElementById('churn-emoji').textContent     = emoji;
  document.getElementById('churn-prediction').textContent = data.prediction;
  document.getElementById('churn-prediction').style.background   = color;
  document.getElementById('churn-prediction').style.webkitBackgroundClip = 'unset';
  document.getElementById('churn-prediction').style.webkitTextFillColor  = color;

  // Risk badge
  const riskClass = { Low: 'risk-low', Medium: 'risk-medium', High: 'risk-high' }[data.risk_level] || 'risk-low';
  const riskIcon  = { Low: '🟢', Medium: '🟡', High: '🔴' }[data.risk_level] || '🟢';
  document.getElementById('churn-risk-badge').innerHTML = `
    <span class="risk-badge ${riskClass}">${riskIcon} ${data.risk_level} Risk · ${data.churn_probability.toFixed(1)}% churn probability</span>`;

  // Probability bars
  renderProbBars('churn-prob-bars', {
    'Will Stay':  data.stay_probability,
    'Will Churn': data.churn_probability,
  }, [
    'linear-gradient(135deg,#22c55e,#06b6d4)',
    'linear-gradient(135deg,#ef4444,#ec4899)',
  ]);

  renderImportanceBars('churn-importance', data.feature_importance);
  showResult('churn-result');
}
