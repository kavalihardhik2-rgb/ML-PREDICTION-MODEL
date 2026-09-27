/* ================================================
   house.js — House Price Predictor UI
   ================================================ */

async function predictHouse() {
  hideError('house-error');
  setLoading('house-btn', true);

  const payload = {
    med_inc:    parseFloat(document.getElementById('med_inc').value),
    house_age:  parseFloat(document.getElementById('house_age').value),
    ave_rooms:  parseFloat(document.getElementById('ave_rooms').value),
    ave_bedrms: parseFloat(document.getElementById('ave_bedrms').value),
    population: parseFloat(document.getElementById('population').value),
    ave_occup:  parseFloat(document.getElementById('ave_occup').value),
    latitude:   parseFloat(document.getElementById('latitude').value),
    longitude:  parseFloat(document.getElementById('longitude').value),
  };

  try {
    const res  = await fetch(`${API_BASE}/api/house/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderHouseResult(data);
  } catch (e) {
    showError('house-error', 'house-error-msg', e.message || 'Failed to reach backend. Is the server running?');
  } finally {
    setLoading('house-btn', false);
  }
}

function renderHouseResult(data) {
  document.getElementById('house-price').textContent = data.predicted_price_formatted;
  renderImportanceBars('house-importance', data.feature_importance);
  showResult('house-result');
}

// Range label helpers specific to house page
function updateHouseRange(id, labelId, value, prefix, suffix) {
  document.getElementById(labelId).textContent = `${prefix}${parseFloat(value).toFixed(suffix === 'yrs' ? 0 : 1)}${suffix ? ' ' + suffix : ''}`;
}
