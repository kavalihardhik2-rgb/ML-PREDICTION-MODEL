/* ================================================
   iris.js — Iris Flower Classifier UI
   ================================================ */

let irisChart = null;

const IRIS_SPECIES_EMOJI = {
  'Setosa':     '🌸',
  'Versicolor': '🌺',
  'Virginica':  '🌻',
};

const IRIS_GRADIENTS = [
  'linear-gradient(135deg,#7c3aed,#a855f7)',
  'linear-gradient(135deg,#06b6d4,#22c55e)',
  'linear-gradient(135deg,#f59e0b,#f97316)',
];

async function predictIris() {
  hideError('iris-error');
  setLoading('iris-btn', true);

  const payload = {
    sepal_length: parseFloat(document.getElementById('sepal_length').value),
    sepal_width:  parseFloat(document.getElementById('sepal_width').value),
    petal_length: parseFloat(document.getElementById('petal_length').value),
    petal_width:  parseFloat(document.getElementById('petal_width').value),
  };

  try {
    const res  = await fetch(`${API_BASE}/api/iris/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderIrisResult(data);
  } catch (e) {
    showError('iris-error', 'iris-error-msg', e.message || 'Failed to reach backend. Is the server running?');
  } finally {
    setLoading('iris-btn', false);
  }
}

function renderIrisResult(data) {
  const emoji = IRIS_SPECIES_EMOJI[data.species] || '🌸';
  document.getElementById('iris-emoji').textContent    = emoji;
  document.getElementById('iris-species').textContent  = data.species;
  document.getElementById('iris-confidence-text').textContent = `Confidence: ${data.confidence.toFixed(1)}%`;

  renderProbBars('iris-prob-bars', data.probabilities, IRIS_GRADIENTS);
  showResult('iris-result');

  // Chart
  const ctx = document.getElementById('iris-chart').getContext('2d');
  const labels = Object.keys(data.probabilities);
  const values = Object.values(data.probabilities);

  if (irisChart) irisChart.destroy();
  document.getElementById('iris-chart-card').style.display = 'block';

  irisChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: [
          'rgba(124,58,237,0.8)',
          'rgba(6,182,212,0.8)',
          'rgba(245,158,11,0.8)',
        ],
        borderColor: [
          'rgba(124,58,237,1)',
          'rgba(6,182,212,1)',
          'rgba(245,158,11,1)',
        ],
        borderWidth: 2,
        hoverOffset: 10,
      }],
    },
    options: {
      responsive: true,
      cutout: '65%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color: '#94a3b8',
            font: { family: 'Inter', size: 12 },
            padding: 16,
          },
        },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.parsed.toFixed(1)}%`
          }
        }
      },
    },
  });
}

/** Quick preset examples */
function setIrisPreset(sl, sw, pl, pw) {
  document.getElementById('sepal_length').value = sl;
  document.getElementById('sepal_width').value  = sw;
  document.getElementById('petal_length').value = pl;
  document.getElementById('petal_width').value  = pw;

  updateRangeVal('sl-val', sl, 'cm');
  updateRangeVal('sw-val', sw, 'cm');
  updateRangeVal('pl-val', pl, 'cm');
  updateRangeVal('pw-val', pw, 'cm');
}
