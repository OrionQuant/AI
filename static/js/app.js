// OrionQuant Dashboard JavaScript

let priceChart = null;
let equityChart = null;
let refreshInterval = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupEventListeners();
    initializeCharts();
    startAutoRefresh();
});

// Health check
async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.getElementById('statusText');
        
        if (data.status === 'healthy') {
            statusDot.classList.add('connected');
            statusText.textContent = data.model_loaded ? 'Model Loaded' : 'No Model';
        } else {
            statusDot.classList.remove('connected');
            statusText.textContent = 'Disconnected';
        }
    } catch (error) {
        console.error('Health check failed:', error);
        document.getElementById('statusText').textContent = 'Error';
    }
}

// Setup event listeners
function setupEventListeners() {
    document.getElementById('refreshBtn').addEventListener('click', () => {
        getPrediction();
        updatePriceChart();
    });
    
    document.getElementById('symbolSelect').addEventListener('change', () => {
        getPrediction();
        updatePriceChart();
    });
    
    document.getElementById('timeframeSelect').addEventListener('change', () => {
        updatePriceChart();
    });
    
    document.getElementById('runBacktestBtn').addEventListener('click', runBacktest);
    
    // Set default end date to today
    const endDateInput = document.getElementById('backtestEndDate');
    endDateInput.value = new Date().toISOString().split('T')[0];
}

// Get prediction
async function getPrediction() {
    try {
        const symbol = document.getElementById('symbolSelect').value;
        
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symbol: symbol,
                timeframe: document.getElementById('timeframeSelect').value
            })
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const data = await response.json();
        updatePredictionDisplay(data);
    } catch (error) {
        console.error('Prediction error:', error);
        showError('Failed to get prediction');
    }
}

// Update prediction display
function updatePredictionDisplay(data) {
    const signalDisplay = document.getElementById('signalDisplay');
    const signalIcon = document.getElementById('signalIcon');
    const signalText = document.getElementById('signalText');
    const confidenceText = document.getElementById('confidenceText');
    const expectedReturn = document.getElementById('expectedReturn');
    const currentPrice = document.getElementById('currentPrice');
    const timestamp = document.getElementById('timestamp');
    
    // Update signal
    signalText.textContent = data.signal;
    confidenceText.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
    
    // Update signal display styling
    signalDisplay.className = 'signal-display ' + data.signal.toLowerCase();
    
    // Update icon
    if (data.signal === 'BUY') {
        signalIcon.textContent = '📈';
    } else if (data.signal === 'SELL') {
        signalIcon.textContent = '📉';
    } else {
        signalIcon.textContent = '➡️';
    }
    
    // Update details
    expectedReturn.textContent = `${(data.expected_return * 100).toFixed(2)}%`;
    expectedReturn.style.color = data.expected_return >= 0 ? '#10b981' : '#ef4444';
    currentPrice.textContent = `$${parseFloat(data.current_price).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    
    if (data.timestamp) {
        const date = new Date(data.timestamp);
        timestamp.textContent = date.toLocaleString();
    }
    
    // Update probability bars
    updateProbabilityBars(data.probabilities);
}

// Update probability bars
function updateProbabilityBars(probs) {
    document.getElementById('buyProb').textContent = `${(probs.BUY * 100).toFixed(1)}%`;
    document.getElementById('holdProb').textContent = `${(probs.HOLD * 100).toFixed(1)}%`;
    document.getElementById('sellProb').textContent = `${(probs.SELL * 100).toFixed(1)}%`;
    
    document.getElementById('buyBar').style.width = `${probs.BUY * 100}%`;
    document.getElementById('holdBar').style.width = `${probs.HOLD * 100}%`;
    document.getElementById('sellBar').style.width = `${probs.SELL * 100}%`;
}

// Initialize charts
function initializeCharts() {
    // Price chart
    const priceCtx = document.getElementById('priceChart').getContext('2d');
    priceChart = new Chart(priceCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Price',
                data: [],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        color: '#cbd5e1'
                    },
                    grid: {
                        color: '#334155'
                    }
                },
                x: {
                    ticks: {
                        color: '#cbd5e1'
                    },
                    grid: {
                        color: '#334155'
                    }
                }
            }
        }
    });
    
    // Equity chart
    const equityCtx = document.getElementById('equityChart').getContext('2d');
    equityChart = new Chart(equityCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Equity',
                data: [],
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        color: '#cbd5e1'
                    },
                    grid: {
                        color: '#334155'
                    }
                },
                x: {
                    ticks: {
                        color: '#cbd5e1'
                    },
                    grid: {
                        color: '#334155'
                    }
                }
            }
        }
    });
    
    updatePriceChart();
}

// Update price chart
async function updatePriceChart() {
    try {
        const symbol = document.getElementById('symbolSelect').value;
        const timeframe = document.getElementById('timeframeSelect').value;
        
        const response = await fetch(`/api/data/latest?symbol=${symbol}&timeframe=${timeframe}&limit=100`);
        const data = await response.json();
        
        if (data.data && data.data.length > 0) {
            const labels = data.data.map(d => {
                const date = new Date(d.timestamp);
                return date.toLocaleTimeString();
            });
            const prices = data.data.map(d => d.close);
            
            priceChart.data.labels = labels;
            priceChart.data.datasets[0].data = prices;
            priceChart.update();
        }
    } catch (error) {
        console.error('Chart update error:', error);
    }
}

// Run backtest
async function runBacktest() {
    const btn = document.getElementById('runBacktestBtn');
    btn.disabled = true;
    btn.textContent = 'Running...';
    
    try {
        const startDate = document.getElementById('backtestStartDate').value;
        const endDate = document.getElementById('backtestEndDate').value;
        const symbol = document.getElementById('symbolSelect').value;
        const timeframe = document.getElementById('timeframeSelect').value;
        
        const response = await fetch('/api/backtest', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symbol: symbol,
                timeframe: timeframe,
                start_date: startDate,
                end_date: endDate
            })
        });
        
        if (!response.ok) {
            throw new Error('Backtest failed');
        }
        
        const results = await response.json();
        displayBacktestResults(results);
    } catch (error) {
        console.error('Backtest error:', error);
        showError('Backtest failed: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run Backtest';
    }
}

// Display backtest results
function displayBacktestResults(results) {
    document.getElementById('backtestResults').style.display = 'block';
    
    // Update metrics
    document.getElementById('totalReturn').textContent = 
        `$${results.total_return.toFixed(2)} (${results.total_return_pct.toFixed(2)}%)`;
    document.getElementById('totalReturn').style.color = 
        results.total_return >= 0 ? '#10b981' : '#ef4444';
    
    document.getElementById('sharpeRatio').textContent = results.sharpe_ratio.toFixed(2);
    document.getElementById('winRate').textContent = `${(results.win_rate * 100).toFixed(2)}%`;
    document.getElementById('profitFactor').textContent = results.profit_factor.toFixed(2);
    document.getElementById('maxDrawdown').textContent = `${results.max_drawdown.toFixed(2)}%`;
    document.getElementById('numTrades').textContent = results.num_trades;
    
    // Update equity chart
    if (results.equity_curve && results.equity_curve.length > 0) {
        const labels = results.timestamps && results.timestamps.length > 0
            ? results.timestamps.map(t => new Date(t).toLocaleDateString())
            : results.equity_curve.map((_, i) => `Day ${i + 1}`);
        
        equityChart.data.labels = labels;
        equityChart.data.datasets[0].data = results.equity_curve;
        equityChart.update();
    }
}

// Auto refresh
function startAutoRefresh() {
    getPrediction();
    updatePriceChart();
    
    refreshInterval = setInterval(() => {
        getPrediction();
        updatePriceChart();
    }, 30000); // Refresh every 30 seconds
}

// Show error
function showError(message) {
    // Simple error display - can be enhanced
    alert(message);
}

