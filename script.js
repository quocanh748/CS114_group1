document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById('submitBtn');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span>Đang xử lý...</span>';

    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (result.status === 'success') {
            updateUI(
                result.rf_prediction, 
                result.rf_probability, 
                result.logistic_prediction,
                result.xgb_prediction,
                result.xgb_probability
            );
        } else {
            alert('Có lỗi xảy ra từ backend.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Không thể kết nối đến server.');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<span>Dự Đoán Kết Quả</span>';
    }
});

function updateUI(rfPred, rfProb, logPred, xgbPred, xgbProb) {
    const predictionResult = document.getElementById('predictionResult');
    const logisticResult = document.getElementById('logisticResult');
    const xgbResult = document.getElementById('xgbResult');

    // 1. Random Forest Box (Keep label as requested)
    predictionResult.className = 'prediction-box visible';
    const rfPercent = Math.round(rfProb * 100);
    
    if (rfPred === 1) {
        predictionResult.innerText = `Random Forest: Có nguy cơ. (Xác suất: ${rfPercent}%)`;
        predictionResult.classList.add('positive');
        predictionResult.classList.remove('negative');
    } else {
        predictionResult.innerText = `Random Forest: Bình thường. (Xác suất: ${rfPercent}%)`;
        predictionResult.classList.add('negative');
        predictionResult.classList.remove('positive');
    }

    // 2. Logistic Regression Box (Keep label as requested)
    logisticResult.className = 'prediction-box visible';
    if (logPred === 1) {
        logisticResult.innerText = 'Logistic Regression: Có nguy cơ.';
        logisticResult.classList.add('positive');
        logisticResult.classList.remove('negative');
    } else {
        logisticResult.innerText = 'Logistic Regression: Bình thường.';
        logisticResult.classList.add('negative');
        logisticResult.classList.remove('positive');
    }

    // 3. XGBoost Box
    xgbResult.className = 'prediction-box info visible';
    const xgbPercent = Math.round(xgbProb * 100);
    
    if (xgbPred === 1) {
        xgbResult.innerText = `XGBoost: Có nguy cơ. (Xác suất: ${xgbPercent}%)`;
        xgbResult.style.color = 'var(--danger)';
    } else {
        xgbResult.innerText = `XGBoost: Bình thường. (Xác suất: ${xgbPercent}%)`;
        xgbResult.style.color = 'var(--success)';
    }
}

// Randomize inputs function
document.getElementById('randomBtn').addEventListener('click', () => {
    const randomRange = (min, max, decimals = 0) => {
        const val = Math.random() * (max - min) + min;
        return decimals === 0 ? Math.round(val) : parseFloat(val.toFixed(decimals));
    };

    const randomSelect = (id, options) => {
        const select = document.getElementById(id);
        const randomIdx = Math.floor(Math.random() * options.length);
        select.value = options[randomIdx];
    };

    // Numeric fields with reasonable ranges based on dataset
    document.getElementById('chol').value = randomRange(100, 400);
    document.getElementById('stab_glu').value = randomRange(50, 300);
    document.getElementById('glyhb').value = randomRange(4.0, 12.0, 1);
    document.getElementById('hdl').value = randomRange(20, 100);
    document.getElementById('ratio').value = randomRange(2, 10, 1);
    document.getElementById('age').value = randomRange(19, 90);
    document.getElementById('height').value = randomRange(55, 75);
    document.getElementById('weight').value = randomRange(100, 300);
    document.getElementById('bmi').value = randomRange(15, 50, 1);
    document.getElementById('bp_1s').value = randomRange(100, 200);
    document.getElementById('bp_1d').value = randomRange(60, 110);
    document.getElementById('waist').value = randomRange(25, 55);
    document.getElementById('hip').value = randomRange(30, 65);
    document.getElementById('time_ppn').value = randomRange(10, 1440);

    // Categorical fields
    randomSelect('gender', ['female', 'male']);
    randomSelect('frame', ['small', 'medium', 'large']);
    randomSelect('location', ['Buckingham', 'Louisa']);
});
