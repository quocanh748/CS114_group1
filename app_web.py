"""
Ứng dụng Web Dự Đoán Bệnh Tiểu Đường
Chạy: python app_web.py  → mở trình duyệt tại http://127.0.0.1:5000
"""
import os, webbrowser, threading
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Model")
TRAIN_DIR = os.path.join(BASE_DIR, "Train")

FEATURE_COLS = [
    "chol","stab.glu","hdl","ratio","age",
    "height","weight","bp.1s","bp.1d",
    "waist","hip","time.ppn","BMI",
    "location_Louisa","gender_male","frame_medium","frame_small",
]

LR_SCALER = None  # StandardScaler cho Logistic Regression

def load_models():
    global LR_SCALER
    m = {}
    paths = {
        "Random Forest":       os.path.join(MODEL_DIR, "diabetes_rf_model.pkl"),
        "XGBoost":             os.path.join(TRAIN_DIR, "diabetes_xgb_model.pkl"),
        "Logistic Regression": os.path.join(MODEL_DIR, "diabetes_logistic_classifier.pkl"),
    }
    for name, p in paths.items():
        if os.path.exists(p):
            try:
                m[name] = joblib.load(p)
                print(f"[OK] Da nap: {name}")
            except Exception as e:
                print(f"[ERR] Loi nap {name}: {e}")

    # Nap scaler cho Logistic Regression
    scaler_path = os.path.join(MODEL_DIR, "scaler_logistic.pkl")
    if os.path.exists(scaler_path):
        try:
            LR_SCALER = joblib.load(scaler_path)
            print("[OK] Da nap scaler cho Logistic Regression")
        except Exception as e:
            print(f"[ERR] Loi nap scaler: {e}")
    else:
        print("[WARN] Khong tim thay scaler_logistic.pkl")
    return m

MODELS = load_models()

HTML = r"""
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dự Đoán Bệnh Tiểu Đường</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:#0f172a;color:#f1f5f9;min-height:100vh}
.header{background:linear-gradient(135deg,#0c4a6e,#0e7490);padding:24px 32px;text-align:center;border-bottom:1px solid #164e63}
.header h1{font-size:1.8rem;font-weight:700;letter-spacing:-0.5px}
.header p{color:#7dd3fc;margin-top:6px;font-size:.9rem}
.container{max-width:1100px;margin:0 auto;padding:24px 20px;display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:760px){.container{grid-template-columns:1fr}}
.card{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px}
.card-title{font-size:.85rem;font-weight:600;color:#38bdf8;text-transform:uppercase;letter-spacing:1px;margin-bottom:14px}
.field{margin-bottom:12px}
.field label{display:block;font-size:.82rem;color:#94a3b8;margin-bottom:4px}
.field input,.field select{width:100%;background:#0f172a;border:1px solid #334155;color:#f1f5f9;border-radius:6px;padding:8px 10px;font-size:.9rem;font-family:inherit;outline:none;transition:.2s}
.field input:focus,.field select:focus{border-color:#38bdf8;box-shadow:0 0 0 2px rgba(56,189,248,.2)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.btn{width:100%;padding:12px;background:#0ea5e9;color:#fff;border:none;border-radius:8px;font-size:1rem;font-weight:600;cursor:pointer;margin-top:6px;transition:.2s;font-family:inherit}
.btn:hover{background:#0284c7;transform:translateY(-1px)}
.btn-reset{background:#334155;margin-top:6px}
.btn-reset:hover{background:#475569}
/* BMI */
.bmi-box{text-align:center;padding:16px;background:#0f172a;border-radius:10px;margin-bottom:14px}
.bmi-value{font-size:3rem;font-weight:700;line-height:1}
.bmi-label{font-size:.9rem;margin-top:4px}
.bmi-bar{display:grid;grid-template-columns:repeat(4,1fr);gap:3px;margin-top:10px;border-radius:6px;overflow:hidden}
.bmi-seg{padding:5px 2px;text-align:center;font-size:.65rem;font-weight:600}
.seg0{background:#0ea5e9}.seg1{background:#4ade80}.seg2{background:#fbbf24}.seg3{background:#f87171}
/* Results */
.result-item{background:#0f172a;border-radius:8px;padding:12px 14px;margin-bottom:10px;border-left:4px solid #334155;transition:.3s}
.result-item.positive{border-color:#f87171}
.result-item.negative{border-color:#4ade80}
.result-model{font-size:.75rem;color:#64748b;margin-bottom:4px}
.result-verdict{font-size:1.05rem;font-weight:700;margin-bottom:2px}
.result-prob{font-size:.78rem;color:#64748b}
.verdict-pos{color:#f87171}
.verdict-neg{color:#4ade80}
/* Prob bar */
.prob-bar-wrap{margin-top:6px;background:#1e293b;border-radius:99px;height:6px;overflow:hidden}
.prob-bar-fill{height:100%;border-radius:99px;transition:width .6s ease}
.bar-pos{background:#f87171}
.bar-neg{background:#4ade80}
.note-list{list-style:none;padding:0}
.note-list li{font-size:.78rem;color:#64748b;padding:3px 0;border-bottom:1px solid #1e293b}
.note-list li:last-child{border:none}
.spinner{display:none;text-align:center;padding:20px;color:#38bdf8}
.no-model{color:#f87171;text-align:center;padding:20px;font-size:.9rem}
</style>
</head>
<body>
<div class="header">
  <h1>🩺 Hệ Thống Dự Đoán Bệnh Tiểu Đường</h1>
  <p>Nhập thông số lâm sàng → Nhận kết quả dự đoán ngay lập tức</p>
</div>

<div class="container">
  <!-- LEFT COLUMN: INPUTS -->
  <div>
    <div class="card" style="margin-bottom:16px">
      <div class="card-title">📋 Thông Tin Cơ Bản</div>
      <div class="grid2">
        <div class="field">
          <label>Tuổi (năm)</label>
          <input type="number" id="age" value="45" min="1" max="120">
        </div>
        <div class="field">
          <label>Giới tính</label>
          <select id="gender">
            <option value="female">Nữ (female)</option>
            <option value="male">Nam (male)</option>
          </select>
        </div>
        <input type="hidden" id="location" value="Buckingham">
        <div class="field">
          <label>Khung người</label>
          <select id="frame">
            <option value="medium">Vừa (medium)</option>
            <option value="large">To (large)</option>
            <option value="small">Nhỏ (small)</option>
          </select>
        </div>
        <div class="field">
          <label>Chiều cao (cm)</label>
          <input type="number" id="height" value="165" step="1" min="100" max="230">
        </div>
        <div class="field">
          <label>Cân nặng (kg)</label>
          <input type="number" id="weight" value="73" step="0.5" min="20" max="250">
        </div>
      </div>
    </div>

    <div class="card" style="margin-bottom:16px">
      <div class="card-title">🩸 Chỉ Số Máu</div>
      <div class="grid2">
        <div class="field">
          <label>Cholesterol toàn phần (mg/dL)</label>
          <input type="number" id="chol" value="200" min="50" max="600">
        </div>
        <div class="field">
          <label>HDL Cholesterol (mg/dL)</label>
          <input type="number" id="hdl" value="50" min="5" max="150">
        </div>
        <div class="field">
          <label>Đường huyết ổn định – stab.glu (mg/dL)</label>
          <input type="number" id="stab_glu" value="90" min="40" max="500">
        </div>
        <div class="field">
          <label>Tỉ lệ Cholesterol / HDL</label>
          <input type="number" id="ratio" value="4.0" step="0.1" min="0.5" max="20">
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">❤️ Huyết Áp &amp; Vóc Dáng</div>
      <div class="grid2">
        <div class="field">
          <label>Huyết áp tâm thu – Systolic (mmHg)</label>
          <input type="number" id="bp_1s" value="120" min="60" max="250">
        </div>
        <div class="field">
          <label>Huyết áp tâm trương – Diastolic (mmHg)</label>
          <input type="number" id="bp_1d" value="75" min="30" max="150">
        </div>
        <div class="field">
          <label>Vòng eo (cm)</label>
          <input type="number" id="waist" value="91" step="1" min="50" max="200">
        </div>
        <div class="field">
          <label>Vòng hông (cm)</label>
          <input type="number" id="hip" value="102" step="1" min="50" max="200">
        </div>
        <div class="field" style="grid-column:1/-1">
          <label>Thời gian sau bữa ăn – time.ppn (phút)</label>
          <input type="number" id="time_ppn" value="360" min="0" max="1440">
        </div>
      </div>
      <button class="btn" onclick="predict()">🔍 DỰ ĐOÁN NGAY</button>
      <button class="btn btn-reset" onclick="resetForm()">🗑️ Nhập Lại</button>
    </div>
  </div>

  <!-- RIGHT COLUMN: RESULTS -->
  <div>
    <!-- BMI -->
    <div class="card" style="margin-bottom:16px">
      <div class="card-title">📐 Chỉ Số BMI</div>
      <div class="bmi-box">
        <div class="bmi-value" id="bmi-val">—</div>
        <div class="bmi-label" id="bmi-cat" style="color:#64748b">Nhập dữ liệu để tính</div>
      </div>
      <div class="bmi-bar">
        <div class="bmi-seg seg0">Thiếu cân<br>&lt;18.5</div>
        <div class="bmi-seg seg1">Bình thường<br>18.5–24.9</div>
        <div class="bmi-seg seg2">Thừa cân<br>25–29.9</div>
        <div class="bmi-seg seg3">Béo phì<br>≥30</div>
      </div>
    </div>

    <!-- Model results -->
    <div class="card" style="margin-bottom:16px">
      <div class="card-title">🤖 Kết Quả Từ Các Mô Hình</div>
      <div class="spinner" id="spinner">⏳ Đang tính toán...</div>
      <div id="results">
        {% if models %}
          {% for name in models %}
          <div class="result-item" id="res-{{ loop.index }}">
            <div class="result-model">{{ name }}</div>
            <div class="result-verdict" style="color:#475569">Chờ dữ liệu...</div>
          </div>
          {% endfor %}
        {% else %}
          <div class="no-model">⚠️ Không tìm thấy mô hình nào!<br>Kiểm tra thư mục Model/</div>
        {% endif %}
      </div>
    </div>

    <!-- Notes -->
    <div class="card">
      <div class="card-title">📌 Lưu Ý Quan Trọng</div>
      <ul class="note-list">
        <li>• Kết quả chỉ mang tính tham khảo, không thay thế bác sĩ.</li>
        <li>• (*) Logistic Regression cần scaler riêng – kết quả có thể lệch.</li>
        <li>• Chiều cao / cân nặng nhập theo cm / kg (tự chuyển đổi về inches / pounds khi dự đoán).</li>
        <li>• Ngưỡng HbA1c ≥ 6.5% = có nguy cơ tiểu đường.</li>
        <li>• Mô hình Random Forest: Acc ≈ 85.9%</li>
        <li>• Mô hình XGBoost: Acc ≈ 91.0%</li>
      </ul>
    </div>
  </div>
</div>

<script>
const MODEL_NAMES = {{ model_names | tojson }};

function bmiInfo(bmi){
  if(bmi<18.5) return {cat:'Thiếu cân ⚠️', color:'#38bdf8'};
  if(bmi<25)   return {cat:'Bình thường ✅', color:'#4ade80'};
  if(bmi<30)   return {cat:'Thừa cân ⚠️',  color:'#fbbf24'};
  return            {cat:'Béo phì 🔴',      color:'#f87171'};
}

async function predict(){
  const g = id => parseFloat(document.getElementById(id).value)||0;
  const gs = id => document.getElementById(id).value;

  // Giá trị người dùng nhập (đơn vị metric)
  const heightCm = g('height');   // cm
  const weightKg = g('weight');   // kg
  const waistCm  = g('waist');    // cm
  const hipCm    = g('hip');      // cm

  // Tính BMI theo đơn vị metric (kg/m²)
  const heightM = heightCm / 100;
  const bmi = heightM > 0 ? weightKg / (heightM * heightM) : 0;
  const info = bmiInfo(bmi);
  document.getElementById('bmi-val').textContent = bmi.toFixed(2);
  document.getElementById('bmi-val').style.color = info.color;
  document.getElementById('bmi-cat').textContent = info.cat;
  document.getElementById('bmi-cat').style.color = info.color;

  // Chuyển đổi về đơn vị Anh để gửi backend (model được train bằng đơn vị Anh)
  const heightIn = heightCm / 2.54;
  const weightLb = weightKg * 2.20462;
  const waistIn  = waistCm  / 2.54;
  const hipIn    = hipCm    / 2.54;

  const data = {
    age: g('age'), gender: gs('gender'), location: gs('location'),
    frame: gs('frame'),
    height: heightIn, weight: weightLb,   // đã đổi sang inches / pounds
    chol: g('chol'), stab_glu: g('stab_glu'), hdl: g('hdl'),
    ratio: g('ratio'), bp_1s: g('bp_1s'), bp_1d: g('bp_1d'),
    waist: waistIn, hip: hipIn,           // đã đổi sang inches
    time_ppn: g('time_ppn')
  };

  document.getElementById('spinner').style.display='block';

  try{
    const resp = await fetch('/predict', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(data)
    });
    const res = await resp.json();
    document.getElementById('spinner').style.display='none';

    MODEL_NAMES.forEach((name,i)=>{
      const el = document.getElementById('res-'+(i+1));
      if(!el) return;
      const r = res[name];
      if(!r){ el.innerHTML=`<div class="result-model">${name}</div><div class="result-verdict" style="color:#f87171">Lỗi</div>`; return; }
      const pos = r.pred===1;
      el.className = 'result-item '+(pos?'positive':'negative');
      const verdictCls = pos?'verdict-pos':'verdict-neg';
      const verdictTxt = pos?'🔴 CÓ NGUY CƠ TIỂU ĐƯỜNG':'🟢 KHÔNG CÓ NGUY CƠ';
      let probHtml='';
      if(r.prob_pos!==null){
        const pct = (r.prob_pos*100).toFixed(1);
        const barCls = pos?'bar-pos':'bar-neg';
        const barW   = pos?r.prob_pos*100:(1-r.prob_pos)*100;
        probHtml=`
          <div class="result-prob">Xác suất mắc: <b>${pct}%</b>  |  Không mắc: <b>${(100-pct).toFixed(1)}%</b></div>
          <div class="prob-bar-wrap"><div class="prob-bar-fill ${barCls}" style="width:${(r.prob_pos*100).toFixed(1)}%"></div></div>
        `;
      }
      el.innerHTML=`
        <div class="result-model">${name}</div>
        <div class="result-verdict ${verdictCls}">${verdictTxt}</div>
        ${probHtml}
      `;
    });
  }catch(e){
    document.getElementById('spinner').style.display='none';
    alert('Lỗi kết nối server: '+e);
  }
}

function resetForm(){
  const defs={age:45, height:165, weight:73, chol:200, stab_glu:90, hdl:50,
              ratio:4.0, bp_1s:120, bp_1d:75, waist:91, hip:102, time_ppn:360};
  Object.entries(defs).forEach(([k,v])=>{const el=document.getElementById(k);if(el)el.value=v;});
  document.getElementById('gender').value='female';
  // location is hidden, always Buckingham
  document.getElementById('frame').value='medium';
  document.getElementById('bmi-val').textContent='—';
  document.getElementById('bmi-val').style.color='#f1f5f9';
  document.getElementById('bmi-cat').textContent='Nhập dữ liệu để tính';
  document.getElementById('bmi-cat').style.color='#64748b';
  MODEL_NAMES.forEach((_,i)=>{
    const el=document.getElementById('res-'+(i+1));
    if(el) el.innerHTML=`<div class="result-model">${MODEL_NAMES[i]}</div><div class="result-verdict" style="color:#475569">Chờ dữ liệu...</div>`;
    if(el) el.className='result-item';
  });
}
</script>
</body>
</html>
"""

app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML,
        models=list(MODELS.keys()),
        model_names=list(MODELS.keys()))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    h, w = float(data["height"]), float(data["weight"])
    bmi  = (w / (h ** 2)) * 703 if h > 0 else 0

    row = {
        "chol":      float(data["chol"]),
        "stab.glu":  float(data["stab_glu"]),
        "hdl":       float(data["hdl"]),
        "ratio":     float(data["ratio"]),
        "age":       float(data["age"]),
        "height":    h,
        "weight":    w,
        "bp.1s":     float(data["bp_1s"]),
        "bp.1d":     float(data["bp_1d"]),
        "waist":     float(data["waist"]),
        "hip":       float(data["hip"]),
        "time.ppn":  float(data["time_ppn"]),
        "BMI":       bmi,
        "location_Louisa": 1 if data["location"] == "Louisa" else 0,
        "gender_male":     1 if data["gender"]   == "male"   else 0,
        "frame_medium":    1 if data["frame"]    == "medium" else 0,
        "frame_small":     1 if data["frame"]    == "small"  else 0,
    }
    X = pd.DataFrame([row], columns=FEATURE_COLS)

    results = {}
    for name, model in MODELS.items():
        try:
            # Logistic Regression can StandardScaler truoc khi predict
            if name == "Logistic Regression" and LR_SCALER is not None:
                X_input = pd.DataFrame(
                    LR_SCALER.transform(X), columns=FEATURE_COLS
                )
            else:
                X_input = X

            pred = int(model.predict(X_input)[0])
            prob_pos = None
            if hasattr(model, "predict_proba"):
                prob_pos = float(model.predict_proba(X_input)[0][1])
            results[name] = {"pred": pred, "prob_pos": prob_pos}
        except Exception as e:
            results[name] = {"error": str(e)}
    return jsonify(results)

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    threading.Timer(1.2, open_browser).start()
    print("[START] Server dang chay tai http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
