<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<h1 align="center"><b>MÁY HỌC</b></h>
<h2 align="center"><b>Dự đoán bệnh tiểu đường</b></h>

## THÀNH VIÊN NHÓM 1
| STT    | MSSV          | Họ và Tên              
| ------ |:-------------:| ----------------------
| 1      | 23520070      | Phạm Ngô Quốc Anh      
| 2      | 21520930      | Nguyễn Văn Đức Huy        
| 3      | 23520021      | Nguyễn Tri An       
| 4      | 22521188      | Phạm Phú Minh Quân

## GIỚI THIỆU MÔN HỌC
* **Tên môn học:** Máy học
* **Mã môn học:** CS114.Q21
* **Năm học:** HK2 (2025 - 2026)
* **Giảng viên**: Võ Nguyễn Lê Duy

### HƯỚNG DẪN CHẠY DEMO

1. **Cài đặt thư viện**

```bash
pip install flask joblib pandas numpy scikit-learn xgboost
```

2. **Cấu trúc thư mục**

```
├── Model/
│   ├── diabetes_logistic_classifier.pkl
│   ├── diabetes_rf_model.pkl
│   ├── scaler_logistic.pkl
│   └── logistic_feature_cols.pkl
├── Train/
│   ├── Diab_pyth_data_clean.csv
│   ├── diabetes_xgb_model.pkl
│   └── save_scaler.py
├── Preprocess/
│   ├── diabetes.csv
│   └── eda.ipynb
└── app_web.py
```

3. **Khởi chạy ứng dụng**

```bash
python app_web.py
```

Trình duyệt sẽ tự động mở tại `http://127.0.0.1:5000`

4. **Nhập thông số và dự đoán**

Điền các chỉ số lâm sàng (tuổi, chiều cao cm, cân nặng kg, huyết áp, chỉ số máu…) rồi nhấn **DỰ ĐOÁN NGAY** để xem kết quả từ 3 mô hình: Random Forest, XGBoost, Logistic Regression.

> **Lưu ý:** Nếu file `scaler_logistic.pkl` chưa tồn tại, chạy `python Train/save_scaler.py` một lần để tạo scaler cho Logistic Regression.
