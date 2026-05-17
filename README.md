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

1. Yêu cầu hệ thống & Cài đặt thư viện
2. 
Trước khi chạy, hãy đảm bảo bạn đã cài đặt Python (khuyến nghị phiên bản 3.8 trở lên). Sau đó, cài đặt các thư viện cần thiết bằng lệnh sau:

```bash
pip install fastapi uvicorn pydantic joblib pandas numpy scikit-learn xgboost
```

2. Cấu trúc thư mục chuẩn bị
Để ứng dụng hoạt động chính xác, hãy đảm bảo cấu trúc các file trong repo của bạn tuân thủ định dạng sau (đặc biệt là các file mô hình .pkl nằm trong thư mục Model):
```bash
├── Model/
│   ├── diabetes_xgb_model.pkl
│   ├── logistic_regression.pkl
│   └── random_forest_diabetes_model.pkl
├── app.py
├── index.html
├── style.css
└── script.js
```
3. Khởi chạy Ứng dụng
   
  Cách 1: Chạy trực tiếp file python
  ```bash
  python app.py
  ```
  Cách 2: Sử dụng lệnh uvicorn
  ```bash
  uvicorn app:app --host 127.0.0.1 --port 8000 --reload
  ```

4. Truy cập giao diện và sử dụng
   
Sau khi khởi chạy thành công, mở trình duyệt web và truy cập vào đường dẫn:
```bash
http://127.0.0.1:8000
```
