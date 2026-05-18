"""
Ứng dụng Dự đoán Bệnh Tiểu Đường
Sử dụng các mô hình ML đã huấn luyện
"""

import tkinter as tk
from tkinter import ttk, messagebox, font
import joblib
import numpy as np
import pandas as pd
import os
import sys

# ─── Đường dẫn tới thư mục Model ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Model")
TRAIN_DIR = os.path.join(BASE_DIR, "Train")

# ─── Màu sắc ─────────────────────────────────────────────────────────────────
BG_DARK      = "#0f172a"
BG_CARD      = "#1e293b"
BG_INPUT     = "#0f172a"
BORDER       = "#334155"
ACCENT       = "#38bdf8"
ACCENT2      = "#818cf8"
SUCCESS      = "#4ade80"
DANGER       = "#f87171"
WARNING      = "#fbbf24"
TEXT_PRIMARY = "#f1f5f9"
TEXT_MUTED   = "#94a3b8"
BTN_BG       = "#0ea5e9"
BTN_HOVER    = "#0284c7"

# ─── Cột features sau khi One-Hot Encode ─────────────────────────────────────
# Thứ tự phải khớp với lúc training (get_dummies, drop_first=True)
FEATURE_COLS = [
    "chol", "stab.glu", "hdl", "ratio", "age",
    "height", "weight", "bp.1s", "bp.1d",
    "waist", "hip", "time.ppn", "BMI",
    "location_Louisa", "gender_male",
    "frame_medium", "frame_small",
]


def load_models():
    """Nạp tất cả mô hình có sẵn."""
    models = {}

    # Random Forest
    rf_path = os.path.join(MODEL_DIR, "diabetes_rf_model.pkl")
    if os.path.exists(rf_path):
        try:
            models["Random Forest"] = joblib.load(rf_path)
        except Exception as e:
            print(f"Lỗi nạp RF: {e}")

    # XGBoost (nằm trong Train/)
    xgb_path = os.path.join(TRAIN_DIR, "diabetes_xgb_model.pkl")
    if os.path.exists(xgb_path):
        try:
            models["XGBoost"] = joblib.load(xgb_path)
        except Exception as e:
            print(f"Lỗi nạp XGBoost: {e}")

    # Logistic Regression (cần scaler, chỉ tải khi có)
    lr_path = os.path.join(MODEL_DIR, "diabetes_logistic_classifier.pkl")
    if os.path.exists(lr_path):
        try:
            models["Logistic Regression*"] = joblib.load(lr_path)
        except Exception as e:
            print(f"Lỗi nạp LR: {e}")

    return models


def build_feature_vector(inputs: dict) -> pd.DataFrame:
    """Tạo DataFrame input theo đúng thứ tự FEATURE_COLS."""
    height_in = inputs["height"]   # inches
    weight_lb = inputs["weight"]   # pounds
    bmi = (weight_lb / (height_in ** 2)) * 703 if height_in > 0 else 0

    row = {
        "chol":      inputs["chol"],
        "stab.glu":  inputs["stab_glu"],
        "hdl":       inputs["hdl"],
        "ratio":     inputs["ratio"],
        "age":       inputs["age"],
        "height":    height_in,
        "weight":    weight_lb,
        "bp.1s":     inputs["bp_1s"],
        "bp.1d":     inputs["bp_1d"],
        "waist":     inputs["waist"],
        "hip":       inputs["hip"],
        "time.ppn":  inputs["time_ppn"],
        "BMI":       bmi,
        "location_Louisa": 1 if inputs["location"] == "Louisa" else 0,
        "gender_male":     1 if inputs["gender"]   == "male"   else 0,
        "frame_medium":    1 if inputs["frame"]    == "medium" else 0,
        "frame_small":     1 if inputs["frame"]    == "small"  else 0,
    }
    return pd.DataFrame([row], columns=FEATURE_COLS), bmi


# ═════════════════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🩺 Dự Đoán Bệnh Tiểu Đường")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.minsize(900, 680)

        # Fonts
        self.fnt_title  = font.Font(family="Segoe UI", size=18, weight="bold")
        self.fnt_head   = font.Font(family="Segoe UI", size=11, weight="bold")
        self.fnt_label  = font.Font(family="Segoe UI", size=10)
        self.fnt_result = font.Font(family="Segoe UI", size=13, weight="bold")
        self.fnt_small  = font.Font(family="Segoe UI", size=9)

        self.models = load_models()
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg="#0c4a6e", pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🩺  Hệ Thống Dự Đoán Bệnh Tiểu Đường",
                 bg="#0c4a6e", fg=TEXT_PRIMARY, font=self.fnt_title).pack()
        tk.Label(hdr, text="Nhập các thông số lâm sàng để dự đoán nguy cơ mắc bệnh",
                 bg="#0c4a6e", fg=ACCENT, font=self.fnt_small).pack()

        # Body
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=20, pady=16)

        left  = tk.Frame(body, bg=BG_DARK)
        right = tk.Frame(body, bg=BG_DARK)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right.pack(side="right", fill="both", expand=True)

        self._build_input_panel(left)
        self._build_result_panel(right)

    def _card(self, parent, title):
        outer = tk.Frame(parent, bg=BORDER, bd=0)
        outer.pack(fill="both", expand=False, pady=6)
        inner = tk.Frame(outer, bg=BG_CARD, padx=14, pady=12)
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        tk.Label(inner, text=title, bg=BG_CARD, fg=ACCENT,
                 font=self.fnt_head).pack(anchor="w", pady=(0, 8))
        return inner

    def _entry_row(self, parent, label, var, unit=""):
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, bg=BG_CARD, fg=TEXT_PRIMARY,
                 font=self.fnt_label, width=28, anchor="w").pack(side="left")
        e = tk.Entry(row, textvariable=var, bg=BG_INPUT, fg=TEXT_PRIMARY,
                     insertbackground=TEXT_PRIMARY, relief="flat",
                     font=self.fnt_label, width=10,
                     highlightthickness=1, highlightcolor=ACCENT,
                     highlightbackground=BORDER)
        e.pack(side="left", padx=(0, 4))
        if unit:
            tk.Label(row, text=unit, bg=BG_CARD, fg=TEXT_MUTED,
                     font=self.fnt_small).pack(side="left")
        return e

    def _combo_row(self, parent, label, var, values):
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, bg=BG_CARD, fg=TEXT_PRIMARY,
                 font=self.fnt_label, width=28, anchor="w").pack(side="left")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TCombobox",
                        fieldbackground=BG_INPUT, background=BG_INPUT,
                        foreground=TEXT_PRIMARY, bordercolor=BORDER,
                        arrowcolor=ACCENT)
        cb = ttk.Combobox(row, textvariable=var, values=values,
                          style="Dark.TCombobox", width=12, state="readonly")
        cb.pack(side="left")
        return cb

    # ── Input panel ──────────────────────────────────────────────────────────
    def _build_input_panel(self, parent):
        # --- Thông tin cơ bản ---
        card1 = self._card(parent, "📋  Thông Tin Cơ Bản")

        self.v_age      = tk.StringVar(value="45")
        self.v_gender   = tk.StringVar(value="female")
        self.v_location = tk.StringVar(value="Buckingham")
        self.v_frame    = tk.StringVar(value="medium")
        self.v_height   = tk.StringVar(value="65")
        self.v_weight   = tk.StringVar(value="160")

        self._entry_row(card1, "Tuổi",          self.v_age,      "tuổi")
        self._combo_row(card1, "Giới tính",      self.v_gender,   ["female", "male"])
        self._combo_row(card1, "Khu vực",        self.v_location, ["Buckingham", "Louisa"])
        self._combo_row(card1, "Khung người",    self.v_frame,    ["large", "medium", "small"])
        self._entry_row(card1, "Chiều cao",      self.v_height,   "inches (1in≈2.54cm)")
        self._entry_row(card1, "Cân nặng",       self.v_weight,   "pounds (1lb≈0.45kg)")

        # --- Chỉ số máu ---
        card2 = self._card(parent, "🩸  Chỉ Số Máu")

        self.v_chol     = tk.StringVar(value="200")
        self.v_stab_glu = tk.StringVar(value="90")
        self.v_hdl      = tk.StringVar(value="50")
        self.v_ratio    = tk.StringVar(value="4.5")

        self._entry_row(card2, "Cholesterol toàn phần",   self.v_chol,     "mg/dL")
        self._entry_row(card2, "Đường huyết ổn định",     self.v_stab_glu, "mg/dL")
        self._entry_row(card2, "HDL Cholesterol",          self.v_hdl,      "mg/dL")
        self._entry_row(card2, "Tỉ lệ Chol/HDL",          self.v_ratio,    "")

        # --- Huyết áp & Vóc dáng ---
        card3 = self._card(parent, "❤️  Huyết Áp & Vóc Dáng")

        self.v_bp_1s   = tk.StringVar(value="120")
        self.v_bp_1d   = tk.StringVar(value="75")
        self.v_waist   = tk.StringVar(value="36")
        self.v_hip     = tk.StringVar(value="40")
        self.v_time_ppn = tk.StringVar(value="360")

        self._entry_row(card3, "Huyết áp tâm thu (Sys)",  self.v_bp_1s,    "mmHg")
        self._entry_row(card3, "Huyết áp tâm trương (Dia)",self.v_bp_1d,   "mmHg")
        self._entry_row(card3, "Vòng eo",                  self.v_waist,    "inches")
        self._entry_row(card3, "Vòng hông",                self.v_hip,      "inches")
        self._entry_row(card3, "Thời gian sau bữa ăn",    self.v_time_ppn, "phút")

        # Nút dự đoán
        btn_frame = tk.Frame(parent, bg=BG_DARK)
        btn_frame.pack(fill="x", pady=(10, 0))

        btn = tk.Button(btn_frame, text="🔍  DỰ ĐOÁN NGAY",
                        bg=BTN_BG, fg="white", font=self.fnt_head,
                        relief="flat", padx=24, pady=10, cursor="hand2",
                        command=self._predict)
        btn.pack(fill="x")
        btn.bind("<Enter>", lambda e: btn.config(bg=BTN_HOVER))
        btn.bind("<Leave>", lambda e: btn.config(bg=BTN_BG))

        btn2 = tk.Button(btn_frame, text="🗑️  Xóa / Nhập Lại",
                         bg="#334155", fg=TEXT_MUTED, font=self.fnt_small,
                         relief="flat", padx=12, pady=6, cursor="hand2",
                         command=self._reset)
        btn2.pack(fill="x", pady=(4, 0))

    # ── Result panel ─────────────────────────────────────────────────────────
    def _build_result_panel(self, parent):
        # BMI card
        bmi_card = self._card(parent, "📐  Chỉ Số BMI")
        self.lbl_bmi_val  = tk.Label(bmi_card, text="—", bg=BG_CARD,
                                     fg=TEXT_PRIMARY, font=font.Font(family="Segoe UI", size=28, weight="bold"))
        self.lbl_bmi_val.pack()
        self.lbl_bmi_cat  = tk.Label(bmi_card, text="Nhập dữ liệu để tính", bg=BG_CARD,
                                     fg=TEXT_MUTED, font=self.fnt_label)
        self.lbl_bmi_cat.pack(pady=(0, 4))

        bmi_scale = tk.Frame(bmi_card, bg=BG_CARD)
        bmi_scale.pack(fill="x")
        for label, color in [("Thiếu\n<18.5", ACCENT), ("Bình thường\n18.5-24.9", SUCCESS),
                              ("Thừa cân\n25-29.9", WARNING), ("Béo phì\n≥30", DANGER)]:
            tk.Label(bmi_scale, text=label, bg=color, fg="white",
                     font=font.Font(family="Segoe UI", size=8), width=10, pady=4).pack(side="left", padx=1)

        # Kết quả mô hình
        res_card = self._card(parent, "🤖  Kết Quả Dự Đoán")
        self.result_frames = {}

        if not self.models:
            tk.Label(res_card, text="⚠️ Không tìm thấy mô hình nào!\nKiểm tra thư mục Model/",
                     bg=BG_CARD, fg=DANGER, font=self.fnt_label, justify="center").pack(pady=20)
        else:
            for name in self.models:
                f = tk.Frame(res_card, bg="#0f172a", bd=0)
                f.pack(fill="x", pady=3)
                inner = tk.Frame(f, bg="#0f172a", padx=10, pady=8)
                inner.pack(fill="x", padx=1, pady=1)

                tk.Label(inner, text=name, bg="#0f172a", fg=TEXT_MUTED,
                         font=self.fnt_small).pack(anchor="w")
                lbl = tk.Label(inner, text="Chờ dữ liệu...", bg="#0f172a",
                               fg=TEXT_MUTED, font=self.fnt_result)
                lbl.pack(anchor="w")
                prob_lbl = tk.Label(inner, text="", bg="#0f172a",
                                    fg=TEXT_MUTED, font=self.fnt_small)
                prob_lbl.pack(anchor="w")
                self.result_frames[name] = (lbl, prob_lbl)

        # Ghi chú
        note_card = self._card(parent, "📌  Lưu Ý")
        notes = [
            "• Kết quả chỉ mang tính tham khảo.",
            "• (*) Logistic Regression cần scaler riêng.",
            "• Mọi quyết định y tế cần bác sĩ tư vấn.",
            "• Chiều cao/cân nặng dùng đơn vị Mỹ (in/lb).",
        ]
        for n in notes:
            tk.Label(note_card, text=n, bg=BG_CARD, fg=TEXT_MUTED,
                     font=self.fnt_small, anchor="w").pack(anchor="w")

    # ── Logic ─────────────────────────────────────────────────────────────────
    def _get_inputs(self):
        def fval(v):
            return float(v.get().strip())
        return {
            "age":      fval(self.v_age),
            "gender":   self.v_gender.get(),
            "location": self.v_location.get(),
            "frame":    self.v_frame.get(),
            "height":   fval(self.v_height),
            "weight":   fval(self.v_weight),
            "chol":     fval(self.v_chol),
            "stab_glu": fval(self.v_stab_glu),
            "hdl":      fval(self.v_hdl),
            "ratio":    fval(self.v_ratio),
            "bp_1s":    fval(self.v_bp_1s),
            "bp_1d":    fval(self.v_bp_1d),
            "waist":    fval(self.v_waist),
            "hip":      fval(self.v_hip),
            "time_ppn": fval(self.v_time_ppn),
        }

    def _bmi_category(self, bmi):
        if bmi < 18.5:
            return "Thiếu cân", ACCENT
        elif bmi < 25:
            return "Bình thường ✓", SUCCESS
        elif bmi < 30:
            return "Thừa cân", WARNING
        else:
            return "Béo phì", DANGER

    def _predict(self):
        try:
            inputs = self._get_inputs()
        except ValueError:
            messagebox.showerror("Lỗi nhập liệu",
                                 "Vui lòng kiểm tra lại các ô số.\nKhông được để trống hoặc nhập chữ.")
            return

        X, bmi = build_feature_vector(inputs)

        # Cập nhật BMI
        cat, color = self._bmi_category(bmi)
        self.lbl_bmi_val.config(text=f"{bmi:.2f}", fg=color)
        self.lbl_bmi_cat.config(text=cat, fg=color)

        # Dự đoán từng mô hình
        for name, model in self.models.items():
            lbl, prob_lbl = self.result_frames[name]
            try:
                pred = model.predict(X)[0]
                label = "🔴  CÓ NGUY CƠ TIỂU ĐƯỜNG" if pred == 1 else "🟢  KHÔNG CÓ NGUY CƠ"
                fg_col = DANGER if pred == 1 else SUCCESS

                # Xác suất (nếu hỗ trợ)
                prob_text = ""
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0]
                    prob_text = f"Xác suất mắc bệnh: {proba[1]*100:.1f}%  |  Không mắc: {proba[0]*100:.1f}%"

                lbl.config(text=label, fg=fg_col)
                prob_lbl.config(text=prob_text, fg=TEXT_MUTED)
            except Exception as e:
                lbl.config(text=f"⚠️ Lỗi: {e}", fg=WARNING)
                prob_lbl.config(text="")

    def _reset(self):
        defaults = {
            self.v_age: "45", self.v_height: "65", self.v_weight: "160",
            self.v_chol: "200", self.v_stab_glu: "90", self.v_hdl: "50",
            self.v_ratio: "4.5", self.v_bp_1s: "120", self.v_bp_1d: "75",
            self.v_waist: "36", self.v_hip: "40", self.v_time_ppn: "360",
        }
        for var, val in defaults.items():
            var.set(val)
        self.v_gender.set("female")
        self.v_location.set("Buckingham")
        self.v_frame.set("medium")

        self.lbl_bmi_val.config(text="—", fg=TEXT_PRIMARY)
        self.lbl_bmi_cat.config(text="Nhập dữ liệu để tính", fg=TEXT_MUTED)
        for name in self.result_frames:
            lbl, prob_lbl = self.result_frames[name]
            lbl.config(text="Chờ dữ liệu...", fg=TEXT_MUTED)
            prob_lbl.config(text="")


if __name__ == "__main__":
    app = App()
    app.mainloop()