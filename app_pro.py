import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import qrcode
from PIL import Image

# ======================
# 页面设置
# ======================
st.set_page_config(layout="wide")

# ======================
# 加载模型
# ======================
model = joblib.load("catboost_model.pkl")
template = joblib.load("template.pkl")
feature_names = template.columns

# ======================
# 读取数据
# ======================
file_path = r"F:\机器学学射血分数保留\数据处理\mimic_5.xlsx"
df = pd.read_excel(file_path)
target = "death_within_icu_28days"
X_full = df.drop(columns=[target])

numeric_features = X_full.select_dtypes(include=['int64','float64']).columns

# ======================
# ⭐ 单位映射（论文关键）
# ======================
unit_map = {
    "Glu": "Glu (mg/dL)",
    "PLT": "PLT (K/uL)",
    "RBC": "RBC (m/uL)",
    "RDW": "RDW (%)",
    "WBC": "WBC (K/uL)",
    "Ca2+": "Ca (mg/dL)",
    "CL-": "Cl (mmol/L)",
    "Mg2+": "Mg (mg/dL)",
    "K+": "K (mmol/L)",
    "Na+": "Na (mmol/L)",
    "AG": "AG (mEq/L)",
    "Pco2": "PaCO2 (mmHg)",
    "Po2": "PaO2 (mmHg)",
    "PH": "pH",
    "ALT": "ALT (IU/L)",
    "BUN": "BUN (mg/dL)",
    "Creatinine": "Creatinine (mg/dL)"
}

# ======================
# 二分类变量
# ======================
binary_features = [
    "Loop diuretic use",
    "Inotrope use",
    "Vasopressor use"
]

binary_name_map = {
    "Diuretic use": "Diuretic use",
    "Inotrope use": "Inotrope use",
    "Vasopressor use": "Vasopressor use"
}

# ======================
# 标题 + QR
# ======================
title_col, qr_col = st.columns([4,1])

with title_col:
    st.markdown("""
    <h3 style='background-color:#1f77b4;color:white;padding:12px;border-radius:6px;'>
   Prediction of short-term mortality in patients with acute coronary syndrome complicated by heart failure
    </h3>
    """, unsafe_allow_html=True)

with qr_col:
    qr = qrcode.make("http://你的网址")  # 👉改成你部署的网址
    st.image(qr, width=120)

st.info("""
a. Predict 28-day mortality  
b. Based on machine learning  
c. Input within 24h ICU admission  
""")

# ======================
# 布局
# ======================
col1, col2 = st.columns([1,2])

# ======================
# 左侧输入
# ======================
with col1:

    st.subheader("Input Parameters")

    input_data = {}
    cols = st.columns(3)

    for i, col in enumerate(feature_names):
        with cols[i % 3]:

            # ===== Yes/No变量 =====
            if col in binary_features:
                display = binary_name_map[col]
                val = st.selectbox(display, ["No", "Yes"])
                input_data[col] = 1 if val == "Yes" else 0

            # ===== 数值变量（带单位）=====
            else:
                label = unit_map.get(col, col)
                default_val = float(X_full[col].median())
                input_data[col] = st.number_input(label, value=default_val)

    predict_btn = st.button("Calculate", use_container_width=True)

# ======================
# 右侧结果
# ======================
with col2:

    if predict_btn:

        input_df = pd.DataFrame([input_data])

        # ===== 预测 =====
        prob = model.predict_proba(input_df)[0][1]

        # ===== SHAP =====
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # ===== Force Plot =====
        st.subheader("Force Plot")

        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values,
            input_df
        )

        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        st.components.v1.html(shap_html, height=280)

        # ===== Waterfall =====
        st.subheader("Waterfall Plot")

        fig = plt.figure(figsize=(6,4))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value,
            shap_values[0],
            feature_names=feature_names
        )
        st.pyplot(fig)

        # ===== 结果 =====
        st.subheader("Prediction Results")

        if prob > 0.5:
            st.markdown(
                f"<div style='background:#f8d7da;padding:12px;border-radius:6px;'>"
                f"<b>Predicted risk of death: {prob:.2%} &nbsp;&nbsp;&nbsp; Predicted label: Death</b></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='background:#d4edda;padding:12px;border-radius:6px;'>"
                f"<b>Predicted survival probability: {(1-prob):.2%} &nbsp;&nbsp;&nbsp; Predicted label: Survival</b></div>",
                unsafe_allow_html=True
            )