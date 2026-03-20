import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import qrcode
from io import BytesIO

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
numeric_features = template.select_dtypes(include=['int64','float64']).columns

# ======================
# ⭐ 变量单位（论文级）
# ======================
unit_map = {
    "SOFA": "(score)",
    "SAPSII": "(score)",
    "PLT": "(K/μL)",
    "RDW": "(%)",
    "RBC": "(×10^6/μL)",
    "WBC": "(K/μL)",
    "Glu": "(mg/dL)",
    "K+": "(mmol/L)",
    "Na+": "(mmol/L)",
    "AG": "(mEq/L)",
    "Ca2+": "(mg/dL)",
    "CL-": "(mmol/L)",
    "Mg2+": "(mg/dL)",
    "Pco2": "(mmHg)",
    "Po2": "(mmHg)",
    "PH": "",
    "ALT": "(IU/L)",
    "BUN": "(mg/dL)",
    "Creatinine": "(mg/dL)"
}

# ======================
# ⭐ 二维码（自动生成）
# ======================
url = "https://repository-name-mlapp-w5nvkm7csffnjihvpuej5q.streamlit.app/"   # ⭐这里改成你的真实网址

qr = qrcode.make(url)
buf = BytesIO()
qr.save(buf, format="PNG")
qr_img = buf.getvalue()

# ======================
# 标题 + 二维码
# ======================
col_title, col_qr = st.columns([5,1])

with col_title:
    st.markdown("""
    <h3 style='background-color:#1f77b4;color:white;padding:10px;border-radius:5px;'>
    Prediction of short-term mortality in patients with acute coronary syndrome complicated by heart failure
    </h3>
    """, unsafe_allow_html=True)

with col_qr:
    st.image(qr_img, width=120)

st.info("""
a. This tool predicts 28-day mortality risk based on MIMIC-IV database  
b. Based on machine learning (CatBoost model)  
c. Input patient data within 24h of ICU admission  
""")

# ======================
# 布局
# ======================
col1, col2 = st.columns([1, 2])

# ======================
# 左侧输入
# ======================
with col1:

    st.subheader("Input Parameters")

    input_data = {}
    cols = st.columns(3)

    for i, col in enumerate(feature_names):
        with cols[i % 3]:

            # ⭐ 显示变量 + 单位
            display_name = col + " " + unit_map.get(col, "")

            # ⭐ Yes / No变量（重点）
            if col in ["Diuretic use", "Inotrope use", "Vasopressor use"]:
                display_name = col.replace("Loop ", "")
                val = st.selectbox(display_name, ["No", "Yes"])
                input_data[col] = 1 if val == "Yes" else 0

            else:
                input_data[col] = st.number_input(display_name, value=0.0)

    predict_btn = st.button("Calculate", use_container_width=True)

# ======================
# 右侧结果
# ======================
with col2:

    if predict_btn:

        input_df = pd.DataFrame([input_data])

        # ======================
        # 预测
        # ======================
        prob = model.predict_proba(input_df)[0][1]

        # ======================
        # SHAP解释
        # ======================
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # ======================
        # Force Plot
        # ======================
        st.subheader("Force Plot")

        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values,
            input_df
        )

        shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
        st.components.v1.html(shap_html, height=250)

        # ======================
        # Waterfall Plot
        # ======================
        st.subheader("Waterfall Plot")

        fig2 = plt.figure()
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value,
            shap_values[0],
            feature_names=feature_names
        )
        st.pyplot(fig2)

        # ======================
        # 结果显示
        # ======================
        st.subheader("Prediction Results")

        if prob > 0.5:
            st.markdown(
                f"<div style='background:#f8d7da;padding:12px;border-radius:6px;'>"
                f"<b>Predicted risk of death: {prob:.2%} | Label: Death</b></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='background:#d4edda;padding:12px;border-radius:6px;'>"
                f"<b>Predicted survival probability: {(1-prob):.2%} | Label: Survival</b></div>",
                unsafe_allow_html=True
            )