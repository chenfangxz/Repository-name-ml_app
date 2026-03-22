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

# ⭐ 手动固定变量（与你模型一致！！）
feature_names = [
    "Diuretic use",
    "ALT (IU/L)",
    "Mg2+ (mg/dL)",
    "RDW (%)",
    "WBC (K/uL)",
    "Glu (mg/dL)",
    "RBC (m/uL)",
    "PLT (K/uL)",
    "Inotrope use",
    "Po2 (mmHg)",
    "Vasopressor use",
    "BUN (mg/dL)",
    "SAPS II",
    "Creatinine (mg/dL)",
    "SOFA",
    "Anion Gap (mEq/L)",
    "Pco2 (mmHg)",
    "pH",
    "Na+ (mEq/L)",
    "Cl- (mEq/L)"
]

# ⭐ 默认值（仍然用template）
default_values = template.median(numeric_only=True)

# ======================
# 单位（已全部修正）
# ======================
unit_map = {
    "SOFA": "(score)",
    "SAPS II": "(score)",
    "PLT (K/uL)": "(K/uL)",
    "RDW (%)": "(%)",
    "RBC (m/uL)": "(m/uL)",
    "WBC (K/uL)": "(K/uL)",
    "Glu (mg/dL)": "(mg/dL)",
    "Na+ (mEq/L)": "(mEq/L)",
    "Anion Gap (mEq/L)": "(mEq/L)",
    "Cl- (mEq/L)": "(mEq/L)",
    "Mg2+ (mg/dL)": "(mg/dL)",
    "Pco2 (mmHg)": "(mmHg)",
    "Po2 (mmHg)": "(mmHg)",
    "pH": "",
    "ALT (IU/L)": "(IU/L)",
    "BUN (mg/dL)": "(mg/dL)",
    "Creatinine (mg/dL)": "(mg/dL)"
}

# ======================
# 二维码
# ======================
url = "https://repository-name-mlapp-w5nvkm7csffnjihvpuej5q.streamlit.app/"

qr = qrcode.make(url)
buf = BytesIO()
qr.save(buf, format="PNG")
qr_img = buf.getvalue()

# ======================
# 标题
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
a. Based on MIMIC-IV database  
b. Machine learning (CatBoost)  
c. Input data within 24h of ICU admission  
""")

# ======================
# 布局
# ======================
col1, col2 = st.columns([1,2])

# ======================
# 输入区
# ======================
with col1:

    st.subheader("Input Parameters")

    input_data = {}
    cols = st.columns(3)

    for i, col in enumerate(feature_names):
        with cols[i % 3]:

            display_name = col + " " + unit_map.get(col, "")

            # ⭐ 二分类变量（已修复）
            if col in ["Diuretic use", "Inotrope use", "Vasopressor use"]:
                val = st.selectbox(col, ["No", "Yes"])
                input_data[col] = 1 if val == "Yes" else 0

            else:
                default_val = float(default_values.get(col, 0))
                input_data[col] = st.number_input(display_name, value=default_val)

    predict_btn = st.button("Calculate", use_container_width=True)

# ======================
# 结果区
# ======================
with col2:

    if predict_btn:

        input_df = pd.DataFrame([input_data])

        # ⭐ 强制顺序一致（关键）
        input_df = input_df[feature_names]

        # ======================
        # 预测
        # ======================
        prob = model.predict_proba(input_df)[0][1]

        # ======================
        # SHAP
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

        fig = plt.figure()
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value,
            shap_values[0],
            feature_names=feature_names
        )
        st.pyplot(fig)

        # ======================
        # 结果
        # ======================
        st.subheader("Prediction Results")

        if prob > 0.5:
            st.error(f"Predicted risk of death: {prob:.2%} | Label: Death")
        else:
            st.success(f"Predicted survival probability: {(1-prob):.2%} | Label: Survival")