import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ======================
# 页面设置
# ======================
st.set_page_config(layout="wide")

# ======================
# 加载模型
# ======================
model = joblib.load("catboost_final_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# ======================
# 标题（蓝色横条）
# ======================
st.markdown("""
<h2 style='background-color:#2c7fb8;color:white;padding:12px;border-radius:6px;'>
Prediction of short-term mortality in patients with acute coronary syndrome complicated by heart failure
</h2>
""", unsafe_allow_html=True)

# ======================
# 说明框
# ======================
st.markdown("""
<div style='background-color:#e6f0fa;padding:15px;border-radius:6px;'>
a. This tool predicts 28-day mortality risk based on MIMIC-IV database<br>
b. Based on machine learning (CatBoost model)<br>
c. Input patient data within 24h of ICU admission
</div>
""", unsafe_allow_html=True)

# ======================
# 布局
# ======================
col1, col2 = st.columns([1, 2])

# ======================
# 左侧输入（🔥三列排版）
# ======================
with col1:
    st.subheader("Input Parameters")

    input_data = {}

    # ===== Row 1 =====
    c1, c2, c3 = st.columns(3)
    with c1:
        input_data['SOFA'] = st.number_input("SOFA (score)", value=14.0)
    with c2:
        input_data['SAPSII'] = st.number_input("SAPSII (score)", value=60.0)
    with c3:
        input_data['PLT'] = st.number_input("PLT (K/uL)", value=80.0)

    # ===== Row 2 =====
    c1, c2, c3 = st.columns(3)
    with c1:
        input_data['RDW'] = st.number_input("RDW (%)", value=18.0)
    with c2:
        input_data['RBC'] = st.number_input("RBC (m/uL)", value=3.0)
    with c3:
        input_data['WBC'] = st.number_input("WBC (K/uL)", value=18.0)

    # ===== Row 3 =====
    c1, c2, c3 = st.columns(3)
    with c1:
        input_data['Glu'] = st.number_input("Glucose (mg/dL)", value=180.0)
    with c2:
        input_data['Na+'] = st.number_input("Na+ (mmol/L)", value=130.0)
    with c3:
        input_data['AG'] = st.number_input("AG (mEq/L)", value=20.0)

    # ===== Row 4 =====
    c1, c2, c3 = st.columns(3)
    with c1:
        input_data['CL-'] = st.number_input("Cl- (mmol/L)", value=95.0)
    with c2:
        input_data['Mg2+'] = st.number_input("Mg2+ (mg/dL)", value=2.5)
    with c3:
        input_data['Pco2'] = st.number_input("PCO2 (mmHg)", value=55.0)

    # ===== Row 5 =====
    c1, c2, c3 = st.columns(3)
    with c1:
        input_data['Po2'] = st.number_input("PO2 (mmHg)", value=60.0)
    with c2:
        input_data['PH'] = st.number_input("pH", value=7.26)
    with c3:
        input_data['ALT'] = st.number_input("ALT (IU/L)", value=120.0)

    # ===== Row 6 =====
    c1, c2, c3 = st.columns(3)
    with c1:
        input_data['Creatinine'] = st.number_input("Creatinine (mg/dL)", value=2.5)
    with c2:
        input_data['BUN'] = st.number_input("BUN (mg/dL)", value=40.0)
    with c3:
        input_data['Diuretic use'] = st.selectbox("Diuretic use", [0, 1])

    # ===== Row 7 =====
    c1, c2, c3 = st.columns(3)
    with c1:
        input_data['Inotrope use'] = st.selectbox("Inotrope use", [0, 1])
    with c2:
        input_data['Vasopressor use'] = st.selectbox("Vasopressor use", [0, 1])
    with c3:
        st.write("")

    st.markdown("<br>", unsafe_allow_html=True)

    calculate = st.button("🔍 Calculate")

# ======================
# 右侧：结果 + 图
# ======================
with col2:

    if calculate:

        input_df = pd.DataFrame([input_data])

        # ===== 预测 =====
        prob = model.predict_proba(input_df)[0][1]

        # ===== SHAP =====
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        # ===== Force Plot =====
        st.subheader("Force Plot")

        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            input_df,
            matplotlib=True,
            show=False
        )
        st.pyplot(plt.gcf())
        plt.clf()

        # ===== Waterfall =====
        st.subheader("Waterfall Plot")

        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0]
            ),
            show=False
        )
        st.pyplot(plt.gcf())
        plt.clf()

        # ===== 结果条（红色）=====
        st.markdown(f"""
        <div style='background-color:#f8d7da;padding:15px;border-radius:6px;margin-top:20px;'>
        <b>Predicted risk of death: {prob*100:.2f}%</b>
        </div>
        """, unsafe_allow_html=True)