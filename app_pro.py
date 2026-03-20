import streamlit as st
import pandas as pd
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
model = joblib.load("catboost_model.pkl")
template = joblib.load("template.pkl")
feature_names = template.columns

# ⭐ 用template代替Excel（关键修复）
numeric_features = template.select_dtypes(include=['int64','float64']).columns

# ======================
# 标题
# ======================
st.markdown("""
<h3 style='background-color:#1f77b4;color:white;padding:10px;border-radius:5px;'>
Prediction of short-term mortality in patients with acute coronary syndrome complicated by heart failure</h3>
""", unsafe_allow_html=True)

st.info("""
a. This tool predicts 28-day mortality risk  
b. Based on machine learning model  
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

            # ⭐ 正确的二分类变量名称
            if col in ["Diuretic use", "Inotrope use", "Vasopressor use"]:
                
                # 显示名称优化
                display_name = col.replace("Loop ", "")
                
                val = st.selectbox(display_name, ["No", "Yes"])
                input_data[col] = 1 if val == "Yes" else 0

            # ⭐ 数值变量（用0或中位数，不能用X_full）
            elif col in numeric_features:
                input_data[col] = st.number_input(col, value=0.0)

            else:
                input_data[col] = st.number_input(col, value=0.0)

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
        # Waterfall
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
        # 结果
        # ======================
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