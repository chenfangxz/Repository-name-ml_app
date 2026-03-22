import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt

# ======================
# 页面
# ======================
st.set_page_config(layout="wide")
st.title("Mortality Risk Calculator")

# ======================
# 加载模型
# ======================
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

explainer = shap.Explainer(model)

# ======================
# 特征顺序（必须一致🔥）
# ======================
feature_names = [
    "Pco2(mmHg)","Po2(mmHg)","PH","Na+(mmol/L)",
    "CL-(mmol/L)","Mg2+(mg/dL)","AG(mEq/L)","RBC(m/uL)","WBC(K/uL)",
    "PLT(K/uL)","RDW(%)","Glu(mg/dL)","BUN(mg/dL)",
    "Creatinine(mg/dL)","ALT(IU/L)","SOFA","SAPSII",
    "Diuretic use","Inotrope use","Vasopressor use"
]

# ======================
# 布局
# ======================
left, right = st.columns([1,2])

# ======================
# 输入
# ======================
with left:

    st.subheader("Input")

    inputs = {}

    for col in feature_names:

        # ⭐ 关键修复：三分类变量
        if col in ["Diuretic use","Inotrope use","Vasopressor use"]:
            val = st.selectbox(col, ["No","Yes"], key=col)
            inputs[col] = 1 if val == "Yes" else 0   # ← 强制转数字

        else:
            inputs[col] = st.number_input(col, value=0.0, key=col)

    run = st.button("Calculate")

# ======================
# 输出
# ======================
with right:

    if run:

        # 转DataFrame
        input_df = pd.DataFrame([inputs])

        # ⭐ 强制所有列为float（关键修复🔥）
        for c in input_df.columns:
            input_df[c] = pd.to_numeric(input_df[c])

        # 保证顺序
        input_df = input_df[feature_names]

        # ======================
        # 预测
        # ======================
        risk = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction")
        st.error(f"Risk of death: {risk:.2%}")

        # ======================
        # SHAP
        # ======================
        shap_values = explainer(input_df)

        # ===== Force Plot =====
        st.subheader("Force Plot")

        force_html = shap.getjs() + shap.plots.force(
            shap_values[0], matplotlib=False
        ).html()

        st.components.v1.html(force_html, height=200)

        # ===== Waterfall =====
        st.subheader("Waterfall Plot")

        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
