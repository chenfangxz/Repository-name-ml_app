import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
import qrcode
from io import BytesIO

# ======================
# 页面设置
# ======================
st.set_page_config(layout="wide")

# ======================
# 加载模型（改这里🔥）
# ======================
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

# ⭐ 特征顺序（必须和训练一致）
feature_names = [
    "Age(years)","Pco2(mmHg)","Po2(mmHg)","PH","Na+(mmol/L)",
    "CL-(mmol/L)","Mg2+(mg/dL)","AG(mEq/L)","RBC(m/uL)","WBC(K/uL)",
    "PLT(K/uL)","RDW(%)","Glu(mg/dL)","BUN(mg/dL)",
    "Creatinine(mg/dL)","ALT(IU/L)","SOFA","SAPSII",
    "Diuretic use","Inotrope use","Vasopressor use"
]

# ======================
# 二维码
# ======================
url = "https://your-app.streamlit.app"

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
    Prediction of short-term mortality in patients with ACS complicated by heart failure
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

    cols = st.columns(3)

    # ⭐ 输入（简化但完整）
    inputs = {}

    for i, col in enumerate(feature_names):
        with cols[i % 3]:

            if col in ["Diuretic use","Inotrope use","Vasopressor use"]:
                val = st.selectbox(col, ["No","Yes"])
                inputs[col] = 1 if val == "Yes" else 0
            else:
                inputs[col] = st.number_input(col, value=0.0)

    predict_btn = st.button("Calculate", use_container_width=True)

# ======================
# 结果区
# ======================
with col2:

    if predict_btn:

        input_df = pd.DataFrame([inputs])

        # ⭐ 保证顺序一致（关键🔥）
        input_df = input_df[feature_names]

        # ======================
        # 预测
        # ======================
        prob = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Results")
        st.error(f"Predicted risk of death: {prob:.2%}")

        # ======================
        # SHAP（稳定写法🔥）
        # ======================
        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)

        # ======================
        # Force Plot
        # ======================
        st.subheader("Force Plot")

        force_html = shap.getjs() + shap.plots.force(
            shap_values[0], matplotlib=False
        ).html()

        st.components.v1.html(force_html, height=200)

        # ======================
        # Waterfall Plot
        # ======================
        st.subheader("Waterfall Plot")

        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
