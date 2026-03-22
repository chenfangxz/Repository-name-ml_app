import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt

# ====== 页面 ======
st.title("Mortality Risk Calculator")

# ====== 加载模型 ======
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

explainer = shap.Explainer(model)

# ====== 输入区域 ======
st.subheader("Input Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age(years)", 0.0, 120.0, 60.0)
    pco2 = st.number_input("Pco2(mmHg)", 0.0, 100.0, 40.0)
    po2 = st.number_input("Po2(mmHg)", 0.0, 200.0, 80.0)
    ph = st.number_input("PH", 6.5, 8.0, 7.4)
    na = st.number_input("Na+(mmol/L)", 100.0, 180.0, 140.0)
    cl = st.number_input("CL-(mmol/L)", 70.0, 140.0, 100.0)
    mg = st.number_input("Mg2+(mg/dL)", 0.5, 5.0, 2.0)

with col2:
    ag = st.number_input("AG(mEq/L)", 0.0, 40.0, 12.0)
    rbc = st.number_input("RBC(m/uL)", 0.0, 10.0, 4.5)
    wbc = st.number_input("WBC(K/uL)", 0.0, 50.0, 8.0)
    plt_ = st.number_input("PLT(K/uL)", 0.0, 1000.0, 200.0)
    rdw = st.number_input("RDW(%)", 0.0, 30.0, 13.0)
    glu = st.number_input("Glu(mg/dL)", 0.0, 500.0, 100.0)
    bun = st.number_input("BUN(mg/dL)", 0.0, 200.0, 20.0)

with col3:
    creatinine = st.number_input("Creatinine(mg/dL)", 0.0, 10.0, 1.0)
    alt = st.number_input("ALT(IU/L)", 0.0, 1000.0, 30.0)
    sofa = st.number_input("SOFA", 0.0, 30.0, 5.0)
    sapsii = st.number_input("SAPSII", 0.0, 100.0, 30.0)

    diuretic = st.selectbox("Diuretic use", ["Yes", "No"])
    inotrope = st.selectbox("Inotrope use", ["Yes", "No"])
    vasopressor = st.selectbox("Vasopressor use", ["Yes", "No"])

# ====== 计算 ======
if st.button("Calculate"):

    input_data = pd.DataFrame([{
        "Age(years)": age,
        "Pco2(mmHg)": pco2,
        "Po2(mmHg)": po2,
        "PH": ph,
        "Na+(mmol/L)": na,
        "CL-(mmol/L)": cl,
        "Mg2+(mg/dL)": mg,
        "AG(mEq/L)": ag,
        "RBC(m/uL)": rbc,
        "WBC(K/uL)": wbc,
        "PLT(K/uL)": plt_,
        "RDW(%)": rdw,
        "Glu(mg/dL)": glu,
        "BUN(mg/dL)": bun,
        "Creatinine(mg/dL)": creatinine,
        "ALT(IU/L)": alt,
        "SOFA": sofa,
        "SAPSII": sapsii,
        "Diuretic use": diuretic,
        "Inotrope use": inotrope,
        "Vasopressor use": vasopressor
    }])

    # ====== 预测 ======
    risk = model.predict_proba(input_data)[0][1]

    st.success(f"Predicted Risk of Death: {risk:.2%}")

    # ====== SHAP图 ======
    shap_values = explainer(input_data)

    st.subheader("Waterfall Plot")

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)