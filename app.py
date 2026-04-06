import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- LOAD FILES ----------
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")
mean_values = joblib.load("mean_values.pkl")

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Medicare Cost Predictor", layout="wide")

# ---------- TITLE ----------
st.title("💊 Medicare Cost Prediction Dashboard")

st.markdown("""
This dashboard predicts the **Medicare Allowed Amount**, which represents the reimbursement approved by Medicare 
for healthcare services provided by a provider.

The prediction is based on:
- 📊 Service volume
- 👨‍⚕️ Patient demographics
- 🩺 Chronic health conditions

👉 Enter the details below to estimate expected healthcare cost.
""")

# ---------- INPUT LAYOUT ----------
col1, col2 = st.columns(2)

# ---------- SERVICE DETAILS ----------
with col1:
    st.subheader("📊 Service Details")
    st.caption("Represents provider workload and billing activity")

    tot_srvcs = st.number_input(
        "Total Services Provided", min_value=0,
        help="Total number of services performed over a year"
    )

    tot_benes = st.number_input(
        "Total Beneficiaries", min_value=0,
        help="Number of unique patients treated over a year"
    )

    submitted_charge = st.number_input(
        "Submitted Charges ($)", min_value=0.0,
        help="Total billing amount over a year before Medicare approval "
    )

# ---------- PATIENT DETAILS ----------
with col2:
    st.subheader("👨‍⚕️ Patient Profile")
    st.caption("Describes patient characteristics")

    avg_age = st.number_input(
    "Average Patient Age",
    min_value=0,
    help="Represents the average age of patients receiving services over a year. Age is an important factor in healthcare cost prediction, as older patients typically require more frequent medical care, have higher chances of chronic conditions, and utilize more healthcare resources."
)

    risk = st.number_input(
    "Average Risk Score", 
    min_value=0.0,
    help=("""This score represents the overall health complexity of the patient population over a year.
    It is based on:
- Number and severity of chronic conditions  
- Patient age and medical history  
- Overall disease burdens
👉 Higher score = More complex patients → More care needed → Higher Medicare cost
"""))

# ---------- HEALTH CONDITIONS ----------
st.subheader("🩺 Chronic Condition Distribution (%)")
st.caption("Percentage of patients with specific conditions")

col3, col4, col5 = st.columns(3)

with col3:
    diabetes = st.slider("Diabetes %", 0, 100)

with col4:
    hypertension = st.slider("Hypertension %", 0, 100)

with col5:
    copd = st.slider("COPD %(Chronic Obstructive Pulmonary Disease)", 0, 100)

# ---------- CREATE INPUT ----------
input_dict = {}

for col in columns:
    if col in mean_values:
        input_dict[col] = mean_values[col]
    else:
        input_dict[col] = 0

input_dict['Tot_Srvcs'] = tot_srvcs
input_dict['Tot_Benes'] = tot_benes
input_dict['Med_Sbmtd_Chrg'] = submitted_charge
input_dict['Bene_Avg_Age'] = avg_age
input_dict['Bene_Avg_Risk_Scre'] = risk
input_dict['Bene_CC_PH_Diabetes_V2_Pct'] = diabetes
input_dict['Bene_CC_PH_Hypertension_V2_Pct'] = hypertension
input_dict['Bene_CC_PH_COPD_V2_Pct'] = copd

input_df = pd.DataFrame([input_dict])

# ---------- PREDICTION ----------
st.markdown("---")

if st.button("🚀 Predict Medicare Allowed Amount"):

    prediction_log = model.predict(input_df)
    prediction = np.expm1(prediction_log)

    st.success(f"💰 Predicted Medicare Allowed Amount: ${prediction[0]:,.2f}")

    # ---------- COST DIFFERENCE ----------
    difference = prediction[0] - submitted_charge

    st.markdown("### 📊 Cost Comparison Insight")

    colA, colB = st.columns(2)

    with colA:
        st.metric("Predicted Amount ($)", f"{prediction[0]:,.2f}")

    with colB:
        st.metric("Difference ($)", f"{difference:,.2f}")

    if difference > 0:
        st.success(f"📈 Medicare is likely to reimburse **${abs(difference):,.2f} MORE** than submitted charges.")
    elif difference < 0:
        st.error(f"📉 Medicare is likely to reimburse **${abs(difference):,.2f} LESS** than submitted charges.")
    else:
        st.info("✅ Submitted charges and predicted reimbursement are approximately equal.")

    st.caption("""
This metric shows the difference between predicted Medicare Allowed Amount and submitted charges:
- Positive → Higher reimbursement expected  
- Negative → Possible under-reimbursement or overbilling  
- Helps providers understand financial gaps and pricing strategy
""")

    st.info("Prediction is based on healthcare utilization and patient condition patterns.")

    # ---------- GRAPHS ----------
    st.markdown("### 📊 Visual Insights")

    col1, col2, col3 = st.columns(3)

    # ---------- GRAPH 1 ----------
    with col1:
        st.markdown("#### Input Summary")

        features = {
            "Services": tot_srvcs,
            "Beneficiaries": tot_benes,
            "Charges": submitted_charge,
            "Age": avg_age,
            "Risk": risk
        }

        fig, ax = plt.subplots()
        ax.bar(features.keys(), features.values())
        ax.set_title("Input Overview")

        st.pyplot(fig)

    # ---------- GRAPH 2 ----------
    with col2:
        st.markdown("#### Health Conditions")

        health_data = {
            "Diabetes": diabetes,
            "Hypertension": hypertension,
            "COPD": copd
        }

        fig2, ax2 = plt.subplots()
        ax2.bar(health_data.keys(), health_data.values())
        ax2.set_title("Condition %")

        st.pyplot(fig2)

    # ---------- GRAPH 3 ----------
    with col3:
        st.markdown("#### Cost Sensitivity")

        sim_values = list(range(0, 2000, 200))
        sim_predictions = []

        for val in sim_values:
            temp_input = input_df.copy()
            temp_input['Tot_Srvcs'] = val

            pred_log = model.predict(temp_input)
            pred = np.expm1(pred_log)

            sim_predictions.append(pred[0])

        fig3, ax3 = plt.subplots()
        ax3.plot(sim_values, sim_predictions)
        ax3.set_title("Services vs Cost")

        st.pyplot(fig3)