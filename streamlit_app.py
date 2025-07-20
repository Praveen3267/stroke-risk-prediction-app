import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model and encoders
@st.cache_resource
def load_model_and_encoders():
    with open("stroke_model_rf.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    return model, label_encoders

model, label_encoders = load_model_and_encoders()

# UI
st.title("🧠 Stroke Risk Prediction App (Loaded Model)")
st.write("Enter your health data to assess your stroke risk.")

gender = st.selectbox("Gender", label_encoders["gender"].classes_)
age = st.slider("Age", 1, 100, 50)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", label_encoders["ever_married"].classes_)
work_type = st.selectbox("Work Type", label_encoders["work_type"].classes_)
residence_type = st.selectbox("Residence Type", label_encoders["Residence_type"].classes_)
avg_glucose = st.number_input("Average Glucose Level", value=100.0)
bmi = st.number_input("BMI", value=25.0)
smoking_status = st.selectbox("Smoking Status", label_encoders["smoking_status"].classes_)

# Encode input
user_data = pd.DataFrame([{
    "gender": label_encoders["gender"].transform([gender])[0],
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": label_encoders["ever_married"].transform([ever_married])[0],
    "work_type": label_encoders["work_type"].transform([work_type])[0],
    "Residence_type": label_encoders["Residence_type"].transform([residence_type])[0],
    "avg_glucose_level": avg_glucose,
    "bmi": bmi,
    "smoking_status": label_encoders["smoking_status"].transform([smoking_status])[0]
}])

# Predict
def predict_stroke_risk(data, model, threshold=0.3):
    prob = model.predict_proba(data)[0][1]
    risk = "High" if prob >= threshold else "Low"
    return prob, risk

if st.button("Predict Stroke Risk"):
    prob, risk = predict_stroke_risk(user_data, model)
    st.markdown(f"🧠 **Predicted Stroke Risk:** {prob * 100:.2f}%")
    st.markdown(f"➡️ **Risk Category:** {risk} Risk")

    if risk == "High":
        st.error("⚠️ The model predicts stroke is likely (class = 1)")
        st.markdown("### 🩺 Next Steps You Should Consider:")
        st.markdown("""
        - Schedule a checkup with your doctor or cardiologist  
        - Control blood pressure, cholesterol, and glucose  
        - Quit smoking and reduce alcohol  
        - Follow a heart-healthy diet and exercise  
        - Take medications as prescribed
        """)
    else:
        st.success("✅ The model predicts no stroke (class = 0)")

st.caption("ℹ️ This tool is for educational purposes only. Always consult a healthcare provider.")
