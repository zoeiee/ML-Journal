import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Input scaler (use the same one as training)
scaler = StandardScaler()

st.title("Placement Prediction App")
st.write("Enter CGPA and IQ to predict if the student gets placed (1) or not (0).")

# Inputs
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01, value=7.5)
iq = st.number_input("IQ Score", min_value=0, max_value=200, step=1, value=110)

if st.button("Predict"):
    # Prepare input
    X_new = np.array([[cgpa, iq]])
    # Scale input
    X_new_scaled = scaler.fit_transform(X_new)  # optional: you could save and reuse original scaler
    # Predict
    prediction = model.predict(X_new_scaled)
    st.success(f"Prediction: {'Placed (1)' if prediction[0]==1 else 'Not Placed (0)'}")