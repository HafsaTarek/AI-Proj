import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model and preprocessing objects
model = joblib.load('xgb_model.pkl')
selector = joblib.load('selector.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# The correct order of selected features from the model (used during training)
selected_features = [
    'رأس المال', 'نسبة السيولة', 'مضاعف حقوق الملكية', 'صافي الربح', 'معدل دوران الاصول',
    'نسبة الديون الى حقوق الملكية', 'الديون الى اجمالي الخصوم', 'العائد على الأصول ROA', 'العائد على حقوق الملكية ROI'
]

# UI to input new data for prediction
st.title("AI Prediction System")
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        h1 {
            color: #18E1D9;
            font-size: 32px;
        }
        .stButton > button {
            background-color: #0B0B15;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            padding: 10px;
        }
        .stButton > button:hover {
            background-color: #18E1D9;
        }
        .stNumberInput input {
            border: 2px solid #18E1D9;
            border-radius: 8px;
            font-size: 16px;
        }
        .stTextInput input {
            border: 2px solid #18E1D9;
            border-radius: 8px;
            font-size: 16px;
        }
        .stWrite {
            color: #0B0B15;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

st.write("Please enter the data to make a prediction:")

# Create input fields for each feature
رأس_المال = st.number_input('رأس المال', min_value=0.0, format="%.2f")
نسبة_السيولة = st.number_input('نسبة السيولة', min_value=0.0, format="%.2f")
مضاعف_حقوق_الملكية = st.number_input('مضاعف حقوق الملكية', min_value=0.0, format="%.2f")
صافي_الربح = st.number_input('صافي الربح', min_value=0.0, format="%.2f")
معدل_دوران_الاصول = st.number_input('معدل دوران الاصول', min_value=0.0, format="%.2f")
نسبة_الديون_الى_حقوق_الملكية = st.number_input('نسبة الديون الى حقوق الملكية', min_value=0.0, format="%.2f")
الديون_الى_اجمالي_الخصوم = st.number_input('الديون الى اجمالي الخصوم', min_value=0.0, format="%.2f")
العائد_على_الأصول_ROA = st.number_input('العائد على الأصول ROA', min_value=0.0, format="%.2f")
العائد_على_حقوق_الملكية_ROI = st.number_input('العائد على حقوق الملكية ROI', min_value=0.0, format="%.2f")

# Collect all inputs into a DataFrame
input_data_manual = pd.DataFrame({
    'رأس المال': [رأس_المال],
    'نسبة السيولة': [نسبة_السيولة],
    'مضاعف حقوق الملكية': [مضاعف_حقوق_الملكية],
    'صافي الربح': [صافي_الربح],
    'معدل دوران الاصول': [معدل_دوران_الاصول],
    'نسبة الديون الى حقوق الملكية': [نسبة_الديون_الى_حقوق_الملكية],
    'الديون الى اجمالي الخصوم': [الديون_الى_اجمالي_الخصوم],
    'العائد على الأصول ROA': [العائد_على_الأصول_ROA],
    'العائد على حقوق الملكية ROI': [العائد_على_حقوق_الملكية_ROI]
})

# Normalize the input data using the same scaler
input_data_scaled_manual = scaler.transform(input_data_manual)

# Apply feature selection (same as done during training)
input_data_selected_manual = selector.transform(input_data_scaled_manual)

# Add a prediction button
if st.button('Predict Classification'):
    # Make prediction
    prediction = model.predict(input_data_selected_manual)
    predicted_class = label_encoder.inverse_transform(prediction)[0]

    # Display the result with styling
    st.markdown(f"""
        <div style="background-color: #18E1D9; color: white; padding: 20px; border-radius: 10px; font-size: 24px; text-align: center;">
            <strong>Predicted Classification: {predicted_class}</strong>
        </div>
    """, unsafe_allow_html=True)
