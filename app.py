import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model and preprocessing objects
model = joblib.load('xgb_model.pkl')
selector = joblib.load('selector.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define the selected features
selected_features = [
    'العام المالي', 'رأس المال', 'نسبة السيولة', 'مضاعف حقوق الملكية', 'صافي الربح', 'معدل دوران الاصول',
    'نسبة الديون الى حقوق الملكية', 'الديون الى اجمالي الخصوم', 'العائد على الأصول ROA', 'العائد على حقوق الملكية ROI'
]

# UI Design
st.title("AI Prediction System")

st.write("Please enter the data to make a prediction:")

# Create input fields for each feature
user_inputs = {}
for feature in selected_features:
    user_inputs[feature] = st.number_input(feature, format="%.3f")

# Convert inputs into a DataFrame
input_data_manual = pd.DataFrame([user_inputs])

# Ensure feature order matches the training data
try:
    input_data_manual = input_data_manual[scaler.feature_names_in_]
except AttributeError:
    st.error("Error: The scaler object does not have 'feature_names_in_'. Ensure the correct scaler is loaded.")
    st.stop()

# Standardize and transform data
input_data_scaled_manual = scaler.transform(input_data_manual)
input_data_selected_manual = selector.transform(input_data_scaled_manual)

# Predict Classification
if st.button('Predict Classification'):
    try:
        prediction = model.predict(input_data_selected_manual)
        predicted_class = label_encoder.inverse_transform(prediction)[0]

        # Display the result
        st.markdown(f"""
            <div style="background-color: #18E1D9; color: white; padding: 20px; border-radius: 10px; font-size: 24px; text-align: center;">
                <strong>Predicted Classification: {predicted_class}</strong>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# Show feature importance
if st.button("Show Feature Importances"):
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': selected_features[:len(feature_importances)],
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    st.write("### Feature Importances")
    st.dataframe(importance_df)

# Show Encoding Mapping
if st.button("Show Encoding Mapping"):
    encoding_dict = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    encoding_df = pd.DataFrame(list(encoding_dict.items()), columns=['Original Class', 'Encoded Value'])

    st.write("### Encoding Mapping")
    st.dataframe(encoding_df)
