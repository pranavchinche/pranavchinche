import streamlit as st
import pandas as pd
import joblib
import os

st.title("🔍 System Diagnostic & Predictor")

# --- STEP 1: Check if files exist ---
files = ['linear_regression_model.joblib', 'gender_encoder.joblib', 'education_mapping.joblib']
missing_files = [f for f in files if not os.path.exists(f)]

if missing_files:
    st.error(f"❌ Missing files in folder: {missing_files}")
    st.info("Go back to your Notebook and run the joblib.dump() code to save these files.")
    st.stop()
else:
    st.success("✅ All model files found!")

# --- STEP 2: Load Assets ---
@st.cache_resource
def load_all_assets():
    model = joblib.load('linear_regression_model.joblib')
    ohe = joblib.load('gender_encoder.joblib')
    edu_map = joblib.load('education_mapping.joblib')
    return model, ohe, edu_map

model, ohe, edu_map = load_all_assets()

# --- STEP 3: UI ---
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    experience = st.number_input("Years of Experience", 0.0, 40.0, 5.0)
    # We use .get() to handle case sensitivity safely
    education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])

if st.button("Predict Salary"):
    # Create DataFrame
    input_df = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Education Level': [education],
        'Years of Experience': [experience]
    })

    # Apply Mappings
    input_df['Education Level'] = input_df['Education Level'].map(edu_map)
    
    # One-Hot Encode Gender
    gender_encoded = ohe.transform(input_df[['Gender']])
    gender_df = pd.DataFrame(gender_encoded, columns=ohe.get_feature_names_out(['Gender']))

    # Combine & Drop old Gender
    final_df = pd.concat([input_df.drop('Gender', axis=1), gender_df], axis=1)
    
    # Check Column Order (Crucial for Linear Regression)
    # If your model was trained on [Age, Education Level, Years of Experience, Gender_Female, Gender_Male]
    # we must ensure that exact order:
    expected_order = ['Age', 'Education Level', 'Years of Experience', 'Gender_Female', 'Gender_Male']
    
    try:
        # Reorder columns to match training exactly
        final_df = final_df[expected_order] 
        prediction = model.predict(final_df)
        st.balloons()
        st.success(f"### Predicted Salary: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Columns found in app:", final_df.columns.tolist())