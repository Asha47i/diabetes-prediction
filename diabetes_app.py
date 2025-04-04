import numpy as np
import streamlit as st
import pickle
import os

# Correct file path for macOS
MODEL_PATH = 'trained_model.sav'  # Since the file is in the same directory
with open(MODEL_PATH, 'rb') as model_file:
    loaded_model = pickle.load(model_file)


# Load the trained model
try:
    with open(MODEL_PATH, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
except FileNotFoundError:
    st.error(f"Model file not found at: {MODEL_PATH}. Please check the file path.")

# Function for prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)  # Ensure float conversion
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    return 'The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic'

# Main Streamlit App
def main():
    st.title('Diabetes Prediction Web App')

    # Get user input
    Pregnancies = st.text_input('Number of pregnancies', '0')
    Glucose = st.text_input('Glucose level', '0')
    BloodPressure = st.text_input('BloodPressure levels', '0')
    SkinThickness = st.text_input('Skin thickness value', '0')
    Insulin = st.text_input('Insulin levels', '0')
    BMI = st.text_input('BMI value', '0')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', '0')
    Age = st.text_input('Age value', '0')

    # Code for prediction
    diagnosis = ''

    # Create a button for prediction
    if st.button('Get Diabetes Test Result'):
        try:
            input_values = [
                float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)
            ]
            diagnosis = diabetes_prediction(input_values)
        except ValueError:
            diagnosis = "Please enter valid numeric values."

    st.success(diagnosis)

# Run the app
if __name__ == '__main__':
    main()
