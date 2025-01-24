import pandas as pd
import numpy as np
import pyttsx3
import streamlit as st
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import os

# Step 1: Load the dataset
file_path = 'dsg.csv'
if not os.path.exists(file_path):
    st.error("Error: Dataset not found! Please ensure 'dsg.csv' is in the same directory.")
    st.stop()  # Stop execution if the file is not found
else:
    df = pd.read_csv(file_path)
    st.write("### Dataset Loaded Successfully")
    # Removed dataset display here

# Step 2: Prepare the data
X = df.drop('prognosis', axis=1)  # Features (symptoms)
y = df['prognosis']  # Target (disease)

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 3: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 4: Build, Train, and Evaluate the Deep Learning Model
@st.cache_data  # Updated from st.cache to st.cache_data
def build_and_train_model():
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        Dense(len(np.unique(y_encoded)), activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
    return model

model = build_and_train_model()

# Step 5: Prediction Functions
def speak_text(text):
    """Convert text to speech."""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"Error in speech synthesis: {e}")

def predict_disease(symptoms):
    """Predict disease based on selected symptoms."""
    if len(symptoms) > 0:
        input_data = pd.DataFrame(0, index=[0], columns=X_train.columns)
        for symptom in symptoms:
            if symptom in input_data.columns:
                input_data[symptom] = 1
        prediction = model.predict(input_data)
        predicted_disease = label_encoder.inverse_transform([np.argmax(prediction)])
        return predicted_disease[0]
    else:
        return "No symptoms provided."

def predict_disease_with_details(name, age, phone, address, symptoms):
    """Combine patient details with prediction."""
    if not all([name, age, phone, address]):
        return "Please fill in all patient details."
    patient_info = f"**Patient Details:**\nName: {name}\nAge: {age}\nPhone: {phone}\nAddress: {address}"
    predicted_disease = predict_disease(symptoms)
    result_text = f"{patient_info}\n**Symptoms:** {', '.join(symptoms)}\n**Predicted Disease:** {predicted_disease}"
    return result_text  # Removed speak_text from here

# Initialize or load the Excel file
def initialize_excel(file_name='predicted_diseases.xlsx'):
    if not os.path.exists(file_name):
        df = pd.DataFrame(columns=["Name", "Age", "Phone", "Address", "Symptoms", "Predicted Disease"])
        df.to_excel(file_name, index=False)

def save_prediction_to_excel(name, age, phone, address, symptoms, predicted_disease, file_name='predicted_diseases.xlsx'):
    initialize_excel(file_name)
    df = pd.read_excel(file_name)
    new_entry = {
        "Name": name,
        "Age": age,
        "Phone": phone,
        "Address": address,
        "Symptoms": ', '.join(symptoms),
        "Predicted Disease": predicted_disease
    }
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_excel(file_name, index=False)

# Step 6: Streamlit Interface
st.title("🩺 Clinical Disease Detection System")
st.write("Enter patient details and symptoms to predict a disease.")

# Input Fields
name_input = st.text_input("Name")
age_input = st.number_input("Age", min_value=0, max_value=100, step=1)
phone_input = st.text_input("Phone")
address_input = st.text_input("Address")

# Select Symptoms
symptom_options = X_train.columns.tolist()
symptoms_input = st.multiselect("Select Symptoms", symptom_options)

# Prediction Button
if st.button('🔍 Predict Disease'):
    if len(symptoms_input) == 0:
        st.error("Please select at least one symptom.")
    else:
        result = predict_disease_with_details(name_input, age_input, phone_input, address_input, symptoms_input)
        if result.startswith("Please fill in all patient details."):
            st.error(result)
        else:
            predicted_disease = predict_disease(symptoms_input)
            save_prediction_to_excel(name_input, age_input, phone_input, address_input, symptoms_input, predicted_disease)
            st.success(result)
            st.info(f"Prediction saved to `predicted_diseases.xlsx`.")
            # Call speak_text after displaying the result
            speak_text(result)
