# 🩺 Clinical Disease Detection System (CDDS)

The **Clinical Disease Detection System (CDDS)** is a deep learning-based application that predicts diseases based on symptoms, lifestyle factors, and health measurements. It features an interactive user interface and supports text-to-speech capabilities, making it user-friendly and highly functional. The application also stores patient details and prediction history for future reference.

---

## 📋 Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Details](#dataset-details)
- [Deep Learning Model](#deep-learning-model)
- [How It Works](#how-it-works)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features
- **Symptom-Based Predictions**: Predict diseases by selecting symptoms from a list.
- **Patient Information Management**: Capture and store patient details like name, age, phone number, and address.
- **Deep Learning Model**: A multi-class classification neural network for accurate predictions.
- **Speech Feedback**: Text-to-speech functionality reads out predictions for better accessibility.
- **Data Export**: Save prediction results to an Excel file for record-keeping.
- **Interactive Web Interface**: User-friendly interface built using Streamlit.

---

## 🛠️ Technologies Used
- **Python**: Backend logic and deep learning.
- **TensorFlow/Keras**: Building and training the neural network model.
- **Streamlit**: Interactive web-based user interface.
- **Pandas**: Data manipulation and preprocessing.
- **NumPy**: Numerical computations.
- **scikit-learn**: Label encoding and train-test splitting.
- **pyttsx3**: Text-to-speech for predictions.

---

## 🏗️ Project Structure

```plaintext
project/
├── dataset/
│   └── disease_dataset.csv          # Input dataset containing symptoms and disease mappings
├── templates/
│   └── index.html                   # Optional web-based frontend for Flask (if applicable)
├── webapp.py                        # Main Streamlit application file
├── data_preprocessing.py            # Data preprocessing and label encoding
├── train_model.py                   # Code to train the deep learning model
├── disease_detection_model.h5       # Saved deep learning model weights
├── label_encoder.pkl                # Serialized label encoder
├── predicted_diseases.xlsx          # Excel file to store prediction history
└── README.md                        # Project documentation

