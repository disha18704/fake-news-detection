import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from io import StringIO

# Load data
train_acc = np.load("lstm/train_acc.npy", allow_pickle=True)
train_loss = np.load("lstm/train_loss.npy", allow_pickle=True)
val_acc = np.load("lstm/val_acc.npy", allow_pickle=True)
val_loss = np.load("lstm/val_loss.npy", allow_pickle=True)

# Load LSTM image
lstm_image = Image.open("process.png")

logistic_classification_report = """
              precision    recall  f1-score   support

           0       1.00      0.99      0.99      2711
           1       0.99      1.00      0.99      2690

    accuracy                           0.99      5401
   macro avg       0.99      0.99      0.99      5401
weighted avg       0.99      0.99      0.99      5401
"""

rf_classification_report = """
              precision    recall  f1-score   support

           0       0.98      0.97      0.97      2683
           1       0.97      0.98      0.98      2718

    accuracy                           0.97      5401
   macro avg       0.97      0.97      0.97      5401
weighted avg       0.97      0.97      0.97      5401
"""

gb_classification_report = """
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      2683
           1       1.00      1.00      1.00      2718

    accuracy                           1.00      5401
   macro avg       1.00      1.00      1.00      5401
weighted avg       1.00      1.00      1.00      5401
"""

# Function to display Logistic Regression details
def display_logistic_details():
    st.markdown("<h2>Logistic Regression Model Details</h2>", unsafe_allow_html=True)
    st.text(logistic_classification_report)

def display_rf_details():
    st.markdown("<h2>Random Forest Classifier Model Details</h2>", unsafe_allow_html=True)
    st.text(rf_classification_report)

def display_gb_details():
    st.markdown("<h2>Gradient Boosting Classifier Model Details</h2>", unsafe_allow_html=True)
    st.text(gb_classification_report)

# Function to display LSTM model details
def display_lstm_details():

    st.markdown("<h2>LSTM Model Performance</h2>", unsafe_allow_html=True)

    # Display model information
    st.write("Model Name: LSTM")
    st.write("Final Train Accuracy:", train_acc[-1])
    st.write("Final Train Loss:", train_loss[-1])
    st.write("Final Validation Accuracy:", val_acc[-1])
    st.write("Final Validation Loss:", val_loss[-1])


    st.write("## Performance Graph")
    # Plot training & validation accuracy values
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax.plot(train_acc, label='Train')
    ax.plot(val_acc, label='Validation')
    ax.set_title('Model accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper left')

    ax2.plot(train_loss, label='Train')
    ax2.plot(val_loss, label='Validation')
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')


    st.pyplot(fig)

    
    # Add LSTM image
    st.image(lstm_image, caption="LSTM Model", use_column_width=True)

# Sidebar for model selection
selected_model = st.sidebar.radio("Select Model", ("Logistic Regression","Random Forest","Gradient Boosting" ,"LSTM"))

# Main content based on selected model
if selected_model == "Logistic Regression":
    display_logistic_details()
elif selected_model == "Random Forest":
    display_rf_details()
elif selected_model == "Gradient Boosting":
    display_gb_details()
elif selected_model == "LSTM":
    display_lstm_details()
