import streamlit as st
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
from PIL import Image

# Set the device to CPU or GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load category model and tokenizer
category_model_path = "./results/final_model"
category_model = AutoModelForSequenceClassification.from_pretrained(category_model_path)
category_tokenizer = AutoTokenizer.from_pretrained(category_model_path)
category_model.to(device)
category_model.eval()

# Load sub_category model and tokenizer
sub_category_model_path = "./results/sc_final_model"
sub_category_model = AutoModelForSequenceClassification.from_pretrained(sub_category_model_path)
sub_category_tokenizer = AutoTokenizer.from_pretrained(sub_category_model_path)
sub_category_model.to(device)
sub_category_model.eval()

# Load training data for label mappings
train_data = pd.read_csv('train.csv')
train_data['category_label'] = train_data['category'].astype('category').cat.codes
category_mapping = dict(enumerate(train_data['category'].astype('category').cat.categories))

# Check and process sub_category data if present
if 'sub_category' in train_data.columns:
    train_data['sub_category_label'] = train_data['sub_category'].astype('category').cat.codes
    sub_category_mapping = dict(enumerate(train_data['sub_category'].astype('category').cat.categories))
else:
    sub_category_mapping = {}

# Sidebar
st.sidebar.image("hacker.jpg", use_column_width=True)  # Sidebar image
st.sidebar.title("Navigation")
options = st.sidebar.radio("Choose an option:", [
    "Home", "Predict Category & Sub_category", 
    "View Category Training Loss", "View Sub_category Training Loss", 
    "Confusion Matrix - Category","Confusion Matrix - Sub Category","Accuracy vs Loss", 
    "Sub_category Accuracy vs Loss"  # New option added here
])

# Display category training loss graph
def plot_training_loss():
    st.subheader("Category Training Loss over Epochs")
    category_train_loss = [0.6, 0.45, 0.35, 0.25, 0.2]  # Replace with actual category training loss data
    epochs = range(1, len(category_train_loss) + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, category_train_loss, label="Category Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Category Training Loss vs. Epoch")
    ax.legend()
    st.pyplot(fig)

# Display sub_category training loss graph
def plot_sub_category_training_loss():
    st.subheader("Sub_category Training Loss over Epochs")
    sub_category_train_loss = [0.7, 0.5, 0.4, 0.3, 0.25]  # Replace with actual sub_category training loss data
    epochs = range(1, len(sub_category_train_loss) + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, sub_category_train_loss, label="Sub_category Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Sub_category Training Loss vs. Epoch")
    ax.legend()
    st.pyplot(fig)

# Display combined category accuracy vs loss graph
def plot_accuracy_vs_loss():
    st.subheader("Category Accuracy vs Loss")
    accuracy = [0.7, 0.8, 0.85, 0.88, 0.9]  # Replace with actual category accuracy data
    train_loss = [0.6, 0.45, 0.35, 0.25, 0.2]  # Replace with actual loss data
    epochs = range(1, len(train_loss) + 1)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(epochs, train_loss, color="tab:red", label="Category Loss")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.plot(epochs, accuracy, color="tab:blue", label="Accuracy")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()
    st.pyplot(fig)

# Display combined sub_category accuracy vs loss graph
def plot_sub_category_accuracy_vs_loss():
    st.subheader("Sub_category Accuracy vs Loss")
    sub_category_accuracy = [0.65, 0.75, 0.8, 0.85, 0.87]  # Replace with actual sub_category accuracy data
    sub_category_loss = [0.7, 0.5, 0.4, 0.3, 0.25]  # Replace with actual sub_category loss data
    epochs = range(1, len(sub_category_loss) + 1)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(epochs, sub_category_loss, color="tab:red", label="Sub_category Loss")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.plot(epochs, sub_category_accuracy, color="tab:blue", label="Sub_category Accuracy")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()
    st.pyplot(fig)

# Predict category and sub_category for input text
def predict_category_and_sub_category(text):
    # Predict category
    category_inputs = category_tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        category_logits = category_model(**category_inputs).logits
        predicted_category_label = torch.argmax(category_logits, dim=1).item()
    
    predicted_category = category_mapping.get(predicted_category_label, "Unknown Category")
    st.write("Predicted Category Label:", predicted_category_label)
    st.write("Category Mapping:", category_mapping)

    # Predict sub_category only if a category is successfully predicted
    predicted_sub_category = "No sub_category data available"
    if predicted_category != "Unknown Category" and 'sub_category_label' in train_data.columns:
        sub_category_inputs = sub_category_tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            sub_category_logits = sub_category_model(**sub_category_inputs).logits
            predicted_sub_category_label = torch.argmax(sub_category_logits, dim=1).item()
        
        predicted_sub_category = sub_category_mapping.get(predicted_sub_category_label, "Unknown Sub_category")
        st.write("Predicted Sub_category Label:", predicted_sub_category_label)
        st.write("Sub_category Mapping:", sub_category_mapping)
    
    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Category**: {predicted_category}")
    st.write(f"**Sub_category**: {predicted_sub_category}")

# Main section for different app options
if options == "Home":
    st.title("Welcome to the Text Classification App!")
    st.write("""
        This app classifies input information into specific categories and sub_categories using a pre-trained language model.
        \n### Features:
        - Predicts both category and sub_category based on input text.
        - Provides training metrics including loss and accuracy graphs.
        - Visualizes the model's performance with a confusion matrix.
        \nGet started by navigating to **Predict Category & Sub_category** from the sidebar.
    """)

elif options == "Predict Category & Sub_category":
    st.title("Category & Sub_category Prediction")
    user_input = st.text_area("Enter Information for Prediction:", "")
    if st.button("Predict"):
        if user_input:
            predict_category_and_sub_category(user_input)
        else:
            st.warning("Please enter some text for prediction.")

elif options == "View Category Training Loss":
    st.title("Category Training Loss")
    plot_training_loss()

elif options == "View Sub_category Training Loss":
    st.title("Sub_category Training Loss")
    plot_sub_category_training_loss()

elif options == "Confusion Matrix - Category":
    st.title("Confusion Matrix")
    confusion_image = Image.open("confusion_cat.jpg")
    st.image(confusion_image, caption="Confusion Matrix", use_column_width=True)

elif options == "Confusion Matrix - Sub Category":
    st.title("Confusion Matrix")
    confusion_image = Image.open("confusion_sub_cat.png")
    st.image(confusion_image, caption="Confusion Matrix", use_column_width=True)

elif options == "Accuracy vs Loss":
    st.title("Category Accuracy vs Loss")
    plot_accuracy_vs_loss()

elif options == "Sub_category Accuracy vs Loss":  # New option here
    st.title("Sub_category Accuracy vs Loss")
    plot_sub_category_accuracy_vs_loss()

st.sidebar.info("A predictive model app to classify input information into categories and sub_categories.")
