# Ensure necessary imports
import os
import logging
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Set CUDA_LAUNCH_BLOCKING for debugging (optional for more error details on GPU)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Suppress specific warnings from transformers
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Load and preprocess training data
train_data = pd.read_csv('train.csv')
train_data['information'].fillna("Unknown information", inplace=True)
train_data['category'].fillna("Unknown category", inplace=True)
train_data['sub_category'].fillna("Unknown sub_category", inplace=True)  # Ensure no missing sub_category

# Encode labels for train data
train_data['category_label'] = train_data['category'].astype('category').cat.codes
train_data['sub_category_label'] = train_data['sub_category'].astype('category').cat.codes

# Ensure labels are within the range expected by the model
num_labels_category = len(train_data['category_label'].unique())
num_labels_sub_category = len(train_data['sub_category_label'].unique())

print(f"Number of unique labels in category: {num_labels_category}")
print(f"Unique labels in category dataset: {train_data['category_label'].unique()}")
print(f"Number of unique labels in sub_category: {num_labels_sub_category}")
print(f"Unique labels in sub_category dataset: {train_data['sub_category_label'].unique()}")

# Initialize tokenizer for both models
category_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
sub_category_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize data
def tokenize_data(text_series, tokenizer):
    return tokenizer(list(text_series), padding=True, truncation=True, max_length=128, return_tensors="pt")

train_encodings_category = tokenize_data(train_data['information'], category_tokenizer)
train_encodings_sub_category = tokenize_data(train_data['information'], sub_category_tokenizer)

train_labels_category = torch.tensor(train_data['category_label'].values)
train_labels_sub_category = torch.tensor(train_data['sub_category_label'].values)

# Define custom dataset
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets for category and sub_category models
train_dataset_category = TextDataset(train_encodings_category, train_labels_category)
train_dataset_sub_category = TextDataset(train_encodings_sub_category, train_labels_sub_category)

# Load the saved models and specify the number of labels
category_model_path = "./results/final_model"
sub_category_model_path = "./results/sc_final_model"

category_model = AutoModelForSequenceClassification.from_pretrained(category_model_path, num_labels=num_labels_category)
sub_category_model = AutoModelForSequenceClassification.from_pretrained(sub_category_model_path, num_labels=num_labels_sub_category)

# Switch to CPU to debug any issues with CUDA assertions
device = torch.device("cpu")  # Use "cuda" if CUDA issues are resolved
category_model.to(device)
sub_category_model.to(device)

# Define dummy training arguments (to enable Trainer usage)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./results/logs"
)

# Initialize Trainer for both models
trainer_category = Trainer(
    model=category_model,
    args=training_args,
    train_dataset=train_dataset_category
)

trainer_sub_category = Trainer(
    model=sub_category_model,
    args=training_args,
    train_dataset=train_dataset_sub_category
)

# Evaluate the category model to get predictions for confusion matrix
category_model.eval()
with torch.no_grad():
    category_predictions = trainer_category.predict(train_dataset_category)
    predicted_category_labels = np.argmax(category_predictions.predictions, axis=1)

# Get true labels for category
true_category_labels = train_labels_category.numpy()

# Generate and plot confusion matrix for category model
cm_category = confusion_matrix(true_category_labels, predicted_category_labels)
disp_category = ConfusionMatrixDisplay(confusion_matrix=cm_category)
disp_category.plot(cmap=plt.cm.Blues)
plt.title("Category Model Confusion Matrix")
plt.show()

# Evaluate the sub_category model to get predictions for confusion matrix
sub_category_model.eval()
with torch.no_grad():
    sub_category_predictions = trainer_sub_category.predict(train_dataset_sub_category)
    predicted_sub_category_labels = np.argmax(sub_category_predictions.predictions, axis=1)

# Get true labels for sub_category
true_sub_category_labels = train_labels_sub_category.numpy()

# Generate and plot confusion matrix for sub_category model
cm_sub_category = confusion_matrix(true_sub_category_labels, predicted_sub_category_labels)
disp_sub_category = ConfusionMatrixDisplay(confusion_matrix=cm_sub_category)
disp_sub_category.plot(cmap=plt.cm.Blues)
plt.title("Sub_category Model Confusion Matrix")
plt.show()
