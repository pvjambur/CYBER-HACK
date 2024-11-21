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

# Encode labels for train data
train_data['category_label'] = train_data['category'].astype('category').cat.codes

# Ensure labels are within the range expected by the model
num_labels = len(train_data['category_label'].unique())
print(f"Number of unique labels in data: {num_labels}")
print(f"Unique labels in dataset: {train_data['category_label'].unique()}")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize data
def tokenize_data(text_series):
    return tokenizer(list(text_series), padding=True, truncation=True, max_length=128, return_tensors="pt")

train_encodings = tokenize_data(train_data['information'])
train_labels = torch.tensor(train_data['category_label'].values)

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

# Create dataset
train_dataset = TextDataset(train_encodings, train_labels)

# Load the saved model and specify the number of labels
model_path = "./results/final_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

# Switch to CPU to debug any issues with CUDA assertions
device = torch.device("cpu")  # Use "cuda" if CUDA issues are resolved
model.to(device)

# Define dummy training arguments (to enable Trainer usage)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./results/logs"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Load training logs if available and plot loss
try:
    log_history = trainer.state.log_history
    train_loss = [log["loss"] for log in log_history if "loss" in log]
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs. Epoch")
    plt.legend()
    plt.show()
except AttributeError:
    print("No training logs available in trainer.state.log_history")

# Evaluate the model to get predictions for confusion matrix
model.eval()
with torch.no_grad():
    predictions = trainer.predict(train_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)

# Get true labels
true_labels = train_labels.numpy()

# Generate and plot confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
