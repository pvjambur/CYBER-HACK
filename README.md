
### **Part A: About the Project**

---

#### **Project Overview**
This project is dedicated to creating a safer digital ecosystem by utilizing **data-driven insights** to identify, prevent, and mitigate cybercrimes. By analyzing cybercrime patterns, we empower users, organizations, and policymakers with actionable intelligence to strengthen cybersecurity measures. Our approach focuses on:
- Addressing major cyber threats like financial fraud and social media exploitation.
- Identifying and mitigating emerging vulnerabilities in the digital landscape.
- Promoting digital literacy and fostering innovative cybersecurity technologies.

Our mission is to build a connected world where secure transactions and interactions are the norm, free from cybercriminal threats.

---

#### **Vision and Goals**
- **Empowering Stakeholders**: Provide tools and insights for individuals, organizations, and governments.
- **Enhancing Digital Security**: Address critical cybercrime trends using advanced methodologies.
- **Innovation for Safety**: Leverage AI technologies like DistilBERT to develop scalable and efficient solutions.

---

#### **Methodology: DistilBERT and Its Role**
##### **What is DistilBERT?**
DistilBERT is a streamlined version of the BERT model that retains its core capabilities while being:
- **Faster** in execution.
- **Smaller** in size for easier deployment in resource-constrained environments.
- **Efficient** without significant performance trade-offs.

##### **How DistilBERT Works**
DistilBERT employs **knowledge distillation**, where:
1. A **teacher model** (e.g., BERT) transfers its knowledge to a smaller **student model** (DistilBERT).
2. The student learns from both the teacher’s outputs and its intermediate behaviors.

It utilizes:
- **Distillation Loss**: Aligns the student’s predictions with the teacher’s probability distributions for nuanced understanding.
- **Masked Language Modeling (MLM) Loss**: Teaches the model to predict masked words, enhancing contextual understanding.
- **Cosine Embedding Loss**: Aligns hidden layers of the student with the teacher, ensuring robust intermediate knowledge transfer.

---

#### **Exploratory Data Analysis (EDA)**
EDA forms the foundation of this project by uncovering critical patterns, identifying biases, and ensuring data quality. Below is a detailed analysis of the various aspects of the dataset, supported by visualizations.

##### **1. Distribution of Sub-Categories**
The distribution of subcategories highlights the prevalence of various cybercrimes. UPI-related frauds dominate the dataset with over 25,000 reported cases, reflecting vulnerabilities in digital payment systems. Debit/credit card frauds and SIM swap frauds rank as the second most common, followed by internet banking-related frauds. Less frequent categories, such as ransomware attacks, SQL injection, and email phishing, also warrant attention due to their specialized threat profiles. This analysis underscores the need to prioritize mitigation of payment-related and financial crimes.

![Distribution of Sub-Categories](images/Screenshot%202024-11-22%20085622.png)

---

##### **2. Common Words in Crime Descriptions**
The word cloud generated from crime descriptions reveals that terms like "account number," "account," "amount," "refund," and "necessary" are most frequently mentioned. These terms highlight the financial nature of most reported incidents, focusing on unauthorized transactions and fraudulent requests. Keywords like "OTP," "bank," and "call" further indicate the prevalence of financial deception and fraudulent communication.

![Common Words in Crime Descriptions](images/Screenshot%202024-11-22%20085635.png)

---

##### **3. Crime Count Distribution by Sub-Category and Category**
This distribution shows the overwhelming dominance of online financial fraud, with nearly 60,000 cases reported, showcasing its prevalence. Social media-related crimes follow, highlighting misuse of these platforms for malicious purposes. Categories like ransomware, phishing, and SQL injection, though less frequent, remain critical specialized threats.

![Crime Count Distribution by Sub-Category and Category](images/Screenshot%202024-11-22%20085652.png)

---

##### **4. Distribution of Crime Description Lengths**
Analysis of crime description lengths reveals that most reports are concise, ranging from 200-400 words, suitable for straightforward cases. A secondary peak around 800-1,000 words reflects more detailed incidents, while fewer cases exceed 1,400 words, indicating complex situations requiring extensive documentation.

![Distribution of Crime Description Lengths](images/Screenshot%202024-11-22%20085701.png)

---

##### **5. Most Common Keywords in Crime Descriptions**
The bar chart of common keywords emphasizes financial crimes, with terms like "account," "bank," and "fraud" dominating descriptions. Words like "money," "card," and "loan" highlight the prevalence of transactional frauds. Technology-related frauds are also evident from keywords like "app," "id," and "phone."

![Most Common Keywords in Crime Descriptions](images/Screenshot%202024-11-22%20085714.png)

---

##### **6. Heatmap Analysis of Cybercrime Categories and Trends**
The heatmap visualization highlights the distribution of subcategories. Online financial frauds and social media-related crimes are the most reported, emphasizing vulnerabilities in digital financial systems and misuse of platforms. Less frequent but critical threats, such as ransomware and SQL injection, require targeted attention.

![Heatmap Analysis of Cybercrime Categories](images/Screenshot%202024-11-22%20085728.png)

---

## Steps to tain

### **Part A: Model and Training Details**

#### **1. Tokenization Process**  
The tokenization and data preprocessing steps for both **category** and **sub-category** training are identical. We utilize the `DistilBERT` tokenizer to convert text descriptions into tokenized inputs suitable for transformer-based models. Missing values in columns such as `information`, `category`, and `sub_category` are replaced with placeholder values like `"Unknown information"`. 

**Code Snippet: Tokenization**  
```python
from transformers import DistilBertTokenizer
from sklearn.preprocessing import LabelEncoder

# Initialize tokenizer and encoders
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
category_encoder = LabelEncoder()
sub_category_encoder = LabelEncoder()

# Fill missing values in relevant columns
data['information'].fillna("Unknown information", inplace=True)
data['category'].fillna("Unknown category", inplace=True)
data['sub_category'].fillna("Unknown sub-category", inplace=True)

# Encode labels
data['category_label'] = category_encoder.fit_transform(data['category'])
data['sub_category_label'] = sub_category_encoder.fit_transform(data['sub_category'])

# Tokenize the 'information' column
def tokenize_text(text):
    return tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")

data['inputs'] = data['information'].apply(tokenize_text)
```

#### **2. Data Splitting for Training**  
Data is divided into `category_dataset` and `sub_category_dataset` using a custom PyTorch `Dataset` class. Both datasets include tokenized inputs and encoded labels for training classification models.  

#### **3. Training Process**  
The training process involves using `DistilBERT` for sequence classification. Below is a summarized workflow:  

1. Tokenize text input.
2. Prepare datasets using PyTorch `Dataset` class.
3. Define and fine-tune a `DistilBERT` model for sequence classification.
4. Train with custom arguments, without evaluation during training to optimize speed.

**Code Snippet: Training Process**  
```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Define model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(train_data['category_label'].unique())
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01
)

# Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Train Model
trainer.train()
```

#### **4. Saving the Model**  
After training, the model and tokenizer are saved for future use.  
```python
# Save the final model and tokenizer
model.save_pretrained("./results/final_model")
tokenizer.save_pretrained("./results/final_model")
```

#### **5. Category and Sub-Category Training Loss**  
The training loss graphs for both **category** and **sub-category** models are shown below:

---

### Training Images

- **Category Loss**  
![Category Loss](images/Screenshot%202024-11-22%20090726.png)  

- **Sub-Category Loss**  
![Sub-Category Loss](images/Screenshot%202024-11-22%20090742.png)  

#### **6. Confusion Matrix**  
The confusion matrices demonstrate the model's classification performance across various labels:

- **Category Confusion Matrix**  
![Category Confusion Matrix](images/Screenshot%202024-11-22%20090802.png)  

- **Sub-Category Confusion Matrix**  
![Sub-Category Confusion Matrix](images/Screenshot%202024-11-22%20090824.png)  

#### **7. Training Progression**  
The training progress for both models is illustrated below:

- **Category Training Progression**  
![Category Training](images/Screenshot%202024-11-22%20090852.png)  

- **Sub-Category Training Progression**  
![Sub-Category Training](images/Screenshot%202024-11-22%20090904.png)  

---

Let me know if further refinements are needed!

---

#### **Training Results and Insights**
The following graphs and matrices provide detailed insights into the training process:

- **Category Loss Trend**:
  Demonstrates the decrease in training loss for the category model, confirming the model's convergence.
  ![Category Loss](images/Screenshot%202024-11-22%20090726.png)

- **Sub-Category Loss Trend**:
  Highlights the loss reduction for the sub-category model, emphasizing effective training.
  ![Sub-Category Loss](images/Screenshot%202024-11-22%20090742.png)

- **Category Confusion Matrix**:
  Evaluates model performance by showcasing correct and incorrect predictions across categories.
  ![Category Confusion Matrix](images/Screenshot%202024-11-22%20090802.png)

- **Sub-Category Confusion Matrix**:
  Details the misclassifications in sub-categories, aiding further data refinement.
  ![Sub-Category Confusion Matrix](images/Screenshot%202024-11-22%20090824.png)

- **Category Training Accuracy**:
  Illustrates accuracy improvements during category training across epochs.
  ![Category Training Accuracy](images/Screenshot%202024-11-22%20090852.png)

- **Sub-Category Training Accuracy**:
  Displays the accuracy trend for sub-category training, confirming steady progress.
  ![Sub-Category Training Accuracy](images/Screenshot%202024-11-22%20090904.png)

---

#### **App Features**
The project integrates the trained model into a **Streamlit-based application** that offers:
- **User-Friendly Interface**: Simplifies category and sub-category predictions through a text area and interactive buttons.
- **Real-Time Feedback**:
  - Predicts and displays the most likely category and sub-category for given text.
  - Maps predictions to human-readable labels for clarity.
- **Educational Insights**:
  - Displays internal mappings and prediction probabilities for transparency.

---

#### **Key Outcomes**
This project is a step forward in:
1. **Building Trust**: Promoting secure digital environments through actionable insights.
2. **Empowering Users**: Offering robust tools to mitigate cybercrime threats.
3. **Scalable Solutions**: Leveraging lightweight models like DistilBERT for efficient real-world deployment.

Through detailed analysis, innovative methodologies, and a user-centric application, this project delivers practical solutions to combat cybercrime effectively.

--- 
