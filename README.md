# CyberGaurd AI Hackathon

## Part A: About the Project

---

### **Project Overview**
This project is dedicated to creating a safer digital ecosystem by utilizing **data-driven insights** to identify, prevent, and mitigate cybercrimes. By analyzing cybercrime patterns, we empower users, organizations, and policymakers with actionable intelligence to strengthen cybersecurity measures. Our approach focuses on:
- Addressing major cyber threats like financial fraud and social media exploitation.
- Identifying and mitigating emerging vulnerabilities in the digital landscape.
- Promoting digital literacy and fostering innovative cybersecurity technologies.

Our mission is to build a connected world where secure transactions and interactions are the norm, free from cybercriminal threats.

---

### **Vision and Goals**
- **Empowering Stakeholders**: Provide tools and insights for individuals, organizations, and governments.
- **Enhancing Digital Security**: Address critical cybercrime trends using advanced methodologies.
- **Innovation for Safety**: Leverage AI technologies like DistilBERT to develop scalable and efficient solutions.

---

### **Methodology: DistilBERT and Its Role**
#### **What is DistilBERT?**
DistilBERT is a streamlined version of the BERT model that retains its core capabilities while being:
- **Faster** in execution.
- **Smaller** in size for easier deployment in resource-constrained environments.
- **Efficient** without significant performance trade-offs.

#### **How DistilBERT Works**
DistilBERT employs **knowledge distillation**, where:
1. A **teacher model** (e.g., BERT) transfers its knowledge to a smaller **student model** (DistilBERT).
2. The student learns from both the teacher’s outputs and its intermediate behaviors.

It utilizes:
- **Distillation Loss**: Aligns the student’s predictions with the teacher’s probability distributions for nuanced understanding.
- **Masked Language Modeling (MLM) Loss**: Teaches the model to predict masked words, enhancing contextual understanding.
- **Cosine Embedding Loss**: Aligns hidden layers of the student with the teacher, ensuring robust intermediate knowledge transfer.

---

## **Exploratory Data Analysis (EDA)**
EDA forms the foundation of this project by uncovering critical patterns, identifying biases, and ensuring data quality. Below is a detailed analysis of the various aspects of the dataset, supported by visualizations.

---

#### **1. Distribution of Sub-Categories**
The distribution of subcategories highlights the prevalence of various cybercrimes. UPI-related frauds dominate the dataset with over 25,000 reported cases, reflecting vulnerabilities in digital payment systems. Debit/credit card frauds and SIM swap frauds rank as the second most common, followed by internet banking-related frauds. Less frequent categories, such as ransomware attacks, SQL injection, and email phishing, also warrant attention due to their specialized threat profiles. This analysis underscores the need to prioritize mitigation of payment-related and financial crimes.

![Distribution of Sub-Categories](images/Screenshot%202024-11-22%20085622.png)

---

#### **2. Common Words in Crime Descriptions**
The word cloud generated from crime descriptions reveals that terms like "account number," "account," "amount," "refund," and "necessary" are most frequently mentioned. These terms highlight the financial nature of most reported incidents, focusing on unauthorized transactions and fraudulent requests. Keywords like "OTP," "bank," and "call" further indicate the prevalence of financial deception and fraudulent communication.

![Common Words in Crime Descriptions](images/Screenshot%202024-11-22%20085635.png)

---

#### **3. Crime Count Distribution by Sub-Category and Category**
This distribution shows the overwhelming dominance of online financial fraud, with nearly 60,000 cases reported, showcasing its prevalence. Social media-related crimes follow, highlighting misuse of these platforms for malicious purposes. Categories like ransomware, phishing, and SQL injection, though less frequent, remain critical specialized threats.

![Crime Count Distribution by Sub-Category and Category](images/Screenshot%202024-11-22%20085652.png)

---

#### **4. Distribution of Crime Description Lengths**
Analysis of crime description lengths reveals that most reports are concise, ranging from 200-400 words, suitable for straightforward cases. A secondary peak around 800-1,000 words reflects more detailed incidents, while fewer cases exceed 1,400 words, indicating complex situations requiring extensive documentation.

![Distribution of Crime Description Lengths](images/Screenshot%202024-11-22%20085701.png)

---

#### **5. Most Common Keywords in Crime Descriptions**
The bar chart of common keywords emphasizes financial crimes, with terms like "account," "bank," and "fraud" dominating descriptions. Words like "money," "card," and "loan" highlight the prevalence of transactional frauds. Technology-related frauds are also evident from keywords like "app," "id," and "phone."

![Most Common Keywords in Crime Descriptions](images/Screenshot%202024-11-22%20085714.png)

---

#### **6. Heatmap Analysis of Cybercrime Categories and Trends**
The heatmap visualization highlights the distribution of subcategories. Online financial frauds and social media-related crimes are the most reported, emphasizing vulnerabilities in digital financial systems and misuse of platforms. Less frequent but critical threats, such as ransomware and SQL injection, require targeted attention.

![Heatmap Analysis of Cybercrime Categories](images/Screenshot%202024-11-22%20085728.png)

---

### **Project Implementation**
#### **Tokenization**
The text data is processed using DistilBERT's tokenizer:
- Converts raw text into numerical tokens.
- Adds padding and truncation to ensure uniform input size.
- Generates PyTorch-compatible tensors for efficient processing.

#### **Custom Dataset Class**
A PyTorch `Dataset` class, `TextDataset`, is implemented to:
- Map tokenized inputs to their corresponding labels.
- Provide modularity for integration with the Hugging Face `Trainer` API.

#### **Model Training**
1. **Model Initialization**:
   - DistilBERT is loaded with `AutoModelForSequenceClassification`.
   - Configured for multi-class classification tasks.
2. **Training Configuration**:
   - Batch size of 8 to optimize memory usage.
   - Training for 5 epochs to balance efficiency and performance.
   - Checkpoint saving for incremental progress monitoring.
3. **Evaluation**:
   - Loss and accuracy trends are tracked during training.
   - Confusion matrices are generated post-training to analyze classification errors.

---

### **Visualization and Insights**
- **Training Loss and Accuracy Graphs**:
  - Demonstrate the learning progression across epochs.
  - Ensure the model converges effectively with reduced loss and increased accuracy.

- **Confusion Matrices**:
  - Highlight model misclassifications.
  - Provide actionable insights for data refinement or additional model tuning.

---

### **App Features**
The project integrates the trained model into a **Streamlit-based application** that offers:
- **User-Friendly Interface**: Simplifies category and sub-category predictions through a text area and interactive buttons.
- **Real-Time Feedback**:
  - Predicts and displays the most likely category and sub-category for given text.
  - Maps predictions to human-readable labels for clarity.
- **Educational Insights**:
  - Displays internal mappings and prediction probabilities for transparency.

---

### **Key Outcomes**
This project is a step forward in:
1. **Building Trust**: Promoting secure digital environments through actionable insights.
2. **Empowering Users**: Offering robust tools to mitigate cybercrime threats.
3. **Scalable Solutions**: Leveraging lightweight models like DistilBERT for efficient real-world deployment.

Through detailed analysis, innovative methodologies, and a user-centric application, this project delivers practical solutions to combat cybercrime effectively.
