# Dataset-Agnostic ML Classification Pipeline for Network Traffic

A scalable **machine learning pipeline** designed to detect **malicious network activity vs normal traffic**.

This project demonstrates how **data preprocessing, feature scaling, model comparison, and hyperparameter tuning** improve predictive performance in classification models.

The pipeline is **dataset-agnostic**, meaning it can ingest **any tabular dataset** and automatically apply preprocessing and model evaluation.

---

# Project Objectives

This project demonstrates the impact of:

• Data preprocessing for heterogeneous datasets
• Feature scaling for improving model performance
• Comparison of multiple classification models
• Hyperparameter tuning for optimization

The goal is to build a **reusable ML pipeline** that can be applied to intrusion detection datasets.

---

# Key Features

Dataset-agnostic machine learning pipeline

Automatic preprocessing for numeric and categorical features

Missing value handling

Feature scaling using **StandardScaler**

Automatic **model comparison leaderboard**

Hyperparameter tuning with **GridSearchCV**

Model evaluation using:

• Confusion Matrix
• Classification Report
• ROC Curve

---

# Machine Learning Workflow

```
Dataset
   ↓
Data Preprocessing
   ↓
Feature Scaling
   ↓
Train/Test Split
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
```

---

# Supported Models

The pipeline compares multiple classification models automatically:

• Logistic Regression
• Decision Tree
• Random Forest
• Support Vector Machine (SVM)

Models are ranked using an **automatic performance leaderboard**.

---

# Visual Analysis

The notebook includes visual analysis such as:

### Traffic Distribution

Shows **attack vs normal traffic distribution**.

### PCA Visualization

Projects high-dimensional network traffic into **2D space** to visualize attack clusters.

### Confusion Matrix

Shows classification performance across classes.

### ROC Curve

Measures the model’s ability to distinguish malicious traffic.

---

# Demonstrating Feature Scaling

The project experimentally compares model performance:

| Model               | Without Scaling | With Scaling      |
| ------------------- | --------------- | ----------------- |
| Logistic Regression | Lower accuracy  | Improved accuracy |

This demonstrates that **feature scaling improves predictive performance** for many machine learning algorithms.

---

# Project Structure

```
dataset-agnostic-ml-classifier
│
├── src
│   └── pipeline.py
│
├── notebooks
│   └── colab_pipeline_reference.ipynb
│
├── requirements.txt
├── run_model.py
└── README.md
```

---

# Installation

Clone the repository:

```
git clone https://github.com/AditiNayak-S/dataset-agnostic-ml-classifier.git
```

Move into the project directory:

```
cd dataset-agnostic-ml-classifier
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running the Pipeline

Edit **run_model.py** and provide:

• dataset path
• target column

Example:

```
run_pipeline(
    dataset_path="data/network_dataset.csv",
    target_column="label"
)
```

The pipeline will automatically:

1. preprocess the dataset
2. compare multiple models
3. select the best model
4. evaluate performance

---

# Example Use Cases

Intrusion detection systems

Network anomaly detection

Security log analysis

Malware traffic classification

---

# Technologies Used

Python

Scikit-learn

Pandas

NumPy

Matplotlib

Seaborn

---

# Future Improvements

AutoML model selection

Integration with benchmark intrusion datasets (CICIDS, UNSW-NB15)

Real-time network traffic detection

Deep learning models for anomaly detection

---

# Author

Aditi S Nayak

Machine Learning • Security • Data Engineering
