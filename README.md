📘 Credit Card Fraud Detection

Machine-learning pipeline for detecting fraudulent transactions in highly imbalanced credit-card datasets.
This project includes EDA, preprocessing, sampling techniques, model training, hyperparameter search, and performance evaluation using best practices for imbalanced classification.

📌 Project Overview

Credit-card fraud detection is a binary classification problem with extremely imbalanced classes.
This project follows a clean ML workflow:

✔ Exploratory Data Analysis (EDA)

✔ Data preprocessing (scaling, imputation, feature engineering)

✔ Handling imbalance (SMOTE / SMOTEENN / RandomUnderSampler /class weighting)

✔ Modeling (Logistic Regression, Random Forest, Neural Network, Voting Classifier)

✔ Grid Search with cross-validation

✔ Saving final models and metrics

✔ Configurable training using command-line arguments

📂 Project Structure
Credit-Card-Fraud-Detection/
│

├── Data/ 

│   ├── newtrain.csv

│   ├── val.csv

│   └── test.csv

│
├── EDA/                      → Notebook for exploration  
│
├── Modeling/                 →Python scripts for training and testing
│   ├──credit_fraud_train.py       
│   ├──credit_fraud_test.py
|   ├──credit_fraud_utils_data.py
|
├── requirements.txt          → Python dependencies  
├── README.md                 → You are here  
└── results.docx              → Model results summary

📥 Dataset

This project works with the popular Credit Card Fraud Detection dataset (284,807 transactions).

📌 Dataset Source: Search “Credit Card Fraud Detection dataset (2013)”.

schema includes:

Time, Amount, and 28 PCA-transformed features (V1–V28)

Class → 0 = normal, 1 = fraud

Place the dataset here:

Data/creditcard.csv

🛠 Installation
1. Create a Python environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

2. Install dependencies
pip install -r requirements.txt


🚀 How to Run Training
Basic training (example)
python credit_fraud_train.py --model RandomForest  --scaler StandardScaler

With options
python train.py \
    --model RandomForest \
    --scaler StandardScaler \
    --train Data/newtrain.csv \
    --val Data/val.csv \


Arguments supported
Argument	Description
--model	LogisticRegression / RandomForest / NeuralNetwork / VotingClassifier
--scaler	StandardScaler / MinMaxScaler / RobustScaler
--train	Path to training CSV
--val	Path to validation CSV
--gridsearch	Enable GridSearchCV
--sampling	Enable SMOTE / SMOTEENN / UnderSampler / None
--factor	Sampling factor
📊 Evaluation Metrics

Because the dataset is highly imbalanced, accuracy is misleading.
This project uses robust metrics:

F1-score

Average Precision (AP)

Confusion matrix

Metrics per fold during Stratified K-Fold cross-validation

🧠 Machine Learning Models

This project supports several models:

🔹 Logistic Regression

Useful baseline

Supports class weighting

🔹 Random Forest

Strong performance on tabular fraud data

Handles non-linear relationships

Supports class_weight='balanced'

🔹 Neural Network (MLP)

Multi-layer perception using sklearn

Tunable via grid search

🔹 Voting Classifier

Combines predictions from multiple models

Supports soft voting

⚖ Handling Imbalanced Data

The project provides multiple strategies:

SMOTE

SMOTEENN (SMOTE + Edited Nearest Neighbors)

RandomUnderSampler

Class-weighting


Sampling is safely applied inside CV folds only to avoid data leakage.

📁 Output Files

After training, the project outputs:

artifacts/
│
├── model.joblib            → Serialized trained model  


🧩 Future Improvements

Build a FastAPI inference service
