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

│   ├── newtrain.csv

│   ├── val.csv

│   └── test.csv

│

├── EDA/                     → Notebooks for exploratory analysis

│

├── Modeling/                → Python scripts for training & testing

│   ├── credit_fraud_train.py

│   ├── credit_fraud_test.py

│   └── credit_fraud_utils_data.py

│
├── requirements.txt         → Python dependencies

├── README.md                → You are here

└── results.docx             → Model results summary



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
pip install -r requirements.txt


🚀 How to Run Training
Basic training (example)
python credit_fraud_train.py


With options
python credit_fraud_train.py \
    --model RandomForest \
    --scaler StandardScaler \
    --train Data/newtrain.csv \
    --val Data/val.csv \


⚙️ Command-Line Arguments
| Argument       | Description                                                          |
| -------------- | -------------------------------------------------------------------- |
| `--model`      | LogisticRegression / RandomForest / NeuralNetwork / VotingClassifier |
| `--scaler`     | StandardScaler / MinMaxScaler / RobustScaler                         |
| `--train`      | Path to training CSV                                                 |
| `--val`        | Path to validation CSV                                               |
| `--gridsearch` | Enable GridSearchCV                                                  |
| `--sampling`   | SMOTE / SMOTEENN / UnderSampler / None                               |
| `--factor`     | Sampling factor for SMOTE                                            |

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
model.joblib            → Serialized trained model and best threshold  


🧩 Future Improvements

Build a FastAPI inference service
