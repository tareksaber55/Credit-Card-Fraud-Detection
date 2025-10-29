📂 Project Structure
├── credit_fraud_utils_data.py   # Helper functions (data loading, report generation)
├── credit_fraud_utils_models.py # Model training utilities (Logistic, RandomForest, etc.)
├── credit_fraud_train.py        # Training script (build, train, save model)
├── credit_fraud_test.py         # Evaluation script on unseen test data
├── data/                        # CSV datasets (train, val, test)
├── models/                      # Saved models (.joblib)
└── results/                     # Model reports & metrics
⚙️ Workflow
1️⃣ Data Preparation (credit_fraud_utils_data.py)

Load and preprocess CSV files.

Create new feature Hour = Time // 3600 % 24.

Split data into train/validation sets (stratified).

Apply scaling (StandardScaler or NormalScaler).

2️⃣ Model Training (credit_fraud_train.py)

Choose between:

Logistic Regression

Random Forest

Train using X_train, t_train.

Compute the best threshold maximizing F1-score.

Save model, scaler, and threshold with joblib.

3️⃣ Evaluation (credit_fraud_test.py)

Load saved model/scaler.

Apply same preprocessing on test data.

Predict fraud probabilities.

Generate a classification report, confusion matrix, and F1 score.

🧮 Features Used

Time-based: Hour (derived from transaction time).

V1–V28: PCA-transformed anonymized features.

Amount: Transaction amount.

Target: Class → 0 = Non-Fraud, 1 = Fraud.

📊 Metrics

Evaluated using metrics suited for imbalanced datasets:

Precision

Recall

F1-Score

ROC-AUC

The model reports the best threshold for fraud detection rather than using 0.5 blindly.

🚀 How to Run

1️⃣ Install dependencies

pip install -r requirements.txt


2️⃣ Run training

python credit_fraud_train.py


3️⃣ Run testing

python credit_fraud_test.py


4️⃣ View results

Reports and confusion matrix printed in console or saved to results/.

📌 Next Steps

Add more models (XGBoost, LightGBM).

Perform hyperparameter tuning.

Experiment with SMOTE or undersampling.

Try anomaly detection or neural approaches.

👤 Author
Tarek Saber
📎 GitHub Profile[https://github.com/tareksaber55]

