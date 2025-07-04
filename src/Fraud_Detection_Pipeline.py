# fraud_detection_pipeline.py
# ✅ Full pipeline: Load daily .pkl files, feature engineering, model training, fraud detection

import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ====== Config ======
DATA_DIR = "D:/Unified_Mentors/fraud_detection/fraud_detection/dataset/data"
MODEL_PATH = "D:/Unified_Mentors/fraud_detection/fraud_detection/saved_models/model.pkl"
FEATURES_PATH = "D:/Unified_Mentors/fraud_detection/fraud_detection/saved_models/features.pkl"
SCALER_PATH = "D:/Unified_Mentors/fraud_detection/fraud_detection/saved_models/scaler.pkl"
REPORT_PATH = "D:/Unified_Mentors/fraud_detection/fraud_detection/outputs/report.txt"
CONF_MATRIX_IMG_PATH = REPORT_PATH.replace(".txt", "_confusion_matrix.png")

# ====== Ensure directories exist ======
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

# ====== Load all daily data files ======
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pkl")))
print("🗂 Found files:", len(files))

# Load and combine
all_dfs = [pd.read_pickle(f) for f in files]
df = pd.concat(all_dfs).reset_index(drop=True)
print(f"✅ Loaded {df.shape[0]} transactions")

# ====== Feature Engineering ======
df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
df['TX_DAY'] = df['TX_DATETIME'].dt.day
df['TX_HOUR'] = df['TX_DATETIME'].dt.hour

# Customer-based features
df['CUSTOMER_TX_COUNT'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform('count')
df['CUSTOMER_AVG_AMOUNT'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform('mean')
df['CUSTOMER_STD_AMOUNT'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform('std').fillna(0)
df['TX_AMOUNT_DIFF'] = df['TX_AMOUNT'] - df['CUSTOMER_AVG_AMOUNT']

# ⚠️ Do NOT use terminal fraud count (leakage)
# df['TERMINAL_FRAUD_COUNT'] = df.groupby('TERMINAL_ID')['TX_FRAUD'].transform('sum')

# Encode categorical IDs
X = df.drop(columns=['TRANSACTION_ID', 'TX_DATETIME', 'TX_FRAUD'])
y = df['TX_FRAUD']
X['CUSTOMER_ID'] = X['CUSTOMER_ID'].astype('category').cat.codes
X['TERMINAL_ID'] = X['TERMINAL_ID'].astype('category').cat.codes

# Save feature names
joblib.dump(X.columns.tolist(), FEATURES_PATH)

# ====== Split and Train Model ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ====== Evaluation ======
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

print("\n📋 Classification Report:")
print(report)
print("\n🧮 Confusion Matrix:")
print(matrix)

# Save model and report
joblib.dump(model, MODEL_PATH)
with open(REPORT_PATH, "w") as f:
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(matrix))

# Optional visualization
plt.figure(figsize=(5, 4))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(CONF_MATRIX_IMG_PATH)
plt.close()

print("\n✅ Model training and evaluation completed.")
print("🚨 Total fraud transactions:", df['TX_FRAUD'].sum())
