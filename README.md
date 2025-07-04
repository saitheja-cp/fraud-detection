
# 💳 Fraud Transaction Detection

This project builds a machine learning model to detect fraudulent transactions based on simulated transaction data using daily `.pkl` files. The model is trained and deployed through a Streamlit web app.

---

## 🎯 Objective

To classify whether a transaction is **fraudulent or legitimate** using core transactional features like amount, customer activity, and terminal details.

---

## 📦 Dataset

The dataset consists of transaction records stored as daily `.pkl` files.

## 📂 Dataset

Due to size constraints, the full dataset is hosted externally:

👉 [⬇️ Download dataset.zip](https://drive.google.com/uc?export=download&id=1CPSnT7qCBMOEQbD8r-RuUWib6Q555Mi7)

Please download and extract it into the `dataset/` folder:

---

## 🛠 Project Structure

```
fraud_detection/
├── app/
│   └── predict_fraud.py       # Streamlit web app
├── dataset/
│   └── data/                  # Daily transaction .pkl files (NOT pushed to GitHub)
├── outputs/
│   └── report.txt             # Classification report and confusion matrix
├── saved_models/
│   ├── model.pkl              # Trained model (ignored in GitHub)
│   └── features.pkl           # Feature column order
├── src/
│   └── fraud_detection_pipeline.py   # Full training pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 Instructions

### 1. 📦 Install dependencies
```bash
pip install -r requirements.txt
```

### 2. 🧠 Run training pipeline
```bash
python src/fraud_detection_pipeline.py
```

This will:
- Load all `.pkl` files
- Perform feature engineering
- Train and evaluate a RandomForest model
- Save the model and report

### 3. 🌐 Launch Streamlit app
```bash
streamlit run app/predict_fraud.py
```

---

## 📌 Notes

- ✅ Make sure to **change the dataset/model paths** in the code based on your environment.
- 📁 Do not upload large files (`model.pkl`, `dataset`, or `.venv`) to GitHub.
- 🔒 All files over 100MB are auto-rejected by GitHub. Use [Git LFS](https://git-lfs.github.com/) if needed.

---

## 👨‍💻 Author

- **Name**: C.P. Sai Theja  
- **Email**: [cpsaitheja@gmail.com](mailto:cpsaitheja@gmail.com)  
- **GitHub**: [github.com/saitheja-cp](https://github.com/saitheja-cp)

---

> This project was developed as part of a machine learning workflow demonstration. It includes feature engineering, model training, evaluation, and a Streamlit UI for real-time fraud prediction.
