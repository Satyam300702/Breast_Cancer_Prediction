🧠 Breast Cancer Prediction using Stacking Ensemble

This project predicts whether a breast tumor is malignant (cancerous) or benign (non-cancerous) using the Breast Cancer Wisconsin Dataset.
The model uses an advanced Stacking Ensemble approach that combines CatBoost, XGBoost, LightGBM, and Random Forest to achieve high accuracy and reliability.

📌 Project Overview

Early and accurate breast cancer detection is crucial for patient survival.
This project applies multiple ensemble algorithms to enhance prediction accuracy using the Breast Cancer Wisconsin Dataset (available in sklearn.datasets).

Model Pipeline
🔹 Base Models

CatBoostClassifier — Handles categorical and numerical data automatically.

XGBClassifier — Efficient gradient boosting with regularization.

LGBMClassifier — Lightweight and fast gradient boosting.

🔹 Meta Model

RandomForestClassifier — Aggregates base model outputs for the final decision.

🔹 Training Setup

RandomizedSearchCV used for hyperparameter tuning.

ROC-AUC as the main scoring metric.

StackingClassifier combines base models with the meta learner (passthrough=True).

Cross-Validation (cv=5) ensures model generalization.

🧪 Model Performance
Metric	Score
Accuracy	0.9726
ROC-AUC	0.9447
Confusion Matrix	[[41, 2], [0, 71]]

✅ The model achieves high recall and precision, minimizing false negatives — crucial for medical diagnostics.

git clone https://github.com/your-Satyam300702/breast-cancer-ensemble.git
cd breast-cancer-ensemble

pip install -r requirements.txt

Live Demo:-