# 💳 Credit Card Fraud Detection using Ensemble Machine Learning

## 📌 Overview
This project focuses on detecting fraudulent credit card transactions using machine learning techniques on a highly imbalanced dataset.  
Multiple models were built, compared, and optimized to achieve the best balance between fraud detection and minimizing false alerts.

---

## 📊 Dataset Information
- Source: European cardholders dataset (September 2013)
- Total Transactions: ~284,000
- Fraud Cases: 492 (~0.17%)
- Features:
  - V1–V28: PCA-transformed features
  - Amount: Transaction amount
  - Time: Converted into Hour feature

---

## ⚠️ Key Challenges
- Extreme class imbalance
- Hidden patterns due to PCA-transformed features
- Trade-off between precision and recall
- Risk of misleading accuracy metric

---

## 🔍 Exploratory Data Analysis

- Fraud transactions show distinct distribution patterns across features
- Transaction Amount is highly skewed
- Fraud is more frequent during certain hours of the day
- PCA features show low correlation but strong distribution differences

---

## 📈 Modeling Approach

### 1️⃣ Logistic Regression
- Baseline model
- Required feature scaling
- High recall but very low precision

---

### 2️⃣ Random Forest (Baseline)
- Captured non-linear relationships
- Used `class_weight='balanced'`
- Improved precision significantly

---

### 3️⃣ Random Forest + SMOTE
- Applied SMOTE to balance dataset
- Increased recall
- Introduced more false positives

---

### 4️⃣ Tuned Random Forest (RandomizedSearchCV)
- Performed hyperparameter tuning using cross-validation
- Optimized:
  - Number of trees
  - Tree depth
  - Splitting criteria
- Achieved very high precision and improved F1-score
- Reduced false positives significantly

---

### 5️⃣ XGBoost (Final Model)
- Used `scale_pos_weight` to handle class imbalance
- Captured complex non-linear patterns
- Achieved best balance between precision and recall

---

## 🏆 Model Performance Comparison

| Model | Precision | Recall | F1-score |
|------|----------|--------|----------|
| Logistic Regression | 0.22 | 0.81 | 0.34 |
| Random Forest | 0.89 | 0.74 | 0.80 |
| RF + SMOTE | 0.58 | 0.82 | 0.68 |
| Tuned Random Forest | 0.96 | 0.73 | 0.83 |
| **XGBoost (Final)** | **0.95** | **0.80** | **0.87** |

---

## 📊 Key Observations

- Tuned Random Forest achieved **very high precision (0.96)** → minimal false positives
- XGBoost improved **recall (0.80)** → better fraud detection
- XGBoost achieved the **highest F1-score (0.87)** → best overall balance
- There is a clear trade-off:
  - Precision vs Recall
  - False alarms vs missed fraud

---

## 📌 Feature Importance

Top features identified by both models:
- V14
- V10
- V4
- V12
- V17

### 🔍 Insight:
- Both Random Forest and XGBoost identified similar important features
- Indicates strong underlying fraud patterns
- Fraud detection depends on **hidden (non-linear) signals rather than simple correlations**

---

## 📊 Visualizations

### 🔹 Feature Importance Comparison
<img width="1630" height="701" alt="image" src="https://github.com/user-attachments/assets/3c7e85b8-6dce-4a3e-9631-a20caba53b60" />


### 🔹 Transaction Amount Distribution
<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/ffe40788-6f3e-402a-9573-32ad7ca50ea0" />


### 🔹 Fraud Distribution by Hour
<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/25347014-f599-4815-81ac-d1cd54761671" />


---

## 💡 Key Insights

- Fraud detection relies heavily on **non-linear relationships**
- A small subset of features drives most predictions
- Handling class imbalance is critical for meaningful results
- Threshold tuning significantly improves model performance

---

## 🚀 Technologies Used

- Python
- Scikit-learn
- XGBoost
- Pandas & NumPy
- Matplotlib & Seaborn

---

## ⚙️ Deployment

The model can be deployed using:
- FastAPI / Flask for API-based prediction
- Streamlit for interactive UI

⚠️ Note: PCA-transformed features limit direct real-world deployment without access to original feature engineering pipeline.

---

## 📈 Future Improvements

- Use SHAP values for better explainability
- Experiment with deep learning models
- Build real-time fraud detection system
- Incorporate streaming data

---

## 👨‍💻 Author

**Hitesh Kumar**

---

## ⭐ Final Conclusion

Both Tuned Random Forest and XGBoost performed strongly:

- Tuned Random Forest provided **very high precision**, making it suitable for minimizing false alerts
- XGBoost provided a **better balance between precision and recall**, improving fraud detection capability

👉 **XGBoost was selected as the final model** due to its superior F1-score and overall performance.
