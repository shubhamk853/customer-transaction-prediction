# 💳 Customer Transaction Prediction

A Machine Learning project to predict whether a customer will make a transaction using Santander Bank dataset.

## 📌 Overview
Built a binary classification model on a dataset of **200,000 observations and 202 features**. Handled extreme class imbalance using SMOTE and achieved **AUC-ROC of 0.89**.

## 🛠️ Tools & Technologies
- Python, Pandas, NumPy
- Scikit-Learn (Logistic Regression, Random Forest)
- imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

## 📊 Results
| Model | AUC-ROC |
|-------|---------|
| Logistic Regression | 0.78 |
| Random Forest | **0.89** |

- Resolved extreme class imbalance using **SMOTE**
- Reduced noise by **30%** using feature importance analysis

## 📁 Dataset
Santander Customer Transaction Prediction Dataset (Kaggle)  
- 200,000 rows × 202 features  
- Binary target: 1 = made transaction, 0 = did not  
- Download: https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

## 🔍 Project Steps
1. Exploratory Data Analysis (EDA)
2. Class Imbalance handling using SMOTE
3. Feature Scaling (StandardScaler)
4. Model Training (Logistic Regression + Random Forest)
5. Feature Importance Analysis
6. Confusion Matrix Evaluation

## 📷 Output Plots
- `class_distribution.png` — Target class imbalance
- `feature_mean_by_target.png` — Feature means by class
- `feature_importance.png` — Top 15 important features
- `confusion_matrix.png` — Model performance matrix

## ⚙️ How to Run
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
python customer_transaction_prediction.py
```
> Place `train.csv` (from Kaggle) in the same folder before running.

## 👤 Author
**Shubham Bhagwan Kale**  
GitHub: https://github.com/shubhamk853  
LinkedIn: https://www.linkedin.com/in/shubham-kale-0bb103347
