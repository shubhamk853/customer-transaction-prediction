# Customer Transaction Prediction
# Author: Shubham Bhagwan Kale

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATASET
# ============================================================
# Dataset: Santander Customer Transaction Prediction (Kaggle)
# Download from: https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
# After download, place train.csv in the same folder as this script

print("Loading dataset...")
df = pd.read_csv('train.csv')
print("Shape:", df.shape)
print("Target distribution:\n", df['target'].value_counts())

# ============================================================
# 2. EDA
# ============================================================
# Class imbalance check
plt.figure(figsize=(5, 4))
df['target'].value_counts().plot(kind='bar', color=['steelblue', 'salmon'])
plt.title('Class Distribution (Target)')
plt.xlabel('Target')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.savefig('class_distribution.png', bbox_inches='tight')
plt.close()
print("Saved: class_distribution.png")

# Mean feature values by target
feat_cols = [c for c in df.columns if c.startswith('var_')]
mean_by_target = df.groupby('target')[feat_cols[:10]].mean().T
plt.figure(figsize=(10, 5))
mean_by_target.plot(kind='bar', ax=plt.gca())
plt.title('Mean of First 10 Features by Target')
plt.ylabel('Mean Value')
plt.legend(['Not Transacted (0)', 'Transacted (1)'])
plt.savefig('feature_mean_by_target.png', bbox_inches='tight')
plt.close()
print("Saved: feature_mean_by_target.png")

# ============================================================
# 3. FEATURE ENGINEERING & PREPROCESSING
# ============================================================
X = df[feat_cols]
y = df['target']

print(f"\nFeatures: {X.shape[1]}, Samples: {X.shape[0]}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (use subset for speed)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 4. HANDLE CLASS IMBALANCE WITH SMOTE
# ============================================================
print("\nApplying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("After SMOTE:", pd.Series(y_train_sm).value_counts().to_dict())

# ============================================================
# 5. MODEL TRAINING
# ============================================================
# Logistic Regression
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sm, y_train_sm)
lr_pred = lr.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_pred)
print(f"Logistic Regression AUC-ROC: {lr_auc:.4f}")

# Random Forest (on sample for speed)
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_sm, y_train_sm)
rf_pred = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_pred)
print(f"Random Forest AUC-ROC: {rf_auc:.4f}")

print("\n--- Model Results ---")
print(f"Logistic Regression -> AUC-ROC: {lr_auc:.4f}")
print(f"Random Forest       -> AUC-ROC: {rf_auc:.4f}")

# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================
feat_imp = pd.Series(rf.feature_importances_, index=feat_cols)
plt.figure(figsize=(10, 5))
feat_imp.sort_values(ascending=False).head(15).plot(kind='bar', color='steelblue')
plt.title('Top 15 Feature Importances')
plt.ylabel('Importance Score')
plt.xticks(rotation=45)
plt.savefig('feature_importance.png', bbox_inches='tight')
plt.close()
print("Saved: feature_importance.png")

# ============================================================
# 7. CONFUSION MATRIX
# ============================================================
rf_labels = rf.predict(X_test)
cm = confusion_matrix(y_test, rf_labels)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Transaction', 'Transaction'],
            yticklabels=['No Transaction', 'Transaction'])
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix.png")

print("\nDone! Best Model: Random Forest | AUC-ROC:", round(rf_auc, 4))
