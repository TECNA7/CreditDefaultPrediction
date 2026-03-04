# roc curve comparison: logistic regression vs naive bayes

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# load data
df = pd.read_csv('data/credit_card_default.csv', skiprows=1)

continuous_features = [
    'LIMIT_BAL', 'AGE',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

categorical_features = [
    'SEX', 'EDUCATION', 'MARRIAGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'
]

X = df[continuous_features + categorical_features]
y = df['default payment next month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# logistic regression
preprocessor = ColumnTransformer([
    ('num', 'passthrough', continuous_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
])
X_train_lr = preprocessor.fit_transform(X_train)
X_test_lr = preprocessor.transform(X_test)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_lr, y_train)
y_proba_lr = lr_model.predict_proba(X_test_lr)[:, 1]

# naive bayes
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
X_train_nb = X_train.copy()
X_test_nb = X_test.copy()
X_train_nb[continuous_features] = discretizer.fit_transform(X_train[continuous_features])
X_test_nb[continuous_features] = discretizer.transform(X_test[continuous_features])

for col in categorical_features:
    X_train_nb[col] = X_train_nb[col].astype(int)
    X_test_nb[col] = X_test_nb[col].astype(int)
    min_val = X_train_nb[col].min()
    if min_val < 0:
        X_train_nb[col] -= min_val
        X_test_nb[col] -= min_val

nb_model = CategoricalNB()
nb_model.fit(X_train_nb.astype(int), y_train)
y_proba_nb = nb_model.predict_proba(X_test_nb.astype(int))[:, 1]

# plot roc curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_proba_nb)
auc_lr = roc_auc_score(y_test, y_proba_lr)
auc_nb = roc_auc_score(y_test, y_proba_nb)

plt.figure()
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={auc_lr:.3f})')
plt.plot(fpr_nb, tpr_nb, linestyle='--', label=f'Naive Bayes (AUC={auc_nb:.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()