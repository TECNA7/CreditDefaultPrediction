# run naive bayes models comparison

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, roc_curve
from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
import matplotlib.pyplot as plt

from Mixed_NB import MixedNaiveBayes
from Discretized_NB import DiscretizedNaiveBayes

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dataset: {len(df)} examples")
print(f"Training: {len(X_train)}, Test: {len(X_test)}")
print(f"No default: {(y==0).sum()}, Default: {(y==1).sum()}")

# custom mixed naive bayes
print("\nCustom Mixed Naive Bayes")
nb_mixed = MixedNaiveBayes(X_train, y_train, continuous_features, categorical_features)
y_pred_mixed = nb_mixed.predict(X_test)
y_proba_mixed = nb_mixed.predict_proba(X_test)[:, 1]

acc_mixed = accuracy_score(y_test, y_pred_mixed)
auc_mixed = roc_auc_score(y_test, y_proba_mixed)
conf_mixed = confusion_matrix(y_test, y_pred_mixed)
ber_mixed = 0.5 * (conf_mixed[0][1]/(conf_mixed[0][0]+conf_mixed[0][1]) + 
                   conf_mixed[1][0]/(conf_mixed[1][0]+conf_mixed[1][1]))
precision_mixed = precision_score(y_test, y_pred_mixed)
recall_mixed = recall_score(y_test, y_pred_mixed)

print(f"Accuracy: {acc_mixed:.4f}, AUC: {auc_mixed:.4f}")
print(f"Precision: {precision_mixed:.4f}, Recall: {recall_mixed:.4f}, BER: {ber_mixed:.4f}")

# custom discretized naive bayes
print("\nCustom Discretized Naive Bayes (5 bins)")
nb_disc = DiscretizedNaiveBayes(X_train, y_train, continuous_features, n_bins=5, strategy='quantile')
y_pred_disc = nb_disc.predict(X_test)
y_proba_disc = nb_disc.predict_proba(X_test)[:, 1]

acc_disc = accuracy_score(y_test, y_pred_disc)
auc_disc = roc_auc_score(y_test, y_proba_disc)
conf_disc = confusion_matrix(y_test, y_pred_disc)
ber_disc = 0.5 * (conf_disc[0][1]/(conf_disc[0][0]+conf_disc[0][1]) + 
                  conf_disc[1][0]/(conf_disc[1][0]+conf_disc[1][1]))
precision_disc = precision_score(y_test, y_pred_disc)
recall_disc = recall_score(y_test, y_pred_disc)

print(f"Accuracy: {acc_disc:.4f}, AUC: {auc_disc:.4f}")
print(f"Precision: {precision_disc:.4f}, Recall: {recall_disc:.4f}, BER: {ber_disc:.4f}")

# prepare data for sklearn models
X_train_gaussian = X_train.copy()
X_test_gaussian = X_test.copy()

label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X_train_gaussian[col] = le.fit_transform(X_train[col])
    X_test_gaussian[col] = le.transform(X_test[col])
    label_encoders[col] = le

discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
X_train_categorical = X_train.copy()
X_test_categorical = X_test.copy()

X_train_categorical[continuous_features] = discretizer.fit_transform(X_train[continuous_features])
X_test_categorical[continuous_features] = discretizer.transform(X_test[continuous_features])

for col in categorical_features:
    X_train_categorical[col] = X_train_categorical[col].astype(int)
    X_test_categorical[col] = X_test_categorical[col].astype(int)
    min_val = X_train_categorical[col].min()
    if min_val < 0:
        X_train_categorical[col] = X_train_categorical[col] - min_val
        X_test_categorical[col] = X_test_categorical[col] - min_val

X_train_categorical = X_train_categorical.astype(int)
X_test_categorical = X_test_categorical.astype(int)

# sklearn gaussian naive bayes
print("\nSklearn Gaussian Naive Bayes")
gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train_gaussian, y_train)
y_pred_gaussian = gaussian_nb.predict(X_test_gaussian)
y_proba_gaussian = gaussian_nb.predict_proba(X_test_gaussian)[:, 1]

acc_gaussian = accuracy_score(y_test, y_pred_gaussian)
auc_gaussian = roc_auc_score(y_test, y_proba_gaussian)
conf_gaussian = confusion_matrix(y_test, y_pred_gaussian)
ber_gaussian = 0.5 * (conf_gaussian[0][1]/(conf_gaussian[0][0]+conf_gaussian[0][1]) + 
                      conf_gaussian[1][0]/(conf_gaussian[1][0]+conf_gaussian[1][1]))
precision_gaussian = precision_score(y_test, y_pred_gaussian)
recall_gaussian = recall_score(y_test, y_pred_gaussian)

print(f"Accuracy: {acc_gaussian:.4f}, AUC: {auc_gaussian:.4f}")
print(f"Precision: {precision_gaussian:.4f}, Recall: {recall_gaussian:.4f}, BER: {ber_gaussian:.4f}")

# sklearn categorical naive bayes
print("\nSklearn Categorical Naive Bayes (5 bins)")
categorical_nb = CategoricalNB()
categorical_nb.fit(X_train_categorical, y_train)
y_pred_categorical = categorical_nb.predict(X_test_categorical)
y_proba_categorical = categorical_nb.predict_proba(X_test_categorical)[:, 1]

acc_categorical = accuracy_score(y_test, y_pred_categorical)
auc_categorical = roc_auc_score(y_test, y_proba_categorical)
conf_categorical = confusion_matrix(y_test, y_pred_categorical)
ber_categorical = 0.5 * (conf_categorical[0][1]/(conf_categorical[0][0]+conf_categorical[0][1]) + 
                         conf_categorical[1][0]/(conf_categorical[1][0]+conf_categorical[1][1]))
precision_categorical = precision_score(y_test, y_pred_categorical)
recall_categorical = recall_score(y_test, y_pred_categorical)

print(f"Accuracy: {acc_categorical:.4f}, AUC: {auc_categorical:.4f}")
print(f"Precision: {precision_categorical:.4f}, Recall: {recall_categorical:.4f}, BER: {ber_categorical:.4f}")

# sklearn multinomial naive bayes
print("\nSklearn Multinomial Naive Bayes (5 bins)")
multinomial_nb = MultinomialNB()
multinomial_nb.fit(X_train_categorical, y_train)
y_pred_multinomial = multinomial_nb.predict(X_test_categorical)
y_proba_multinomial = multinomial_nb.predict_proba(X_test_categorical)[:, 1]

acc_multinomial = accuracy_score(y_test, y_pred_multinomial)
auc_multinomial = roc_auc_score(y_test, y_proba_multinomial)
conf_multinomial = confusion_matrix(y_test, y_pred_multinomial)
ber_multinomial = 0.5 * (conf_multinomial[0][1]/(conf_multinomial[0][0]+conf_multinomial[0][1]) + 
                         conf_multinomial[1][0]/(conf_multinomial[1][0]+conf_multinomial[1][1]))
precision_multinomial = precision_score(y_test, y_pred_multinomial)
recall_multinomial = recall_score(y_test, y_pred_multinomial)

print(f"Accuracy: {acc_multinomial:.4f}, AUC: {auc_multinomial:.4f}")
print(f"Precision: {precision_multinomial:.4f}, Recall: {recall_multinomial:.4f}, BER: {ber_multinomial:.4f}")

# plot roc curves
plt.figure()

fpr_categorical, tpr_categorical, _ = roc_curve(y_test, y_proba_categorical)
plt.plot(fpr_categorical, tpr_categorical, label=f'Sklearn Categorical NB (AUC={auc_categorical:.3f})', linestyle='-.')

fpr_disc, tpr_disc, _ = roc_curve(y_test, y_proba_disc)
plt.plot(fpr_disc, tpr_disc, label=f'Custom Discretized NB (AUC={auc_disc:.3f})', linestyle='--')

fpr_mixed, tpr_mixed, _ = roc_curve(y_test, y_proba_mixed)
plt.plot(fpr_mixed, tpr_mixed, label=f'Custom Mixed NB (AUC={auc_mixed:.3f})')

fpr_multinomial, tpr_multinomial, _ = roc_curve(y_test, y_proba_multinomial)
plt.plot(fpr_multinomial, tpr_multinomial, label=f'Sklearn Multinomial NB (AUC={auc_multinomial:.3f})')

fpr_gaussian, tpr_gaussian, _ = roc_curve(y_test, y_proba_gaussian)
plt.plot(fpr_gaussian, tpr_gaussian, label=f'Sklearn Gaussian NB (AUC={auc_gaussian:.3f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: Naive Bayes Comparison')
plt.legend(loc='lower right')
plt.show()

# summary table
print("\nFinal Comparison")
results = pd.DataFrame({
    'Model': ['Custom Mixed NB', 'Custom Discretized NB', 'Sklearn Gaussian NB', 
              'Sklearn Categorical NB', 'Sklearn Multinomial NB'],
    'Accuracy': [acc_mixed, acc_disc, acc_gaussian, acc_categorical, acc_multinomial],
    'AUC': [auc_mixed, auc_disc, auc_gaussian, auc_categorical, auc_multinomial],
    'BER': [ber_mixed, ber_disc, ber_gaussian, ber_categorical, ber_multinomial],
    'Precision': [precision_mixed, precision_disc, precision_gaussian, precision_categorical, precision_multinomial],
    'Recall': [recall_mixed, recall_disc, recall_gaussian, recall_categorical, recall_multinomial]
})
print(results.to_string(index=False))