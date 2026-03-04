"""
CS260 Credit Default Project: Main Analysis Script
Authors: Ruth Tilahun, Kripa Lamichhane
Description: Compare Logistic Regression and Naive Bayes for credit default prediction
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Import our implementations
from logistic_regression import LogisticRegression
from Mixed_NB import MixedNaiveBayes
from Discretized_NB import DiscretizedNaiveBayes

# LOAD AND PREPARE DATA
print("CREDIT CARD DEFAULT PREDICTION")
print("Comparing Logistic Regression vs Naive Bayes")

# Load data
print("\nLoading data...")
data = pd.read_excel('credit_card_default.xls', header=1)

print(f"Dataset shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")

# Remove ID column
data = data.drop('ID', axis=1)

# Define feature categories
continuous_features = [
    'LIMIT_BAL', 'AGE',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

categorical_features = [
    'SEX', 'EDUCATION', 'MARRIAGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'
]

all_features = continuous_features + categorical_features

# Separate features and labels
X = data[all_features]
y = data['default payment next month']

print(f"\nFeatures: {X.shape}")
print(f"Labels: {y.shape}")
print(f"Class distribution:")
print(f"  No default (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"  Default (1): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")

# TRAIN-TEST SPLIT
print("\nSplitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"Training set: {len(X_train)} examples")
print(f"Test set: {len(X_test)} examples")


# NORMALIZE DATA FOR LOGISTIC REGRESSION
print("\nNormalizing features for Logistic Regression...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Normalization complete (StandardScaler: mean=0, std=1)")


# TRAIN LOGISTIC REGRESSION


print("LOGISTIC REGRESSION")


print("\nTraining Logistic Regression with SGD...")
lr_model = LogisticRegression()

# Train with optimized hyperparameters
train_info = lr_model.fit_SGD(
    X_train_scaled, 
    y_train,
    alpha=0.1,         # Learning rate (higher for faster convergence)
    eps=1e-6,          # Convergence threshold
    max_iter=500,      # Max iterations (reduced - should converge much faster)
    batch_size=256,    # Mini-batch size
    verbose=True
)

print(f"\nTraining completed:")
print(f"  Iterations: {train_info['iterations']}")
print(f"  Final cost: {train_info['final_cost']:.6f}")

# Evaluate on test set
print("\n--- OUR IMPLEMENTATION ---")
lr_results = lr_model.evaluate(X_test_scaled, y_test, verbose=True)

# Compare with sklearn's Logistic Regression
print("\n--- SKLEARN COMPARISON ---")
from sklearn.linear_model import LogisticRegression as SklearnLR
sklearn_model = SklearnLR(max_iter=1000, random_state=42)
sklearn_model.fit(X_train_scaled, y_train)
y_pred_sklearn = sklearn_model.predict(X_test_scaled)
y_proba_sklearn = sklearn_model.predict_proba(X_test_scaled)[:, 1]

# Calculate sklearn metrics
tp_sk = np.sum((y_test == 1) & (y_pred_sklearn == 1))
tn_sk = np.sum((y_test == 0) & (y_pred_sklearn == 0))
fp_sk = np.sum((y_test == 0) & (y_pred_sklearn == 1))
fn_sk = np.sum((y_test == 1) & (y_pred_sklearn == 0))

acc_sk = (tp_sk + tn_sk) / len(y_test)
prec_sk = tp_sk / (tp_sk + fp_sk) if (tp_sk + fp_sk) > 0 else 0
rec_sk = tp_sk / (tp_sk + fn_sk) if (tp_sk + fn_sk) > 0 else 0
auc_sk = roc_auc_score(y_test, y_proba_sklearn)

print(f"\nAccuracy:  {acc_sk:.4f}")
print(f"Precision: {prec_sk:.4f}")
print(f"Recall:    {rec_sk:.4f}")
print(f"AUC-ROC:   {auc_sk:.4f}")

print("\nConfusion Matrix:")
print("           Predicted")
print("           0      1")
print("         +-----------")
print(f"Actual 0 | {tn_sk:5d}  {fp_sk:5d}")
print(f"Actual 1 | {fn_sk:5d}  {tp_sk:5d}")

print("\n--- COMPARISON SUMMARY ---")
print(f"Accuracy:  Our={lr_results['accuracy']:.4f} | Sklearn={acc_sk:.4f} | Diff={abs(lr_results['accuracy']-acc_sk):.4f}")
print(f"Precision: Our={lr_results['precision']:.4f} | Sklearn={prec_sk:.4f} | Diff={abs(lr_results['precision']-prec_sk):.4f}")
print(f"Recall:    Our={lr_results['recall']:.4f} | Sklearn={rec_sk:.4f} | Diff={abs(lr_results['recall']-rec_sk):.4f}")

if abs(lr_results['accuracy'] - acc_sk) < 0.02:
    print("\nOur implementation matches sklearn closely! (within 2%)")
else:
    print("\n Some difference from sklearn - may need tuning")

# Get feature importance
feature_importance = lr_model.get_feature_importance(all_features)
print("\n--- Top 10 Most Important Features ---")
for i in range(min(10, len(feature_importance['features']))):
    feat = feature_importance['features'][i]
    imp = feature_importance['importance'][i]
    print(f"{i+1:2d}. {feat:15s}: {imp:.4f}")

# TRAIN NAIVE BAYES (MIXED)


print("NAIVE BAYES (MIXED: GAUSSIAN + CATEGORICAL)")


print("\nTraining Mixed Naive Bayes...")
nb_model = MixedNaiveBayes(
    X_train, y_train,
    continuous_features,
    categorical_features
)

# Predictions
y_pred_nb = nb_model.predict(X_test)
y_proba_nb = nb_model.predict_proba(X_test)[:, 1]

# Calculate metrics
tp = np.sum((y_test == 1) & (y_pred_nb == 1))
tn = np.sum((y_test == 0) & (y_pred_nb == 0))
fp = np.sum((y_test == 0) & (y_pred_nb == 1))
fn = np.sum((y_test == 1) & (y_pred_nb == 0))

acc_nb = (tp + tn) / len(y_test)
precision_nb = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_nb = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_nb = 2 * (precision_nb * recall_nb) / (precision_nb + recall_nb) if (precision_nb + recall_nb) > 0 else 0
ber_nb = 0.5 * (fp/(tn+fp) + fn/(tp+fn))

print("\nNAIVE BAYES EVALUATION")

print(f"Accuracy:  {acc_nb:.4f}")
print(f"Precision: {precision_nb:.4f}")
print(f"Recall:    {recall_nb:.4f}")
print(f"F1 Score:  {f1_nb:.4f}")
print(f"BER:       {ber_nb:.4f}")

print("\nConfusion Matrix:")
print("           Predicted")
print("           0      1")
print("         +-----------")
print(f"Actual 0 | {tn:5d}  {fp:5d}")
print(f"Actual 1 | {fn:5d}  {tp:5d}")


# COMPARISON SUMMARY

print("MODEL COMPARISON SUMMARY")


# Calculate AUC scores
auc_lr = roc_auc_score(y_test, lr_results['probabilities'])
auc_nb = roc_auc_score(y_test, y_proba_nb)

print("\n{:<25s} {:>12s} {:>12s} {:>12s}".format("Metric", "Our LR", "Sklearn LR", "Naive Bayes"))
print("-----------------------------------------------------------------------")
print("{:<25s} {:>12.4f} {:>12.4f} {:>12.4f}".format("Accuracy", lr_results['accuracy'], acc_sk, acc_nb))
print("{:<25s} {:>12.4f} {:>12.4f} {:>12.4f}".format("Precision", lr_results['precision'], prec_sk, precision_nb))
print("{:<25s} {:>12.4f} {:>12.4f} {:>12.4f}".format("Recall", lr_results['recall'], rec_sk, recall_nb))
print("{:<25s} {:>12.4f} {:>12.4f} {:>12.4f}".format("F1 Score", lr_results['f1_score'], 2*(prec_sk*rec_sk)/(prec_sk+rec_sk) if (prec_sk+rec_sk)>0 else 0, f1_nb))
print("{:<25s} {:>12.4f} {:>12.4f} {:>12.4f}".format("AUC-ROC", auc_lr, auc_sk, auc_nb))
print("{:<25s} {:>12.4f} {:>12.4f} {:>12.4f}".format("Balanced Error Rate", lr_results['ber'], 0.5*(fp_sk/(tn_sk+fp_sk)+fn_sk/(tp_sk+fn_sk)), ber_nb))

# Determine winner
if lr_results['accuracy'] > acc_nb:
    print("\nLogistic Regression performs better!")
else:
    print("\nNaive Bayes performs better !")


# VISUALIZATIONS
print("\n Generating visualizations...")

# --- Plot 1: Cost History ---
plt.figure(figsize=(10, 6))
plt.plot(lr_model.cost_history, linewidth=2, color='#2E86AB')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Cost (Negative Log-Likelihood)', fontsize=12)
plt.title('Logistic Regression: Training Cost History', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/lr_cost_history.png', dpi=300, bbox_inches='tight')
print("figures/lr_cost_history.png is saved")

# --- Plot 2: ROC Curves Comparison ---
plt.figure(figsize=(10, 7))

# Logistic Regression ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_results['probabilities'])
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={auc_lr:.3f})', 
         linewidth=2.5, color='#A23B72')

# Naive Bayes ROC
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_proba_nb)
plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC={auc_nb:.3f})', 
         linewidth=2.5, color='#F18F01', linestyle='--')

# Random classifier baseline
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.4)

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison: Logistic Regression vs Naive Bayes', 
          fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/lr_vs_nb_roc.png', dpi=300, bbox_inches='tight')
print("figures/lr_vs_nb_roc.png is saved")

# --- Plot 3: Feature Importance (Top 15) ---
plt.figure(figsize=(10, 8))
top_n = 15
features = feature_importance['features'][:top_n]
importance = feature_importance['importance'][:top_n]

colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
bars = plt.barh(range(top_n), importance, color=colors)
plt.yticks(range(top_n), features)
plt.xlabel('Absolute Weight (Importance)', fontsize=12)
plt.title(f'Top {top_n} Most Important Features (Logistic Regression)', 
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/lr_feature_importance.png', dpi=300, bbox_inches='tight')
print("figures/lr_feature_importance.png is saved")

# --- Plot 4: Confusion Matrices Side-by-Side ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Logistic Regression
cm_lr = lr_results['confusion_matrix']
im1 = axes[0].imshow(cm_lr, cmap='Blues', aspect='auto')
axes[0].set_title('Logistic Regression', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Predicted', fontsize=11)
axes[0].set_ylabel('Actual', fontsize=11)
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['No Default', 'Default'])
axes[0].set_yticklabels(['No Default', 'Default'])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, f'{cm_lr[i, j]}', ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='white' if cm_lr[i,j] > cm_lr.max()/2 else 'black')
plt.colorbar(im1, ax=axes[0], fraction=0.046)

# Naive Bayes
cm_nb = np.array([[tn, fp], [fn, tp]])
im2 = axes[1].imshow(cm_nb, cmap='Oranges', aspect='auto')
axes[1].set_title('Naive Bayes', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted', fontsize=11)
axes[1].set_ylabel('Actual', fontsize=11)
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['No Default', 'Default'])
axes[1].set_yticklabels(['No Default', 'Default'])
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f'{cm_nb[i, j]}', ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='white' if cm_nb[i,j] > cm_nb.max()/2 else 'black')
plt.colorbar(im2, ax=axes[1], fraction=0.046)

plt.tight_layout()
plt.savefig('figures/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
print(" figures/confusion_matrices_comparison.png is saved")



print("\nfolowing files are created:")
print("figures/lr_cost_history.png")
print("figures/lr_vs_nb_roc.png")
print("figures/lr_feature_importance.png")
print("figures/confusion_matrices_comparison.png")