# feature importance analysis

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from rename_features import (
    rename_dataframe, 
    CONTINUOUS_FEATURES_NEW,
    CATEGORICAL_FEATURES_NEW,
    DEMOGRAPHIC_FEATURES_NEW,
    rename_onehot_features
)
from Discretized_NB import DiscretizedNaiveBayes
from information_gain import calculate_information_gain_all_features, visualize_information_gain

# load and rename data
df = pd.read_csv('data/credit_card_default.csv', skiprows=1)
df = rename_dataframe(df)

continuous_features = CONTINUOUS_FEATURES_NEW
categorical_features = CATEGORICAL_FEATURES_NEW

X = df[continuous_features + categorical_features]
y = df['Default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dataset: {len(df)} examples")
print(f"Training: {len(X_train)}, Test: {len(X_test)}")

# method 1: information gain
print("\nMethod 1: Information Gain")
ig_results = calculate_information_gain_all_features(
    X_train, y_train, continuous_features, categorical_features, n_bins=5
)
print("\nTop 15 features:")
print(ig_results.head(15).to_string(index=False))

visualize_information_gain(ig_results, top_n=15)

# method 2: logistic regression coefficients
print("\nMethod 2: Logistic Regression Coefficients")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', continuous_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

X_train_lr = preprocessor.fit_transform(X_train)
X_test_lr = preprocessor.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
log_reg.fit(X_train_lr, y_train)

feature_names_encoded = preprocessor.get_feature_names_out()
feature_names_intuitive = rename_onehot_features(feature_names_encoded)

coefficients = pd.DataFrame({
    'Feature': feature_names_intuitive,
    'Coefficient': log_reg.coef_[0],
    'Abs_Coefficient': np.abs(log_reg.coef_[0])
})
coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)
coefficients['Rank'] = range(1, len(coefficients) + 1)

print("\nTop 15 features by coefficient magnitude:")
print(coefficients.head(15)[['Feature', 'Coefficient', 'Abs_Coefficient']].to_string(index=False))

plt.figure()
top_coef = coefficients.head(20)
colors = ['green' if c > 0 else 'red' for c in top_coef['Coefficient']]
plt.barh(range(len(top_coef)), top_coef['Coefficient'], color=colors)
plt.yticks(range(len(top_coef)), top_coef['Feature'])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Top 20 Features by Logistic Regression Coefficient')
plt.gca().invert_yaxis()
plt.show()

