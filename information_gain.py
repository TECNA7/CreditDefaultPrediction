# information gain analysis for feature importance

import numpy as np
import pandas as pd
from math import log2

def entropy(labels):
    # calculate entropy of class labels
    if len(labels) == 0:
        return 0
    
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    
    h = 0
    for p in probabilities:
        if p > 0:
            h -= p * log2(p)
    
    return h


def conditional_entropy(feature_values, labels):
    # calculate conditional entropy
    if len(feature_values) == 0:
        return 0
    
    n_total = len(labels)
    h_conditional = 0
    
    for value in np.unique(feature_values):
        mask = (feature_values == value)
        labels_subset = labels[mask]
        p_x = len(labels_subset) / n_total
        h_y_given_x = entropy(labels_subset)
        h_conditional += p_x * h_y_given_x
    
    return h_conditional


def information_gain(feature_values, labels):
    # calculate information gain: ig = h(y) - h(y|x)
    h_y = entropy(labels)
    h_y_given_x = conditional_entropy(feature_values, labels)
    return h_y - h_y_given_x


def discretize_continuous(values, n_bins=5, strategy='quantile'):
    # discretize continuous features for information gain calculation
    if strategy == 'quantile':
        bin_edges = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    else:
        bin_edges = np.linspace(values.min(), values.max(), n_bins + 1)
    
    bin_edges = np.unique(bin_edges)
    discretized = np.digitize(values, bin_edges[1:-1])
    return discretized


def calculate_information_gain_all_features(X_train, y_train, 
                                           continuous_features, 
                                           categorical_features,
                                           n_bins=5):
    # calculate information gain for all features
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    
    results = []
    
    for feature in continuous_features:
        values = X_train[feature].values
        discretized = discretize_continuous(values, n_bins=n_bins)
        ig = information_gain(discretized, y_train)
        results.append({
            'Feature': feature,
            'Type': 'Continuous',
            'Information_Gain': ig
        })
    
    for feature in categorical_features:
        values = X_train[feature].values
        ig = information_gain(values, y_train)
        results.append({
            'Feature': feature,
            'Type': 'Categorical',
            'Information_Gain': ig
        })
    
    ig_df = pd.DataFrame(results)
    ig_df = ig_df.sort_values('Information_Gain', ascending=False)
    ig_df['Rank'] = range(1, len(ig_df) + 1)
    
    return ig_df


def visualize_information_gain(ig_df, top_n=15):
    # visualize information gain results
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    plt.figure()
    top_features = ig_df.head(top_n)
    colors = ['steelblue' if t == 'Continuous' else 'coral' 
              for t in top_features['Type']]
    
    plt.barh(range(len(top_features)), top_features['Information_Gain'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Information Gain (bits)')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Features by Information Gain')
    
    legend_elements = [
        Patch(facecolor='steelblue', label='Continuous'),
        Patch(facecolor='coral', label='Categorical')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    plt.gca().invert_yaxis()
    plt.show()