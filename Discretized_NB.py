"""
CS260 Credit Default Project: Discretized Naive Bayes Classifier
Author: Ruth Tilahun, Kripa Lamichhane
Description:
Implements Naive Bayes that discretizes all continuous features into bins,
then treats all features as categorical using the original NaiveBayes implementation.
"""
# discretized naive bayes classifier
# implements naive bayes that discretizes continuous features into bins

from math import log, exp
import numpy as np

class DiscretizedNaiveBayes:
    # naive bayes classifier that discretizes continuous features into bins
    
    def __init__(self, X_train, y_train, continuous_features, 
                 n_bins=10, strategy='quantile'):
        self.alpha = 1.0
        self.n_bins = n_bins
        self.strategy = strategy
        self.continuous_features = continuous_features
        
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        y_train = np.array(y_train)
        
        self.K = len(np.unique(y_train))
        self.n = len(y_train)
        
        # count examples per class
        self.class_counts = [0] * self.K
        for label in y_train:
            self.class_counts[label] += 1
        
        # compute class priors
        self.log_prior = []
        for k in range(self.K):
            pk = (self.class_counts[k] + self.alpha) / (self.n + self.K * self.alpha)
            self.log_prior.append(log(pk))
        
        # discretize continuous features
        self.bin_edges = {}
        X_discretized = self._discretize_training_data(X_train)
        
        # collect all feature values
        self.feature_values = {}
        if isinstance(X_discretized, dict):
            for fname in X_discretized:
                self.feature_values[fname] = list(set(X_discretized[fname]))
        else:
            for fname in X_discretized.columns:
                self.feature_values[fname] = list(X_discretized[fname].unique())
        
        # count feature-value occurrences per class
        fv_counts = [{} for _ in range(self.K)]
        for i in range(len(y_train)):
            k = y_train[i]
            if isinstance(X_discretized, dict):
                features = {fname: X_discretized[fname][i] for fname in X_discretized}
            else:
                features = X_discretized.iloc[i].to_dict()
            for fname, v in features.items():
                if fname not in fv_counts[k]:
                    fv_counts[k][fname] = {}
                fv_counts[k][fname][v] = fv_counts[k][fname].get(v, 0) + 1
        
        # calculate log-likelihoods with laplace smoothing
        self.log_likelihood = [{} for _ in range(self.K)]
        self.denominators = [{} for _ in range(self.K)]
        for k in range(self.K):
            nk = self.class_counts[k]
            for fname in self.feature_values:
                V = len(self.feature_values[fname])
                den = nk + self.alpha * V
                self.denominators[k][fname] = den
                self.log_likelihood[k][fname] = {}
                for v in self.feature_values[fname]:
                    count = fv_counts[k].get(fname, {}).get(v, 0)
                    num = count + self.alpha
                    self.log_likelihood[k][fname][v] = log(num / den)
    
    def _discretize_training_data(self, X_train):
        # discretize continuous features and store bin edges
        if isinstance(X_train, dict):
            X_discretized = {k: list(v) for k, v in X_train.items()}
        else:
            X_discretized = X_train.copy()
        
        for fname in self.continuous_features:
            if isinstance(X_train, dict):
                values = np.array(X_train[fname])
            else:
                values = X_train[fname].values
            
            if self.strategy == 'uniform':
                min_val, max_val = values.min(), values.max()
                self.bin_edges[fname] = np.linspace(min_val, max_val, self.n_bins + 1)
            elif self.strategy == 'quantile':
                self.bin_edges[fname] = np.percentile(
                    values, np.linspace(0, 100, self.n_bins + 1))
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            
            self.bin_edges[fname] = np.unique(self.bin_edges[fname])
            discretized = np.digitize(values, self.bin_edges[fname][1:-1])
            
            if isinstance(X_discretized, dict):
                X_discretized[fname] = discretized.tolist()
            else:
                X_discretized[fname] = discretized
        
        return X_discretized
    
    def _discretize_test_data(self, X_test):
        # discretize test data using stored bin edges
        if isinstance(X_test, dict):
            X_discretized = {k: list(v) for k, v in X_test.items()}
        else:
            X_discretized = X_test.copy()
        
        for fname in self.continuous_features:
            if isinstance(X_test, dict):
                values = np.array(X_test[fname])
            else:
                values = X_test[fname].values
            
            discretized = np.digitize(values, self.bin_edges[fname][1:-1])
            
            if isinstance(X_discretized, dict):
                X_discretized[fname] = discretized.tolist()
            else:
                X_discretized[fname] = discretized
        
        return X_discretized
    
    def classify(self, x_test):
        # classify a single example
        scores = list(self.log_prior)
        for k in range(self.K):
            score = scores[k]
            for fname, v in x_test.items():
                if v in self.log_likelihood[k].get(fname, {}):
                    score += self.log_likelihood[k][fname][v]
                else:
                    den = self.denominators[k][fname]
                    score += log(self.alpha / den)
            scores[k] = score
        return np.argmax(scores)
    
    def predict(self, X_test):
        # classify multiple examples
        X_discretized = self._discretize_test_data(X_test)
        predictions = []
        
        if hasattr(X_discretized, 'iterrows'):
            for _, row in X_discretized.iterrows():
                predictions.append(self.classify(row))
        elif isinstance(X_discretized, dict):
            n_samples = len(X_discretized[list(X_discretized.keys())[0]])
            for i in range(n_samples):
                x_test = {fname: X_discretized[fname][i] for fname in X_discretized}
                predictions.append(self.classify(x_test))
        
        return np.array(predictions)
    
    def predict_proba(self, X_test):
        # predict class probabilities
        X_discretized = self._discretize_test_data(X_test)
        probas = []
        
        if hasattr(X_discretized, 'iterrows'):
            for _, row in X_discretized.iterrows():
                scores = self._get_scores(row)
                probas.append(self._scores_to_proba(scores))
        elif isinstance(X_discretized, dict):
            n_samples = len(X_discretized[list(X_discretized.keys())[0]])
            for i in range(n_samples):
                x_test = {fname: X_discretized[fname][i] for fname in X_discretized}
                scores = self._get_scores(x_test)
                probas.append(self._scores_to_proba(scores))
        
        return np.array(probas)
    
    def _get_scores(self, x_test):
        # get log-probability scores for all classes
        scores = list(self.log_prior)
        for k in range(self.K):
            score = scores[k]
            for fname, v in x_test.items():
                if v in self.log_likelihood[k].get(fname, {}):
                    score += self.log_likelihood[k][fname][v]
                else:
                    den = self.denominators[k][fname]
                    score += log(self.alpha / den)
            scores[k] = score
        return scores
    
    def _scores_to_proba(self, scores):
        # convert log-probability scores to normalized probabilities
        max_score = max(scores)
        exp_scores = [exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        return [p / total for p in exp_scores]