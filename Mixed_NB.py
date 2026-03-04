"""
CS260 Credit Default Project: Mixed Naive Bayes Classifier
Author: Ruth Tilahun, Kripa Lamichhane
Description:
Implements Mixed Naive Bayes that handles both continuous (Gaussian) 
and categorical features for credit default prediction.
"""
from math import log, sqrt, pi, exp
import numpy as np

class MixedNaiveBayes:
    """
    Naive Bayes classifier that handles mixed feature types:
    - Continuous features: Uses Gaussian (Normal) distribution
    - Categorical features: Uses multinomial distribution with Laplace smoothing
    
    Attributes:
        alpha (float): Laplace smoothing parameter for categorical features
        K (int): Number of classes (2 for binary classification)
        continuous_features (list): Names of continuous features
        categorical_features (list): Names of categorical features
        class_counts (list): Count of examples for each class
        log_prior (list): Log probabilities of class priors
        
        # For continuous features (Gaussian parameters)
        means (list of dict): Mean for each continuous feature per class
        stds (list of dict): Standard deviation for each continuous feature per class
        
        # For categorical features (frequency-based probabilities)
        log_likelihood (list of dict): Log-likelihoods for categorical features
        denominators (list of dict): Denominators for Laplace smoothing
        categorical_values (dict): Possible values for each categorical feature
    """
    
    def __init__(self, X_train, y_train, continuous_features, categorical_features):
        """
        Initialize and train the Mixed Naive Bayes model.
        
        Args:
            X_train: Training data (pandas DataFrame or dict of feature arrays)
            y_train: Training labels (numpy array or list)
            continuous_features: List of column names that are continuous
            categorical_features: List of column names that are categorical
        """
        self.alpha = 1.0  # Laplace smoothing
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        
        # Convert to numpy if needed
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        y_train = np.array(y_train)
        
        self.K = len(np.unique(y_train))  # Number of classes
        self.n = len(y_train)  # Total training examples
        
        # Count examples per class
        self.class_counts = [0] * self.K
        for label in y_train:
            self.class_counts[label] += 1
        
        # Compute class priors: P(y=k)
        self.log_prior = []
        for k in range(self.K):
            pk = (self.class_counts[k] + self.alpha) / (self.n + self.K * self.alpha)
            self.log_prior.append(log(pk))
        
        # === TRAIN CONTINUOUS FEATURES (Gaussian) ===
        self.means = [{} for _ in range(self.K)]
        self.stds = [{} for _ in range(self.K)]
        
        for k in range(self.K):
            # Get all examples of class k
            class_mask = (y_train == k)
            
            for fname in continuous_features:
                # Get feature values for this class
                if isinstance(X_train, dict):
                    feature_vals = np.array(X_train[fname])[class_mask]
                else:
                    feature_vals = X_train[fname].values[class_mask]
                
                # Calculate mean and standard deviation
                self.means[k][fname] = np.mean(feature_vals)
                self.stds[k][fname] = np.std(feature_vals) + 1e-6  # Add small value to avoid division by zero
        
        # === TRAIN CATEGORICAL FEATURES (Multinomial with Laplace) ===
        self.categorical_values = {}
        
        # First, collect all possible values for each categorical feature
        for fname in categorical_features:
            if isinstance(X_train, dict):
                self.categorical_values[fname] = list(set(X_train[fname]))
            else:
                self.categorical_values[fname] = list(X_train[fname].unique())
        
        # Count feature-value occurrences for each class
        fv_counts = [{} for _ in range(self.K)]
        
        for i in range(len(y_train)):
            k = y_train[i]
            
            for fname in categorical_features:
                if isinstance(X_train, dict):
                    v = X_train[fname][i]
                else:
                    v = X_train[fname].iloc[i]
                
                if fname not in fv_counts[k]:
                    fv_counts[k][fname] = {}
                fv_counts[k][fname][v] = fv_counts[k][fname].get(v, 0) + 1
        
        # Calculate log-likelihoods with Laplace smoothing
        self.log_likelihood = [{} for _ in range(self.K)]
        self.denominators = [{} for _ in range(self.K)]
        
        for k in range(self.K):
            nk = self.class_counts[k]
            
            for fname in categorical_features:
                V = len(self.categorical_values[fname])
                den = nk + self.alpha * V
                self.denominators[k][fname] = den
                self.log_likelihood[k][fname] = {}
                
                for v in self.categorical_values[fname]:
                    count = fv_counts[k].get(fname, {}).get(v, 0)
                    num = count + self.alpha
                    self.log_likelihood[k][fname][v] = log(num / den)
    
    def _gaussian_log_likelihood(self, x, mean, std):
        """
        Calculate log of Gaussian probability density function.
        
        P(x|μ,σ) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
        log P(x|μ,σ) = -log(σ) - 0.5*log(2π) - (x-μ)²/(2σ²)
        """
        variance = std ** 2
        log_coef = -log(std) - 0.5 * log(2 * pi)
        log_exp = -((x - mean) ** 2) / (2 * variance)
        return log_coef + log_exp
    
    def classify(self, x_test):
        """
        Classify a single example.
        
        Args:
            x_test: Dictionary or pandas Series with feature name-value pairs
            
        Returns:
            int: Predicted class label (0 or 1)
        """
        scores = list(self.log_prior)  # Start with class priors
        
        for k in range(self.K):
            score = scores[k]
            
            # Add log-likelihoods from continuous features (Gaussian)
            for fname in self.continuous_features:
                x_val = x_test[fname]
                mean = self.means[k][fname]
                std = self.stds[k][fname]
                score += self._gaussian_log_likelihood(x_val, mean, std)
            
            # Add log-likelihoods from CATEGORICAL features
            for fname in self.categorical_features:
                v = x_test[fname]
                
                # If value was seen in training
                if v in self.log_likelihood[k].get(fname, {}):
                    score += self.log_likelihood[k][fname][v]
                else:
                    # Unseen value: use Laplace-smoothed probability (count=0)
                    den = self.denominators[k][fname]
                    score += log(self.alpha / den)
            
            scores[k] = score
        
        # Return class with highest score
        return np.argmax(scores)
    
    def predict(self, X_test):
        """
        Classify multiple examples.
        
        Args:
            X_test: DataFrame or dict of feature arrays
            
        Returns:
            numpy array: Predicted class labels
        """
        predictions = []
        
        # Handle DataFrame
        if hasattr(X_test, 'iterrows'):
            for _, row in X_test.iterrows():
                predictions.append(self.classify(row))
        # Handle dict
        elif isinstance(X_test, dict):
            n_samples = len(X_test[list(X_test.keys())[0]])
            for i in range(n_samples):
                x_test = {fname: X_test[fname][i] for fname in X_test}
                predictions.append(self.classify(x_test))
        else:
            raise ValueError("X_test must be DataFrame or dict")
        
        return np.array(predictions)
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities.
        
        Returns:
            numpy array: Shape (n_samples, n_classes) with probabilities
        """
        probas = []
        
        # Handle DataFrame
        if hasattr(X_test, 'iterrows'):
            for _, row in X_test.iterrows():
                scores = self._get_scores(row)
                probas.append(self._scores_to_proba(scores))
        # Handle dict
        elif isinstance(X_test, dict):
            n_samples = len(X_test[list(X_test.keys())[0]])
            for i in range(n_samples):
                x_test = {fname: X_test[fname][i] for fname in X_test}
                scores = self._get_scores(x_test)
                probas.append(self._scores_to_proba(scores))
        
        return np.array(probas)
    
    def _get_scores(self, x_test):
        """Helper to get log-probability scores for all classes."""
        scores = list(self.log_prior)
        
        for k in range(self.K):
            score = scores[k]
            
            # Continuous features
            for fname in self.continuous_features:
                x_val = x_test[fname]
                mean = self.means[k][fname]
                std = self.stds[k][fname]
                score += self._gaussian_log_likelihood(x_val, mean, std)
            
            # Categorical features
            for fname in self.categorical_features:
                v = x_test[fname]
                if v in self.log_likelihood[k].get(fname, {}):
                    score += self.log_likelihood[k][fname][v]
                else:
                    den = self.denominators[k][fname]
                    score += log(self.alpha / den)
            
            scores[k] = score
        
        return scores
    
    def _scores_to_proba(self, scores):
        """Convert log-probability scores to normalized probabilities."""
        # Subtract max for numerical stability
        max_score = max(scores)
        exp_scores = [exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        return [p / total for p in exp_scores]