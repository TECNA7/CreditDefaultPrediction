"""
CS260 Credit Default Project: Logistic Regression
Author: Kripa Lamichhane
Date: Dec 2024
Description: Logistic regression implementation using SGD for credit card default prediction
"""
from math import log, exp
import numpy as np

class LogisticRegression():
    """
    Logistic Regression classifier using Mini-Batch Stochastic Gradient Descent
    
    Attributes:
        X (np.array): Training features with bias term appended
        y (np.array): Training labels (0 or 1)
        weights (np.array): Learned weights after training
        cost_history (list): Cost at each iteration
    """

    def __init__(self):
        """Initialize empty logistic regression model"""
        self.weights = None
        self.cost_history = []

    def _add_bias(self, X):
        """Add bias term (column of 1s) to feature matrix"""
        bias = np.ones((X.shape[0], 1))
        return np.concatenate((bias, X), axis=1)

    def sigmoid(self, z):
        """
        Sigmoid function: converts linear combination to probability
        σ(z) = 1 / (1 + e^(-z))
        
        Args:
            z: Linear combination w^T * x
        Returns:
            Probability between 0 and 1
        """
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        """
        Compute probability of positive class for each example
        
        Args:
            X: Feature matrix (n_samples, n_features)
        Returns:
            Array of probabilities
        """
        X_bias = self._add_bias(X)
        z = np.dot(X_bias, self.weights)
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        Make binary predictions
        
        Args:
            X: Feature matrix
            threshold: Classification threshold (default 0.5)
        Returns:
            Array of 0s and 1s
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def cost(self, X, y):
        """
        Compute logistic regression cost (negative log-likelihood)
        
        Cost = -1/n * Σ[y*log(h) + (1-y)*log(1-h)]
        
        Args:
            X: Feature matrix with bias
            y: True labels
        Returns:
            Average cost over all examples
        """
        n = len(y)
        epsilon = 1e-15  # To avoid log(0)
        
        z = np.dot(X, self.weights)
        h = self.sigmoid(z)
        
        # Clip predictions to avoid log(0)
        h = np.clip(h, epsilon, 1 - epsilon)
        
        cost = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost

    def fit_SGD(self, X_train, y_train, alpha=0.01, eps=1e-6, max_iter=1000, 
                batch_size=128, verbose=True):
        """
        Train logistic regression using Mini-Batch Stochastic Gradient Descent
        
        OPTIMIZED: Uses mini-batches instead of single examples for faster training
        
        Args:
            X_train: Training features (should be normalized)
            y_train: Training labels
            alpha: Learning rate
            eps: Convergence threshold
            max_iter: Maximum iterations (epochs)
            batch_size: Number of examples per mini-batch
            verbose: Print progress
            
        Returns:
            Dictionary with training info (iterations, final cost, weights)
        """
        # Add bias term
        X = self._add_bias(X_train)
        y = np.array(y_train)
        
        n_samples, n_features = X.shape
        
        # Initialize weights to zero
        self.weights = np.zeros(n_features)
        
        # Track cost history
        self.cost_history = []
        cost_prev = float('inf')
        
        # Calculate number of batches
        n_batches = int(np.ceil(n_samples / batch_size))
        
        if verbose:
            print(f"Using mini-batch SGD with batch_size={batch_size}")
            print(f"Training on {n_samples} examples, {n_batches} batches per epoch")
        
        for iteration in range(max_iter):
            # Shuffle data each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process mini-batches
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Compute predictions for batch
                z = np.dot(X_batch, self.weights)
                h = self.sigmoid(z)
                
                # Compute gradient (averaged over batch)
                gradient = np.dot(X_batch.T, (h - y_batch)) / len(y_batch)
                
                # Update weights
                self.weights -= alpha * gradient
            
            # Compute cost after full epoch
            cost_curr = self.cost(X, y)
            self.cost_history.append(cost_curr)
            
            # Check convergence
            if abs(cost_prev - cost_curr) < eps:
                if verbose:
                    print(f"\n Converged after {iteration + 1} iterations!")
                    print(f"Final cost: {cost_curr:.6f}")
                break
            
            cost_prev = cost_curr
            
            # Print progress every 50 iterations
            if verbose and (iteration + 1) % 50 == 0:
                print(f"Iteration {iteration + 1:4d}: Cost = {cost_curr:.6f}")
        
        if verbose and iteration == max_iter - 1:
            print(f"\nReached max iterations ({max_iter})")
            print(f"Final cost: {cost_curr:.6f}")
        
        return {
            'iterations': iteration + 1,
            'final_cost': cost_curr,
            'weights': self.weights
        }

    def evaluate(self, X_test, y_test, verbose=True):
        """
        Evaluate model performance on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            verbose: Print results
            
        Returns:
            Dictionary with accuracy, confusion matrix, and metrics
        """
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Confusion matrix components
        tp = np.sum((y_test == 1) & (y_pred == 1))  # True positives
        tn = np.sum((y_test == 0) & (y_pred == 0))  # True negatives
        fp = np.sum((y_test == 0) & (y_pred == 1))  # False positives
        fn = np.sum((y_test == 1) & (y_pred == 0))  # False negatives
        
        # Calculate metrics
        accuracy = (tp + tn) / len(y_test)
        
        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall (Sensitivity): TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Balanced Error Rate
        ber = 0.5 * (fp / (tn + fp) + fn / (tp + fn)) if (tn + fp) > 0 and (tp + fn) > 0 else 0
        
        if verbose:
            print("LOGISTIC REGRESSION EVALUATION")
            print(f"\nAccuracy:  {accuracy:.4f} ({tp + tn}/{len(y_test)} correct)")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            print(f"BER:       {ber:.4f}")
            
            print("\nConfusion Matrix:")
            print("           Predicted")
            print("           0      1")
            print("         +-----------")
            print(f"Actual 0 | {tn:5d}  {fp:5d}")
            print(f"Actual 1 | {fn:5d}  {tp:5d}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'ber': ber,
            'confusion_matrix': np.array([[tn, fp], [fn, tp]]),
            'predictions': y_pred,
            'probabilities': y_proba
        }

    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance based on absolute weight values
        (excluding bias term)
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.weights is None:
            raise ValueError("Model hasn't been trained yet!")
        
        # Exclude bias term (first weight)
        importance = np.abs(self.weights[1:])
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importance))]
        
        # Sort by importance
        sorted_indices = np.argsort(importance)[::-1]
        
        return {
            'features': [feature_names[i] for i in sorted_indices],
            'importance': importance[sorted_indices]
        }