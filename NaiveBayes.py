"""
CS260 Lab 5: Naive Bayes Classifier
Author: Ruth Tilahun
Date: 10/21/25
Description:
This program implements a Naive Bayes classifier.
"""
from math import log

class NaiveBayes:
    """
    Implements a Naive Bayes classifier for categorical data.

    Attributes:
        alpha (float): Laplace smoothing parameter.
        K (int): Number of classes.
        n (int): Number of training examples.
        F (dict): Dictionary mapping feature names to possible values.
        class_counts (list): Count of examples for each class.
        log_prior (list): Log probabilities of class priors.
        log_likelihood (list): Nested dict of log-likelihoods for each feature value.
        denominators (list): Denominators for Laplace-smoothed probabilities.
    """
    def __init__(self, partition):
        """ 
        Initializes the Naive Bayes model with training data.
        partition (Partition): The training dataset containing examples
        """
        self.alpha = 1.0
        self.K = partition.K
        self.n = partition.n
        self.F = {fname: list(vals) for fname, vals in partition.F.items()} 

        # class counts and priors(how common each class is)
        self.class_counts =[0] * self.K
        
        for example in partition.data:
            self.class_counts[example.label] += 1

        self.log_prior = []
        for k in range(self.K):
            pk = (self.class_counts[k] + self.alpha) / (self.n + self.K)
            self.log_prior.append(log(pk))

        #feature-value count for each K
        fv_counts = []
        for _ in range(self.K):
            fv_counts.append({})

        for example in partition.data:
            k = example.label
            for fname, v in example.features.items():
                if fname not in fv_counts[k]:
                    fv_counts[k][fname] = {}
                fv_counts[k][fname][v]= fv_counts[k][fname].get(v, 0) + 1
        
        #likelihood probabilities
        self.log_likelihood = []
        self.denominators= []
        for _ in range(self.K):
            self.log_likelihood.append({}) 
            self.denominators.append({})  

        for k in range(self.K):
            nk = self.class_counts[k]
            for fname, values in self.F.items():
                V = len(values)
                den = nk + self.alpha * V
                self.denominators[k][fname] = den
                self.log_likelihood[k][fname] = {}
                for v in values:
                    num = fv_counts[k].get(fname, {}).get(v, 0) + self.alpha
                    self.log_likelihood[k][fname][v] = log(num / den)
    
    def classify(self, x_test):
        """
        Classifies a new example using the trained Naive Bayes model.
        x_test (dict): Dictionary of feature name-value pairs for the example to be classified.
        """
        # score(k) = log P(y=k) + sum_j log P(x_j=v | y=k)
        scores = [p for p in self.log_prior]  # copy
        for k in range(self.K):
            s = scores[k]
            nk = self.class_counts[k]
            for fname, v in x_test.items():
                # If value was unseen in training for this feature:
                # use Laplace-smoothed probability as if count = 0
                if v in self.log_likelihood[k].get(fname, {}):
                    s += self.log_likelihood[k][fname][v]
                else:
                    den = self.denominators[k][fname]
                    s += log(self.alpha / den)
            scores[k] = s
        
        #argmax
        best_k = 0
        best_score = scores[0]
        for k in range(1, self.K):
            if scores[k] > best_score:
                best_score = scores[k]
                best_k = k

        return best_k
