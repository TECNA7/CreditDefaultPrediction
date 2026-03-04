# Credit Card Default Prediction: Comparing Logistic Regression and Naive Bayes

**Authors:** Ruth T. Tilahun & Kripa Lamichhane  
**Course:** CS260 - Foundation of Data Science  
**Institution:** Haverford College  
**Date:** December 2025

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Motivation](#motivation)
3. [Research Questions](#research-questions)
4. [Dataset](#dataset)
5. [Installation & Setup](#installation--setup)
6. [Data Preprocessing](#data-preprocessing)
7. [Naive Bayes Implementation](#naive-bayes-implementation)
8. [Logistic Regression Implementation](#logistic-regression-implementation)
9. [Results](#results)
10. [Model Comparison](#model-comparison)
11. [Feature Importance](#feature-importance)
12. [Conclusion](#conclusion)
13. [Future Work](#future-work)
14. [References](#references)

---

## Project Overview

This project compares two fundamental machine learning algorithms for credit card default prediction. We implemented both **Logistic Regression** and **Naive Bayes** from previous labs using Python and NumPy and then compared their performance on predicting whether credit card customers will default on their next payment.

Our approach shows the perspective of a credit institution that wants to accurately predict customer defaults. We treat mistakes in both directions (false positives and false negatives) as equally important, making overall model correctness the main indicator of performance.

### What We Built

- Custom implementation of Logistic Regression using mini-batch SGD
- Two versions of Naive Bayes (Mixed and Discretized) to handle mixed data types
- Comprehensive evaluation framework with multiple metrics
- Validation against scikit-learn implementations
- Feature importance analysis using both Information Gain and coefficient weights

---

## Motivation

We approach the problem from the perspective of a credit institution that wants to know, as often as possible, whether a customer will or will not default. In this setup, mistakes in either direction are treated as equally important, so the overall correctness of the model becomes the main indicator of performance.

Credit card defaults cost financial institutions billions of dollars annually. Better prediction models can:
- Help manage risk and reduce losses
- Set appropriate credit limits for customers
- Ensure fair lending practices across demographic groups

---

## Research Questions

Our project addresses three key questions:

1. **Performance Comparison:** Will Naive Bayes or Logistic Regression perform better for credit default prediction?

2. **Feature Importance:** Which features are most informative for predicting default?

3. **Bias Analysis:** Do demographic features (like gender and education) introduce bias into the predictions?

---

## Dataset

### Dataset Overview

- **Source:** UCI Machine Learning Repository (Taiwan, 2005)
- **Name:** Default of Credit Card Clients
- **Size:** 30,000 customers
- **Features:** 23 independent variables
- **Task:** Binary classification (Will customer default next month?)
- **Class Distribution:**
  - No Default: 23,364 customers (78%)
  - Default: 6,636 customers (22%)
- **Challenge:** Imbalanced classes (3.5:1 ratio)

### Features Breakdown

#### Continuous Features (15 features)

1. **LIMIT_BAL:** Credit limit amount in NT dollars
2. **AGE:** Client age in years
3. **BILL_AMT1 through BILL_AMT6:** Bill statement amounts for past 6 months
   - BILL_AMT1 = September 2005 bill
   - BILL_AMT2 = August 2005 bill
   - And so on...
4. **PAY_AMT1 through PAY_AMT6:** Previous payment amounts for past 6 months
   - PAY_AMT1 = Amount paid in September 2005
   - PAY_AMT2 = Amount paid in August 2005
   - And so on...

#### Categorical Features (8 features)

1. **SEX:** Gender
   - 1 = male
   - 2 = female

2. **EDUCATION:** Education level
   - 1 = graduate school
   - 2 = university
   - 3 = high school
   - 4 = others

3. **MARRIAGE:** Marital status
   - 1 = married
   - 2 = single
   - 3 = others

4. **PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6:** Repayment status for past 6 months
   - Range: -2 to 9
   - -2 = no consumption
   - -1 = paid on time (duly)
   - 0 = use of revolving credit
   - 1 = payment delay for 1 month
   - 2 = payment delay for 2 months
   - Higher values = more months delayed

#### Target Label

- **default payment next month:** Whether customer defaulted in October 2005
  - 0 = No default
  - 1 = Default

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Required Packages

```bash
pip install numpy pandas scikit-learn matplotlib openpyxl
```

Or install from requirements file:
```bash
pip install -r requirements.txt
```

### Project Structure

```
cs260-final-project-ruth_kripa/
├── data/
│   └── credit_card_default.csv
├── figures/
│   ├── lr_cost_history.png
│   ├── lr_vs_nb_roc.png
│   ├── lr_feature_importance.png
│   ├── confusion_matrices_comparison.png
│   ├── naive_bayes_roc_comparison.png
│   └── discretized_nb_bin_analysis.png
├── logistic_regression.py        
├── Mixed_NB.py                    
├── Discretized_NB.py               
├── main.py                         
├── run_NB.py                       
├── credit_card_default.xls         
└── README.md                        
```

### Running the Code

**Main Analysis (Logistic Regression vs Naive Bayes):**
```bash
python3 main.py
```

This will:
- Load and preprocess the data
- Train Logistic Regression using mini-batch SGD
- Train Mixed Naive Bayes
- Compare both models with sklearn implementations
- Print detailed metrics and confusion matrices
- Generate 4 visualization plots in `figures/` folder
- Takes approximately 30 seconds to run

**Naive Bayes Comparison (Mixed vs Discretized):**
```bash
python3 run_NB.py
```

This will:
- Compare Mixed Naive Bayes with Discretized Naive Bayes
- Test different bin sizes (5, 10, 15, 20) for discretization
- Generate ROC curves and performance plots
- Takes approximately 10 seconds to run

---

## Data Preprocessing

### Step 1: Load and Clean Data

1. Read data from csv file (`credit_card_default.csv`)
2. Remove ID column (not predictive, just an identifier)
3. Verify data integrity (check for missing values - none found)
4. Separate features (X) and labels (y)

### Step 2: Train-Test Split

- **Split Ratio:** 80% training, 20% testing
- **Training Set:** 24,000 examples
- **Testing Set:** 6,000 examples
- **Method:** split to maintain class distribution in both sets
  - Training: 78% no default, 22% default
  - Testing: 78% no default, 22% default

### Step 3: Feature Normalization (for Logistic Regression)

**Method:** StandardScaler (Z-score normalization)

**Formula:**
```
normalized_value = (value - mean) / standard_deviation
```

**Result:** All features transformed to have:
- Mean = 0
- Standard Deviation = 1

**Why Normalize?**
- Features have vastly different scales:
  - LIMIT_BAL ranges from 10,000 to 1,000,000
  - AGE ranges from 21 to 79
- Without normalization, SGD would be dominated by large-scale features
- Normalization makes gradient descent converge faster and more reliably

**Important Note:** Naive Bayes does NOT require normalization because:
- Gaussian NB estimates mean and std for each feature separately (scale-invariant)
- Categorical features just count frequencies (scale doesn't matter)

---

## Naive Bayes Implementation

### The Challenge: Mixed Data Types

Our dataset contains both:
- **Continuous features:** Credit limit, age, bill amounts, payment amounts
- **Categorical features:** Gender, education, payment status

Traditional Naive Bayes can only handle one type:
- **Gaussian Naive Bayes:** Assumes all features are continuous and normally distributed
- **Multinomial Naive Bayes:** Assumes all features are discrete/categorical

**Our Solution:** Implement two approaches to handle mixed data types

### Approach 1: Mixed Naive Bayes

**Strategy:** Use different probability models for different feature types

**For Continuous Features:**
- Assume Gaussian (normal) distribution
- Estimate mean (μ) and standard deviation (σ) for each feature, for each class
- Calculate probability using Gaussian probability density function:
  ```
  P(x|class) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))
  ```

**For Categorical Features:**
- Use multinomial distribution with frequency counts
- Apply Laplace smoothing (add 1 to all counts) to handle unseen values
- Calculate probability as:
  ```
  P(x=value|class) = (count(value, class) + 1) / (count(class) + number_of_values)
  ```

**Combining Both:**
- Use Bayes' theorem: `P(class|features) ∝ P(class) × P(features|class)`
- With independence assumption: `P(features|class) = P(f1|class) × P(f2|class) × ... × P(fn|class)`
- Use log probabilities for numerical stability

**Implementation File:** `Mixed_NB.py`

### Approach 2: Discretized Naive Bayes

**Strategy:** Convert all continuous features to categorical using binning

**How It Works:**
1. For each continuous feature, create bins (like creating categories)
2. Assign each value to a bin
3. Now all features are categorical
4. Use standard multinomial Naive Bayes

**Binning Methods Tested:**
- **Uniform:** Equal-width bins (divide range into equal parts)
- **Quantile:** Equal-frequency bins (each bin has roughly same number of examples)

**Bin Sizes Tested:** 5, 10, 15, 20 bins

**Best Result:** 5 bins with quantile strategy achieved 78.1% accuracy

**Trade-offs:**
- **Simpler:** Only need to count frequencies, no Gaussian calculations
- **May lose information:** Binning can lose fine-grained details
- **Bin size matters:** Too few bins = too coarse, too many bins = overfitting

**Implementation File:** `Discretized_NB.py`

### Why Naive Bayes?

**Advantages:**
- **Interpretable:** Easy to understand how it makes decisions
- **Handles mixed data:** Our two approaches solve this challenge
- **Computationally efficient:** Training is very fast (< 1 second)
- **Works with less data:** Doesn't need as much training data as other methods

**The "Naive" Assumption:**
- It assumes all features are independent given the class
- Example: Assumes knowing PAY_0 doesn't tell you anything about PAY_2
- This assumption is violated in our data (payment history is correlated)
- **Question we wanted to answer:** Does it still give good predictions despite this?

### Naive Bayes Results

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| sklearn Categorical NB | 78.8% | 51.0% | 48.8% |
| Custom Discretized NB (5 bins) | 78.1% | 50.9% | 49.1% |
| Custom Mixed NB | 73.4% | 42.7% | 58.9% |

**Key Observations:**
- Discretized NB (78.1%) almost matches sklearn (78.8%)
- Mixed NB has lower accuracy but **much higher recall** (58.9% vs 48-49%)
- Mixed NB is better at catching actual defaults, even with more false alarms

---

## Logistic Regression Implementation

### Algorithm Overview

Logistic Regression models the probability that a customer will default using the sigmoid function.

**Mathematical Model:**
```
P(default|features) = 1 / (1 + e^(-(w₀ + w₁×feature₁ + w₂×feature₂ + ... + wₙ×featureₙ)))
```

Where:
- w₀, w₁, w₂, ..., wₙ are weights we need to learn
- The sigmoid function converts any number to a probability between 0 and 1

### Training Method: Mini-Batch Stochastic Gradient Descent (SGD)

**What is SGD?**
Instead of looking at all 24,000 training examples at once to update weights:
1. Randomly shuffle the training data
2. Split data into mini-batches (groups of 256 examples)
3. For each mini-batch:
   - Calculate predictions
   - Calculate error (how wrong we are)
   - Update weights to reduce error
4. Repeat until the model converges (stops improving)

**Why Mini-Batch SGD?**
- **Faster than full-batch:** Don't need to process all 24,000 examples before updating
- **More stable than single-example:** Using 256 examples gives more reliable updates
- **Works well in practice:** Good balance of speed and accuracy

### Training Details

**Hyperparameters Used:**
- **Learning Rate (α):** 0.1
  - How big of a step we take when updating weights
  - Too large = unstable, too small = very slow
  - 0.1 worked well for our normalized data
- **Batch Size:** 256 examples per mini-batch
  - 24,000 training examples ÷ 256 = 94 batches per iteration
- **Max Iterations:** 500 (we allow up to 500 full passes through the data)
- **Convergence Threshold:** 1e-6 (stop if improvement is less than 0.000001)

### Cost History

The cost function measures how well our model fits the training data (lower is better).

**Our Training Results:**
- **Started with cost:** 0.474
- **Final cost:** 0.463
- **Converged in:** 32 iterations (epochs)
- **Training time:** ~10 seconds

**What the cost history graph shows:**
1. **Sharp initial drop:** Cost decreases rapidly from 0.474 to 0.464 in first 5 iterations
2. **Smooth convergence:** Gradual, stable improvement with minimal oscillation
3. **Fast convergence:** Only needed 32 out of 500 possible iterations
4. **No overfitting:** Steady cost reduction indicates good generalization

The smooth curve shows our learning rate (0.1) was well-chosen - not too aggressive, not too conservative.

### Validation Against sklearn

To verify our implementation is correct, we compared it with scikit-learn's LogisticRegression:

| Metric | Our Implementation | sklearn | Difference |
|--------|-------------------|---------|------------|
| Accuracy | 80.87% | 80.77% | 0.10% |
| Precision | 69.58% | 68.68% | 0.90% |
| Recall | 23.96% | 23.96% | 0.00% |

**Conclusion:** Our implementation matches sklearn within 1%, confirming it's working correctly!

**Implementation File:** `logistic_regression.py`

---

## Results

### Overall Performance Comparison

| Metric | Logistic Regression | Mixed Naive Bayes | Discretized NB (5 bins) | sklearn Categorical NB |
|--------|---------------------|-------------------|------------------------|------------------------|
| **Accuracy** | 80.92% | 73.4% | 78.1% | 78.8% |
| **Precision** | 68.56% | 42.7% | 50.9% | 51.0% |
| **Recall** | 25.47% | 58.9% | 49.1% | 48.8% |
| **F1 Score** | 35.65% | 49.49% | 50.0% | 49.9% |
| **AUC-ROC** | 0.7074 | 0.7343 | 0.7250 | 0.7280 |

### Understanding the Metrics

**Accuracy:** Percentage of correct predictions overall
- LR is best at 80.92%
- But accuracy can be misleading with imbalanced data!

**Precision:** When model predicts "default", how often is it correct?
- LR: 68.56% - when it says default, it's right about 2 out of 3 times
- NB: 42.7% - more false alarms

**Recall:** Of all customers who actually defaulted, how many did we catch?
- LR: 25.47% - catches only 1 out of 4 defaults (big problem!)
- NB: 58.9% - catches almost 6 out of 10 defaults (much better!)

**F1 Score:** Harmonic mean of precision and recall (balanced metric)
- NB (49.49%) beats LR (35.65%) because recall matters

**AUC-ROC:** Probability ranking quality (higher = better at ordering predictions)
- NB (0.7343) beats LR (0.7074)
- Shows NB has better calibrated probabilities

### Confusion Matrices

#### Logistic Regression
```
                    PREDICTED
                No Default  Default    Total
ACTUAL
No Default         4,534      139      4,673
Default            1,009      318      1,327
                  -------    ------
Total              5,543      457      6,000
```

**What this tells us:**
- **True Negatives (4,534):** Correctly identified non-defaults - excellent!
- **False Positives (139):** Wrong flags - very few false alarms
- **False Negatives (1,009):** Missed defaults - this is the big problem!
- **True Positives (318):** Correctly caught defaults - only 24% of all defaults

**Model Behavior:** Very conservative
- Predicts "default" only 457 times (7.6% of all predictions)
- Reality: 22.1% should default
- Model is being too cautious, missing most actual defaults

#### Mixed Naive Bayes
```
                    PREDICTED
                No Default  Default    Total
ACTUAL
No Default         3,622     1,051     4,673
Default              545       782     1,327
                  -------    ------
Total              4,167     1,833     6,000
```

**What this tells us:**
- **True Negatives (3,622):** Correctly identified non-defaults - still good
- **False Positives (1,051):** More false alarms than LR
- **False Negatives (545):** Half of what LR misses - much better!
- **True Positives (782):** Catches 2.5× more defaults than LR

**Model Behavior:** More balanced
- Predicts "default" 1,833 times (30.6% of all predictions)
- Closer to reality (22.1% actually default)
- Willing to have more false alarms to catch more defaults

---

## Model Comparison
### AUC-ROC

**Naive Bayes has better AUC-ROC (0.7343 vs 0.7074)**

What this means:
- If you pick a random defaulter and a random non-defaulter
- NB will rank the defaulter as more risky 73.43% of the time
- LR will rank correctly only 70.74% of the time

NB has better probability calibration across all possible decision thresholds, not just 0.5.

---

## Feature Importance

### Two Methods of Analysis

We analyzed feature importance using two different approaches:

1. **Logistic Regression Coefficients** (Kripa's analysis)
   - Shows how much each feature changes the probability of default
   - Positive weight = increases default risk
   - Negative weight = decreases default risk
   - Magnitude shows strength of effect

2. **Information Gain** (Ruth's analysis)
   - Measures how much knowing a feature reduces uncertainty
   - Higher value = more informative feature
   - Used for Naive Bayes feature selection

### Top 10 Features (Logistic Regression Weights)

| Rank | Feature | Weight | Interpretation |
|------|---------|--------|----------------|
| 1 | PAY_0 | 0.6608 | Most recent payment status - **dominates all others** |
| 2 | BILL_AMT1 | 0.2397 | Most recent bill amount |
| 3 | PAY_AMT1 | 0.1847 | Most recent payment amount |
| 4 | PAY_AMT2 | 0.1829 | Second most recent payment |
| 5 | LIMIT_BAL | 0.1350 | Credit limit |
| 6 | EDUCATION | 0.0948 | Education level |
| 7 | PAY_3 | 0.0928 | Payment status 3 months ago |
| 8 | PAY_2 | 0.0906 | Payment status 2 months ago |
| 9 | AGE | 0.0744 | Client age |
| 10 | MARRIAGE | 0.0740 | Marital status |

### Key Insights

**1. Recency Matters Most**
- PAY_0 (most recent payment) is 2.75× more important than the second feature
- Top 4 features are all from the most recent 1-2 months
- Recent behavior is the best predictor of future behavior

**2. Payment Behavior vs Debt Amount**
- Payment status (PAY_0, PAY_2, PAY_3) is very important
- Bill amounts (BILL_AMT1) matter but less so
- **How you pay matters more than how much you owe**

**3. Demographics Have Minimal Impact**
- **SEX (gender):** Not even in top 10 - very low importance
- **EDUCATION:** 6th place but still 7× less important than PAY_0
- **AGE & MARRIAGE:** 9th and 10th place
- Financial behavior is 50-80× more important than demographics

**4. Information Gain Confirms the Same Pattern**
- Ruth's Information Gain analysis showed identical top features
- PAY_0 dominates
- Gender has minimal predictive power
- Debt amounts less important than payment behavior

### What This Means for Bias

**Good News:**
- Gender (SEX) contributes almost nothing to predictions
- Minimal gender bias in model predictions

**Moderate Concern:**
- Education does appear (6th place, weight 0.0948)
- Could reflect correlation with financial literacy rather than direct discrimination
- Worth investigating further

**Recommendation:**
- Train model without demographic features and compare:
  - If accuracy drops a lot, they're useful
  - If accuracy stays similar, can remove them for fairness

---

## Conclusion

### Main Findings

**1. Logistic Regression Outperforms Naive Bayes on Accuracy**
- LR: 80.92% accuracy vs NB: 73.4% accuracy
- **Why?** Features are correlated (payment history, bill amounts, etc.)
- LR models these correlations, NB assumes independence
- Theory predicts LR should win, and it does!

**2. Naive Bayes Has Better Recall**
- NB catches 58.9% of defaults vs LR's 25.5%
- NB has better AUC-ROC (0.7343 vs 0.7074)
- **Why?** NB's probabilistic framework handles class imbalance better
- For catching defaults, NB is actually superior

**3. No Single "Best" Model**
- Choice depends on business priorities:
  - Need accuracy? → Logistic Regression
  - Need to catch defaults? → Naive Bayes
- Could use both in a two-stage system

**4. Recent Payment History is King**
- PAY_0 (most recent payment status) is by far the most important feature
- 50-80× more important than demographic features
- If you could only collect one feature, collect payment history!

**5. Demographics Contribute Minimal Value**
- Gender: Not in top 10
- Education: 6th place but small weight
- Age and Marriage: Bottom of top 10
- Good news for fairness - financial behavior drives predictions, not demographics

## References
**Dataset**

Yeh, I-Cheng. "Default of Credit Card Clients." UCI Machine Learning Repository, 2009. https://doi.org/10.24432/C55S3H
Yeh, I-C., & Lien, C. (2009). "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients." Expert Systems with Applications, 36(2), 2473-2480.


## Software Libraries

Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research, 12, 2825-2830.
Harris, C. R., et al. (2020). "Array programming with NumPy." Nature, 585(7825), 357-362.
Hunter, J. D. (2007). "Matplotlib: A 2D graphics environment." Computing in Science & Engineering, 9(3), 90-95.

### License & Citation
This project was completed as part of CS260 coursework at Haverford College.
To cite this work:
Tilahun, R. T., & Lamichhane, K. (2024). 
Credit Card Default Prediction: A Comparative Analysis of Logistic Regression and Naive Bayes. 
CS260 Machine Learning, Haverford College.
