# Credit Card Fraud Prediction

A beginner-friendly machine-learning project that trains a **logistic regression** classifier to distinguish fraudulent from legitimate credit-card transactions. The work is presented as an exploratory Jupyter notebook, from loading data through balancing, training, and evaluation.

> **Important:** This repository is an educational experiment, not a production fraud-detection system. Fraud detection is a highly imbalanced, high-risk classification problem; a model should not be used to make financial decisions without rigorous validation, monitoring, privacy controls, and human review.

## Project overview

The notebook follows this workflow:

1. Loads the transaction dataset from `sample_data/creditcard.csv`.
2. Inspects the schema, summary statistics, missing values, and class distribution.
3. Separates legitimate (`Class = 0`) and fraudulent (`Class = 1`) transactions.
4. Addresses the class imbalance with random under-sampling: it samples 492 legitimate transactions to match the 492 fraud cases.
5. Splits the balanced data into training and test sets.
6. Trains a scikit-learn `LogisticRegression` model.
7. Reports accuracy for both training and test predictions.

## Repository structure

```text
.
├── credit_card_fraud_detection.ipynb  # Data exploration, training, and evaluation
├── README.md                          # Project documentation
└── .gitignore                         # Excludes the local dataset file
```

## Dataset

The notebook expects the [Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains anonymized PCA-transformed features (`V1`–`V28`), along with `Time`, `Amount`, and the target column `Class`:

| Value | Meaning |
| --- | --- |
| `0` | Legitimate transaction |
| `1` | Fraudulent transaction |

The original dataset contains 284,807 transactions, including 492 fraud cases. Because the dataset is large, it is deliberately not committed to this repository.

## Getting started

### Prerequisites

- Python 3.9 or newer
- Jupyter Notebook or JupyterLab
- `pandas`
- `numpy`
- `scikit-learn`

### Installation

1. Clone the repository and enter it:

   ```bash
   git clone <repository-url>
   cd Credit-Card-fraud-prediction
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell, use `.venv\\Scripts\\Activate.ps1` instead.

3. Install the required packages:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install jupyter numpy pandas scikit-learn
   ```

4. Download `creditcard.csv` from Kaggle and place it here:

   ```text
   sample_data/creditcard.csv
   ```

   Create the `sample_data` directory if it does not already exist. Keep the dataset local: it is excluded by `.gitignore`.

5. Start Jupyter and open the notebook:

   ```bash
   jupyter notebook credit_card_fraud_detection.ipynb
   ```

6. Run the notebook cells from top to bottom.

### Notebook setup note

The train/test split cell is currently commented out in the notebook. Before running it in a fresh kernel, remove the leading `#` from this line:

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2, stratify=Y
)
```

Without this step, the later training and evaluation cells will not have the `X_train`, `X_test`, `Y_train`, and `Y_test` variables they require.

## Reported notebook result

The committed notebook records a test accuracy of approximately **92.4%** on its balanced, under-sampled test split. Accuracy alone is not enough to assess a fraud model: it can obscure performance on the minority class when evaluated against the natural transaction distribution. Treat this value as a reproducibility reference for the notebook, not as a production-quality benchmark.

## Limitations and next steps

- **Sampling variability:** the under-sampling call has no fixed random seed, so repeat runs can select different legitimate transactions and produce different results.
- **Convergence:** the recorded run reports a logistic-regression convergence warning. Feature scaling and a larger `max_iter` value should be considered before interpreting results.
- **Evaluation:** add precision, recall, F1 score, PR-AUC, ROC-AUC, and a confusion matrix. In fraud detection, recall and precision are often more informative than accuracy.
- **Data splitting:** use a reproducible, stratified split and consider temporal validation when transaction time is meaningful.
- **Imbalance handling:** compare class weighting, over-sampling, and other resampling strategies with the current under-sampling baseline.
- **Deployment readiness:** production use requires threshold selection based on business cost, calibration, drift monitoring, security, and compliance review.

## License

No license has been specified for this repository. Add a license file before redistributing or reusing the project.
