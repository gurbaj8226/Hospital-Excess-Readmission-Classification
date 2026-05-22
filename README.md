# Hospital Excess Readmission Classification Using CMS HRRP Data

## Project Overview

This project seeks to answer whether publicly reported CMS hospital readmission performance data can be used to classify hospital-measure records with excess readmission ratios above 1.0. The workflow focuses on healthcare analytics modeling, using structured public reporting data from the CMS Hospital Readmissions Reduction Program (HRRP), while avoiding any sort of leakage.

The project includes:

* structural data validation
* missing value analysis
* Leakage-aware feature selection 
* binary target construction
* exploratory analysis
* preprocessing pipelines
* supervised classification modeling
* model comparison and evaluation

Two classification models were evaluated:

* Logistic Regression
* Decision Tree

The project focuses on having a realistic healthcare analytics methodology, instead of maximizing predictive performance using variables exposing the model to leakage.

---

# Business / Healthcare Context

Hospital readmission performance is an important healthcare quality and reimbursement metric tracked by the Centers for Medicare & Medicaid Services (CMS). Excess readmissions may symbolize operational inefficiencies, discharge planning issues, care coordination gaps, or broader healthcare population challenges.

CMS publicly reports hospital readmission performance through the Hospital Readmissions Reduction Program (HRRP), which evaluates hospitals across several clinical measures.

This project investigates whether broad operational and contextual variables can help classify hospital-measure records with excess readmission performance above expected levels.

---

# Dataset

**Dataset:** CMS Hospital Readmissions Reduction Program (HRRP)
**Source:** CMS Provider Data
**Link:** [https://data.cms.gov/provider-data/dataset/9n3s-kdb3#data-table](https://data.cms.gov/provider-data/dataset/9n3s-kdb3#data-table)

### Original Dataset

* **Rows:** 18,330
* **Columns:** 12

### Final Modeling Dataset

* **Rows:** 8,037
* **Columns:** 4

### Data Grain

Each row represents a:

> hospital-measure reporting record

This is not patient-level data.

---

# Project Objective

The objective of this project was to build a binary classification model that predicts whether a hospital-measure record has an excess readmission ratio above 1.0.

### Binary Target Definition

```python
high_excess_readmission = 1 if ERR > 1.0
high_excess_readmission = 0 if ERR <= 1.0
```

The classification portion identifies hospital-measure records associated with higher-than-expected readmission performance.

---

# Warning & Key Disclaimer 

This project intentionally serves as a:

> hospital-measure-level healthcare analytics classification task

It is **not**:

* patient-level readmission prediction
* clinical diagnosis prediction
* causal inference analysis

The models classify publicly reported CMS hospital-measure records using operational/contextual variables. The results reflect associations within aggregated public data, rather than direct explanations of clinical quality outcomes.

---

# Tools Used

* Python
* pandas
* scikit-learn
* matplotlib
* Jupyter Notebook
* Parquet
* Git / GitHub

---

# Repository Structure

```text
Hospital-Excess-Readmission-Classification/
│
├── notebooks/
│   ├── 01_data_cleaning_and_structural_validation.ipynb
│   ├── 02_target_definition_and_eda.ipynb
│   └── 03_classification_modeling.ipynb
│
├── visuals/
│   └── roc_curve_logistic_regression.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

> The `data/` directory is excluded from GitHub because the original dataset is publicly available from CMS.

---

# Data Cleaning and Structural Validation

The project began with structural validation and cleaning of the CMS HRRP dataset.

### Cleaning Steps Included

* standardized column names
* duplicate checks
* missing value analysis
* structural validation of data types
* investigation of CMS footnote reporting behavior
* filtering rows with valid excess readmission ratio values

### Missing Value Investigation

Several variables contained missing values which can be attributed to CMS reporting rules and suppression logic.

Common CMS footnotes included:

* “Too few cases”
* “Results not available”
* “Reporting periods incomplete”

Rows missing the target variable (`excess_readmission_ratio`) were excluded from supervised modeling.

### Dataset Reduction

* Original rows: **18,330**
* Rows with valid ERR values: **11,720**
* Final modeling dataset: **8,037**

---

# Target Definition

The target variable was created using the hospital’s excess readmission ratio (ERR).

### Classification Logic

* `ERR > 1.0` → high excess readmission
* `ERR <= 1.0` → non-high excess readmission

The target classes remained reasonably balanced after filtering:

* approximately 54% positive class
* approximately 46% negative class

Because ERR values were densely concentrated near 1.0, the classification boundary was inherently difficult.

---

# Feature Selection and Leakage Prevention

A conservative feature selection strategy was intentionally used to reduce target leakage.

## Included Features

* `state`
* `measure_name`
* `number_of_discharges`

These variables represent broad operational and contextual hospital-measure characteristics.

## Excluded Variables

The following variables were excluded due to leakage concerns or weak analytical value:

* `excess_readmission_ratio`
* `predicted_readmission_rate`
* `expected_readmission_rate`
* `number_of_readmissions`
* `facility_id`
* `facility_name`
* `footnote`
* reporting date fields

### Why Leakage Prevention Matters

Several excluded variables are directly tied, and can therefore reveal the target variable, either mathematically or operationally. Including them could artificially inflate model performance without actually improving the real generalization ability.

This project prioritizes methodological defensibility over maximizing predictive accuracy.

---

# Modeling Approach

The dataset was split using:

* 80/20 train-test split
* stratified target distribution (maintain the ratio of data despite splitting it).
* fixed random state for reproducibility (Fixed random state for reproducibility).

The same train-test split was used across all models.

## Logistic Regression

The Logistic Regression workflow included:

* `OneHotEncoder` for categorical features (The models cannot interpret text, so it needs to be converted to numeric values. If the numeric values are sequenced, such as 1,2, or 3, the model might be biased towards larger numbers. OneHotEncoder avoids this by creating binary columns with 1s and 0s).
* `StandardScaler` for numeric features (To avoid the model from being biased towards bigger values, as some categories like salary would obviously be greater than age, StandardScaler makes sure they are all treated equally).
* scikit-learn preprocessing pipeline (Manually processing data each time can be prone to reproducibility issues manual inconsistency or preprocessing errors. A pipeline maintains consistent results with computer precision).

This model served as the primary baseline classifier due to:

* interpretability
* stability
* strong generalization behavior

## Decision Tree

Several Decision Tree configurations were tested:

* default one-hot encoded tree
* constrained tree with reduced depth
* ordinal-encoded tree

The Decision Tree experiments explored:

* non-linear split behavior (Instead of drawing one line through data like a linear model, a decision tree breaks down a problem into hundreds or even thousands of split decisions of branches stacked on top of each other).
* preprocessing tradeoffs (Decision Trees do not require StandardScalar because the trees find the best split based on an order or rank, instead of the magnitude).
* overfitting behavior (The model has memorized the outcomes of the data rather than learning a pattern for recognizing the target)
* model complexity control (The max-depth of a tree was limited to five to control overfitting)

---

# Model Results

| Model                   | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ----------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression     |    0.591 |      0.59 |   0.77 |     0.67 |   0.618 |
| Decision Tree (Ordinal) |    0.579 |      0.57 |   0.84 |     0.68 |   0.606 |

### Overfitting Observation

The unconstrained Decision Tree achieved:

* approximately **97.4% training accuracy**
* approximately **56.1% testing accuracy**

This large train-test gap demonstrated severe overfitting behavior.

---

# Key Findings

* Logistic Regression provided the strongest overall balance of interpretability, stability, and generalization performance.
* The Decision Tree achieved stronger positive-class recall but produced substantially more false positives.
* Model performance remained modest across all evaluated approaches.
* The classification task was structurally difficult because ERR values clustered densely around the 1.0 threshold.
* Conservative feature selection reduced leakage but limited predictive signal.

The project demonstrates that realistic healthcare analytics workflows often produce modest performance when leakage-heavy variables are intentionally excluded.

---

# Limitations

Several limitations should be considered when interpreting this project.

## Aggregated Public Reporting Data

The dataset contains hospital-measure-level reporting records rather than patient-level clinical data.

## Limited Feature Space

Only a small set of operational/contextual variables was used:

* state
* measure name
* discharge volume

## Leakage-Aware Constraints

Several potentially predictive variables were intentionally excluded because they were directly tied to target construction.

## Structurally Difficult Classification Boundary

ERR values were tightly concentrated near 1.0, creating substantial overlap between target classes.

## Structural Missingness

CMS suppression rules created structurally missing values for some hospitals and measures.

## No Causal Interpretation

The models identify statistical classification patterns within public reporting data and should not be interpreted as causal explanations of hospital quality.

---

# How to Run This Project

1. Clone the repository

```bash
git clone <https://github.com/gurbaj8226/Hospital-Excess-Readmission-Classification>
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Download the CMS HRRP dataset from:

[https://data.cms.gov/provider-data/dataset/9n3s-kdb3#data-table](https://data.cms.gov/provider-data/dataset/9n3s-kdb3#data-table)

4. Place the dataset inside:

```text
data/raw/
```

5. Run notebooks in order:

```text
01_data_cleaning_and_structural_validation.ipynb
02_target_definition_and_eda.ipynb
03_classification_modeling.ipynb
```

---

# Future Improvements

Potential future improvements include:

* integrating additional CMS hospital characteristics datasets
* adding socioeconomic or regional healthcare variables
* testing Random Forest or Gradient Boosting models
* performing systematic hyperparameter tuning (Using algorithms like grid or randomized search instead of manually guessing to determine complexity controls like max_depth).
* using cross-validation (Instead of slicing the data one time into an 80/20 split, Cross-Validation slices the training data and trains the model multiple times and averages the performance).
* exploring probability threshold tuning for recall/precision tradeoffs

---

# Author

Created by Gurbaj Singh

Healthcare Data Analytics Portfolio Project
