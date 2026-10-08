
# ETAI 2026/27 | Customer Repurchase & Spend Prediction

## Project Overview

This project was developed as part of the **ETAI 2026/27 Group Project**.


The project focuses on **Everything & Then Some, Ltd.**, a diversified wholesale company whose customers have different purchasing patterns. The objective is to use historical transaction data to support more targeted commercial actions.

The project aims to answer two questions for each customer purchase:

1. **Will the customer make another purchase within the next 90 days?**
2. **If the customer returns, how much will they spend during those 90 days?**

The answers to these two questions can support different commercial actions. Customers who are unlikely to return can be targeted with incentives, while customers expected to spend more can be prioritized for retention and account management. The final solution should therefore be capable of making both predictions for a new purchase.

---

## Project Members

20231693 David Carrilho
20260620 Diogo Padilha
20211536 Luís Soeiro
20231630 Maria Inês Santos


## Project Objectives

The project is divided into several main objectives:

* Explore and clean the historical transaction data.
* Develop and optimize a **classification pipeline** to predict whether a customer will purchase again within 90 days.
* Develop and optimize a **regression pipeline** to predict the amount the customer will spend during the following 90 days.
* Generate predictions for the Kaggle competition.
* Build terminal-based pipelines for training and prediction.
* Provide predictions for individual new purchases through a client interface.
* Apply explainability methods to understand the predictions of both models.

---

## Project Structure

The current repository is organized as follows:

```text
ETAI_Project/
│
├── .gitignore
├── config.yaml
├── main.py
├── README.md
├── requirements.txt
│
├── classification/
│   ├── evaluate.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── results.py
│   └── tuning.py
│
├── data/
│   ├── processed/
│   └── raw/
│
├── notebooks/
│
├── regression/
│   ├── evaluate.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── results.py
│   └── tuning.py
│
└── results/
```

### Main files

**`main.py`**
The main entry point of the project. It will be used to run the project's pipelines from the terminal.

**`config.yaml`**
Configuration file for project settings and parameters. The project specification requires configurable choices to be kept outside the code whenever possible.

**`requirements.txt`**
Contains the Python dependencies required to run the project.

**`README.md`**
Project documentation, including installation, usage instructions, project structure and other relevant information.

### Data

**`data/raw/`**
Contains the original project datasets.

The project provides:

* `train.csv`
* `test.csv`
* `clientes.csv`
* `sample_submission.csv`

The training data contains the classification and regression targets, while the test data contains the predictors without the targets. `clientes.csv` is an optional customer-level table that can be joined using `customer_id`.

**`data/processed/`**
Reserved for processed or transformed datasets generated during the project.

### Classification

The `classification/` directory contains the components related to predicting whether a customer will make another purchase within 90 days.

* `preprocessing.py` — preprocessing steps
* `model.py` — classification models
* `tuning.py` — model and hyperparameter tuning
* `evaluate.py` — model evaluation
* `results.py` — results reporting

### Regression

The `regression/` directory contains the components related to predicting the customer's spend during the following 90 days.

* `preprocessing.py` — preprocessing steps
* `model.py` — regression models
* `tuning.py` — model and hyperparameter tuning
* `evaluate.py` — model evaluation
* `results.py` — results reporting

### Notebooks

**`notebooks/`**
Reserved for exploratory analysis and experimentation during the development of the project.

### Results

**`results/`**
Reserved for experiment results, evaluation outputs, figures and other generated project results.

---

## Data

Each row in the dataset represents a **customer purchase occasion**.

The available variables include information about:

* Current transaction values
* Purchase details
* Categorical transaction information
* Historical purchase timing
* Historical customer value
* Rolling purchase frequency
* Rolling customer spend
* Additional numerical variables

The project also requires that all features used for prediction are available at the time of the purchase, avoiding information from the future or from other test observations.

---

## Development Status

This repository currently contains the **initial project structure**.


