# Puffco Customer Behavior Prediction Project

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository presents a complete workflow for predicting the optimal time to send promotional emails and recommending the next product to advertise to Puffco customers. The pipeline includes data synthesis, cleaning, augmentation, model selection, hyperparameter tuning, and inference.

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Data Generation and Preparation](#data-generation-and-preparation)
  - [Synthetic Data Creation](#synthetic-data-creation)
  - [Data Cleaning and Augmentation](#data-cleaning-and-augmentation)
- [Model Development](#model-development)
  - [Model Selection and Training](#model-selection-and-training)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Testing and Inference](#model-testing-and-inference)
- [Additional Improvements](#additional-improvements)
- [License](#license)

## Introduction

This project aims to identify the best time to send a promotional email to Puffco customers and determine which product to promote next. By combining regression (time prediction) and classification (product recommendation) models, this project showcases a full machine learning pipeline from data creation to inference.

## Project Overview

1. **Data Generation**: Create and simulate synthetic customer behavior data, including purchase frequency, average/lowest times between orders, and product ownership.
2. **Data Cleaning & Augmentation**: Preprocess and enrich the dataset, ensuring missing values are handled and categorical targets are encoded.
3. **Model Selection**: Train baseline and advanced models for timing (regression) and product prediction (classification).
4. **Model Tuning**: Use Grid Search to fine-tune hyperparameters of the models for optimal performance.
5. **Testing & Inference**: Validate models on test data and demonstrate inference on new customers.

## Project Structure

- **`synthetic_data.csv`**: A synthetic dataset generated for this project.
- **`train_models.ipynb`**: Jupyter notebook detailing data exploration, cleaning, training, and tuning of the models.
- **`sample_predictions.ipynb`**: Jupyter notebook that demonstrates how to load the trained models and make predictions on new customer data.
- **`rf_model.pkl`**, **`product_model.pkl`**, **`label_encoder.pkl`**: Saved trained models and encoders.
- **`README.md`**: This file, providing an overview and instructions.
  
*(If you have a `LICENSE` or `requirements.txt`, consider referencing them here.)*

## Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab
- Packages: `pandas`, `numpy`, `scikit-learn`, `torch`, `matplotlib`, `joblib`

You can install dependencies via:

```bash
pip install -r requirements.txt
```

*(If you don’t have a `requirements.txt`, consider adding one.)*

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/puffco-promotions-ml.git
cd puffco-promotions-ml
```

## Data Generation and Preparation

### Synthetic Data Creation

In this project, data is synthetically generated to simulate Puffco customers’ buying patterns. This includes one-time purchasers, repeat buyers, and customers owning multiple products (peak, knife, pivot).

You can find the initial dataset generation code and logic in the `train_models.ipynb` notebook, where we load `synthetic_data.csv` and begin data exploration and cleaning.

### Data Cleaning and Augmentation

Missing values are handled, non-numeric values (`"N/A"`) are replaced with `NaN`, and columns are coerced into numeric types. The dataset is augmented with additional columns (e.g., `Next_Product`) based on product ownership rules.  
All these steps ensure a robust and consistent dataset for model training.

## Model Development

### Model Selection and Training

- **Regression (Timing Model)**: A `RandomForestRegressor` is trained to predict the best time (in days) to send out a promotional email.
- **Classification (Product Model)**: A `RandomForestClassifier` is trained to recommend the next product to advertise, using features that indicate which products the customer already owns.

The entire model training workflow, including baseline model comparisons and initial fits, is documented in `train_models.ipynb`.

### Hyperparameter Tuning

The regression model (timing model) undergoes hyperparameter tuning using Grid Search (`GridSearchCV`). By testing different configurations of `n_estimators`, `max_depth`, and `min_samples_split`, we find the best performing model.

## Model Testing and Inference

After training and tuning:

- The best performing models are saved as `.pkl` files (`rf_model.pkl` for timing, `product_model.pkl` for product recommendation, and `label_encoder.pkl` for target encoding).
- In `sample_predictions.ipynb`, we demonstrate how to load these models, provide a new customer’s data, and generate a prediction. The predictor returns both the recommended time and product, formatted into a human-readable sentence.

## Additional Improvements

Consider adding:

- **Unit Tests**: To ensure code robustness.
- **Continuous Integration (CI)**: Automated tests and style checks using GitHub Actions.
- **More Complex Features**: Incorporate additional features such as total spend, customer demographics, or purchase channels.
- **Model Explainability Tools**: Use SHAP or LIME to explain model predictions.

These additions would further enhance the professionalism and maintainability of your project.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and adapt the code in your own projects.

---

If you have any questions or suggestions, feel free to open an issue or submit a pull request. Your contributions and feedback are welcome!