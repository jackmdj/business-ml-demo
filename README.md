# Puffco Customer Behavior Prediction Project

## Introduction

Puffco has solidified itself as the top maker of cannabis consumption devices. However, it has been noted that they struggle with customer retention, with many customers buying once or infrequently. This project aims to identify the best time to send a promotional email to Puffco customers and determine which product to promote next. By combining random forest regression (time prediction) and classification (product recommendation) models, this project showcases a full machine learning pipeline from data creation to inference.

## Project Overview

1. **Data Generation**: Create and simulate synthetic customer behavior data, including purchase frequency, average/lowest times between orders, and product ownership.
2. **Data Cleaning & Augmentation**: Preprocess the dataset, ensuring missing values are handled and categorical targets are encoded.
3. **Model Selection**: Train baseline and advanced models for timing (regression) and product prediction (classification).
4. **Model Tuning**: Use Grid Search to fine-tune hyperparameters of the models for optimal performance.
5. **Testing & Inference**: Validate models on test data and demonstrate inference on new customers.

## Project Structure

- **`synthetic_data.csv`**: A dataset generated for this project.
- **`train_models.ipynb`**: Jupyter notebook detailing data exploration, cleaning, training, and tuning of the models.
- **`sample_predictions.ipynb`**: Jupyter notebook that demonstrates how to load the trained models and make predictions on new customer data.
- **`rf_model.pkl`**, **`product_model.pkl`**, **`label_encoder.pkl`**: Saved trained models and encoders.
- **`README.md`**: This file, providing an overview and instructions.

## Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab
- Packages: `pandas`, `numpy`, `scikit-learn`, `joblib`

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/puffco-promotions-ml.git
cd puffco-ml-demo
```

You can install dependencies via:

```bash
pip install -r requirements.txt
```

## Data Generation and Preparation

### Data Creation

In this project, data is synthetically generated using an LLM to simulate Puffco customers’ buying patterns. This includes Customer_ID, Name, Order_Frequency, Average_Time_Between_Orders, Lowest_Time_Between_Orders, has_peak, has_knife, has_pivot.

I used Excel to briefly clean the data before loading it into Python for further analysis.

Missing values are handled, non-numeric values (`"N/A"`) are replaced with `NaN`, and columns are coerced into numeric types. The dataset is augmented with additional columns (e.g., `Next_Product`) based on product ownership rules.  

## Model Development

### Model Selection and Training

- **Regression (Timing Model)**: A `RandomForestRegressor` is trained to predict the best time (in days) to send out a promotional email.
- **Classification (Product Model)**: A `RandomForestClassifier` is trained to recommend the next product to advertise, using features that indicate which products the customer already owns.

In training the regression model, the target variable is `Lowest_Time_Between_Orders` because we want to predict the minimum amount of time it will take for a customer to buy again. The features are `Order_Frequency` and `Average_Time_Between_Orders`.

For the classification model, the target variable is `Next_Product`, a created column with one of three values: `pivot`, `knife`, and `peak`. The features are the binary variables `has_peak`, `has_knife`, and `has_pivot`.

The entire model training workflow, including baseline model comparisons and initial fits, is documented in `train_models.ipynb`.

### Hyperparameter Tuning

The regression model undergoes hyperparameter tuning using Grid Search (`GridSearchCV`). By testing different configurations of `n_estimators`, `max_depth`, and `min_samples_split`, we find the best performing model.

## Model Testing and Inference

After training and tuning:

- The best performing models are saved as `.pkl` files (`rf_model.pkl` for timing, `product_model.pkl` for product recommendation, and `label_encoder.pkl` for encoding).
- In `sample_predictions.ipynb`, we demonstrate how to load these models, provide a new customer’s data, and generate a prediction. The predictor returns both the recommended time and product, formatted into a readable sentence.

## Conclusion

This project demonstrates a complete workflow for identifying the optimal timing for promotional emails and recommending the next product to market, using entirely synthetic data. The resulting models show encouraging performance under these controlled conditions. However, the limitations of the dataset mean that these findings are primarily illustrative. If integrated into a real-world setting with authentic customer purchase histories and refined with domain knowledge, these models could become more accurate, reliable, and ultimately more valuable for business decision-making.

