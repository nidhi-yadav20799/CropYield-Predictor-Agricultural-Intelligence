# 🌾 Crop Yield Predictor - Agricultural Intelligence


## Overview

Crop Yield Predictor is a machine learning project that predicts agricultural crop yield using historical weather, rainfall, pesticide usage, and temperature data. The project follows a complete end-to-end machine learning workflow, including data preprocessing, exploratory data analysis, feature engineering, model development, model evaluation, and dashboard deployment.

---

## Project Objectives

- Analyze agricultural production data.
- Perform data cleaning and validation.
- Conduct exploratory data analysis (EDA).
- Engineer meaningful domain-specific features.
- Build reproducible machine learning pipelines.
- Compare multiple regression models.
- Optimize model performance using hyperparameter tuning.
- Deploy an interactive prediction dashboard using Streamlit.

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-Learn, XGBoost |
| Dashboard | Streamlit |
| Experiment Tracking | MLflow |
| Model Serialization | Joblib |

---

## Dataset

**Source:** Kaggle – Crop Yield Prediction Dataset

### Target Variable

- hg/ha_yield

### Features

- Area
- Item (Crop)
- Year
- Average Rainfall (mm/year)
- Pesticides (tonnes)
- Average Temperature

---

## Project Structure

```text
CropYield-Predictor/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── images/
│
├── models/
│
├── notebooks/
│   └── 01_Data_Preprocessing_EDA.ipynb
│
├── reports/
│
├── src/
│
├── README.md
└── .gitignore
```

---

## Workflow

### Data Preparation

- Dataset acquisition
- Data loading
- Data validation
- Data dictionary generation
- Missing value analysis

### Data Cleaning

- Duplicate removal
- Data type validation
- Outlier analysis using the IQR method
- Clean dataset generation

### Exploratory Data Analysis

The project includes comprehensive exploratory data analysis with statistical interpretation.

Visualizations include:

- Histograms
- KDE Plots
- Boxplots
- Correlation Heatmap
- Feature Correlation Analysis
- Crop Yield by Region
- Rainfall vs Crop Yield
- Temperature vs Crop Yield
- Pair Plot

### Feature Engineering

Engineered features include:

- Rainfall Category
- Growing Degree Days (GDD)
- Temperature Category
- Pesticide Category
- Rainfall–Temperature Ratio
- Yield per Unit of Pesticide

### Machine Learning Pipeline

Implemented using Scikit-Learn Pipeline and ColumnTransformer.

Current baseline models:

- Linear Regression
- Ridge Regression
- Random Forest Regressor

---

## Progress

### Completed

- Data acquisition and validation
- Data cleaning and preprocessing
- Exploratory Data Analysis (Part 1)
- Exploratory Data Analysis (Part 2)
- Feature engineering
- Scikit-Learn preprocessing pipeline
- Train-test split
- Baseline regression models
  - Linear Regression
  - Ridge Regression
  - Random Forest
- Advanced regression models
  - Gradient Boosting
  - XGBoost
- 5-fold cross-validation
- Hyperparameter tuning using RandomizedSearchCV

### Upcoming

- Model evaluation
- Feature importance analysis
- Residual analysis
- Streamlit dashboard
- Final report
---

## Results

The project currently includes:

- Cleaned dataset
- Feature-engineered dataset
- Reproducible preprocessing pipeline
- Baseline machine learning models

Model evaluation, optimization, and deployment will be completed in the remaining development stages.

---

## Author

Nidhi Yadav

