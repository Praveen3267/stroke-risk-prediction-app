# Stroke Risk Prediction App

A machine learning web application for exploring stroke risk prediction from patient health information using a trained Random Forest classifier and an interactive Streamlit interface.

## Overview

This project demonstrates an end-to-end machine learning workflow for a binary stroke prediction task.

The project includes:

- Data preprocessing and missing-value handling
- Categorical feature encoding
- Train/test splitting
- Class-imbalance handling with SMOTE during model experimentation
- Comparison of multiple classification algorithms
- Random Forest model training
- Model evaluation using accuracy, recall, precision, and F1 score
- Interactive prediction through Streamlit

> **Note:** The notebook contains multiple experimental training and evaluation workflows. The application uses the saved Random Forest model included in the repository.

## Application Features

- Enter patient health information through an interactive web interface
- Predict stroke probability using the trained model
- Display a risk category using a configurable probability threshold
- Reuse saved model and label-encoder artifacts
- Run locally with Streamlit

## Machine Learning Workflow

### 1. Data Preparation

The project uses the `healthcare-dataset-stroke-data.csv` dataset.

The notebook performs:

- Removal of the dataset `id` column
- Missing-value handling
- Categorical feature encoding
- Train/test splitting
- Exploratory data analysis

### 2. Class Imbalance

Stroke datasets can contain substantially fewer positive stroke cases than non-stroke cases.

The notebook experiments with **SMOTE (Synthetic Minority Over-sampling Technique)** on training data to address class imbalance.

### 3. Model Experiments

The notebook experiments with several classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- XGBoost
- Gradient Boosting

### 4. Evaluation

The notebook evaluates models using:

- Accuracy
- Recall
- Precision
- F1 Score
- Confusion Matrix

The notebook also explores different probability thresholds for classification.

## Application Model

The Streamlit application loads:

- `stroke_model_rf.pkl` — trained Random Forest model
- `label_encoders.pkl` — categorical feature encoders

The application calculates the predicted probability of stroke and categorizes the result using a probability threshold.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- imbalanced-learn
- XGBoost
- Streamlit
- Jupyter Notebook
- Matplotlib
- Seaborn

## Project Structure

```text
stroke-risk-prediction-app/
├── healthcare-dataset-stroke-data.csv
├── heart.ipynb
├── label_encoders.pkl
├── requirements.txt
├── streamlit_app.py
└── stroke_model_rf.pkl