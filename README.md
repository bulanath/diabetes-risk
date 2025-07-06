# Diabetes Risk Assessment Quiz

An end-to-end machine learning project that predicts a user's risk of having diabetes based on health indicators. The final product is an interactive web application built with Streamlit.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)

---

## Project Overview

This project was developed as a hands-on exercise to understand and implement the complete end-to-end machine learning lifecycle. It starts with a raw dataset, proceeds through data cleaning, model training, and evaluation, and concludes with the deployment of the trained model as an interactive web application.

The final application is a user-friendly quiz that asks for various health-related inputs and provides an instant diabetes risk assessment based on a predictive model.

## Dataset

The data for this project is the **"Diabetes Health Indicators"** dataset, which is a balanced version derived from the Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey.

- **Source:** [Kaggle Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data)
- **Target Variable:** `Diabetes_binary` (0 = No Diabetes, 1 = Prediabetes or Diabetes)
- **Features:** 21 health indicators such as High Blood Pressure, High Cholesterol, BMI, Smoking Status, and more.

## Workflow

The project followed a structured machine learning workflow:

#### 1. Data Preparation
- **Loading Data:** The dataset was loaded using Pandas.
- **Data Cleaning:** Duplicate rows were identified and removed to ensure the integrity of the dataset. A check for missing values was performed, and none were detected.

#### 2. Data Splitting
- The dataset was split into training and testing sets using an **80/20 ratio**. This ensures that the model is evaluated on data it has never seen before.
- A stratified split was used to maintain a consistent distribution of the target classes in both the training and testing sets.

#### 3. Model Training
- An **XGBoost Classifier** was selected for its high performance and efficiency with tabular data.
- The model was trained on the 80% training data. Hyperparameter tuning was performed using `RandomizedSearchCV` to find the optimal model configuration.

#### 4. Deployment
- **Model Serialization:** The best-performing trained model was saved to a file using `joblib`. This allows the model to be easily loaded into another application without retraining.
- **Web Application:** An interactive front-end quiz was built using **Streamlit**. This application loads the saved model, collects user input through a form, and provides a real-time risk prediction.

## Model Performance

The final XGBoost model achieved **75% accuracy** on the unseen test set.

## Technologies Used

- **Data Analysis:** Pandas
- **Machine Learning:** Scikit-learn, XGBoost
- **Web Framework:** Streamlit
- **Model Serialization:** Joblib