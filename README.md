# 🏠 USA Housing Price Prediction

## 📌 Project Overview

This project is a **Machine Learning Regression project** that predicts house prices in the USA using the **USA Housing dataset**.

The project follows an end-to-end machine learning workflow, including data exploration, preprocessing, feature scaling, training multiple regression algorithms, model evaluation, comparison, and saving trained models for future predictions.

## 📊 Dataset

The dataset contains **5,000 housing records** and includes the following features:

- **Avg. Area Income** – Average income of the area
- **Avg. Area House Age** – Average age of houses in the area
- **Avg. Area Number of Rooms** – Average number of rooms
- **Avg. Area Number of Bedrooms** – Average number of bedrooms
- **Area Population** – Population of the area
- **Price** – Target variable representing the house price
- **Address** – Address information, dropped during preprocessing because it is not required for the numerical prediction model

### Target Variable

`Price`

The goal is to predict the price of a house based on the available numerical housing and demographic features.

## 🔄 Machine Learning Workflow

1. Load the USA Housing dataset
2. Explore and understand the data
3. Remove the `Address` column
4. Separate features (`X`) and target (`y`)
5. Split the data into training and testing sets
6. Apply feature standardization where required
7. Train multiple regression models
8. Evaluate model performance
9. Compare the models using MAE, MSE, and R²
10. Select the best-performing model
11. Save trained models using Pickle (`.pkl`)

## 🤖 Models Used

The following regression algorithms were trained and compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Polynomial Regression
- Random Forest Regressor
- Support Vector Regression (SVR)
- LightGBM Regressor
- XGBoost Regressor
- K-Nearest Neighbors Regressor

## 📈 Model Evaluation

The models were evaluated using:

- **MAE (Mean Absolute Error)** – Measures the average absolute difference between actual and predicted prices. Lower is better.
- **MSE (Mean Squared Error)** – Penalizes larger prediction errors. Lower is better.
- **R² Score** – Measures how well the model explains variation in house prices. Higher is better.

### 🏆 Best Performing Model

Based on the current evaluation results, **Ridge Regression** performed best overall, achieving an **R² score of approximately 0.915**.

The model evaluation results are stored in:

`model_evaluation_results.csv`

## ⚙️ Preprocessing & Pipeline

Feature standardization is performed using **StandardScaler** for models that benefit from scaling.

For Polynomial Regression, a Scikit-learn **Pipeline** is used so that polynomial feature generation and scaling are applied consistently to new/unseen data before prediction.

This makes the saved model more suitable for future deployment.

## 💾 Model Serialization

The trained models are saved using **Pickle (`.pkl`)**.

A saved pipeline/model can be loaded later and used to make predictions on new housing data without retraining the model.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- Pickle

## 📁 Project Files

```text
USA-Housing-Price-Prediction/
│
├── regression_model.py
├── model_evaluation_results.csv
├── best_model.pkl
├── requirements.txt
└── README.md
```

## ▶️ How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python regression_model.py
```

## 🚀 Future Improvements

- Build a **FastAPI** prediction API
- Create a web interface for house price prediction
- Store prediction history using **MySQL**
- Dockerize the application
- Deploy the ML model/API to the cloud

## 🎯 Project Objective

The objective of this project is to build a reliable regression model for **USA house price prediction** and create a foundation for deploying the trained model as a real-world machine learning application.
