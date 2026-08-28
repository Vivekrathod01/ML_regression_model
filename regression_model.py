#importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load dataset
data = pd.read_csv(r"C:\Users\vivek\Downloads\USA_Housing.csv")

# Preprocessing
X = data.drop(['Price', 'Address'], axis=1) 
y = data['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Define models
models = {
    'LinearRegression': Pipeline([("scaler",StandardScaler()),("model",LinearRegression())]),
    'RidgeRegression': Pipeline([("scaler",StandardScaler()),("model",Ridge())]),
    'LassoRegression': Pipeline([("scaler",StandardScaler()),("model",Lasso())]),
    'ElasticNet': Pipeline([("scaler",StandardScaler()),("model",ElasticNet())]),
    'PolynomialRegression': Pipeline([
        ('poly', PolynomialFeatures(degree=4)),
        ('linear', LinearRegression())
    ]),
    'RandomForest': Pipeline([("scaler",StandardScaler()),("model",RandomForestRegressor())]),
    'SVM': Pipeline([("scaler",StandardScaler()),("model",SVR())]),
    'LGBM': Pipeline([("scaler",StandardScaler()),("model",lgb.LGBMRegressor())]),
    'XGBoost':  Pipeline([("scaler",StandardScaler()),("model",xgb.XGBRegressor())]),
    'KNN': Pipeline([("scaler",StandardScaler()),("model",KNeighborsRegressor())]),
}

# Train and evaluate models
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    })
    
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('model_evaluation_results.csv', index=False)

print("Models have been trained and saved as pickle files. Evaluation results have been saved to model_evaluation_results.csv.")
