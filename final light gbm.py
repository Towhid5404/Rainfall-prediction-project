# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load your rainfall forecasting dataset from the specified path
data = pd.read_csv("C:/Users/Arshad/Desktop/project/towhid.csv")

# Assuming you have a feature matrix 'X' and a target variable 'y'
features = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
            'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'RainToday']
X = data[features]
y = data['RainTomorrow'].map({'No': 0, 'Yes': 1})

# Identify categorical features
categorical_features = ['Location', 'RainToday']

# Create a transformer for categorical features
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline with preprocessing and XGBoost model
xgboost_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for XGBoost
param_grid = {
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__subsample': [0.8, 0.9, 1.0]
}

# Use GridSearchCV for hyperparameter tuning for XGBoost
grid_search = GridSearchCV(xgboost_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters for XGBoost
best_params_xgboost = grid_search.best_params_
print(f"Best Hyperparameters for XGBoost: {best_params_xgboost}")

# Fit the XGBoost model with the best hyperparameters
xgboost_model.set_params(**best_params_xgboost)
xgboost_model.fit(X_train, y_train)

# Make predictions with XGBoost on the test data
y_pred_xgboost_tuned = xgboost_model.predict(X_test)

# Fit the LightGBM model to the training data
lightgbm_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier())
])

# Fit the LightGBM model to the training data
lightgbm_model.fit(X_train, y_train)

# Make predictions with LightGBM on the test data
y_pred_lightgbm = lightgbm_model.predict(X_test)

accuracy_xgboost_tuned = accuracy_score(y_test, y_pred_xgboost_tuned)
print(f"Tuned XGBoost Accuracy: {accuracy_xgboost_tuned * 100:.2f}%")

# Evaluate the accuracy of LightGBM
accuracy_lightgbm = accuracy_score(y_test, y_pred_lightgbm)
print(f"LightGBM Accuracy: {accuracy_lightgbm * 100:.2f}%")
