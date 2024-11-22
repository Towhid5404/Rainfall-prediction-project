# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('C:/Users/Arshad/Desktop/project/towhid.csv')

# Assuming the dataset has columns like 'Location', 'MinTemp', ..., 'RainToday', 'RainTomorrow'
# Replace these with the actual column names in your dataset
features = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
            'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'RainToday']
X = data[features]

# Map 'No' to 0 and 'Yes' to 1 in the target variable
y = data['RainTomorrow'].map({'No': 0, 'Yes': 1})

# Check for NaN values in the target variable
print(f"NaN values in target variable (y): {y.isnull().sum()}")

# Separate numerical and categorical columns
numeric_features = X.select_dtypes(include=['float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create a ColumnTransformer for one-hot encoding and imputation
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Apply one-hot encoding and imputation to features
X = preprocessor.fit_transform(X)

# Print summary statistics of the imputed data
print(pd.DataFrame(X).describe())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.72, random_state=28)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

try:
    rf_classifier.fit(X_train, y_train)
    # Make predictions with the Random Forest model
    rf_predictions = rf_classifier.predict(X_test)
    # Evaluate the accuracy of the Random Forest model
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

    # Train an XGBoost classifier
    xgb_classifier = XGBClassifier(random_state=42)
    xgb_classifier.fit(X_train, y_train)
    # Make predictions with the XGBoost model
    xgb_predictions = xgb_classifier.predict(X_test)
    # Evaluate the accuracy of the XGBoost model
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

except ValueError as e:
    print(f"Error: {e}")
