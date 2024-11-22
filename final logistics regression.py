# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load your dataset
df = pd.read_csv('C:/Users/Arshad/Desktop/project/towhid.csv')

# Exclude date columns from the features
X = df.drop(['RainTomorrow', 'Date'], axis=1)
y = df['RainTomorrow']

# Convert string values in the target variable to numeric
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessor to handle categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Location', 'RainToday'])
    ],
    remainder='passthrough'
)

# Create a Logistic Regression classifier pipeline
logistic_regression_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Fit the Logistic Regression model to the training data
logistic_regression_model.fit(X_train, y_train)

# Make predictions with Logistic Regression on the test data
y_pred_logistic = logistic_regression_model.predict(X_test)

# Calculate and print the accuracy of Logistic Regression
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print(f'Logistic Regression Accuracy: {accuracy_logistic:.2%}')

# Use the same preprocessor for XGBoost
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# Create an XGBoost classifier
xgboost_model = XGBClassifier()

# Fit the XGBoost model to the training data
xgboost_model.fit(X_train_encoded, y_train)

# Make predictions with XGBoost on the test data
y_pred_xgboost = xgboost_model.predict(X_test_encoded)

# Calculate and print the accuracy of XGBoost
accuracy_xgboost = accuracy_score(y_test, y_pred_xgboost)
print(f'XGBoost Accuracy: {accuracy_xgboost:.2%}')
