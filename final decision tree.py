# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load your rainfall forecasting dataset from the specified path
data = pd.read_csv("C:/Users/Arshad/Desktop/project/towhid.csv")

# One-hot encode the 'Location' and 'RainToday' features
data = pd.get_dummies(data, columns=['Location', 'RainToday'])

# Assuming you have a feature matrix 'X' and a target variable 'y'
features = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
            'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm']
X = data[features]
y = data['RainTomorrow'].map({'No': 0, 'Yes': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier
decision_tree_model = DecisionTreeClassifier()

# Fit the Decision Tree model to the training data
decision_tree_model.fit(X_train, y_train)

# Make predictions with the Decision Tree on the test data
y_pred_decision_tree = decision_tree_model.predict(X_test)

# Evaluate the accuracy of the Decision Tree
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
print(f"Decision Tree Accuracy: {accuracy_decision_tree * 100:.2f}%")

# Create an XGBoost classifier
xgboost_model = XGBClassifier()

# Fit the XGBoost model to the training data
xgboost_model.fit(X_train, y_train)

# Make predictions with XGBoost on the test data
y_pred_xgboost = xgboost_model.predict(X_test)

# Evaluate the accuracy of XGBoost
accuracy_xgboost = accuracy_score(y_test, y_pred_xgboost)
print(f"XGBoost Accuracy: {accuracy_xgboost * 100:.2f}%")
