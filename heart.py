import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix , f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv('Heart_Disease_Dataset.csv')

# Separate features and target variable
x = df.iloc[:, :-2]
y = df.iloc[:, -1]

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.35)

# Standardize the features using StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Create a SVM model with grid search for hyperparameter tuning
svm_model = SVC()

# Define the parameter grid for grid search
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(svm_model, param_grid, scoring='accuracy', cv=5)
grid_search.fit(x_train, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the SVM model with the best hyperparameters
best_svm_model = SVC(**best_params)
best_svm_model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = best_svm_model.predict(x_test)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate accuracy in percentage
accuracy_percentage = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", f"{accuracy_percentage:.2f}%")

# You can also print other evaluation metrics like F1-score if needed
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)
