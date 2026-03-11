# Credit Card Fraud Detection Project
# Author: Nitesh Kumar Gangwar
# AI & Machine Learning Project

# Step 1 – Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2 – Load Dataset
df = pd.read_csv("creditcard.csv")

# Step 3 – Basic Information
print("Dataset Shape:", df.shape)
print(df.head())

# Step 4 – Check Missing Values
print("Missing Values:")
print(df.isnull().sum())

# Step 5 – Check Fraud vs Normal Transactions
print("Transaction Count:")
print(df['Class'].value_counts())

# Step 6 – Data Visualization
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Normal Transactions")
plt.show()

# Step 7 – Feature Selection
X = df.drop("Class", axis=1)
y = df["Class"]

# Step 8 – Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 9 – Model Training (Random Forest)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Step 10 – Prediction
y_pred = model.predict(X_test)

# Step 11 – Accuracy
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 12 – Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Step 13 – Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 14 – Confusion Matrix Visualization
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Fraud Detection Model Completed")
