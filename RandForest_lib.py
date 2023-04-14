# import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# load the dataset
df = pd.read_csv(r'C:/Users/abi3c/Desktop/creditcard.csv')

#balancing datasets
non_fraud = df[df['Class']==0]
fraud = df[df['Class']==1]
legit = non_fraud.sample(n=508)
ndf = pd.concat([legit,fraud], axis = 0)

X = ndf.drop(columns="Class", axis=1)
y = ndf["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # split the dataset into train and test sets

# instantiate the Random Forest model with hyperparameters
rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42)

# train the model
rf.fit(X_train, y_train)

# make predictions on the test set
y_pred = rf.predict(X_test)

# evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))
