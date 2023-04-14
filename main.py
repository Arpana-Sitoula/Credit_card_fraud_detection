import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import LogisticRegression


fraud_dataset = pd.read_csv(r'C:/Users/abi3c/Desktop/creditcard.csv')

X = fraud_dataset.drop(columns="Class", axis=1)
Y = fraud_dataset["Class"]

#For balancing unbalanced data set

from imblearn.under_sampling import RandomUnderSampler
rus=RandomUnderSampler(sampling_strategy=1)
x_res,y_res=rus.fit_resample(X,Y)

scalar = StandardScaler()
scalar.fit(x_res)
standardized_data = scalar.transform(x_res)
x_res= standardized_data

# y_res= fraud_dataset["Class"]
X_train, X_test, Y_train, Y_test = train_test_split(
    x_res, y_res, test_size=0.2, random_state=2)
classifier = LogisticRegression.logistic_regression(
    learning_rate=0.01, no_of_iterations=1000)
classifier.fit(X_train, Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print("accuracy of train data is:", training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print("accuracy of test data is:", test_data_accuracy)



# Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm=confusion_matrix(Y_test,X_test_prediction)
print("confusion_matrix : ")
print(cm)
print("accuracy_score : ",accuracy_score(Y_test,X_test_prediction))
print(classification_report(Y_test,X_test_prediction))
y_pred_logistic=classifier.predict(X_test)



from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

logistic_fpr, logistic_tpr, threshold=roc_curve(Y_test,y_pred_logistic)
auc_logistic=auc(logistic_fpr,logistic_tpr)



plt.figure(figsize=(5,5),dpi=100)
plt.plot(logistic_fpr,logistic_tpr,marker=".",label="logistic (auc= %0.3f)" % auc_logistic)
plt.xlabel("false positive rate -->")
plt.ylabel("true positive rate -->")

plt.legend()
plt.show()

