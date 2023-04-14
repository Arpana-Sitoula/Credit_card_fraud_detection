from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , confusion_matrix
import numpy as np
import pandas as pd
from RandomForest import RandomForest
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(r'C:/Users/abi3c/Desktop/creditcard.csv')
df.drop_duplicates(inplace=True)

data = df.drop('Class' ,axis = 1)
target = df['Class']
X = data.to_numpy()
y = target.to_numpy()

#For balancing unbalanced data set

from imblearn.under_sampling import RandomUnderSampler
rus=RandomUnderSampler(sampling_strategy=1)
x_res,y_res=rus.fit_resample(X,y)

scalar = StandardScaler()
scalar.fit(x_res)
standardized_data = scalar.transform(x_res)
x_res= standardized_data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

clf = RandomForest(n_trees=20)
clf.fit(X_train, y_train)


for i, tree in enumerate(clf.trees):
    print(f"Tree {i+1}:")
    print(tree)

predictions = clf.predict(X_test)
print("Predicted values:", predictions)
print("Actual values:", y_test)

