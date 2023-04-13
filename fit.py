from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from RandomForest import RandomForest

df = pd.read_csv(r'C:/Users/abi3c/Desktop/creditcard.csv')
df.drop_duplicates(inplace=True)

data = df.drop('Class' ,axis = 1)
target = df['Class']
X = data.to_numpy()
y = target.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

clf = RandomForest(n_trees=20)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc =  accuracy(y_test, predictions)
print(acc)
