import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load the dataset
df = pd.read_csv(r'C:/Users/abi3c/Desktop/creditcard.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('Class', axis=1), df['Class'], test_size=0.2, random_state=42)

# Train a Gradient Boosting Machine (GBM) model
gbm = GradientBoostingClassifier()
gbm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gbm.predict(X_test)

# Evaluate the performance of the model
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
print('ROC AUC Score:', roc_auc_score(y_test, y_pred))

## for balanced datasets
non_fraud = df[df['Class']==0]
fraud = df[df['Class']==1]

legit = non_fraud.sample(n=508)

ndf = pd.concat([legit,fraud], axis = 0)

X = ndf.drop(columns='Class', axis=1)
Y = ndf['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train a Gradient Boosting Machine (GBM) model
gbm = GradientBoostingClassifier()
gbm.fit(X_train, Y_train)

# Make predictions on the test set
y_pred = gbm.predict(X_test)

# Evaluate the performance of the model
print('Confusion Matrix:\n', confusion_matrix(Y_test, y_pred))
print('Classification Report:\n', classification_report(Y_test, y_pred))
print('ROC AUC Score:', roc_auc_score(Y_test, y_pred))
