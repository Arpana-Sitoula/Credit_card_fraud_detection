from tkinter import *
import pandas as pd
from sklearn.model_selection import train_test_split
import LogisticRegression
from sklearn.preprocessing import StandardScaler
import UserInterface
import Barchart
from Barchart import barchart
# import genetic_algorithm_xgb
from xgboost import XGBClassifier
from genetic_algorithm_xgb import DataFrame



# Load the credit card fraud dataset
data =DataFrame
X = data.drop(columns="Class", axis=1)
Y = data["Class"]

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
    x_res, y_res, test_size=0.2, random_state=42)



#For LogisticRegression
classifier = LogisticRegression.logistic_regression(
    learning_rate=0.01, no_of_iterations=1000)
classifier.fit(X_train, Y_train)
X_test_prediction = classifier.predict(X_test)

#for Xgboost
xgboost=XGBClassifier(n_jobs = 4,random_state = 123)
xgboost.fit(X_train,Y_train)
xg_predict=xgboost.predict(X_test)


# Create a window for displaying the output
window = Tk()
window.title("Credit Card Fraud Detection")
window.geometry("750x450")

#first frame :1st algorithm
frame = Frame(window)
frame.pack(side=LEFT,padx=10,anchor="n")

#second frame :2nd algorithm
frame1=Frame(window)
frame1.pack(side=LEFT,padx=10,anchor="n")

#Third frame :3rd algorithm
frame2=Frame(window,padx=10)
frame2.pack(side=LEFT,anchor="n")

#Fourth frame :Barchart button
frame3=Frame(window)
frame3.pack(side=BOTTOM,pady=20)
frame3.place(relx=0.5, rely=1.0, anchor="s")


    

UserInterface.uifunction(Y_test,X_test_prediction,frame,"logistic regression1","Logistic Regression")
UserInterface.uifunction(Y_test,xg_predict,frame1,"XGBClassifier","XGBClassifier")
# UserInterface.uifunction(Y_test,predsXgboostwithGA,frame2,"logistic regression 3","Logistic Regression")
UserInterface.genetic_algorithm_UserInterface(frame2,"GA with XGBClassifier")


#button for barchart
update_button = Button(frame3 ,text="BarChart", command=Barchart.barchart)
update_button.pack(pady=2)

# Run the window
window.mainloop()
