from tkinter import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve
from genetic_algorithm_xgb import paramvalue

#Dictionary for holding value of accuracy,precision,recall,f1-score of each algorithm
performance_metrics = {}

#Function For frame,frame1,frame2
def uifunction(x,y,display,name_of_curve,title):
    # Create labels for displaying the output
    Title_Algo= Label(display, text=title,font=("Times New Roman",16,"bold"))
    accuracy_label = Label(display, text="Accuracy:", font=("Arial", 12))
    precision_label = Label(display, text="Precision:", font=("Arial", 12))
    recall_label = Label(display, text="Recall:", font=("Arial", 12))
    f1_score_label = Label(display, text="F1 score:", font=("Arial", 12))
    Title_Algo.pack(pady=5)
    accuracy_label.pack(pady=2)
    precision_label.pack(pady=2)
    recall_label.pack(pady=2)
    f1_score_label.pack(pady=2)

#for confusion matrix
    screen=Canvas(display,width=200, height=200)
    screen.pack(pady=2)

    box_width=100
    box_height=100


# Define a function to update the output labels with the performance metrics
# def update_output():
    # Make predictions on the test set using the logistic regression model
    
    tn, fp, fn, tp=confusion_matrix(x,y).ravel()
    # Calculate the performance metrics
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = (2*precision*recall)/(precision+recall)


    # Update the output labels with the performance metrics
    accuracy_label.config(text=f"Accuracy: {accuracy*100:.2f}%")
    precision_label.config(text=f"Precision: {precision*100:.2f}%")
    recall_label.config(text=f"Recall: {recall*100:.2f}%")
    f1_score_label.config(text=f"F1 score: {f1*100:.2f}%")

     # Add the performance metrics to the dictionary
    performance_metrics[title] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

    

    #drawing matrix cells
    matrix_value=[[tn,fp],[fn,tp]]
    for i in range(len(matrix_value)):
        for j in range(len(matrix_value[0])):
                a0=j*box_width
                b0=i*box_height
                a1=a0+box_width
                b1=b0+box_height
                screen.create_rectangle(a0,b0,a1,b1,fill="white")
                screen.create_text((a0+a1)/2,(b0+b1)/2,text=matrix_value[i][j])

               

    screen.create_text(box_width/2,box_height/2)
    screen.create_text(3*box_width/2,box_height/2)
    screen.create_text(box_width/2,3*box_height/2)
    screen.create_text(3*box_width/2,3*box_height/2)
    
    



    def curve():
        logistic_fpr, logistic_tpr, threshold=roc_curve(x,y)
        auc_logistic=auc(logistic_fpr,logistic_tpr)


        
        plt.figure(name_of_curve,figsize=(4,4),dpi=100)
        plt.plot(logistic_fpr,logistic_tpr,marker=".",label="logistic (auc= %0.3f)" % auc_logistic)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("false positive rate -->")
        plt.ylabel("true positive rate -->")

        plt.legend()
        plt.show()
#create button to show roc_auc curve
    update_button = Button(display, text="ROC-AUC curve", command=curve)
    update_button.pack(pady=2)


#this function is used for displaying best Parameter from genetic algorithm
def genetic_algorithm_UserInterface(display,title):
    Title_Algo= Label(display, text=title,font=("Times New Roman",16,"bold"))
    best_fitness_f1_score = Label(display, text="Best_Fitness(F1-Score):", font=("Arial", 12))
    Best_parameter=Label(display,text="Best_parameter_are: ",font=("Arial", 12))
    learning_rate = Label(display, text="Learning_rate:", font=("Arial", 12))
    n_estimators = Label(display, text="N_estimators:", font=("Arial", 12))
    max_depth = Label(display, text="Max_depth:", font=("Arial", 12))
    min_child_weight = Label(display, text="Min_child_weight:", font=("Arial", 12))
    gamma = Label(display, text="Gamma:", font=("Arial", 12))
    subsample = Label(display, text="Subsample:", font=("Arial", 12))
    colsample_bytree = Label(display, text="Colsample_bytree:", font=("Arial", 12))
    Title_Algo.pack(pady=5)
    best_fitness_f1_score.pack(pady=2)
    Best_parameter.pack(pady=2)
    learning_rate.pack(pady=2)
    n_estimators.pack(pady=2)
    max_depth.pack(pady=2)
    min_child_weight.pack(pady=2)
    gamma.pack(pady=2)
    subsample.pack(pady=2)
    colsample_bytree.pack(pady=2)

   
    value_of_parameter= np.array(list(paramvalue.values()))

    best_fitness_f1_score.config(text=f"Best_Fitness(F1-Score): {value_of_parameter[0]*100:.2f}%")
    learning_rate.config(text=f"Learning_rate: {value_of_parameter[1]:.2f}")
    n_estimators.config(text=f"N_estimators: {value_of_parameter[2]:.2f}")
    max_depth.config(text=f"Max_depth: {int(value_of_parameter[3]):.2f}")
    min_child_weight.config(text=f"Min_child_weight: {value_of_parameter[4]:.2f}")
    gamma.config(text=f"Gamma: {value_of_parameter[5]:.2f}")
    subsample.config(text=f"Subsample: {value_of_parameter[6]:.2f}")
    colsample_bytree.config(text=f"Colsample_bytreee: {value_of_parameter[7]:.2f}")

