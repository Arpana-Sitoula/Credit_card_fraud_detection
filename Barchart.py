

#Function for creating Barchart
from tkinter import *
from matplotlib import pyplot as plt
import numpy as np
from UserInterface import performance_metrics

def barchart():
    #Create a list of the algorithms that you want to compare:
    algorithms = ['Logistic Regression','XGBClassifier']

#Create a list of the evaluation metrics that you want to plot:
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']

#Create a numpy array of the values for each metric for each algorithm. For example, you might have a 3x4 array where the rows represent the algorithms and the columns represent the metrics:
    value= np.array(list(performance_metrics.values()))

#Create the bar chart using the bar() function:
    x = np.arange(len(metrics))  # the x locations for the metrics
    width = 0.2  # the width of the bars

#Add more customizations to the chart, such as adjusting the size, color, and font:
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, list(value[0].values()), width, label=algorithms[0])
    rects2 = ax.bar(x,list(value[1].values()), width, label=algorithms[1])
    # rects3 = ax.bar(x + width, list(value[2].values()), width, label=algorithms[2])

# Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by algorithm and metric')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()


# Customize the chart
    fig.set_size_inches(10, 6)
    ax.set_facecolor('#f2f2f2')
    ax.grid(True, axis='y', color='w', linewidth=1.5, alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12, width=1)
    ax.spines['bottom'].set_color('#444444')
    ax.spines['top'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    ax.spines['right'].set_color('#444444')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.set_ylim([0, 1.2])
    ax.set_yticks(np.arange(0, 1.2, 0.1))
    ax.set_yticklabels(['{:,.1%}'.format(x) for x in np.arange(0, 1.2, 0.1)])


    plt.show()

