import numpy as np


class logistic_regression():
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape

        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.no_of_iterations):
            self.update_weight()

    def update_weight(self):
        Y_hate = 1/(1+np.exp(-(self.X.dot(self.w)+self.b)))

        dw = (1/self.m)*np.dot(self.X.T, (Y_hate-self.Y))
        db = (1/self.m)*np.sum(Y_hate-self.Y)

        self.w = self.w-self.learning_rate*dw
        self.b = self.b-self.learning_rate*db

    def predict(self, X):
        Y_pred = 1/(1+np.exp(-(X.dot(self.w)+self.b)))
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
        return Y_pred
    
        
