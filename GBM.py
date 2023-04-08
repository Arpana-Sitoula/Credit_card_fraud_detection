import numpy as np


class GradientBoostingModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        
        self.base_model = DecisionTree(max_depth=max_depth)
        self.ensemble = []
        
    def fit(self, X, y):
        # Initialize the model with a constant prediction
        y_pred = np.full(y.shape, np.mean(y))
        
        for i in range(self.n_estimators):
            # Calculate the gradient of the loss function
            gradient = 2 * (y - y_pred)
            
            # Fit the base model to the gradient of the loss function
            self.base_model.fit(X, gradient)
            
            # Make predictions with the base model
            y_pred_tree = self.base_model.predict(X)
            
            # Update the ensemble predictions
            y_pred += self.learning_rate * y_pred_tree
            
            # Store the base model in the ensemble
            self.ensemble.append(self.base_model)
    
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        
        for model in self.ensemble:
            y_pred += self.learning_rate * model.predict(X)
        
        return y_pred
    
    def score(self, X, y):
        return 1 - prediction_error(y, self.predict(X))
    
    #To use gradient boosting model with our datasets we need to call the fit and predict methods.
    model = GradientBoostingModel(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
