'''
This file is designed for the homework 2 of AIT 736;
Author: Xu He
Date: 2019.Oct.15
Content: Linear Regerssion algorithm.
'''

import numpy as np

class linearRegression(object):
    
    # single prediction function
    def predict_(self,x):
        y = np.dot(self.w,x) # calculate the result of decision function
        
        return np.sign(y) # make a classfication with the sign function
    
    # the function train is designed for train the model
    def train(self, features, labels):
        x0 = np.ones((len(labels),1)) 
        x = np.hstack((features,x0)) # add the bias
        
        pinv = np.linalg.pinv(x) # cpmpulte the pseudo inverse
        self.w = np.dot(pinv,labels) # calculate the weight + bias
        
        return self.w
    
    # the function predict is designed for classfication task
    def predict(self, features):
        labels = []
        row = features.shape[0]
        
        for i in range(row):
            x = features[i]
            x = np.append(x,1.0) # add the bias
            labels.append(self.predict_(x))
            # Call a single prediction function and add the result to the labels
            
        return labels
    
    # the function predict is designed for regression task
    def predict_regression(self, features):
        Results = []
        row = features.shape[0]
        
        for i in range(row):
            x = features[i]
            x = np.append(x,1.0)
            Results.append(np.dot(self.w,x))
            
        return Results
        
        
