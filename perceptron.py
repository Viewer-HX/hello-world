'''
This file is designed for the homework 2 of AIT 736;
Author: Xu He
Date: 2019.Oct.15
Content: PLA algorithm.
'''

import numpy as np

class Perceptron(object):
    # Initialization function
    def __init__(self):
        self.learning_ratio = 0.00001 # learning rate, which define the step length of updating the weights
        self.max_iteration = 5000 # the max times of iteration, which limit the times of iteration
    
    # single prediction function
    def predict_(self, x):
        wx = np.dot(self.w, x) # calculate the result of decision function
        return np.sign(wx) # make a classfication with the sign function

    # make a classfication with the sign function
    def train(self, features, labels):
        self.w = np.zeros(len(features[0]) + 1) #weight + bias
        correct_count = 0 # count the number of correct classfication
        times = 0 # count the total number of iteration
        
        while times < self.max_iteration:
            index = np.random.randint(0, len(labels) - 1) #select a sample randomly
            x = features[index]
            x = np.append(x,1.0) # add the bias
            y = labels[index]
            wx = np.dot(self.w,x)

            if wx * y > 0: # Determine if the classification is correct
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue
           
            self.w = self.w + self.learning_ratio*y*x #update the weights 

            times += 1 # Cumulative number of iterations
    
    # the function predict is designed for classfication task
    def predict(self,features):
        labels = []
        row = features.shape[0]
        
        for i in range(row):
            x = features[i]
            x = np.append(x,1.0)
            labels.append(self.predict_(x))
            # Call a single prediction function and add the result to the labels
        return labels