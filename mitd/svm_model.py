# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:27:53 2021

@author: GJK
"""

from sklearn import svm
from sklearn import metrics

class SVM_Model:
    def __init__(self):
        self.__C            = None
        self.__kernel       = None
        self.__degree       = None
        self.__shape        = None
        self.__model        = None
        self.__groundTruths = []
        self.__predictions  = []
    
    """
        Sets the parameters of the SVM model to specified values.
        @param c regularization parameter.
        @param degree degree of the polynomial kernel function.
        @param shape whether to return "one-vs-rest" or "one-vs-one" decision function.
    """
    def parametrize(self, C, degree, shape):
        self.__C = C
        self.__degree = degree
        self.__shape = shape     
        return
        
    """
        TODO
        Trains the SVM model with the given training data.
        @param dbTrain list containing all training instances
    """
    def train(dbTrain):
        self.__model = svm.SVC(C = self.__C,\
                               kernel = self.__kernel,\
                               degree = self.__degree,\
                               decision_function_shape = self.__shape)
        return
    
    """
        TODO
        Uses the current SVM model to classify all test instances.
        @param dbTest list containing all test instances.
        
        @note
            __groundTruths shall be populated with the ground truths in dbTest.
            __predictions shall be populated with all the predictions made. 
    """
    def predict(dbTest):
        return
    
    """
        Calculates the accuracy of the model.
        @return accuracy the accuracy (type float) of the model based on the 
            predictions made against the ground truths.
    """
    def accuracy():
        accuracy = metrics.accuracy_score(self.__groundTruths, self.__predictions)
        return accuracy
        