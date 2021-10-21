# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 18:00:23 2021

@author: GJK
"""

from sklearn import svm
from sklearn import metrics

class RandomForest_Model:
    def __init__(self):
        self.__model        = None
        self.__groundTruths = []
        self.__predictions  = []
    
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