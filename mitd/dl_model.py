# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:59:52 2021

@author: GJK
"""

class DeepLearning_Model:
    def __init__(self):
        self.__nx = None
        self.__nh = None
        self.__ny = None
        
    """ 
        TODO
        @param n_x size of the input layer
        @param n_h size of the hidden layer
        @param n_y size of the output layer
        
        @return parameters python dictionary containing your parameters:
            W1 -- weight matrix of shape (n_h, n_x)
            b1 -- bias vector of shape (n_h, 1)
            W2 -- weight matrix of shape (n_y, n_h)
            b2 -- bias vector of shape (n_h, 1)
            ...
            Wn -- weight matrix of shape (n_y, n_h)
            bn -- bias vector of shape (n_y, 1)
        
        @note
            __nx, __nh, __ny shall be changed appropriately.    
    """
    def parametrize(n_x, n_h, n_y):
        parameters = {}
        return parameters    
    
    
    """ 
        TODO
        @param layer_dims python array containing the dimensions of each layer in our network
        
        @return parameters  python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
            Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
            bl -- bias vector of shape (layer_dims[l], 1)
    """
    def initialize_parameters_deep(layer_dims):
        parameters = {}
        return parameters