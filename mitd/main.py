# -*- coding: utf-8 -*-
"""
Title: Medical Image Tamper Detection
Authors: Gary Kim,
Date: 10/13/21
Description:
    
"""

""" TODO
    @param filePath path of the file which contains medical dcm images
    
    @return imageMatrix np array of size (n, 1)
    
    @note
        In our dataset, we have a collection of 2D images per instance.
        Each image will have the same dimensions, x pixels by y pixels.
        Say we have n of these images.
        The dimension of each image in imageMatrix should be (x * y * z, 1).
        
        Every instance has a set of 2D images and the number of these images
        per instance ranges from [~100, ~300] according to researchers' README.
        You will need to decide how many of these images we will include in our
        dataset and how we will pick them out.
        e.g. If we know every instance has at least 80 images, we can just pick
        first 80 for every instance to be included in our dataset.
        Then, each instance in our data will be an np array of dimension 
        (x * y * 80, 1).
"""
def convertDcmToMatrix(filePath):
    return imageMatrix

""" TODO
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
"""
def initialize_parameters(n_x, n_h, n_y):
    parameters = {
    
    }
    
    return parameters    


""" TODO
    @param layer_dims python array containing the dimensions of each layer in our network
    
    @return parameters  python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
        bl -- bias vector of shape (layer_dims[l], 1)
"""
def initialize_parameters_deep(layer_dims):
    return parameters


def main():


if __name__ == "__main__":
    main()