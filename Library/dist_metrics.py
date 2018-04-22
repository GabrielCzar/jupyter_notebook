#-*- encoding=utf-8 -*-
import numpy as np 

def minkowski_distance (X,row,p):
    sum = np.abs(X-row) ** p
    return np.sum(sum, axis = 1) ** (1 / p)

def euclidean_distance(X, row): 
    return minkowski_distance(X, row, 2)

def manhattan_distance(X, row): 
    return minkowski_distance(X, row, 1)

def chebyshev_distance(X, row):
    return np.max(np.abs(X - row))
