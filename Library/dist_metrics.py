#-*- encoding=utf-8 -*-
import numpy as np 

def minkowski_distance(X, row, p=2):
    return np.sum(abs(X - row) ** p) ** (1/p)

def euclidean_distance(X, row): 
    # Dica: usar minkowski_distance com p=2.
    return minkowski_distance(X, row, 2)

def manhattan_distance(X, row): 
    # Dica: usar minkowski_distance com p=1.
    return minkowski_distance(X, row, 1)

def chebyshev_distance(X, row):
    return np.max(abs(X - row))
