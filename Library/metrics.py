#-*- coding=utf-8 -*-
import numpy as np

def is_ndarray(array):
    return type(array) != np.ndarray
    
def mse(y, y_pred):
    '''Calcula a media de erro quadrado'''
    n = len(y); npred = len(y_pred)
    if n != npred:
        raise Exception('y e y_pred tem tamanhos diferentes')
    if is_ndarray(y) and is_ndarray(y_pred):
        raise Exception('y e y_pred devem ser do tipo numpy.ndarray')
    return np.sum((y - y_pred) ** 2) / n

def rmse(y_true, y_pred):
    '''Retorna a raiz quadrado do mse'''
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    '''Calcula a media de erro absoluto'''
    return np.sum(abs(y_true - y_pred)) / len(y_true)