#-*- coding=utf-8 -*-
import numpy as np
from sklearn import metrics

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

def accuracy(y_true,y_pred):
    num = np.where(y_true == y_pred)
    return num[0].shape[0]/y_pred.shape[0]

def precision(y_true, y_pred):
    precision = []
    matriz = metrics.confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    for i in range(matriz.shape[1]):
        precision.append(matriz[i,i]/(sum(matriz[:,i])))

    return precision

def f1_measure(y, y_pred):
    mat =  metrics.confusion_matrix(y, y_pred, labels=np.unique(y))
    return 2*((precision(y, y_pred)*recall(y, y_pred))/(precision(y, y_pred)+recall(y, y_pred)))

def recall(y_true, y_pred):
    precision = []
    matriz = metrics.confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    print (matriz)
    for i in range(matriz.shape[1]):
        precision.append(matriz[i,i]/(sum(matriz[i,:])))
        
    return precision
