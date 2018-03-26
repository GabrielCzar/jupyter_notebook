#-*- coding=utf-8 -*-
import numpy as np

def mean(x):
    '''Calcula a média dos valores de x'''
    return np.sum(x) / len(x)

def stdev(x):
    '''Calcula o desvio padrão'''
    return np.sqrt(np.var(x))

def var(y):
    '''Calcula a variancia'''
    media = mean(y)
    return np.sum((y - media) ** 2) / len(y)
