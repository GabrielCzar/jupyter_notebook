#-*- coding=utf-8 -*-
import linear_regression as lr

class SimpleLinearRegression:
    
    def __init__(self):
        '''Linear model'''
        self.b0 = 0
        self.b1 = 0
        
    def fit(self, x, y):
        '''Treina o model baseado no gradiente descendente'''
        self.b0, self.b1 = lr.gradient_descendent(x, y, learning_rate=0.00000001, num_iterations=100000)
    
    def predict(self, x):
        '''Cria a predição dos valores de x'''
        return self.b0 + self.b1 * x
        
    