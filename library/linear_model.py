#-*- coding=utf-8 -*-
import numpy as np


class SimpleLinearRegression:
            
    def __isNumpyArray(self, arr):
        return type(arr) == np.ndarray
    
    def fit(self, x, y):
        '''Treina o model baseado no gradiente descendente'''
        if self.__isNumpyArray(x):
            x = x[:, 0]
            
        b1 = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2) 
        b0 = np.mean(y) - b1 * np.mean(x)
        print ('b0={}, b1={}'.format(b0, b1))
        
        self.b0, self.b1 = b0, b1
    
    def predict(self, x):
        '''Cria a predição dos valores de x'''
        if self.__isNumpyArray(x):
            x = x[:, 0]
            
        print('b0={}, b1={}'.format(self.b0, self.b1))
        
        return self.b0 + self.b1 * x
        

class LogisticRegression:
	np.random.seed(3)
	num_pos = 5000
	epochs = 0
	learning_rate = 0
	beta = 0
    
	def __init__(self,epochs,learning_rate):
		self.epochs = epochs
		self.learning_rate = learning_rate

	def fit(self,X,y): 
		beta = np.zeros(X.shape[1]).reshape(X.shape[1], 1)
		for step in np.arange(self.epochs):
		    x_beta = np.dot(X, beta)
		    y_hat = 1 / (1 + np.exp(-x_beta))
		    likelihood = np.sum(np.log(1 - y_hat)) + np.dot(y.T, x_beta)
		    preds = np.round( y_hat )
		    accuracy = np.sum(preds == y)*1.00/len(preds)
		    gradient = np.dot(np.transpose(X), y - y_hat)
		    beta = beta + self.learning_rate*gradient
		    if( step % 5000 == 0):
		    	print("After step {}, likelihood: {}; accuracy: {}".format(step+1, likelihood, accuracy))
		self.beta = beta
		return beta
	
	def predict(self, X):
		b0 = self.beta[0]
		x1 = X[:, 0]
		x2 = X[:, 1]

		print(self.beta[1])

		rt = 1.0 + np.exp((b0+(self.beta[1]*x1)+(self.beta[2]*x2))*(-1))
		return 1.0/rt	