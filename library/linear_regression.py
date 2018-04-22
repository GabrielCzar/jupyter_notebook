# y = b0 + b1 * x
# b1 é a inclinação, b0 é y-intercept
import numpy as np

def compute_error_for_line_given_points(b0, b1, x, y):
    totalError = np.sum((y - (b1 * x + b0)) ** 2)
    return totalError / float(len(y))

def step_gradient(b0_current, b1_current, x, y, learning_rate):
    N = float(len(y))
    b0_gradient = 2/N * np.sum(-(y - ((b1_current * x) + b0_current)))
    b1_gradient = 2/N * np.sum(-x * (y - ((b1_current * x) + b0_current)))
    new_b0 = b0_current - (learning_rate * b0_gradient)
    new_b1 = b1_current - (learning_rate * b1_gradient)
    return new_b0, new_b1

def gradient_descent_runner(x, y, b0, b1, learning_rate, num_iterations):
    for _ in range(num_iterations):
        b0, b1 = step_gradient(b0, b1, x, y, learning_rate)
    return b0, b1

def run(x, y, initial_b0, initial_b1, learning_rate, num_iterations):
    print("Inicia com b0 = {}, b1 = {}, error = {}".format(initial_b0, initial_b1, compute_error_for_line_given_points(initial_b0, initial_b1, x, y)))  
    print("Executando...")
    b0, b1 = gradient_descent_runner(x, y, initial_b0, initial_b1, learning_rate, num_iterations)
    print("Após {} iterações, obtemos b0 = {}, b1 = {}, error = {}".format(num_iterations, b0, b1, compute_error_for_line_given_points(b0, b1, x, y)))
    
def gradient_descendent(x, y, init_b0=0, init_b1=0, learning_rate=0.0001, num_iterations=10000):
    b0, b1 = gradient_descent_runner(x, y, init_b0, init_b1, learning_rate, num_iterations)
    print ('learning_rate={}, number_iterations={}'.format(learning_rate, num_iterations))
    print ('Após {} iterações, obtemos b0 = {}, b1 = {}, error = {}'.format(num_iterations, b0, b1, compute_error_for_line_given_points(b0, b1, x, y)))
    return b0, b1