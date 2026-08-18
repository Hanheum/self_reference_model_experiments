import numpy as np

def generate_random_matrix(shape):
    return np.random.randn(*shape).astype(np.float32)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return (1+np.exp(-x))**(-2)*np.exp(-x)

def to_column(x):
    return np.reshape(x, [*x.shape, 1])

def to_row(x):
    return np.reshape(x, [1, len(x)])

def to_single_vector(x):
    return np.reshape(x, [len(x), ])

def distance(A, B):
    return np.sum((A-B)**2)**0.5

def square(A, B):
    return np.sum((A-B)**2)

def MSE(y_pred, y):
    n = len(y_pred)
    loss = (1/n)*np.sum((y_pred - y)**2)
    return loss

def MSE_derivation(y_pred, y):
    n = len(y_pred)
    return (2/n)*(y_pred - y)

def one_hot(arr, size):
    N = len(arr)
    one_hot_version = np.zeros([N, size]).astype(np.float32)
    for i, number in enumerate(arr):
        one_hot_version[i][number] = 1

    return one_hot_version