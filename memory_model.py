import numpy as np
from ml_utils import *

class memory_model:
    def __init__(self, input_size=4, output_size=4, memory_size=12):
        self.memory_size = 12
        self.input_size = input_size
        self.output_size = output_size
        self.memory_size = memory_size

        self.W0 = generate_random_matrix((32, self.input_size+self.memory_size))
        self.b0 = generate_random_matrix((32, 1))
        self.W1 = generate_random_matrix((32, 32))
        self.b1 = generate_random_matrix((32, 1))
        self.W2 = generate_random_matrix((32, 32))
        self.b2 = generate_random_matrix((32, 1))
        self.W3 = generate_random_matrix((self.memory_size, 32))
        self.b3 = generate_random_matrix((self.memory_size, 1))
        self.W4 = generate_random_matrix((self.output_size, self.memory_size))
        self.b4 = generate_random_matrix((self.output_size, 1))

        self.x0, self.g0, self.x1, self.g1, self.x2, self.g2, self.x3, self.g3, self.x4, self.g4 = None, None, None, None, None, None, None, None, None, None
        self.learning_rate = 1e-5

    def forward(self, x):
        #x be like: [sample]
        self.x0 = to_column(x)
        self.g0 = np.matmul(self.W0, self.x0) + self.b0
        self.x1 = sigmoid(self.g0)
        self.g1 = np.matmul(self.W1, self.x1) + self.b1
        self.x2 = sigmoid(self.g1)
        self.g2 = np.matmul(self.W2, self.x2) + self.b2
        self.x3 = sigmoid(self.g2)
        self.g3 = np.matmul(self.W3, self.x3) + self.b3
        self.x4 = sigmoid(self.g3)
        self.g4 = np.matmul(self.W4, self.x4) + self.b4
        return to_single_vector(self.g4)

    def forward_train(self, x):
        #x be like: [[sample1], [sample2], ...]
        self.x0 = x.T
        self.g0 = np.matmul(self.W0, self.x0) + self.b0
        self.x1 = sigmoid(self.g0)
        self.g1 = np.matmul(self.W1, self.x1) + self.b1
        self.x2 = sigmoid(self.g1)
        self.g2 = np.matmul(self.W2, self.x2) + self.b2
        self.x3 = sigmoid(self.g2)
        self.g3 = np.matmul(self.W3, self.x3) + self.b3
        self.x4 = sigmoid(self.g3)
        self.g4 = np.matmul(self.W4, self.x4) + self.b4

        self.x0 = self.x0.T
        self.g0 = self.g0.T
        self.x1 = self.x1.T
        self.g1 = self.g1.T
        self.x2 = self.x2.T
        self.g2 = self.g2.T
        self.x3 = self.x3.T
        self.g3 = self.g3.T
        self.x4 = self.x4.T
        self.g4 = self.g4.T
        return self.g4

    def backward(self, x, y):
        #x, y be like: [[sample1], [sample2], ...]
        N = len(x)
        y_pred = self.forward_train(x)
        dLdx5 = MSE_derivation(y_pred, y)

        dLdb4 = to_column(dLdx5)
        dLdW4 = np.matmul(dLdb4, np.reshape(self.x4, [N, 1, self.x4.shape[1]]))
        dLdx4 = np.matmul(self.W4.T, dLdb4)

        dLdb3 = dLdx4 * sigmoid_prime(to_column(self.g3))
        dLdW3 = np.matmul(dLdb3, np.reshape(self.x3, [N, 1, self.x3.shape[1]]))
        dLdx3 = np.matmul(self.W3.T, dLdb3)

        dLdb2 = dLdx3 * sigmoid_prime(to_column(self.g2))
        dLdW2 = np.matmul(dLdb2, np.reshape(self.x2, [N, 1, self.x2.shape[1]]))
        dLdx2 = np.matmul(self.W2.T, dLdb2)

        dLdb1 = dLdx2 * sigmoid_prime(to_column(self.g1))
        dLdW1 = np.matmul(dLdb1, np.reshape(self.x1, [N, 1, self.x1.shape[1]]))
        dLdx1 = np.matmul(self.W1.T, dLdb1)

        dLdb0 = dLdx1 * sigmoid_prime(to_column(self.g0))
        dLdW0 = np.matmul(dLdb0, np.reshape(self.x0, [N, 1, self.x0.shape[1]]))

        self.b0 -= self.learning_rate * np.sum(dLdb0, axis=0)
        self.b1 -= self.learning_rate * np.sum(dLdb1, axis=0)
        self.b2 -= self.learning_rate * np.sum(dLdb2, axis=0)
        self.b3 -= self.learning_rate * np.sum(dLdb3, axis=0)
        self.b4 -= self.learning_rate * np.sum(dLdb4, axis=0)

        self.W0 -= self.learning_rate * np.sum(dLdW0, axis=0)
        self.W1 -= self.learning_rate * np.sum(dLdW1, axis=0)
        self.W2 -= self.learning_rate * np.sum(dLdW2, axis=0)
        self.W3 -= self.learning_rate * np.sum(dLdW3, axis=0)
        self.W4 -= self.learning_rate * np.sum(dLdW4, axis=0)

        return MSE(y_pred, y)