import numpy as np
from ml_utils import distance

class gridworld:
    def __init__(self, size=5):
        self.size = size

        self.player = None
        self.target = None

        self.moving_dictionary = {0:np.array([-1, 0]),
                             1:np.array([1, 0]),
                             2:np.array([0, 1]),
                             3:np.array([0, -1])}

        self.distance = None
        self.terminated = False

    def reset(self):
        self.player = np.random.randint(0, self.size, size=[2, ])
        self.target = np.random.randint(0, self.size, size=[2, ])
        
        while np.sum((self.player - self.target)**2) == 0:
            print(self.target)
            self.target = np.random.randint(0, self.size, size=[2, ])

        self.terminated = False
        self.distance = distance(self.player, self.target)

        return self.player, self.target

    def step(self, action):
        #0~3, 0:up 1:down 2:right 3:left
        self.player += self.moving_dictionary[action]
        self.player[0] = min([self.size-1, max([0, self.player[0]])])
        self.player[1] = min([self.size-1, max([0, self.player[1]])])

        if np.sum((self.player - self.target)**2) == 0:
            self.terminated = True
            reward = 1
        else:
            new_distance = distance(self.player, self.target)
            if new_distance >= self.distance:
                self.distance = new_distance
                reward = -0.1
            else:
                self.distance = new_distance
                reward = 0.1

        return self.player, self.target, reward, self.terminated

    def show(self):
        board = np.zeros([self.size, self.size])
        board[int(self.player[0])][int(self.player[1])] = 1
        board[int(self.target[0])][int(self.target[1])] = 2

        board_txt = ''
        for i in range(self.size):
            for j in range(self.size):
                board_txt += f"{int(board[i][j])}"
            board_txt += '\n'

        print('='*10)
        print(board_txt)