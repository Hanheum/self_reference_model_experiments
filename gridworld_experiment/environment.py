import numpy as np
from ml_utils import distance
import pygame

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

        self.count = 1
        self.reached_target = False

        self.window = None
        self.window_size = 100 * self.size

    def reset(self):
        self.player = np.random.randint(0, self.size, size=[2, ])
        self.target = np.random.randint(0, self.size, size=[2, ])
        
        while np.sum((self.player - self.target)**2) == 0:
            self.target = np.random.randint(0, self.size, size=[2, ])

        self.terminated = False
        self.distance = distance(self.player, self.target)
        self.count = 1
        self.reached_target = False

        return self.player, self.target

    def step(self, action):
        #0~3, 0:up 1:down 2:right 3:left
        #before_moving = self.player.copy()
        self.player += self.moving_dictionary[action]
        self.player[0] = min([self.size-1, max([0, self.player[0]])])
        self.player[1] = min([self.size-1, max([0, self.player[1]])])

        if np.sum((self.player - self.target)**2) == 0:
            self.terminated = True
            reward = 1
            self.reached_target = True
        else:
            new_distance = distance(self.player, self.target)
            if new_distance >= self.distance:
                self.distance = new_distance
                reward = -0.1
            else:
                self.distance = new_distance
                reward = 0.1

        self.count += 1

        if self.count >= 200:
            self.terminated = True

        return self.player, self.target, reward, self.terminated

    def make_coord_for_show(self, original_coord):
        return ((original_coord.astype(np.float32)+0.5)*100).astype(int)

    def init_show(self):
        pygame.init()

        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption('Gridworld')
        self.window.fill((255, 255, 255))
        pygame.display.flip()

        for i in range(self.size+1):
            pygame.draw.line(self.window, (0, 0, 0), [i*100, 0], [i*100, self.window_size], 5)
            pygame.draw.line(self.window, (0, 0, 0), [0, i*100], [self.window_size, i*100], 5)

        pygame.draw.circle(self.window, (255, 0, 0), self.make_coord_for_show(self.player), 50)
        pygame.draw.circle(self.window, (0, 0, 255), self.make_coord_for_show(self.target), 50)

        pygame.display.update()

    def show(self):
        self.window.fill((255, 255, 255))
        pygame.display.flip()

        for i in range(self.size+1):
            pygame.draw.line(self.window, (0, 0, 0), [i*100, 0], [i*100, self.window_size], 5)
            pygame.draw.line(self.window, (0, 0, 0), [0, i*100], [self.window_size, i*100], 5)
        
        pygame.draw.circle(self.window, (255, 0, 0), self.make_coord_for_show(self.player), 50)
        pygame.draw.circle(self.window, (0, 0, 255), self.make_coord_for_show(self.target), 50)
        
        pygame.display.update()