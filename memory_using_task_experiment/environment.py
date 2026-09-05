import numpy as np
from ml_utils import distance
import pygame
from random import sample, randint

class memory_using_task:
    def __init__(self, size=5):
        self.size = size

        self.player = None
        self.target = None

        self.red_spot = np.array([0, 0])
        self.green_spot = np.array([(self.size//2), 0]).astype(int)
        self.blue_spot = np.array([self.size-1, 0])

        self.spots = [self.red_spot, self.green_spot, self.blue_spot]

        self.spot_colors = [(100, 100, 100), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

        self.right_spot = None

        self.target_type = None

        self.count = 1
        self.reached_target = False
        self.is_right_spot = False

        self.window = None
        self.window_size = 100 * self.size

        self.terminated = False
        self.distance = None

        self.moving_dictionary = {0:np.array([-1, 0]),
                                     1:np.array([1, 0]),
                                     2:np.array([0, 1]),
                                     3:np.array([0, -1])}

    def reset(self):
        options = list(range(1, self.size**2+1))
        options.remove(self.red_spot[0]+1)
        options.remove(self.green_spot[0]+1)
        options.remove(self.blue_spot[0]+1)

        picked_positions = sample(options, k=2)

        player_number, target_number = picked_positions

        self.player = np.array([player_number//self.size, player_number%self.size-1])
        self.target = np.array([target_number//self.size, target_number%self.size-1])

        self.terminated = False
        self.count = 1
        self.reached_target = False
        self.is_right_spot = False
        self.distance = distance(self.player, self.target)

        self.target_type = randint(1, 3)
        self.right_spot = self.spots[self.target_type-1]

        return self.player, self.target, self.target_type

    def step(self, action):
        #0~3, 0:up 1:down 2:right 3:left
        self.player += self.moving_dictionary[action]
        self.player[0] = min([self.size-1, max([0, self.player[0]])])
        self.player[1] = min([self.size-1, max([0, self.player[1]])])

        if self.reached_target:
            self.target = self.player

        if distance(self.player, self.target) == 0:
            if not self.reached_target:
                self.reached_target = True
                reward = 1
                returning_taget_type = 0
                self.distance = distance(self.player, self.right_spot)
            else:
                returning_taget_type = 0
                new_distance = distance(self.player, self.right_spot)

                if new_distance < self.distance:
                    reward = 0.1
                else:
                    reward = -0.2

                if new_distance == 0:
                    reward = 3
                    self.terminated = True
                    self.is_right_spot = True

                for i in range(3):
                    if i+1 == self.target_type:
                        continue

                    if distance(self.player, self.spots[i]) == 0:
                        reward = -10
                        self.terminated = True

                self.distance = new_distance

        else:
            new_distance = distance(self.player, self.target)
            if new_distance < self.distance:
                reward = 0.1
            else:
                reward = -0.2

            returning_taget_type = self.target_type
            self.distance = new_distance

        self.count += 1
        if self.count >= 200:
            self.terminated = True

        return self.player, self.target, reward, returning_taget_type, self.terminated
    
    def make_coord_for_show(self, original_coord):
        return ((original_coord.astype(np.float32)+0.5)*100).astype(int)

    def init_show(self):
        pygame.init()

        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption('memory using task')
        self.window.fill((255, 255, 255))
        pygame.display.flip()

        for i in range(self.size+1):
            pygame.draw.line(self.window, (0, 0, 0), [i*100, 0], [i*100, self.window_size], 5)
            pygame.draw.line(self.window, (0, 0, 0), [0, i*100], [self.window_size, i*100], 5)

        pygame.draw.circle(self.window, (255, 255, 0), self.make_coord_for_show(self.player), 50)
        pygame.draw.circle(self.window, self.spot_colors[self.target_type], self.make_coord_for_show(self.target), 50)

        for i in range(3):
            pygame.draw.rect(self.window, self.spot_colors[i+1], pygame.Rect(*(self.make_coord_for_show(self.spots[i])-50), 100, 100))

        pygame.display.update()

    def show(self):
        self.window.fill((255, 255, 255))
        pygame.display.flip()

        for i in range(self.size+1):
            pygame.draw.line(self.window, (0, 0, 0), [i*100, 0], [i*100, self.window_size], 5)
            pygame.draw.line(self.window, (0, 0, 0), [0, i*100], [self.window_size, i*100], 5)

        for i in range(3):
            pygame.draw.rect(self.window, self.spot_colors[i+1], pygame.Rect(*(self.make_coord_for_show(self.spots[i])-50), 100, 100))
                
        pygame.draw.circle(self.window, (255, 255, 0), self.make_coord_for_show(self.player), 50)
        pygame.draw.circle(self.window, self.spot_colors[int(1-int(self.reached_target))*self.target_type], self.make_coord_for_show(self.target), 50)
        
        pygame.display.update()
                

        