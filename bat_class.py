import random
from math import pi, atan, atan2
from constants import *
import numpy as np

class Bat:

    def __init__(self):
        self.has_goal = False

        self.goal_X = 0
        self.goal_Y = 0
        self.goal_speed_X = 0
        self.goal_speed_Y = 0
   
        self.X = random.uniform(MIN_X, MAX_X)
        self.Y = random.uniform(MIN_Y, MAX_Y)
        self.hd = random.uniform(0, 360)
        self.walked = False

    def set_randomized_goal(self):
        self.has_goal = True
        self.goal_X = random.uniform(MIN_X, MAX_X)
        self.goal_Y = random.uniform(MIN_Y, MAX_Y)
        self.goal_speed_X = random.uniform(MIN_FV, MAX_FV)
        self.goal_speed_Y = random.uniform(MIN_FV, MAX_FV)

        # TODO: head direction might change too sharply!
        #  set the new goal to be in a reasonable degree to the
        #  head's direction

        dx = self.goal_X - self.X
        dy = self.goal_Y - self.Y
        self.hd = round((atan2(dy, dx) % (2*pi)) * 180 / pi)
            
        if dx < 0:
            self.goal_speed_X *= -1
        if dy < 0:
            self.goal_speed_Y *= -1
                    

    def fly_toward_goal(self):
        if self.has_goal:
            #print("Flying!")
            if self.goal_X == self.X:
                self.goal_speed_X = 0

            if self.goal_Y == self.Y:
                self.goal_speed_Y = 0

            if self.goal_Y == self.Y and self.goal_X == self.X:
                self.has_goal = False
           
            if self.goal_speed_X > 0:
                self.X = min(self.X + self.goal_speed_X, self.goal_X)

            if self.goal_speed_X < 0:
                self.X = max(self.X + self.goal_speed_X, self.goal_X)

            if self.goal_speed_Y > 0:
                self.Y = min(self.Y + self.goal_speed_Y, self.goal_Y)

            if self.goal_speed_Y < 0:
                self.Y = max(self.Y + self.goal_speed_Y, self.goal_Y)

    def change_head_direction(self):
        self.hd += random.randint(-1, 1)
        self.hd %= 360

    def change_hd_within_range(self):
        self.hd += np.tanh(np.random.normal(0, 0.5, 1)[0]) * HD_CHANGE_RANGE
        self.hd %= 360
        
    def walk_randomly(self):
        self.change_hd_within_range()
        walking_speed = abs(np.random.normal(2, 1, 1)[0])
        
        self.X += np.cos(self.hd / 180 * np.pi) * walking_speed
        self.Y += np.sin(self.hd / 180 * np.pi) * walking_speed

        
    def behave(self):
        if random.random() < TARGET_P:
            self.set_randomized_goal()

        if random.random() < HD_P:
            self.change_head_direction()

        if not self.has_goal:
            if not self.walked and random.random() < START_W_P:
                self.change_hd_within_range()
                self.walked = True
            if self.walked and random.random() < KEEP_W_P:
                self.walk_randomly()
                self.walked = True
            else:
                self.walked = False

        self.fly_toward_goal()
        
        # don't cross boundaries
        if self.X <= MIN_X or self.X >= MAX_X or self.Y <= MIN_Y or self.Y >= MAX_Y:
            self.X = min(self.X, MAX_X)
            self.X = max(self.X, MIN_X)

            self.Y = min(self.Y, MAX_Y)
            self.Y = max(self.Y, MIN_Y)
            
            self.change_hd_within_range()
            # self.walked=True