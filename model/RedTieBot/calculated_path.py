from __future__ import print_function, division
from builtins import range
import gym
import numpy as np
import RedTieBot

class Path_Calculator:
    def __init__(self, env):
        self.env = env
        self.our_tan = self.calctan()
        self.sm = {'not facing': self.rotate,
                   'facing': self.move2,
                   'reached': self.rotate}
        self.state = 'not facing'
        self.angle = None

    def calctan(self):
        tano = []
        for x in range(24):
            tano.append(np.tan((.5+x)*np.pi/12))
        return tano

    def reset(self):
        self.target = None
        self.state = 'not facing'

    def get_target(self,x,y):
        if self.target is None:
            self.target = self.env.get_a_target(x,y)
        return self.target

    def calculated_path(self,observation):
        return self.sm[self.state](observation)  
        
    def rotate(self,observation):
        cx, cy, cfacing, l_speed, r_speed = observation.values()
        x, y, facing = self.get_target(cx,cy)
        action = [1,-1]
        if self.angle is None:
            if self.state == 'not facing':
                if cx != x:
                    if cx - x < 0:
                        self.angle = int(np.arctan((cy-y)/(cx-x))*12/np.pi)
                    else:
                        self.angle = int((np.arctan(((cy-y)/(cx-x))))*12/np.pi)
                        self.angle += 12
                    if self.angle < 0:
                        self.angle += 24
                else:
                    self.angle = 6
            else:
                self.angle = facing
            #print(self.angle)
        if np.round(l_speed,2) == 0 and np.round(r_speed,2) > 0:
            action = ([0,-1])
        elif np.round(r_speed,2) == 0 and np.round(l_speed,2) > 0:
            action = ([-1,0])
        elif self.angle != cfacing:
            if l_speed * r_speed < 0:
                action = ([0,0])
            else:
                action = ([-1,1])
        else:
            if self.state == 'not facing':
                self.state = 'facing'
                self.g_angle = self.angle
            self.angle = None    
        return action

    def move2(self,observation):
        cx, cy, cfacing, l_speed, r_speed = observation.values()
        x, y, facing = self.get_target(cx,cy)
        action = ([0,0])
        goal = ((y-cy)/(self.our_tan[cfacing])+cx)
        if (x, y) == (cx, cy):
            action = ([-1,-1])
            self.state = 'reached'
        elif abs(l_speed * r_speed) <= 0.001:
            action = ([1,1])
        elif np.round(l_speed,2) > np.round(r_speed,2):
            action = ([-1,0])
        elif np.round(l_speed,2) < np.round(r_speed,2):
            action = ([0,-1])
        elif goal > x:
            action = ([0,1])
        elif goal < x:
            action = ([1,0])
        return action
