import numpy as np
import gym.spaces.box as b

class BotModel(gym.env):
    def __init__(self):
        self.w=1
        self.t=0.5
        self.x0=0
        self.y0=0
        self.x=self.x0
        self.y=self.y0
        self.facing=0
        self.l_speed=0
        self.r_speed=0
        self.is_over=False

        self.observation_space = b.Box(0, 1.0, shape=(int(821/10), int(1598/10),36))
        self.action_space = [-1,-.25,0,.25,1] #b.Box(0, 1.0, shape=(int(-128/2), int(127/2)))

    def step(self, action):
        self.l_speed += action[0]
        self.r_speed += action[1]
        if self.l_speed > 127:
            self.l_speed = 127
        elif self.l_speed < -128:
            self.l_speed = -128
        if self.r_speed > 127:
            self.r_speed = 127
        elif self.r_speed < -128:
            self.r_speed = -128
        
        radius = (self.w/2)*(self.l_speed+self.r_speed)/(self.l_speed-self.r_speed)
        z = (self.l_speed-self.r_speed)*self.t/self.w
        self.x = self.x+(radius*np.sin(facing))-(radius*np.sin(facing-z))
        self.y = self.y-(radius*np.cos(facing))+(radius*np.cos(facing-z))
        self.facing -= z
        
        while z<0:
            z+=2*np.pi
        while z>2*np.pi:
            z-=2*np.pi
        
        ob = dict(x=self.x, y=self.y, facing=self.facing, l_speed=self.l_speed, r_speed=self.r_speed)
        reward = self.checkreward()
        episode_over = self.is_over()
        info = dict(3)
        return ob
        
    def reset(self):
        self.x=self.x0
        self.y=self.y0
        self.facing=np.pi/2
        self.l_speed=0
        self.r_speed=0
        return dict(x=self.x, y=self.y, facing=self.facing, l_speed=self.l_speed, r_speed=self.r_speed)

    def checkreward(self):
        if self.l_speed == 0.0 and self.r_speed == 0.0 and self.x < 821 and self.x > 621 and self.y < 1398 and self.y > 1348:
            if np.round(self.facing,1) == np.round(np.tan((1598-self.y)/(638-self.x)),3):
                self.is_over = True
                return 1000.0
        return 0.0

    def render(self, mode='human'):
        return

    def close(self):
        return
