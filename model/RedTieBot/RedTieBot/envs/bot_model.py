import numpy as np
import gym.spaces.box as b
#we need numpy and Gym present here.

class BotModel(gym.env):
    def __init__(self):
        self.w=1
        #width of robot
        self.t=0.5
        #round to the nearest .5 seconds when dealing with time
        self.x0=0
        #robot's starting x-position
        self.y0=0
        #robot's starting y-position
        self.x=self.x0
        #robot's current x-position.
        self.y=self.y0
        #robot's current y-position.
        self.facing=0
        #the direction that the robot is facing in radians, in standard position
        self.l_speed=0
        #left wheel speed
        self.r_speed=0
        #right wheel speed
        self.is_over=False
        #the game is not over yet.

        self.observation_space = b.Box(0, 1.0, shape=(int(821/10), int(1598/10),36))
        #The dimensions of the field. The 360 degree vision is split into 36 parts, 10 degrees each.
        #The box is technically a 1x1x1 cube.
        self.action_space = b.Box(0, 1.0, shape=(int(-128/2), int(127/2)))
        #The range of speeds that the wheel can have.

    def step(self, action):
        self.l_speed += action[0]
        #in the list "action", the first value is the left wheel speed.
        self.r_speed += action[1]
        #in the list "action", the second value is the right wheel speed.
        
        if self.l_speed > 127:
            self.l_speed = 127
        elif self.l_speed < -128:
            self.l_speed = -128
        if self.r_speed > 127:
            self.r_speed = 127
        elif self.r_speed < -128:
            self.r_speed = -128

        if self.l_speed == self.r_speed:
          distance = l_speed * t
          self.x = self.x + (distance * np.sin(facing))
          self.y = self.y + (distance * np.cos(facing))

        else:
            #above lines limit the speed of the wheels to 128 cm/s backwards or 127 cm/s forward.
            radius = (self.w/2)*(self.l_speed+self.r_speed)/(self.l_speed-self.r_speed)
            #this is the physical radius of the robot.
            z = (self.l_speed-self.r_speed)*self.t/self.w
            self.x = self.x+(radius*np.sin(facing))-(radius*np.sin(facing-z))
            self.y = self.y-(radius*np.cos(facing))+(radius*np.cos(facing-z))
            self.facing -= z
            #see desmos link on slack for explanation of above three lines. It's essentially direction calculations
        
            while z<0:
                z+=2*np.pi
                
            while z>2*np.pi:
                z-=2*np.pi

            while self.facing<0:
                self.facing+=2*np.pi

            while self.facing>2*np.pi:
                self.facing-=2*np.pi
            #making sure that the z-angle measurement doesn't go below 0 or above 2pi
                
        ob = dict(x=self.x, y=self.y, facing=self.facing, l_speed=self.l_speed, r_speed=self.r_speed)
        #when it's training, it takes the data from the environment and says "I have this to use now."
        reward = self.checkreward()
        #returns the amount of reward achieved.
        episode_over = self.is_over()
        #checks to see if it's over.
        info = dict(3)
        #openai needs that line to be happy. means nothing
        return ob, reward, episode_over, info
        #spit back all that data.
        
    def reset(self):
        self.x=self.x0
        self.y=self.y0
        #reset my position. Where I am now is now (0,0) from my perspective.
        self.facing=np.pi/2
        #I am facing forward.
        self.l_speed=0
        #stop the left wheel
        self.r_speed=0
        #stop the right wheel
        return dict(x=self.x, y=self.y, facing=self.facing, l_speed=self.l_speed, r_speed=self.r_speed)
        #spit back all the data about what I'm doing right now.

    def checkreward(self):
        if self.l_speed == 0.0 and self.r_speed == 0.0 and self.x < 821 and self.x > 621 and self.y < 1398 and self.y > 1348:
        #If I'm in position in front of the goal and facing the right way,
            if np.round(self.facing,1) == np.round(np.tan((1598-self.y)/(638-self.x)),3):
            #If I'm in position in front of the goal and facing the right way (but with extra parameters)
                self.is_over = True
                #end the game!
                return 1000.0
                #i get a lot of points
        return 0.0
        #if im not in the right position, i get no points.; :(

    def render(self, mode='human'):
        #graphics; nothing yet
        return

    def close(self):
        #closing the graphics; nothing yet
        return
