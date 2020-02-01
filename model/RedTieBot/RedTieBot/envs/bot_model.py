#coding: utf-8
import numpy as np
import gym.spaces.box as b
import gym
import turtle as tu
#we need numpy, gym and pygame present here. np and pg is merely shorthand for numpy and pygame, respectively.
a=tu.Turtle()
a.speed(0)
class ActionSpace:
    def __init__(self):
        self._spaces = np.array([(-1,-1), (-1,0), (-1,1), (0,-1),
                                (0,0), (0,1), (1,-1), (1,0), (1,1)])

    def sample(self):
        return self._spaces[np.random.choice(len(self._spaces))]

    def fromQ(self, val):
        print(val)
        return self._spaces[np.digitize(val, np.linspace(-1.0, 1.0, len(self._spaces))) - 1]
    
class BotModel(gym.Env):
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
        self.minShootDist = 5 #This is the MINIMUM Distance from away the target
        self.maxShootDist = 10 #This is the MAXIMUM Distance away from the target
        self.reward = 0#the points rewarded to the robot during the simulation


        self.observation_space = b.Box(0, 1.0, shape=(int(821/10), int(1598/10), 24))
        #The structure of the data that will be returned by the environment. It's the dimensions of the field (without obstacles at the moment)
        #The box is technically a 1x1x1 cube.
        self.action_space = ActionSpace()
        #The range of speeds that the wheel can have.

        self.path = []

    def step(self, action):
        try:
            self.l_speed += action[0]
            #in the list "action", the first value is the left wheel speed.
            self.r_speed += action[1]
            #in the list "action", the second value is the right wheel speed.
        except Exception as e:
            print(e)
            import pdb; pdb.set_trace()
        if self.l_speed > 127:
            self.l_speed = 127
        elif self.l_speed < -128:
            self.l_speed = -128
        if self.r_speed > 127:
            self.r_speed = 127
        elif self.r_speed < -128:
            self.r_speed = -128
        #above lines limit the speed of the wheels to 128 cm/s backwards or 127 cm/s forward
        self.checkreward()
        if not self.is_over:
            if self.l_speed == self.r_speed:
              distance = self.l_speed * self.t
              #calculate the distance traveled.
              self.x = self.x + (distance * np.sin(self.facing))
              self.y = self.y + (distance * np.cos(self.facing))
              #update my x and y positions, now that I know how far I've traveled.

            else:
                radius = (self.w/2)*(self.l_speed+self.r_speed)/(self.l_speed-self.r_speed)
                #this is the physical radius of the robot.
                z = (self.l_speed-self.r_speed)*self.t/self.w
                self.x = self.x+(radius*np.sin(self.facing))-(radius*np.sin(self.facing-z))
                self.y = self.y-(radius*np.cos(self.facing))+(radius*np.cos(self.facing-z))
                self.facing -= z
                #see desmos link on slack for explanation of above three lines. It's essentially direction calculationswhile z<0:
                while z<0:
                    z+=2*np.pi
                
                while z>2*np.pi:
                    z-=2*np.pi

                while self.facing<0:
                    self.facing+=2*np.pi

                while self.facing>2*np.pi:
                    self.facing-=2*np.pi
                #making sure that the z-angle measurement doesn't go below 0 or above 2pi
                
        ob = dict(x=int(self.x), y=int(self.y), facing=int(self.facing*12/np.pi), l_speed=self.l_speed, r_speed=self.r_speed)
        #when it's training, it takes the data from the environment and says "I have this to use now."
        episode_over = self.is_over
        #checks to see if it's over.
        info = dict()
        #openai needs that line to be happy. means nothing
        return ob, self.reward, episode_over, info
        #spit back all that data.
        
    def reset(self):
        T = True
        while T:
            self.x0 = 82 * np.random.random_sample()
            self.y0 = 159.8 * np.random.random_sample()
            if not self.invalid_point(self.x0,self.y0):
                T = False
        self.x = self.x0
        self.y = self.y0
        #set position to a random point
        self.facing = 2*np.pi * np.random.random_sample()
        #set facing to a random position
        self.l_speed = 0
        #stop the left wheel
        self.r_speed = 0
        #stop the right wheel
        return dict(x=int(self.x), y=int(self.y), facing=int(self.facing*12/np.pi), l_speed=self.l_speed, r_speed=self.r_speed)
        #spit back all the data about what I'm doing right now.
        
        
    def checkreward(self):
        if self.l_speed == 0 and self.r_speed == 0 and ((58 - self.x) ** 2 + (159 - self.y) ** 2 >= self.minShootDist ** 2 and (58 - self.x) ** 2 + (159 - self.y) ** 2 <= self.maxShootDist ** 2 and self.y <= self.x + 101 and self.y <= -self.x + 217):
        #If I'm in position in front of the goal and facing the right way,
            if np.round(self.facing,1) <= np.round(np.tan((1598-self.y)/(638-self.x)),3):
            #If I'm in position in front of the goal and facing the right way (but with extra parameters)
                self.is_over = True
                #end the game!
                self.reward += 100
                #i get a lot of points
        x = self.x
        y = self.y
        t = 0
        facing = self.facing
        for check in range(100):
            t+=self.t/100

            if self.l_speed == self.r_speed:
                distance = self.l_speed * t
                #calculate the distance traveled.
                x = x + (distance*np.sin(facing))
                y = y + (distance*np.cos(facing))
            #update my x and y positions, now that I know how far I’ve traveled.
            else:
                radius = (self.w/2)*(self.l_speed+self.r_speed)/(self.l_speed-self.r_speed)
                  #this the radius the robot travels.
                z2 = (self.l_speed-self.r_speed)*t/self.w
                x =  x+(radius*np.sin(facing))-(radius*np.sin(facing-z2))
                y =  y-(radius*np.cos(facing))+(radius*np.cos(facing-z2))
                facing -= z2
                #see desmos link on slack for explanation of above three lines. It’s essentially direction calculations
                while z2<0:
                    z2+=2*np.pi
                while z2>2*np.pi:
                    z2-=2*np.pi
                while  facing<0:
                    facing+=2*np.pi
                while  facing>2*np.pi:
                    facing-=2*np.pi
                #making sure that the z-angle measurement doesn’t go below 0 or above 2pi
            if self.invalid_point(x,y):
                self.reward -= 100
                self.is_over = True
                print("crash: ("+str(x)+","+str(y)+")")
                return
            else:
                '''
                a.penup()
                a.goto(x,y)
                a.pendown()
                a.circle(0.05)
                '''
                print("not crash: ("+str(x)+","+str(y)+")")
            
    def invalid_point(self, x, y):
        if (y <= -0.364 * x + 6.255) or (y <= 0.364 * x - 23.626) or (y >= 0.364 * x + 153.545) or (y >= -0.364 * x + 183.426):
            return True
            #robot ran into the triangles in the corners and loses points

        if y > 87.526 and y < 95.146 and x > 0 and x < 14.1:
            return True 
            #robot ran into the north spinner and loses points

        if y > 64.68 and y < 72.3 and x > 68 and x < 82:
            return True 
            #robot ran into the south spinner and loses points
            
        if x > 82 or y > 159.8 or x < 0 or y<0:
            return True 
            #robot went outside the barrier

        if (y-105.979)>=((106.403-105.979)/(50.871-49.91))*(x-49.91) and (y-106.936)<=((107.36-106.936)/(50.439-49.478))*(x-49.478):
          if (y-105.979)>=((106.936-105.979)/(49.478-49.91))*(x-49.91) and (y-106.403)<=((107.36-106.403)/(50.439-50.871))*(x-50.871):
            return True 
            #robot ran into the top right pillar of the rendezvous point

        if (y-52.469)>=((52.883-52.469)/(32.604-31.666))*(x-31.666) and (y-53.403)<=((53.817-53.403)/(32.182-31.244))*(x-31.244):
          if (y-52.469)>=((53.403-52.469)/(31.244-31.666))*(x-31.666) and (y-52.883)<=((53.817-52.883)/(32.182-32.604))*(x-32.604):
            return True 
            #robot ran into the bottom left pillar of the rendezvous point

        if (y-90.379)>=((90.799-90.379)/(15.42-14.529))*(x-14.529) and (y-91.336)<=((91.76-91.336)/(15.056-14.097))*(x-14.097):
          if (y-90.379)>=((91.336-90.379)/(14.097-14.529))*(x-14.529) and (y-90.799)<=((91.76-90.799)/(15.056-15.42))*(x-15.42):
            return True 
            #robot ran into the top left pillar of the rendezvous point

        if (y-68.07)>=((68.494-68.07)/(68-67.039))*(x-67.039) and (y-69.027)<=((69.451-69.027)/(67.568-66.607))*(x-66.607):
           if (y-68.07)>=((69.027-68.07)/(66.607-67.039))*(x-67.039) and (y-68.494)<=((69.451-68.494)/(67.568-68))*(x-68):
             return True
             #robot ran into the bottom right pillar of the rendezvous point
        return False

    def render(self, mode='human'):
        pg.init()
        screen = pg.display.set_mode([1000, 1000])
        #initialize the whole thing and create a window.

        screen.fill((255, 255, 255))
        #fill the background with white



    def close(self):
        pg.quit()
        return
