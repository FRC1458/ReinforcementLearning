#coding: utf-8
import numpy as np
import gym.spaces.box as b
import gym
import turtle
#we need numpy, gym and pygame present here. np and pg is merely shorthand for numpy and pygame, respectively.
#a=tu.Turtle()
#a.speed(0)
class ActionSpace:
    def __init__(self):
        self._spaces = np.array([(-1,-1), (-1,0), (-1,1), (0,-1),
                                (0,0), (0,1), (1,-1), (1,0), (1,1)])

    def sample(self):
        return self._spaces[np.random.choice(len(self._spaces))]

    def fromQ(self, val):
       # print(val)
        return self._spaces[np.digitize(val, np.linspace(-1.0, 1.0, len(self._spaces))) - 1]

class BotModel(gym.Env):
    def __init__(self):
        self.graphics = False
        self.s = 2
        #the scale of the graphics
        self.minShootDist = 5 #This is the MINIMUM Distance from away the target
        self.maxShootDist = 10 #This is the MAXIMUM Distance away from the target
        self.a=self.reward_point()
        self.w=5
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
        #the direction that the robot is facing, in standard position
        self.l_speed=0
        #left wheel speed
        self.r_speed=0
        #right wheel speed
        self.is_over=False
        #the game is not over yet.
        self.reward = 0#the  a rewarded to the robot during the simulation
        self.counter = 0
        self.observation_space = b.Box(0, 1.0, shape=(int(821/10), int(1598/10), 24))
        #The structure of the data that will be returned by the environment. It's the dimensions of the field (without obstacles at the moment)
        #The box is technically a 1x1x1 cube.
        self.action_space = ActionSpace()
        #The range of speeds that the wheel can have.
        self.path = []
        self.fast_mode = 0

    def step2(self, m, n):
        for i in range(m):
            self.step([1, 1])
        for i in range(n):
            self.step([0,0])
        r = None
        for i in range(m):
            r = self.step([-1, -1])
        return r

    def step(self, action):
        s = self.s
        try:
            self.l_speed += 0.3*action[0]
            #in the list "action", the first value is the left wheel speed.
            self.r_speed += 0.3*action[1]
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
        #above lines limit the speed of the wheels to 128 cm/s backwards or 127 cm/s forward.
        self.checkreward()
        if not self.is_over:
            self.x, self.y, self.facing = self.moving(self.x,self.y,self.facing,self.t)
        ob = dict(x=int(self.x), y=int(self.y), facing=int(self.facing), l_speed=self.l_speed, r_speed=self.r_speed)
        #when it's training, it takes the data from the environment and says "I have this to use now."
        episode_over = self.is_over
        #checks to see if it's over.
        info = dict()
        self.render()
        #openai needs that line to be happy. means nothing
        return ob, self.reward, episode_over, info
        #spit back all that data.

    def reset(self):
        s = self.s
        self.x0, self.y0, self.facing = self.generate_point()      
        self.x = self.x0
        self.y = self.y0
        #set position to a random point
        #set facing to a random position
        self.l_speed = 0
        #stop the left wheel
        self.r_speed = 0
        self.reward = 0
        self.is_over = False
        if self.graphics:
            self.trt.penup()
            self.trt.goto(self.x0*s, self.y0*s)
        self.counter += 1
        self.checkreward()
        return dict(x=int(self.x), y=int(self.y), facing=int(self.facing), l_speed=self.l_speed, r_speed=self.r_speed)

    def checkreward(self):
        d0 = np.sqrt((58-self.x0)**2+(((self.maxShootDist + self.minShootDist)/2)-self.y0)**2)
        d = np.sqrt((58-self.x)**2+(((self.maxShootDist + self.minShootDist)/2)-self.y)**2)
        if self.is_over:
            self.reward += ((1/d)*100)-((1/d0)*100)
        s = self.s
        if abs(self.l_speed) <= 0.01 and abs(self.r_speed) <= 0.01 and ((int(self.x), int(self.y), int(self.facing)) in self.a):
        #If I'm in position in front of the goal and facing the right way,
            self.is_over = True
            #end the game!
            #print(20*'>' + 'Reached')
            self.reward += 100
            #i get a lot of points
        x = self.x
        y = self.y
        t = 0
        facing = self.facing
        N = 10
        for check in range(N):
            t+=self.t/N
            x, y, facing = self.moving(x,y,facing,t)
            if self.fast_mode:
                if self.invalid_point_fast(x,y):
                    self.reward -= 100
                    self.is_over = True
            else:
                if self.invalid_point(x, y):
                    self.reward -= 100
                    self.is_over = True

    def invalid_point(self, x, y):
        if (y <= -0.364 * x + 6.255) or (y <= 0.364 * x - 23.626) or (y >= 0.364 * x + 153.545) or (y >= -0.364 * x + 183.426):
            print("ran into the triangle corners")
            return True
            #robot ran into the triangles in the corners and loses points

        if y > 87.526 and y < 95.146 and x > 0 and x < 14.1:
            print('ran into the east spinner')
            return True
            #robot ran into the east spinner and loses points

        if y > 64.68 and y < 72.3 and x > 68 and x < 82:
            print('ran into the west spinner')
            return True
            #robot ran into the west spinner and loses points

        if x > 82.1 or y > 159.8 or x < 0 or y<0:
            print('into the barrier')
            return True
            #robot went outside the barrier

        #lines past this point are in point-slope form, not slope-intercept.

        if (y-105.979)>=((106.403-105.979)/(50.871-49.91))*(x-49.91) and (y-106.936)<=((107.36-106.936)/(50.439-49.478))*(x-49.478):
          if (y-105.97)>=((106.936-105.979)/(49.478-49.91))*(x-49.91) and (y-106.403)<=((107.36-106.403)/(50.439-50.871))*(x-50.871):
            print('ran into the top pillar')
            return True
            #robot ran into the top right pillar of the rendezvous point

        if (y-52.469)>=((52.883-52.469)/(32.604-31.666))*(x-31.666) and (y-53.403)<=((53.817-53.403)/(32.182-31.244))*(x-31.244):
          if (y-52.469)>=((53.403-52.469)/(31.244-31.666))*(x-31.666) and (y-52.883)<=((53.817-52.883)/(32.182-32.604))*(x-32.604):
            print('ran into the south pillar')
            return True
            #robot ran into the bottom left pillar of the rendezvous point

        if (y-90.379)>=((90.799-90.379)/(15.42-14.529))*(x-14.529) and (y-91.336)<=((91.76-91.336)/(15.056-14.097))*(x-14.097):
          if (y-90.379)>=((91.336-90.379)/(14.097-14.529))*(x-14.529) and (y-90.799)<=((91.76-90.799)/(15.056-15.42))*(x-15.42):
            print('ran into the west pillar')
            return True
            #robot ran into the top left pillar of the rendezvous point

        if (y-68.07)>=((68.494-68.07)/(68-67.039))*(x-67.039) and (y-69.027)<=((69.451-69.027)/(67.568-66.607))*(x-66.607):
           if (y-68.07)>=((69.027-68.07)/(66.607-67.039))*(x-67.039) and (y-68.494)<=((69.451-68.494)/(67.568-68))*(x-68):
             print('ran into the east pillar')
             return True
             #robot ran into the bottom right pillar of the rendezvous point
        return False

    def invalid_point_fast(self, x, y):
        if (y <= -0.364 * x + 6.255) or (y <= 0.364 * x - 23.626) or (y >= 0.364 * x + 153.545) or (y >= -0.364 * x + 183.426):
            return True
            #robot ran into the triangles in the corners and loses points

        if y > 87.526 and y < 95.146 and x > 0 and x < 14.1:
            return True
            #robot ran into the east spinner and loses points

        if y > 64.68 and y < 72.3 and x > 68 and x < 82:
            return True
            #robot ran into the west spinner and loses points

        if x > 82.1 or y > 159.8 or x < 0 or y<0:
            return True
            #robot went outside the barrier

        if (y-105.979)>=((106.403-105.979)/(50.871-49.91))*(x-49.91) and (y-106.936)<=((107.36-106.936)/(50.439-49.478))*(x-49.478) and ((y-105.97)>=((106.936-105.979)/(49.478-49.91))*(x-49.91) and (y-106.403)<=((107.36-106.403)/(50.439-50.871))*(x-50.871)):
            return True
            #robot ran into the top right pillar of the rendezvous point

        if ((y-52.469)>=((52.883-52.469)/(32.604-31.666))*(x-31.666) and (y-53.403)<=((53.817-53.403)/(32.182-31.244))*(x-31.244)) and ((y-52.469)>=((53.403-52.469)/(31.244-31.666))*(x-31.666) and (y-52.883)<=((53.817-52.883)/(32.182-32.604))*(x-32.604)):
            return True
            #robot ran into the bottom left pillar of the rendezvous point

        if ((y-90.379)>=((90.799-90.379)/(15.42-14.529))*(x-14.529) and (y-91.336)<=((91.76-91.336)/(15.056-14.097))*(x-14.097)) and ((y-90.379)>=((91.336-90.379)/(14.097-14.529))*(x-14.529) and (y-90.799)<=((91.76-90.799)/(15.056-15.42))*(x-15.42)):
            return True
            #robot ran into the top left pillar of the rendezvous point

        if ((y-68.07)>=((68.494-68.07)/(68-67.039))*(x-67.039) and (y-69.027)<=((69.451-69.027)/(67.568-66.607))*(x-66.607)) and ((y-68.07)>=((69.027-68.07)/(66.607-67.039))*(x-67.039) and (y-68.494)<=((69.451-68.494)/(67.568-68))*(x-68)):
             return True
             #robot ran into the bottom right pillar of the rendezvous point
        return False

    def render(self, mode='human'):
        s = self.s
        if self.graphics == True:
            self.trt.width(1)
            self.trt.pendown()
            self.trt.setheading(self.facing*15)
            self.trt.goto(self.x*s, self.y*s)
            #these few lines just turn the turtle as needed and go to the new point. Accounts for the scalar.

    def generate_point(self):
        s = self.s
        if self.counter >0:
            T = True
            facing = np.random.randint(24)
            while T:
                x = np.random.randint(82)
                y = np.random.randint(159)
                if not self.fast_mode:
                    if not self.invalid_point(x,y):
                        return x,y,facing
                else:
                    if not self.invalid_point_fast(x,y):
                        return x,y,facing
        else:
            return self.a[np.random.choice(len(self.a))]

    def get_a_target(self):
        s = self.s
        return self.a[np.random.choice(len(self.a))]

    def reward_point(self):
        s = self.s
        a=[]
        for x in range(81):
            for y in range(160):
                if ((58 - x) ** 2 + (159 - y) ** 2 >= self.minShootDist ** 2 and (58 - x) ** 2 + (159 - y) ** 2 <= self.maxShootDist ** 2 and y <= x + 101 and y <= -x + 217):
                    if x != 58:
                        facing=np.arctan((58-x)/(158-y))+np.pi/2
                    else:
                        facing=np.pi/2
                    if facing<0:
                        facing+=np.pi*2
                    facing = int(facing*12/np.pi)
                    if facing > 2:
                        a.append((x,y,facing))
        return a

    def moving(self, x,y,facing,t):
        #this function calculates the robot's new point.
        #takes the direction and each of the wheel speeds to calculate the point change.
        facing = facing*np.pi/12
        if abs(self.l_speed - self.r_speed) <= 0.01:
            distance = (self.l_speed + self.r_speed) * .5 * t
            #calculate the distance traveled.
            x = x + (distance*np.cos(facing))
            y = y + (distance*np.sin(facing))
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
        return x, y, facing*12/np.pi
        
    def close(self):
        #self.trt.bye()
        pass

    def clearAndDraw(self):
        s = self.s
        if self.graphics:
            self.trt = turtle.Turtle()
            self.trt.shape('square')
            self.trt.speed(0)
            
            self.trt.clear()
            self.trt.penup()
            self.trt.pencolor('red')
            self.trt.fillcolor('red')
            #reset and set turtle scale

            self.trt.goto(0*s,6.26*s)
            self.trt.pendown()
            self.trt.goto(0*s, 153.55*s)
            self.trt.goto(17.33*s, 159.85*s)
            self.trt.goto(64.77*s, 159.85*s)
            self.trt.goto(82.18*s, 153.52*s)
            self.trt.goto(82.18*s, 6.28*s)
            self.trt.goto(64.91*s, 0)
            self.trt.goto(17.19*s, 0)
            self.trt.goto(0*s, 6.26*s)
            #draw the field square with corners

            self.trt.penup()
            self.trt.goto(50.44*s, 107.36*s)
            self.trt.pendown()
            self.trt.begin_fill()
            self.trt.goto(50.87*s, 106.4*s)
            self.trt.goto(49.91*s, 105.98*s)
            self.trt.goto(49.48*s, 106.94*s)
            self.trt.end_fill()
            #draw the north pillar

            self.trt.penup()
            self.trt.goto(67.57*s, 69.45*s)
            self.trt.pendown()
            self.trt.begin_fill()
            self.trt.goto(68*s, 68.49*s)
            self.trt.goto(67.04*s, 68.07*s)
            self.trt.goto(66.61*s, 69.03*s)
            self.trt.goto(67.57*s, 69.45*s)
            self.trt.end_fill()
            #draw the east pillar

            self.trt.penup()
            self.trt.goto(32.18*s, 53.82*s)
            self.trt.pendown()
            self.trt.begin_fill()
            self.trt.goto(32.60*s, 52.88*s)
            self.trt.goto(31.67*s, 52.47*s)
            self.trt.goto(31.24*s, 53.40*s)
            self.trt.goto(32.18*s, 53.82*s)
            self.trt.end_fill()
            #draw the south pillar

            self.trt.penup()
            self.trt.goto(15.06*s, 91.76*s)
            self.trt.pendown()
            self.trt.begin_fill()
            self.trt.goto(15.48*s, 90.8*s)
            self.trt.goto(14.53*s, 90.38*s)
            self.trt.goto(14.1*s, 91.34*s)
            self.trt.goto(15.06*s, 91.76*s)
            self.trt.end_fill()
            #drw the west pillar

            self.trt.penup()
            self.trt.goto(14.1*s, 95.15*s)
            self.trt.pendown()
            self.trt.begin_fill()
            self.trt.goto(14.1*s, 87.53*s)
            self.trt.goto(0, 87.53*s)
            self.trt.goto(0, 95.15*s)
            self.trt.goto(14.1*s, 95.15*s)
            self.trt.end_fill()
            #draw the west spinner

            self.trt.penup()
            self.trt.goto(82.1*s, 72.3*s)
            self.trt.pendown()
            self.trt.begin_fill()
            self.trt.goto(82.1*s, 64.68*s)
            self.trt.goto(68*s, 64.68*s)
            self.trt.goto(68*s, 72.3*s)
            self.trt.goto(82.1*s, 72.3*s)
            self.trt.end_fill()
            #draw the east spinner

            self.trt.pencolor('black')
