from __future__ import print_function, division
from builtins import range
import gym
import os
import sys
import pandas as pd
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import RedTieBot
#from ttictoc import TicToc

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]

class FeatureTransformer:
    def __init__(self):
        '''
        self.bot_xposition_bins = np.linspace(0, 821, 82)
        self.bot_yposition_bins = np.linspace(0, 1598, 160)
        self.bot_facing_bins = np.linspace(0,2*np.pi,24)
        self.bot_lspeed_bins = np.linspace(-128,127,256)
        self.bot_rspeed_bins = np.linspace(-128,127,256)
        '''
    def transform(self, observation):
        x, y, facing, l_speed, r_speed = observation.values()
        '''
        return build_state([
            to_bin(x, self.bot_xposition_bins),
            to_bin(y, self.bot_yposition_bins),
            to_bin(facing, self.bot_facing_bins),
            to_bin(l_speed, self.bot_lspeed_bins),
            to_bin(r_speed, self.bot_rspeed_bins),
        ])
        '''
        return int(x*24*160+y*24+facing)
class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.try_model = gym.make('redtiebot-v0')
        self.feature_transformer = feature_transformer
        self.our_tan = self.calctan()
        self.num_states = 82*160*24#10**env.observation_space.shape[0]
        ############
        self.num_actions = 9
        self.load()
        self.counter = 0
        self.sm = {'not facing': self.rotate,
                   'facing': self.move2,
                   'reached': self.rotate}
        self.state = 'not facing'
        self.spun = False
        self.angle = None
        self.g_angle = None

    def calctan(self):
        tano = []
        for x in range(24):
            tano.append(np.tan((.5+x)*np.pi/12))
        return tano

    def reset(self):
        self.counter = 0
        self.target = None
        self.state = 'not facing'
        self.spun = False

    def get_target(self,x,y):
        if self.target is None:
            self.target = self.env.get_a_target(x,y)
        return self.target

    def save(self):
        np.save('save_Q.npy', self.Q)

    def load(self):
        if os.path.exists('save_Q.npy'):
            self.Q = np.load('save_Q.npy')
        else:
            self.Q = np.random.uniform(low=-1, high=1, size=(self.num_states, self.num_actions))

    def predict(self, s):
        x=self.feature_transformer.transform(s)
        return self.Q[int(x)]

    def update(self,s,aa,G):
        a = self.env.action_space.get_idx(aa)
        x=self.feature_transformer.transform(s)
        #print("update: ", str(x), str(a))
        self.Q[x,a] += 10e-3*(G-self.Q[x,a])

    def sample_action(self,s,eps):
        self.counter += 1
        if self.counter < 2000:
            return self.calculated_path(s)
        if np.random.random() < eps:
            #print('random')
            return self.env.action_space.sample()
        else:
            p=self.predict(s)
            #print('prob: {}'.format(p))
            return self.env.action_space.fromQ(np.argmax(p))
        
    def setGraphics(self, num):
        env.num_graph = num
        env.graphics = True

    def stopGraphics(self):
        env.graphics = False

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

    def move(self,observation):
        cx, cy, cfacing, l_speed, r_speed = observation.values()
        x, y, facing = self.get_target(cx,cy)
        last_dist = dist = (cx-x)**2 + (cy-y)**2
        if self.angle is None:
            mn = []
            for m in range(1,6):
                n = 0
                reached = False
                while not reached and n < 100:
                    self.set_my_model()
                    r = self.try_model.step2(m, n)
                    n += 1
                    if (r[0]['x'], r[0]['y']) == (x, y):
                        reached = True
                        mn.append((m,n))
                    d = (r[0]['x']-x)**2 + (r[0]['y']-y)**2
                    if last_dist < d:
                        reached = True
            if mn ==[]:
                mn = [1,20]
            else:
                mn = max(mn, key=lambda p: p[0])
            self.angle = list(([1,1],)*mn[0] + ([0,0],)*mn[1] + ([-1, -1],)*mn[0])
            #print(self.angle)
        r = self.angle.pop(0)
        if not self.angle:
            self.angle = None
            if (x, y) == (cx, cy):
                self.state = 'reached'
            else:
                self.state = 'not facing'
        return r

    def move2(self,observation):
        cx, cy, cfacing, l_speed, r_speed = observation.values()
        x, y, facing = self.get_target(cx,cy)
        '''
        if cfacing >= 18:
            cfacing -= 24
        '''
        action = ([0,0])
        '''
        if cx != x:
            
            if cx - x < 0:
                self.g_angle = int(np.arctan((cy-y)/(cx-x))*12/np.pi)
            else:
                self.g_angle = int((np.arctan(((cy-y)/(cx-x))+np.pi))*12/np.pi)
                self.g_angle += 12
            
        else:
            self.g_angle = 6
        '''
        goal = ((y-cy)/(self.our_tan[cfacing])+cx)
        #print(cfacing, self.g_angle)
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
        '''
        elif self.g_angle > cfacing:
            action = ([0,1])
        elif self.g_angle < cfacing:
            action = ([1,0])
        '''
        return action
            
    def set_my_model(self):
        self.try_model.x = self.env.x
        self.try_model.y = self.env.y
        self.try_model.facing = self.env.facing
        self.try_model.l_speed = self.env.l_speed
        self.try_model.r_speed = self.env.r_speed

def play_one(model,eps,gamma):
    ticc = time.time()
    observation=env.reset()
    model.reset()
    done=False
    totalreward=0
    iters=0
    path = []
    while not done and iters<10000:
        action=model.sample_action(observation, eps)
        prev_observation=observation
        observation, reward, done, info = env.step(action)
        path.append((observation, reward, action))
        totalreward+= reward
        #if done and iters<299==0:
        #    reward=-300

        G=reward+gamma*np.max(model.predict(observation))
        model.update(prev_observation, action, G)
        iters+=1

    #if totalreward > 0:
        #print(path)

    tics.append(time.time()-ticc)

    return totalreward

def plot_running_avg(totalrewards):
    N=len(totalrewards)
    running_avg=np.empty(N)
    for t in range(N):
        running_avg[t]=totalrewards[max(0,t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title('Rewards')
    plt.show()
    #print(totalrewards)
    nameeverything = input("save file? ")
    if nameeverything in ["yes", "y"]:
        model.save()

'''
def mytest(env):
    tn = []
    for j in range(1,5):
        for i in range(7):
            for k in range(6, 19, 6):
                env.facing, env.x, env.y, env.l_speed, env.r_speed = i, 10, 10, 0.0, 0.0
                d = env.step2(j,k)[0]
                nd = {'accn': j, 'speed': k, 'x': d['x']-10, 'y': d['y']-10, 'ang': d['facing']}
                tn.append(nd)
                #print(nd)
            #print('====')
        #print('>>>')
    #import pdb; pdb.set_trace()
    
    import pprint
    #pprint.pprint(tn)
    sys.exit(0)
'''

if __name__ == '__main__':
    env = gym.make('redtiebot-v0')
    ft = FeatureTransformer()
    model = Model(env,ft)
    gamma = 0.9

    #mytest(env)

    if 'monitor' in sys.argv:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = './' + filename + '_' + str(datetime.now())
        env = wrappers.Monitor(env, monitor_dir)

    N=100
    totalrewards=np.empty(N)
    import pdb; pdb.set_trace()
    
    show = 'never'; env.graphics = False; env.show = 'never'
    #show = 'thousand'; env.graphics = True; env.show = 'thousand'
    #show = 'last'; env.graphics = False; env.show = 'last'
    env.fast_mode = True
    #env.fast_mode = False
    '''
    fast = input("Fast mode? (y or n): ")
    if fast == 'y':
        env.fast_mode = True
    else:
        env.fast_mode = False
    
    graphics = input("Show graphics? (y or n): ")
    if graphics == 'y':
        show = 'thousand'
        num_g = 250
    elif graphics == 'a':
        show = 'thousand'
        num_g = 1
    elif graphics == 'l':
        show = 'last'
        num_g = 1000
    else:
        show = 'never'    
    '''

    tics = []

    for n in range(N):
        print(n)

        '''
        if show == 'last' and n == N-1:
            env.graphics = True
            model.setGraphics()
            env.start()
            env.clearAndDraw()
            '''

        eps=1.0/np.sqrt(n+1)
        totalreward=play_one(model, eps, gamma)
        totalrewards[n] = totalreward

        if n%100==0:
            if not env.fast_mode:
                print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
                print("total rewards:", totalrewards.sum())
                if n%1000==0:
                    #print("1000 episodes have passed")
                    if show == 'thousand':
                    	model.setGraphics()
                    	if n == 0:
                        	env.start()
                        	env.clearAndDraw()
    
    print(str((sum(tics)/len(tics))) + " seconds average episode runtime.")

    '''
    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()
    plot_running_avg(totalrewards)
    print('Total rewards are ' + str(totalreward) + '!')
    '''
