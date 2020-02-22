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
import RedTieBot
import time

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
        self.feature_transformer = feature_transformer

        self.num_states = 82*160*24#10**env.observation_space.shape[0]
        ############
        self.num_actions = 9
        self.load()
        self.counter = 0

    def reset(self):
        self.counter = 0
        self.target = None

    def get_target(self):
        if self.target is None:
            self.target = self.env.get_a_target
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

    def update(self,s,a,G):
        x=self.feature_transformer.transform(s)
        #print("update: ", str(x), str(a))
        self.Q[x,a] += 10e-3*(G-self.Q[x,a])

    def sample_action(self,s,eps):
        self.counter += 1
        #if self.counter < 2000:
            #return self.calculated_path(s)
        if np.random.random() < eps:
            #print('random')
            return self.env.action_space.sample()
        else:
            p=self.predict(s)
            #print('prob: {}'.format(p))
            return self.env.action_space.fromQ(np.argmax(p))
    def setGraphics(self):
        env.graphics = True

    def stopGraphics(self):
        env.graphics = False

        cx, cy, cfacing, cl_speed, cr_speed = observation
        a=self.env.reward_point()
        x, y, facing = self.get_target()
        if facing == np.arctan((cy-y)/(cx-x))*12/np.pi:
            if l_speed == -1 and r_speed == 1:
                return ([0,0])
            elif l_speed == 1 and r_speed == -1:
                return([0,0])
            elif l_speed == 0 and r_speed == 0:
                return ([1,-1])
            elif l_speed == 1 and r_speed == 1:
                return ([-1,-1])
            elif l_speed == -1 and r_speed == -1:
                return ([1,1])

    def check_turn(l_speed, r_speed):
        l_speed = int(10*l_speed/3)
        r_speed = int(10*r_speed/3)
        if l_speed == -1 and r_speed == 1:
                return ([0,0])
        elif l_speed == 1 and r_speed == -1:
            return([0,0])
        elif l_speed == 0 and r_speed == 0:
            return ([1,-1])
        elif abs(l_speed) >= 1:
            l=-1*abs(l_speed)/l_speed
        elif r_speed >= 1:
            r=-1*abs(r_speed)/r_speed
        return([l,r])

def play_one(model,eps,gamma):
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
    return totalreward

def plot_running_avg(totalrewards):
    N=len(totalrewards)
    running_avg=np.empty(N)
    for t in range(N):
        running_avg[t]=totalrewards[max(0,t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title('Rewards')
    plt.show()
    print(totalrewards)
    nameeverything = input("save file? ")
    if nameeverything in ["yes", "y"]:
        model.save()

lookuptable = [{'accn': 1, 'ang': 0, 'speed': 6, 'x': 1, 'y': 0},
 {'accn': 1, 'ang': 0, 'speed': 12, 'x': 1, 'y': 0},
 {'accn': 1, 'ang': 0, 'speed': 18, 'x': 2, 'y': 0},
 {'accn': 1, 'ang': 1, 'speed': 6, 'x': 1, 'y': 0},
 {'accn': 1, 'ang': 1, 'speed': 12, 'x': 1, 'y': 0},
 {'accn': 1, 'ang': 1, 'speed': 18, 'x': 2, 'y': 0},
 {'accn': 1, 'ang': 2, 'speed': 6, 'x': 0, 'y': 0},
 {'accn': 1, 'ang': 2, 'speed': 12, 'x': 1, 'y': 0},
 {'accn': 1, 'ang': 2, 'speed': 18, 'x': 2, 'y': 1},
 {'accn': 1, 'ang': 3, 'speed': 6, 'x': 0, 'y': 0},
 {'accn': 1, 'ang': 3, 'speed': 12, 'x': 1, 'y': 1},
 {'accn': 1, 'ang': 3, 'speed': 18, 'x': 2, 'y': 2},
 {'accn': 1, 'ang': 4, 'speed': 6, 'x': 0, 'y': 0},
 {'accn': 1, 'ang': 4, 'speed': 12, 'x': 0, 'y': 1},
 {'accn': 1, 'ang': 4, 'speed': 18, 'x': 1, 'y': 2},
 {'accn': 1, 'ang': 5, 'speed': 6, 'x': 0, 'y': 1},
 {'accn': 1, 'ang': 5, 'speed': 12, 'x': 0, 'y': 1},
 {'accn': 1, 'ang': 5, 'speed': 18, 'x': 0, 'y': 2},
 {'accn': 1, 'ang': 6, 'speed': 6, 'x': 0, 'y': 1},
 {'accn': 1, 'ang': 6, 'speed': 12, 'x': 0, 'y': 1},
 {'accn': 1, 'ang': 6, 'speed': 18, 'x': 0, 'y': 2},
 {'accn': 2, 'ang': 0, 'speed': 6, 'x': 2, 'y': 0},
 {'accn': 2, 'ang': 0, 'speed': 12, 'x': 4, 'y': 0},
 {'accn': 2, 'ang': 0, 'speed': 18, 'x': 6, 'y': 0},
 {'accn': 2, 'ang': 1, 'speed': 6, 'x': 2, 'y': 0},
 {'accn': 2, 'ang': 1, 'speed': 12, 'x': 4, 'y': 1},
 {'accn': 2, 'ang': 1, 'speed': 18, 'x': 5, 'y': 1},
 {'accn': 2, 'ang': 2, 'speed': 6, 'x': 2, 'y': 1},
 {'accn': 2, 'ang': 2, 'speed': 12, 'x': 3, 'y': 2},
 {'accn': 2, 'ang': 2, 'speed': 18, 'x': 5, 'y': 3},
 {'accn': 2, 'ang': 3, 'speed': 6, 'x': 1, 'y': 1},
 {'accn': 2, 'ang': 3, 'speed': 12, 'x': 2, 'y': 2},
 {'accn': 2, 'ang': 3, 'speed': 18, 'x': 4, 'y': 4},
 {'accn': 2, 'ang': 4, 'speed': 6, 'x': 1, 'y': 2},
 {'accn': 2, 'ang': 4, 'speed': 12, 'x': 2, 'y': 3},
 {'accn': 2, 'ang': 4, 'speed': 18, 'x': 3, 'y': 5},
 {'accn': 2, 'ang': 5, 'speed': 6, 'x': 0, 'y': 2},
 {'accn': 2, 'ang': 5, 'speed': 12, 'x': 1, 'y': 4},
 {'accn': 2, 'ang': 5, 'speed': 18, 'x': 1, 'y': 5},
 {'accn': 2, 'ang': 6, 'speed': 6, 'x': 0, 'y': 2},
 {'accn': 2, 'ang': 6, 'speed': 12, 'x': 0, 'y': 4},
 {'accn': 2, 'ang': 6, 'speed': 18, 'x': 0, 'y': 6},
 {'accn': 3, 'ang': 0, 'speed': 6, 'x': 4, 'y': 0},
 {'accn': 3, 'ang': 0, 'speed': 12, 'x': 6, 'y': 0},
 {'accn': 3, 'ang': 0, 'speed': 18, 'x': 9, 'y': 0},
 {'accn': 3, 'ang': 1, 'speed': 6, 'x': 3, 'y': 1},
 {'accn': 3, 'ang': 1, 'speed': 12, 'x': 6, 'y': 1},
 {'accn': 3, 'ang': 1, 'speed': 18, 'x': 9, 'y': 2},
 {'accn': 3, 'ang': 2, 'speed': 6, 'x': 3, 'y': 2},
 {'accn': 3, 'ang': 2, 'speed': 12, 'x': 5, 'y': 3},
 {'accn': 3, 'ang': 2, 'speed': 18, 'x': 8, 'y': 4},
 {'accn': 3, 'ang': 3, 'speed': 6, 'x': 2, 'y': 2},
 {'accn': 3, 'ang': 3, 'speed': 12, 'x': 4, 'y': 4},
 {'accn': 3, 'ang': 3, 'speed': 18, 'x': 6, 'y': 6},
 {'accn': 3, 'ang': 4, 'speed': 6, 'x': 2, 'y': 3},
 {'accn': 3, 'ang': 4, 'speed': 12, 'x': 3, 'y': 5},
 {'accn': 3, 'ang': 4, 'speed': 18, 'x': 4, 'y': 8},
 {'accn': 3, 'ang': 5, 'speed': 6, 'x': 1, 'y': 3},
 {'accn': 3, 'ang': 5, 'speed': 12, 'x': 1, 'y': 6},
 {'accn': 3, 'ang': 5, 'speed': 18, 'x': 2, 'y': 9},
 {'accn': 3, 'ang': 6, 'speed': 6, 'x': 0, 'y': 4},
 {'accn': 3, 'ang': 6, 'speed': 12, 'x': 0, 'y': 6},
 {'accn': 3, 'ang': 6, 'speed': 18, 'x': 0, 'y': 9},
 {'accn': 4, 'ang': 0, 'speed': 6, 'x': 5, 'y': 0},
 {'accn': 4, 'ang': 0, 'speed': 12, 'x': 9, 'y': 0},
 {'accn': 4, 'ang': 0, 'speed': 18, 'x': 13, 'y': 0},
 {'accn': 4, 'ang': 1, 'speed': 6, 'x': 5, 'y': 1},
 {'accn': 4, 'ang': 1, 'speed': 12, 'x': 9, 'y': 2},
 {'accn': 4, 'ang': 1, 'speed': 18, 'x': 12, 'y': 3},
 {'accn': 4, 'ang': 2, 'speed': 6, 'x': 5, 'y': 3},
 {'accn': 4, 'ang': 2, 'speed': 12, 'x': 8, 'y': 4},
 {'accn': 4, 'ang': 2, 'speed': 18, 'x': 11, 'y': 6},
 {'accn': 4, 'ang': 3, 'speed': 6, 'x': 4, 'y': 4},
 {'accn': 4, 'ang': 3, 'speed': 12, 'x': 6, 'y': 6},
 {'accn': 4, 'ang': 3, 'speed': 18, 'x': 9, 'y': 9},
 {'accn': 4, 'ang': 4, 'speed': 6, 'x': 3, 'y': 5},
 {'accn': 4, 'ang': 4, 'speed': 12, 'x': 4, 'y': 8},
 {'accn': 4, 'ang': 4, 'speed': 18, 'x': 6, 'y': 11},
 {'accn': 4, 'ang': 5, 'speed': 6, 'x': 1, 'y': 5},
 {'accn': 4, 'ang': 5, 'speed': 12, 'x': 2, 'y': 9},
 {'accn': 4, 'ang': 5, 'speed': 18, 'x': 3, 'y': 12},
 {'accn': 4, 'ang': 6, 'speed': 6, 'x': 0, 'y': 5},
 {'accn': 4, 'ang': 6, 'speed': 12, 'x': 0, 'y': 9},
 {'accn': 4, 'ang': 6, 'speed': 18, 'x': 0, 'y': 13}]

def mytest(env):
    tn = []
    for j in range(1,5):
        for i in range(7):
            for k in range(6, 19, 6):
                env.facing, env.x, env.y, env.l_speed, env.r_speed = i, 10, 10, 0.0, 0.0
                d = env.step2(j,k)[0]
                nd = {'accn': j, 'speed': k, 'x': d['x']-10, 'y': d['y']-10, 'ang': d['facing']}
                tn.append(nd)
                print(nd)
            print('====')
        print('>>>')
    #import pdb; pdb.set_trace()
    import pprint
    pprint.pprint(tn)
    sys.exit(0)

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

    N=10000
    totalrewards=np.empty(N)
    import pdb; pdb.set_trace()
    
    show = 'yes'
    #show = 'no'
    #env.fast_mode = True; show = 'no'
    env.fast_mode = False

    for n in range(N):
        eps=1.0/np.sqrt(n+1)
        totalreward=play_one(model, eps, gamma)
        totalrewards[n] = totalreward
        if n%100==0:
            if not env.fast_mode:
                print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
                print("total rewards:", totalrewards.sum())
            if n%500==0:
                print("500 more iterations have passed.")
                if show == 'yes':
                    model.setGraphics()
                    env.clearAndDraw()
                if show == 'no':
                    model.stopGraphics()
                '''
                word = input("show graphics? (y/n)")
                if word == 'y':
                    model.setGraphics()
                    env.clearAndDraw()
                if word == 'n' :
                    model.stopGraphics()
                '''

    #plt.plot(totalrewards)
    #plt.title("Rewards")
    #plt.show()
    plot_running_avg(totalrewards)
    print('Total rewards are ' + str(totalreward) + '!')
