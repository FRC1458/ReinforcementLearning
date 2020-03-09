# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import calculated_path
import copy
import time
from gym import wrappers
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from q_learning_bins import plot_running_avg

tf.compat.v1.disable_eager_execution()


# a version of HiddenLayer that keeps track of params
class HiddenLayer:
  def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
    self.W = tf.Variable(tf.random.normal(shape=(M1, M2)))
    self.params = [self.W]
    self.use_bias = use_bias
    if use_bias:
      self.b = tf.Variable(np.zeros(M2).astype(np.float32))
      self.params.append(self.b)
    self.f = f

  def forward(self, X):
    if self.use_bias:
      a = tf.matmul(X, self.W) + self.b
    else:
      a = tf.matmul(X, self.W)
    return self.f(a)


class DQN:
  def __init__(self, D, K, hidden_layer_sizes, gamma, env, max_experiences=10000, min_experiences=100, batch_sz=32):
    self.env = env
    self.A = calculated_path.Path_Calculator(self.env)
    self.K = K
    self.history = 4
    self.D = D * self.history
    self.D_orig = D
    self.is_guided = False
    self.max_x = env.max_x
    self.max_y = env.max_y
    self.max_theta = env.max_facing
    self.num_states = self.max_x * self.max_y * self.max_theta

    self.init_run_cnt = 0
    self.init_run_max = 0
    self.obs_space = None
    # create the graph
    
    #self.layers = []
    #self.model = Sequential()
    inputs = tf.keras.Input(shape = (self.D,))
    x = inputs
    #self.model.add(Embedding(D, K, input_length=1))
    M1 = D
    for M2 in hidden_layer_sizes:
      #layer = HiddenLayer(M1, M2)
      #self.layers.append(layer)
      #self.model.add(Dense(M2, activation='relu'))
      x = Dense(M2, activation='relu')(x)
      M1 = M2

    # final layer
    #layer = HiddenLayer(M1, 1, lambda x: x)
    #self.model.add(Dense(K, activation='linear'))
    y_hat = Dense(K, activation='linear')(x)
    self.model = Model(inputs = inputs, outputs = y_hat)
    #self.layers.append(layer)
    self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True), optimizer=Adam(0.01))

    # collect params for copy
   
    # create replay memory
    # self.load()
    self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
    self.max_experiences = max_experiences
    self.min_experiences = min_experiences
    self.batch_sz = batch_sz
    self.gamma = gamma

  def load(self):
    if os.path.exists('save_Q.npy'):
      self.Q = np.load('save_Q.npy')
    else:
      self.Q = np.random.uniform(low=-1, high=1, size=(self.num_states, self.K))

  def set_max_guided_run(self, n):
    self.init_run_max = n
    
  def reset(self, p=.25):
    self.is_guided = (self.init_run_cnt < self.init_run_max) or np.random.random() < p
    self.init_run_cnt += 1
    self.obs_space = None
    self.A.reset()

  def set_session(self, session):
    self.session = session

  def copy_from(self, other):
    self.model.set_weights(other.model.get_weights())

  def predict(self, X):
    X = np.atleast_2d(X)
    return np.argmax(self.model.predict(X))

  def train(self, target_network):
    # sample a random batch from buffer, do an iteration of GD
    if self.init_run_cnt < self.init_run_max:
      self.update_static_reward(target_network)
      return
    if len(self.experience['s']) < self.min_experiences:
      # don't do anything if we don't have enough experience
      return
    
    # randomly select a batch
    idx = np.random.choice(len(self.experience['s'])-4, size=self.batch_sz, replace=False)
    # print("idx:", idx)
    states = [self.experience['s'][i] + self.experience['s'][i+1] \
               + self.experience['s'][i+2] + self.experience['s'][i+3] for i in idx]
    actions = [self.experience['a'][i+3] for i in idx]
    rewards = [self.experience['r'][i+3] for i in idx]
    next_states = [self.experience['s2'][i] + self.experience['s2'][i+1] \
               + self.experience['s2'][i+2] + self.experience['s2'][i+3] for i in idx]
    dones = [self.experience['done'][i+3] for i in idx]
    
    targets = [self.update_reward(self.experience['s'][i+3], self.experience['a'][i+3],
                         self.experience['done'][i+3], self.experience['r'][i+3],
                         self.experience['s2'][i] + self.experience['s2'][i+1] \
               + self.experience['s2'][i+2] + self.experience['s2'][i+3],
                                  target_network) for i in idx ]
    # call optimizer
    
    self.model.fit([states], [targets], verbose = 0)

  def update_static_reward(self, target):
    i = -1
    state, action, is_done, reward, next_state = \
           self.experience['s'][i], self.experience['a'][i], \
                         self.experience['done'][i], self.experience['r'][i], \
                         self.experience['s2'][i]
    target.Q[self.q_state_idx(state[0], state[1], state[2])][action] = reward
    
    
  def update_reward(self, state, action, is_done, reward, next_state, target):
    if is_done:
      target.Q[self.q_state_idx(state[0], state[1], state[2])][action] = reward
      return action
    else:
      a = target.predict(next_state)
      vi = np.argmax(target.Q[self.q_state_idx(next_state[-5], next_state[-4], next_state[-3])])
      target.Q[self.q_state_idx(state[0], state[1], state[2])][
        action] = reward + self.gamma * \
             target.Q[self.q_state_idx(next_state[0], next_state[1], next_state[2])][vi]
      return a

  def q_state_idx(self, x, y, theta):
    return theta + self.max_theta * y + self.max_y * self.max_theta * x
  
  def add_experience(self, s, a, r, s2, done):
    if len(self.experience['s']) >= self.max_experiences:
      self.experience['s'].pop(0)
      self.experience['a'].pop(0)
      self.experience['r'].pop(0)
      self.experience['s2'].pop(0)
      self.experience['done'].pop(0)
    self.experience['s'].append(s)
    self.experience['a'].append(a)
    self.experience['r'].append(r)
    self.experience['s2'].append(s2)
    self.experience['done'].append(done)

  def obs_x(self, x):
    if self.obs_space is None:
      self.obs_space =  list(x.values())+  list(x.values())+\
                       list(x.values())+ list(x.values())
    else:
      for i in range(self.D_orig):
        self.obs_space.pop(0)
      self.obs_space += list(x.values())
    return self.obs_space
      
  def sample_action(self, x, eps):
    if self.is_guided:
      return self.A.calculated_path(x)
    else:
      if np.random.random() < eps:
        return self.env.action_space.sample()
      else:
        p=self.predict(self.obs_x(x))
        return self.env.action_space.fromQ(np.argmax(p))


def play_one(env, model, tmodel, eps, gamma, copy_period):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  model.reset(.2)
  # tmodel.reset()
  
  while not done and iters < 10000:
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, info = env.step(action)
    totalreward += reward
    #a = tf.one_hot(env.action_space.get_idx(action), len(env.action_space))
    a = env.action_space.get_idx(action)
    model.add_experience(list(prev_observation.values()), a, reward,
                         list(observation.values()), done)
    model.train(tmodel)

    iters += 1

    if iters % copy_period == 0:
      tmodel.copy_from(model)

  return totalreward


def main():
  env = gym.make('redtiebot-v0')
  gamma = 0.99
  copy_period = 50
  s_time = time.time()
  D = 5
  K = 9
  sizes = [10,15,20,15,10]
  model = DQN(D, K, sizes, gamma, env)
  tmodel = DQN(D, K, sizes, gamma, env)
  tmodel.load()
  init = tf.compat.v1.global_variables_initializer()
  #session = tf.compat.v1.InteractiveSession()
  #session.run(init)
  #model.set_session(session)
  #tmodel.set_session(session)
  import pdb; pdb.set_trace()

  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)


  N = 500
  model.set_max_guided_run(int(.25*N))
  totalrewards = np.empty(N)
  costs = np.empty(N)
  for n in range(N):
    eps = 1.0/np.sqrt(n+1)
    totalreward = play_one(env, model, tmodel, eps, gamma, copy_period)
    totalrewards[n] = totalreward
    if n % 100 == 0:
      print("episode:", n, "total reward:", totalreward, "eps:", eps, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())
      print("time: "+str(time.time()-s_time))
      s_time = time.time()

  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", totalrewards.sum())

  plt.plot(totalrewards)
  plt.title("Rewards")
  plt.show()

  plot_running_avg(totalrewards)


if __name__ == '__main__':
  main()

