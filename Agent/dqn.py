#################################################################
# Copyright (C)                                                 #
# 2019 Qiskit Team                                              #
# Permission given to modify the code as long as you keep this  #
# declaration at the top                                        #
#################################################################
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

from typing import Dict, Tuple, Sequence, List
import copy

from network.nets import *

class dqn:
  """
  Deep Q Network

  Action Space: {x1, x2, y1, y2, z1, z2, h1, h2, c12, c21}

  Attribute
    self.num_qubit: 
    self.input_dim:

  Methods
    parse_action: convert 0 to 9 to specific gate and its argument
  """
  def __init__(self, num_qubit=2, num_action=10, gamma=0.9, alpha=.01, epsilon=0.01):
    self.num_qubit = num_qubit
    self.input_dim = ( 1,pow(2, num_qubit) )
    self.num_action = num_action 
    self.init = False
    self.gamma = gamma
    self.alpha = alpha
    self.epsilon = epsilon
    self.total_reward = 0
    self.net_instance = vanila_neural_net(self.num_qubit, self.num_action, self.input_dim)
    self.q_network = self.net_instance.model()
    
  def parse_action(self, action_num):
    if action_num == 0 or action_num == 1:
      return ["X", action_num+1]
    elif action_num == 2 or action_num == 3:
      return ["Y", action_num-1]
    elif action_num == 4 or action_num == 5:
      return ["Y", action_num-3]
    elif action_num == 6 or action_num == 7:
      return ["Y", action_num-5]

    return ["CX", [num_action-8, 2-(num_action-9)]]

  def find_max_val_indx(self, q_values):
    init_flag = False
    indx_list = []
    max_val:float = None
    for indx in range(self.num_action):
      if not init_flag:
        max_val = q_values[indx] 
        indx_list.append(indx)
        init_flag = True
      else:
        if max_val < q_values[indx]:
          max_val = q_values[indx]
          indx_list = [indx]
        elif max_val == q_values[indx]:
          indx_list.append(indx)
    
    return np.random.choice(indx_list) 

  def get_action(self, state):
    self.prev_state = copy.deepcopy(state.reshape(self.input_dim))
    favor_action = None
    if np.random.uniform(0, 1) < self.epsilon:
      favor_action = np.random.choice(range(10))
    else:
      q_values = self.q_network.predict(self.prev_state)[0]
      favor_action = self.find_max_val_indx(q_values)

    self.prev_action = self.parse_action(favor_action)
    return self.prev_action

  def learn_from_transition(self, next_state, reward, terminate):
    if not self.init:
      self.init = True
      return

    state = self._prev_state
    n_state = copy.deepcopy(next_state.reshape(self.input_dim))
    action = self.prev_action
    q_table = self.q_network.predict(state)

    q_values = 0
    if not terminate:
      action_list = self.get_action(n_state)
      q_values = np.max(q_table[0][action_list])
    else:
      self._init = False
      self._prev_action = None
      self._prev_state = None

    q_table[0][action] = reward + self.gamma * q_values
    self.q_network.fit(state, q_table, batch_size=1, verbose=0)

  def reset(self):
    self.init = False
    self.q_network = self.net_instance.model()
    self.q_network.save_weights(filepath +'train_' + str(ag_times) + '.h5')
