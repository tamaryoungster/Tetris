import torch
import random
import math
from DQN import DQN
from State import *

S = State()


epsilon_start = 1
epsilon_final = 0.01
epsilon_decay = 5000

class DQN_Agent:
    def __init__(self, parametes_path = None, train = True, env= None):
        self.DQN = DQN()
        if parametes_path:
            self.DQN.load_params(parametes_path)
        self.train = train
        self.setTrainMode()

    def setTrainMode (self):
          if self.train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def get_Action (self, state, epoch = 0, events= None, train = False) -> tuple:
        state = state.toTensor()
        actions = [0,1,2,3]
        if self.train and train:
            epsilon = self.epsilon_greedy(epoch)
            rnd = random.random()
            if rnd < epsilon:
                return random.choice(actions)
        
        with torch.no_grad():
            Q_values = self.DQN(state)
        return torch.argmax(Q_values)

    def get_Actions_Values (self, states):
        with torch.no_grad():
            Q_values = self.DQN(states)
            max_values, max_indices = torch.max(Q_values,dim=1) # best_values, best_actions
        
        return max_indices.reshape(-1,1), max_values.reshape(-1,1)

    def Q (self, states, actions):
        Q_values = self.DQN(states)
        rows = torch.arange(Q_values.shape[0]).reshape(-1,1)
        cols = actions.reshape(-1,1)
        return Q_values[rows, cols]

    def epsilon_greedy(self,epoch, start = epsilon_start, final=epsilon_final, decay=epsilon_decay):
        # res = final + (start - final) * math.exp(-1 * epoch/decay)
        if epoch < decay:
            return start - (start - final) * epoch/decay
        return final
        
    def loadModel (self, file):
        self.model = torch.load(file)
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)

    def __call__(self, events= None, state=None):
        return self.get_Action(state)
    
    def get_end_Action(self, events=None):
        action = 6
        return action



# import pygame
# import random
# from Environment import *
# import math
# from typing import Any
# import torch
# import torch.nn as nn
# import numpy as np
# from DQN import DQN
# from State import State

# env = Environment(state=state)

# # epsilon Greedy
# epsilon_start = 1
# epsilon_final = 0.01
# epsilon_decay = 5000

# # epochs = 1000
# # batch_size = 64
# gamma = 0.99 
# MSELoss = nn.MSELoss()

# class DQN_Agent:
#     def __init__(self, parametes_path = None, train = True, env= None):
#         self.DQN = DQN()
#         if parametes_path is not None:
#             self.DQN.load_params(parametes_path)
#         self.train = train
#         self.setTrainMode()

#     def setTrainMode (self):
#           if self.train:
#               self.DQN.train()
#           else:
#               self.DQN.eval()

    
#     def get_Action (self, state, epoch = 0, events= None, train = False) -> tuple:
#         actions = [0,1,2,3]
#         if self.train and train:
#             # epsilon = self.epsilon_greedy(epoch)
#             # rnd = random.random()
#             # if rnd < epsilon:
#                 return random.choice(actions)
        
#         with torch.no_grad():
#             Q_values = self.DQN(state, actions)
#             # Q_values = self.DQN(state)
#         return actions[Q_values]
    
#     # def epsilon_greedy(self,epoch, start = epsilon_start, final=epsilon_final, decay=epsilon_decay):
#     #     res = final + (start - final) * math.exp(-1 * epoch/decay)
#     #     return res
    
#     def save_param (self, path):
#         self.DQN.save_params(path)

#     def load_params (self, path):
#         self.DQN.load_params(path)

#     def __call__(self, events= None, state=None) -> Any:
#         return self.get_Action(state)
    
#     def get_end_Action(self, events=None):
#         action = 6
#         return action
    

    
#     # def get_Action(self, state, events=None, step=None):
#     #     # Action: left = 1; right=2; rotate=3; no_action = 0; illegal = none
#     #     if step % 15 != 0:
#     #         return None
#     #     action = random.randint(0, 3)
#     #     if env.not_legal(state, action):
#     #         return None
#     #     return action
    
    