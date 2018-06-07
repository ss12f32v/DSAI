import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 




class DQN(nn.Module):
    def __init__(self):
        """
            Pass
        """
        action_size = 3 
        hidden_size = 100
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(2, hidden_size)
        self.input_layer.weight.data.normal_(0, 0.1)   # initialization

        self.dense_layer = nn.Linear(hidden_size, int(hidden_size/2) )
        self.Policy = nn.Linear(hidden_size, action_size)
        self.Policy.weight.data.normal_(0, 0.1)   # initialization

    
            
        self.Dropout = nn.Dropout(p = 0.5)

        self.gamma = 0.99
        self.memory = []
        self.loss = []
        self.loss_function = nn.MSELoss()
   
    def forward(self, inputs):
        first = F.relu(self.input_layer(inputs))
        # second = self.dense_layer(first)
        action = self.Policy(first)

        return action

    def select_action(self, action_prob):
        
        values, indices = torch.max(action_prob, 1)
        indices = indices.data.numpy()[0]
       
        return indices, values


    def learn(self, optimizer):
        # Normalize the reward first 
        # reward_mean = np.mean(self.reward_pool)
        # reward_std = np.std(self.reward_pool)   
        # for i, reward in enumerate(self.reward_pool):
        #         self.reward_pool[i] = (self.reward_pool[i]) / reward_std
        
        memory_chosen_rate = 0
        optimizer.zero_grad()
       

        for i, single_memory in enumerate(self.memory):
            if i == len(self.memory) -1:
                # Final reward is not used in this task.
                pass
            else :
                if np.random.rand(1) < memory_chosen_rate:
                    pass
                else:
                    reward, Q, true_action = self.memory[i]
                    # if i ==0:
                    #     print('here')
                    #     print(reward, Q, true_action)
                    reward = Variable(torch.FloatTensor([reward])) 

                    _, Q_sa,_ = self.memory[i+1]

                   
                    Max_Q = Q[0, true_action]
                    Max_Q_sa, i  = torch.max(Q_sa, dim=-1)
                  
                

                    # one step forward bellman equation
                    greedy_Q = reward + self.gamma * Max_Q_sa.detach()
            
                    loss = self.loss_function(Max_Q, greedy_Q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


       
        self.loss = []
        self.memory = []

