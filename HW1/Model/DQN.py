import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 


state_number = 6 # open, close , high ,low , current_income, status
embedding_size = 50
action_size = 3 

class DQN(nn.Module):
    def __init__(self, batch_size=20):
        """
            Pass
        """
        super(DQN, self).__init__()
        self.income_tran_mat = nn.Linear(1, embedding_size)
        self.status_embed_mat = nn.Embedding(3, embedding_size)       
        self.four_feature_transform = nn.Linear(4, embedding_size)
        self.Policy = nn.Linear(embedding_size * 3, action_size)

        self.Dropout = nn.Dropout(p = 0.5 )
        

        self.reward_discount_factor = 0.8
        self.memory = []
        self.loss = []


    def attenion():
        pass
    def forward(self, inputs):

        four, status, income = inputs
        status = self.status_transform(status)
        income_emdeds = self.income_tran_mat(income).view(1, embedding_size)
        status_embeds = self.status_embed_mat(status)
        four_embeds = self.four_feature_transform(four).view(1, embedding_size)
        total_embeds  = torch.cat([four_embeds, status_embeds, income_emdeds], dim=1)

        # total_embeds = self.Dropout(total_embeds)
        
        action_Q = F.softmax(self.Policy(total_embeds))		
        return action_Q

    def select_action(self, action_prob):
        """ 
            Input : Var action in value [xx,xx,xx]
            Return : int action in value -1, 0, 1
            In Q-learning just to choose the action that maximize the  Q-value. 

            Notice : There's no episilon here now!
        """
        values, indices = torch.max(action_prob, 1)
        indices = indices.data.numpy()[0]
        if indices ==0: action = -1
        elif indices == 1: action = 0
        elif indices == 2: action = 1
        return action


    def learn(self, optimizer):
        # Normalize the reward first 
        # reward_mean = np.mean(self.reward_pool)
        # reward_std = np.std(self.reward_pool)   
        # for i, reward in enumerate(self.reward_pool):
        #         self.reward_pool[i] = (self.reward_pool[i]) / reward_std
        
        # Discount the reward in reverse chronological order
        for i, single_memory in enumerate(self.memory):
            if i == len(self.memory) -1:
                # Final reward is not used in this task.
                pass
            else :
                daily_reward, Q = self.memory[i]
                _, Q_sa = self.memory[i+1]
                Max_Q, i  = torch.max(Q_sa, dim=1)
                Max_Q_sa, i  = torch.max(Q_sa, dim=1)
 
                greedy_Q = daily_reward + self.reward_discount_factor * Max_Q_sa
                daily_loss = greedy_Q - Max_Q
                self.loss.append(daily_loss)

        optimizer.zero_grad()
        for i, loss in enumerate(self.loss):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.loss = []
        self.memory = []

    def status_transform(self, status):
        
        if status == -1:
            return Variable(torch.LongTensor([0]))
        elif status == 0:
            return Variable(torch.LongTensor([1]))
        elif status == 1:
            return Variable(torch.LongTensor([2]))
