import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 
state_number = 6 # open, close , high ,low , current_income, status
embedding_size = 50
hidden_size = 5
action_size = 3 

class Policy(nn.Module):
    def __init__(self, batch_size=20):
        super(Policy, self).__init__()
        self.hidden_size = hidden_size
#         self.linear1 = nn.Linear(embedding_size, hidden_size)
#         self.linear2 = nn.Linear(hidden_size, state_number)
        self.income_tran_mat = nn.Linear(1, embedding_size)
        self.status_embed_mat = nn.Embedding(3, embedding_size)       
        self.four_feature_transform = nn.Linear(4, embedding_size)
        self.Policy = nn.Linear(embedding_size * 3, action_size)

        self.action_pool = []
        self.reward_pool = []
        self.state_pool = []

        self.reward_discount_factor = 0.99

    def attenion():
        pass
    def forward(self, inputs):

        four, status, income = inputs
        status = self.status_transform(status)
        income_emdeds = self.income_tran_mat(income).view(1, embedding_size)
        status_embeds = self.status_embed_mat(status)
        four_embeds = self.four_feature_transform(four).view(1, embedding_size)
        # print(self.four_feature_transform.weight)
        # print("status_embeds size : ",status_embeds.size())
        # print("income_emdeds size : ",income_emdeds.size())
        # print("four_embeds : ", four_embeds.size())
        total_embeds  = torch.cat([four_embeds, status_embeds, income_emdeds], dim=1)
        # print(total_embeds.size())
        action = F.softmax(self.Policy(total_embeds))		
        self.state_pool.append(total_embeds)
        self.action_pool.append(action)
        return action

    def select_action(self, action_prob):
        # return int action in value 0, -1, 1
        values, indices = torch.max(action_prob, 1)
        action_prob = action_prob.data.numpy()[0]
        # print(action_prob)
        indices = np.random.choice(3, 1, p = action_prob / np.sum(action_prob))[0]
        # print(indices)
        if indices ==0: action = -1
        elif indices == 1: action = 0
        elif indices == 2: action = 1
        return action


    def learn(self, optimizer):
        # Normalize the reward first 
        # reward_mean = np.mean(self.reward_pool)
        # reward_std = np.std(self.reward_pool)   
        # for i, reward in enumerate(self.reward_pool):
        #         self.reward_pool[i] = (self.reward_pool[i] - reward_mean) / reward_std
        
        # Discount the reward in reverse chronological order
        
        count = 0
        for i, reward in reversed(list(enumerate(self.reward_pool))):
            self.reward_pool[i] *= pow(self.reward_discount_factor, count)
            count +=1

        optimizer.zero_grad()
        for i, reward in enumerate(self.reward_pool):
            reward = Variable(torch.FloatTensor([reward]))
            # mask = torch.bernoulli(self.action_pool[i])

            
            mask = Variable(torch.zeros(1, 3))

            values, indices = torch.max(self.action_pool[i], dim = 1)
            mask[0,indices.data.numpy()[0]] = 1
            loss = -torch.log(torch.mm(mask ,self.action_pool[i].transpose(0,1))) * reward  # Negtive score function x reward
            loss.backward()
        optimizer.step()

        self.action_pool = []
        self.reward_pool = []
        self.state_pool = []

    def status_transform(self, status):
        
        if status == -1:
            return Variable(torch.LongTensor([0]))
        elif status == 0:
            return Variable(torch.LongTensor([1]))
        elif status == 1:
            return Variable(torch.LongTensor([2]))
