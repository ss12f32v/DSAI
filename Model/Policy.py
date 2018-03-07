import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

state_number = 6 # open, close , high ,low , current_income, status
embedding_size = 5
hidden_size = 5
action_size = 3 

class Policy_Model(nn.Module):
    def __init__(self, batch_size=20):
        super(Policy_Model, self).__init__()
        self.hidden_size = hidden_size
#         self.linear1 = nn.Linear(embedding_size, hidden_size)
#         self.linear2 = nn.Linear(hidden_size, state_number)
        self.income_tran_mat = nn.Linear(1, embedding_size)

        self.status_embed_mat = nn.Embedding(1, embedding_size)
        
        self.four_feature_transform = nn.Linear(4, embedding_size)
        self.Policy = nn.Linear(embedding_size * 3, action_size)
    def attenion():
    	pass
    def forward(self, inputs):

    	four, status, income = inputs
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
    	# print(F.softmax(action))
    	return action
