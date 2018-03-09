import torch
from torch.autograd import Variable

class Trainer():
    def __init__(self, Agent, Env, train_Data, optimizer, Epoch = 3,verbose_round=300):
        self.Agent = Agent
        self.Env = Env
        self.train_Data = train_Data
        self.optimizer = optimizer
        self.Epoch = Epoch
        self.verbose_round = verbose_round
    def train(self):
        for epoch in range(self.Epoch):
            self.Env.reset()
            for i in range(self.train_Data.shape[0]-1):
                four_features = list(self.train_Data.iloc[i])
                tomorrow_open_prize = float(list(self.train_Data.iloc[i+1])[0])
                four_features = [float(i) for i in four_features]
                
                four_features_var = Variable(torch.FloatTensor(four_features)) #DL
                status, income = self.Env.Current() # return current status and income

                income_var = Variable(torch.FloatTensor([income])) #DL

                data = (four_features_var, status, income_var) 
                action_var = self.Agent(data)   # put data inside your model, Return Variable type action
                action = self.Agent.select_action(action_var) # Return -1, 0 ,1
                daily_reward, ter = self.Env.execute_action(action, four_features[0], tomorrow_open_prize) # put current action and open_prize in int 
                
                        
                
                if i % self.verbose_round ==0 :
                    print("Day:%s, Status:%s, income:%s, Choosen-Action:%s, reward:%s, ter : %s "%(i, status, income, action, daily_reward, ter))
                if ter:
                    print("Violate rule in day:", i)
                    self.Agent.reward_pool.append(daily_reward)
                    break
                self.Agent.reward_pool.append(daily_reward)
            # print(self.Agent.reward_pool)
            print("Day:%s, Status:%s, Income:%s,  Action:%s, reward:%s, ter : %s "%(i, status, income, action, daily_reward, ter))
            print("EPOCH Income", self.Env.Epoch_Reward(tomorrow_open_prize))
            print()
            self.Agent.learn(self.optimizer)
        