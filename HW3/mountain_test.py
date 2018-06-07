import gym
import torch
import numpy as np 
from Model.DQN import DQN
from torch.autograd import Variable


if __name__ == "__main__":
    max_postiion = 0.5
    min_position = -1.2
    epochs = 5000
    epsilon = 0.1
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 3000
    # observation = env.reset()
    model = DQN()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    # r_scheduler.StepLR
    model.load_state_dict(torch.load('checkpoint/speed250time/step2408.pt'))

    for epoch in range(epochs):
        observation = env.reset()

        for i in range(250):
            env.render()
            observation = Variable(torch.FloatTensor([observation])) #DL

            prev = observation
            Q = model.forward(observation)
            true_action, Q_max_value = model.select_action(Q)

            # if np.random.rand(1) < epsilon:
            #     true_action = np.random.randint(0,3)

            observation, reward, done, info = env.step(true_action) 
            # reward = 0 
            reward = 0.5 + abs(observation[0])
            
            # print("modified reward", reward)
            # observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
            # if observation[0] >= max_postiion:
            #     reward += 1
            
            # model.memory.append([reward, Q, true_action])

           
            
                # scheduler.step()
           
            # print(i)
        # model.learn(optimizer)
        print("End epoch:", epoch)
