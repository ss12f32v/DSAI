import gym
import torch
import numpy as np 
from Model.DQN import DQN
from torch.autograd import Variable


if __name__ == "__main__":
    max_postiion = 0.5
    min_position = -1.2
    epochs = 5000
    epsilon = 0.13
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 3000
    # observation = env.reset()
    model = DQN()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    

    # Load pre-train model
    # model.load_state_dict(torch.load('step419.pt'))
    model.load_state_dict(torch.load('checkpoint/speed250time/step1763.pt'))

    for epoch in range(epochs):
        observation = env.reset()

        for i in range(250):
            env.render()
            observation = Variable(torch.FloatTensor([observation])) #DL

            prev = observation
            Q = model.forward(observation)
            true_action, Q_max_value = model.select_action(Q)

            if np.random.rand(1) < epsilon:
                true_action = np.random.randint(0,3)

            # print('original_action',Q)
            # print('true action', true_action)
            observation, reward, done, info = env.step(true_action) # take a random action
            # reward = 0 
            reward = 0.5 + abs(observation[0])
            
            # print("modified reward", reward)
            # observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
            if observation[0] >= max_postiion:
                for i, single_memory in enumerate(model.memory):
                    model.memory[i][0] += 10
                reward += 10
            
            model.memory.append([reward, Q, true_action])

            if i %25 ==0:
                # print('prev observation',prev.data.numpy())
                # print('Q Value',Q.data.numpy())
                # print('true_action', true_action)
                # print()
                # torch.save(model.state_dict(), 'step'+str(epoch) +'.pt')

                pass
            if observation[0] >= max_postiion:
                print("Reach the top!")
                print('Done')
                torch.save(model.state_dict(), 'checkpoint/speed250time/step'+str(epoch+1763) +'.pt')

                # model.save('step'+str(epoch) +'.pt')


                break
                # scheduler.step()
           
            # print(i)
        model.learn(optimizer)
        print("End epoch:", epoch)
