hello

# Usage

First, Dont care the code with #DL comment. It's useless.

## 
User can reference the building of the stock environment in the **Stock_Environment** Folder.

You can return the current status and income by calling ```Current```

Calling ```reset``` to reset all the parameter in Environment.


## Model 
I already finish the Model using **Policy Gradient**.
I'm not sure is right, but I will try to validate the model and improve the robustness of it .

You can replace the model you want in the **Model** folder. 

If you don't want to use any ML model in this homework, then just put your discrete decision method in **Model** is also fine. 

Updata 3/9 : Add **DQN** Model in **Model** Folder, but I'm not sure it can get high revenue...

## Trainer 
Trainer is used to contain all the contents used in training.

I highly recommend you write your own trainner when using pynotebook!


## Reference
[1][Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)

[2][莫凡Policy Gradient](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/5-1-policy-gradient-softmax1/)

[3][PyTorch](http://pytorch.org/)