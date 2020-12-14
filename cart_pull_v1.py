import torch 
import torch.nn as nn 
import gym  
from collections import namedtuple 
import torch.optim as optim
import random
import math
import numpy as np

import torch.nn.functional as F

#setup the transition class 
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done')) 

#global params
BATCH_SIZE = 10 

class Replay_Buffer(): 
    def __init__(self, capacity): 
        self.storage = [] 
        self.capacity = capacity
        self.position = 0

    def __len__(self): 
        return len(self.storage) 

    def push(self, *args): 
        # save a tranisiton
        if len(self.storage) < self.capacity: 
            self.storage.append(None)   
        self.storage[self.position] = Transition(*args)
        # circular buffer
        self.position = (self.position + 1) % self.capacity 

    # get a random sample from the dataset of transition s
    def sample(self, batch_size):
        return random.sample(self.storage, batch_size) 
 
class DQN_Network(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(DQN_Network, self) .__init__()
        
        self.input_dim = input_dim 
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim  
        self.l1  = nn.Linear(self.input_dim, self.hidden_dim1)
        self.bn1 = nn.BatchNorm(self.hidden_dim1)
        self.l2  = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.bn2 = nn.BatchNorm(self.hidden_dim2)
        self.l3  = nn.Linear(self.hidden_dim2, self.output_dim)

        """       
        self.conv1  = nn.Conv2d(input_dim, 16, kernel_size=5, stride=2)
        
        self.conv2  = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2    = nn.BatchNorm2d(32)
        self.conv3  = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3    = nn.BatchNorm2d(32)
        self.fc1    = nn.Linear(32, output_dim)
        """ 

    def forward(self, x):
        # x = x.expand(x, dim=1) 
        x = torch.unsqueeze(x, dim=1) 
        print("x shape", x.size())
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))
        x = F.relu(self.bn3(self.l3(x)))
        x = x.view(x.size(0), -1)
        return x
    
class DQNAgent(): 
    def __init__(self, action_dim, input_dim, output_dim): 
        self.action_dim = action_dim
        self.target_net = DQN_Network(input_dim, output_dim,)
        self.policy_net = DQN_Network(input_dim, output_dim)  

        self.eps_start = .9  
        self.eps_end = .05
        self.eps_decay = 200  
        self.steps_done = 0 
        
    def select_action(self, state):
        random_n = random.random() #generate random number
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1  

        if (random_n < eps_threshold): 
        #take random action (random # betwee 0 and 1) (left and right)
            action = torch.tensor([random.randrange(2)]) 
        else: 
        #take the best action  
            with torch.no_grad(): 
                actions = self.policy_net(state)  
                action = torch.argmax(actions).view(1, 1) 
                
        return action.item() 

def compute_loss(memory, optimizer): 
    
    # do nothing if we have not collected enough samples 
    if (len(memory.storage) < BATCH_SIZE): 
        return 
    
    mini_batch = memory.sample(BATCH_SIZE) 
    
    for state, action, next_state, reward, done in mini_batch:
        if done:
            reward = torch.tensor(reward)
            target = reward 
        else: 
            next_state = torch.tensor((next_state)) 
            next_state_values = dqn.target_net(next_state) 
            target = reward + gamma * torch.max(next_state_values)

        expected_state_action_vals.append(target) 
        q_values = dqn.policy_net(torch.tensor(state))
        state_action_val = q_values[action]
        state_action_vals.append(state_action_val)

    expected_state_action_vals = torch.tensor(expected_state_action_vals).clone().detach().requires_grad_(True)
    state_action_vals = torch.tensor(state_action_vals) 
    expected_state_action_vals.requires_grad = True  
    state_action_vals.requires_grad = True  
            
    # compute Huber loss
    loss = F.smooth_l1_loss(state_action_vals, state_action_vals)
    
    ## gradient update
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer.step()

    return loss 
    

def train(dqn, epochs, optimizer, target_update, gamma, env, memory, render):
    running_reward = 0 
    episodes = 100 
    rewards_list = [] 
    loss_list = []
    for ep in range(episodes):  
        state = env.reset()
        score = 0 
        done = False  
        counter = 0 
        while not done: 
            action = dqn.select_action(torch.Tensor(env.state)) 
            next_state, reward, done, _ = env.step(action) 
            memory.storage.append((state, action, next_state, reward, done))
            score += reward 
            rewards_list.append(reward)
            if (render):
                env.render()
            counter += 1
        
        #compute running_reward here
        loss = compute_loss(memory, optimizer)
        loss_list.append(loss)

        if (ep % target_update == 0): 
            dqn.target_net.load_state_dict(v.policy_net.state_dict()) 

        # do some printing here
        print("Episode: {}, Score: {}, Avg. Reward: {}".format(ep, score, score/counter))
        
    return rewards_list
    
#testing purposes 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
 
env_name = "CartPole-v0"
env = gym.make(env_name)
env.reset() 
action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]

dqn = DQNAgent(env, state_dim, 128, 128, action_dim)   

render = True 
gamma = .9  
replay = Replay_Buffer(1000)
epochs = 5
optimizer = optim.Adam(dqn.policy_net.parameters())
train(dqn, epochs, optimizer, 10, gamma, env, replay, render) 