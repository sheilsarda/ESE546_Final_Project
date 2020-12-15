import gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from itertools import count
import torch.nn.functional as F
from collections import namedtuple 
import math


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done')) 

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
    def __init__(self, state_dim, hidden_dim1, hidden_dim2, action_dim):
        super(DQN_Network, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.action_dim = action_dim

        self.l1 = nn.Linear(self.state_dim, self.hidden_dim1)
        self.l2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.l3 = nn.Linear(self.hidden_dim2, self.action_dim)

        self.eps_start = .9  
        self.eps_end = .05
        self.eps_decay = 200  
        self.steps_done = 0 

    def forward(self, state):
        out = F.relu(self.l1(state))
        out = F.relu(self.l2(out))
        action = self.l3(out)

        return action
 
    def select_action(self, state):
        random_n = random.random() #generate random number
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1  

        if (random_n < eps_threshold): 
        #take random action (random # betwee 0 and 1) (left and right)  
            rando = random.randint(0, 1)
            action = torch.tensor([rando]) 
        
        else: 
        #take the best action  
            with torch.no_grad(): 
                action = torch.argmax(self.forward(state), dim=1)
                 
        return action.item() 

def optimize_loss(dqn, optimizer, gamma, memory_replay):
    
    if (len(memory_replay) < BATCH_SIZE):
        return 
 
    batch = memory_replay.sample(BATCH_SIZE) 
    batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

    batch_state = torch.FloatTensor(batch_state)
    batch_next_state = torch.FloatTensor(batch_next_state)
    batch_action = torch.FloatTensor(batch_action).unsqueeze(1)
    batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1)
    batch_done = torch.FloatTensor(batch_done).unsqueeze(1)

    with torch.no_grad():
        q_val_next = dqn.target_net(batch_next_state)
        preds = batch_reward + (1 - batch_done) * gamma * torch.max(q_val_next, dim=1, keepdim=True)[0]

    loss = F.mse_loss(dqn.policy_net(batch_state).gather(1, batch_action.long()), preds)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train(dqn, episodes, optimizer, target_updates, batch_size, gamma, render):
    
    for ep in range(episodes): 
        state = env.reset()
        score = 0
        done = False
        while not done:  
            if render: env.render()
            tensor_state = torch.FloatTensor(state).unsqueeze(0)
            action = dqn.policy_net.select_action(tensor_state)
            next_state, reward, done, _ = env.step(action)
            score += reward            
            memory_replay.push(state, next_state, action, reward, done)
            if (ep % target_updates == 0): 
                dqn.target_net.load_state_dict(dqn.policy_net.state_dict())
   
            optimize_loss(dqn, optimizer, gamma, memory_replay)
            state = next_state
        
        #printing 
        if ep % 10 == 0: 
            print('Ep {}\tTotal Episode score: {:.2f}\t'.format(ep, score)) 

class DQN_Agent():
    def __init__(self, state_dim, h_dim1, h_dim2, action_dim):
        self.policy_net = DQN_Network(state_dim, h_dim1, h_dim2, action_dim)
        self.target_net = DQN_Network(state_dim, h_dim1, h_dim2, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

if __name__ == "__main__": 
    env = gym.make('CartPole-v0')
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n

    state_dim = 4
    action_dim = 2 
    h_dim1 = 64
    h_dim2 = 256 
    dqn_agent = DQN_Agent(state_dim, h_dim1, h_dim2, action_dim)    
    optimizer = torch.optim.Adam(dqn_agent.policy_net.parameters(), lr=1e-4)

    GAMMA = 0.99 
    BATCH_SIZE = 16
    memory_replay = Replay_Buffer(10000) 
    episode_reward = 0
    episodes = 1000 
    target_updates = 10  
    render = False
    train(dqn_agent, episodes, optimizer, target_updates, BATCH_SIZE, GAMMA, render)
    