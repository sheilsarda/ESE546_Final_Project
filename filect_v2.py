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
import datetime
import os
import pandas as pd 
from copy import deepcopy
 
def save_model(model):
    out_dir = 'saved'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    checkpoint = {'net': model.state_dict()}
    torch.save(checkpoint, 'saved/checkpoint' + "DQN_Model" + str(datetime.datetime.now().strftime("%Y-%m-%d %H%M%S")).strip() +'.pt')

def load_model(checkpoint):
    return torch.load(checkpoint)

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

def train(env, dqn, episodes, optimizer, target_updates, batch_size, gamma, render, save, k_iters, meta):
    
    for ep in range(episodes): 
        state = env.reset()
        score = 0
        done = False
        count = 0
        while not done:  
            if render: env.render()

            # save state

            tensor_state = torch.FloatTensor(state).unsqueeze(0)
            action = dqn.policy_net.select_action(tensor_state)
            next_state, reward, done, _ = env.step(action)
            score += reward            
            memory_replay.push(state, next_state, action, reward, done)
            if (ep % target_updates == 0): 
                dqn.target_net.load_state_dict(dqn.policy_net.state_dict())
   
            optimize_loss(dqn, optimizer, gamma, memory_replay) # if meta is true, then only do k-iters of ADAM
            state = next_state
            count += 1
            if meta and count > k_iters:
                break         
            
        #printing 
        if ep % 10 == 0: 
            if save: 
                save_model(dqn.policy_net)
            print('Episode {} | Total Episode score: {:.2f}\t'.format(ep, score)) 

class DQN_Agent():
    def __init__(self, state_dim, h_dim1, h_dim2, action_dim):
        self.policy_net = DQN_Network(state_dim, h_dim1, h_dim2, action_dim)
        self.target_net = DQN_Network(state_dim, h_dim1, h_dim2, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

def create_envs_g(): 
    envs = []
    env_orig = gym.make("CartPole-v0") 
    envs.append(env_orig)
    env_1 = gym.make("CartPole-v2")   
    envs.append(env_1)
    env_2 = gym.make("CartPole-v3")
    envs.append(env_2)
    env_3 = gym.make("CartPole-v4")
    envs.append(env_3)
    env_4 = gym.make("CartPole-v5")
    envs.append(env_4)
    
    # Hold out CartPole-v6
    
    return envs
    

if __name__ == "__main__": 
 
    #hyperparameters
    GAMMA = 0.99 
    BATCH_SIZE = 16
    memory_replay = Replay_Buffer(10000) 
    episodes = 500  
    target_updates = 10
    render = False 
    save = True  

    #SETUP NETWORK
    state_dim = 4
    action_dim = 2 
    h_dim1 = 64
    h_dim2 = 256
    dqn_agent = DQN_Agent(state_dim, h_dim1, h_dim2, action_dim)    

    # ______ REPTILE ______ # 
    #make environments
    envs_g = create_envs_g()   

    # all_params = torch.FloatTensor(np.random.normal(0.0, 1.0, int(1e6)))
    
    meta_step_size_final = .1
    meta_step_size = .1 
    k_iters = 10
    meta_iters = 5 
    goin_meta = True

    o_weights = None
    all_params = None

    for i in range(meta_iters): 
        # Update learning rate
        frac_done = i / meta_iters
        #convex combination of meta_step_size (start, final) --> curr
        cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size
        
        optimizer = torch.optim.Adam(dqn_agent.policy_net.parameters(), lr=1e-4)
                
        # Sample base task
        task_idx = random.choices(envs_g, k=1)   
        print(task_idx)
        o_weights = optimizer.state_dict() 
        single_task_params = dqn_agent.policy_net.parameters()
        
        # Update network
        train(task_idx[0], dqn_agent, episodes, optimizer, target_updates, BATCH_SIZE, GAMMA, render, save, k_iters, goin_meta)
        state = optimizer.state_dict()  
        
        if all_params == None:
            all_params = single_task_params # initialization
        else:
            updated_params = list(all_params)
            for i, (s, a) in enumerate(zip(single_task_params, all_params)):
                dif = s.data - a.data 
                updated_params[i] = a.data + cur_meta_step_size*(dif)
                
            all_params = updated_params

    print("---------Doing Validation--------")
    val_env= gym.make("CartPole-v6")
    print(val_env)
    render = True
    train(val_env, dqn_agent, episodes, optimizer, target_updates, BATCH_SIZE, GAMMA, render, save, 1, True) # one-shot learning
    
    
    """
    #load models back 
    PATH = "saved/" + str() + "checkpointDQN_Model2020-12-14 200727.pt" 
    PATH_2 = "saved/checkpointDQN_Model2020-12-14 200726.pt"
    
    for name, param in dqn_agent.policy_net.named_parameters():
        print(name)
        print(param)
    
    
    checkpoint = load_model(PATH)
    checkpoint_2 = load_model(PATH_2)
    
    def Merge(dict1, dict2):
        res = {**dict1, **dict2}
        return res
    
    check_merged = Merge(checkpoint, checkpoint_2)
    #print(check_merged)
    
    l1_w = checkpoint['net']['l1.weight']
    l2_w = checkpoint['net']['l2.weight']
    l3_w = checkpoint['net']['l3.weight'] 
    l3_b = checkpoint['net']['l3.weight']
    """