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
BATCH_SIZE = 16

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
        self.l2  = nn.Linear(self.hidden_dim1, self.hidden_dim2) 
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
        x = x.squeeze(0) 
        print("x shape", x)

        x = x.view(4) 
        print("x shape", x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = x.view(x.size(0), -1)
        return x
    
class DQNAgent(): 
    def __init__(self, input_dim, action_dim): 
        self.action_dim = action_dim
        self.target_net = DQN_Network(input_dim, 128, 128, self.action_dim)
        self.policy_net = DQN_Network(input_dim, 128, 128, self.action_dim)  

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
            rando = random.randint(0, 1)
            action = torch.tensor([rando]) 
        
        else: 
        #take the best action  
            with torch.no_grad(): 
                # print("State:", state)
                actions = self.policy_net(state) 
                # print("actions.size(): ") 
                # print(actions.size())
                action = torch.argmax(self.policy_net(state)).view(1,1)
                #action = .max(1)[1].view(1, 1)
                
        return action.item() 

"""
def compute_loss(memory, model, optimizer): 
    
    # do nothing if we have not collected enough samples 
    if (len(memory.storage) < BATCH_SIZE): 
        return 
    
    mini_batch = memory.sample(BATCH_SIZE) 
    
    state_action_vals = [] 
    expected_state_action_vals = [] 
    
    for state, action, next_state, reward, done in mini_batch:
        if done:
            reward = torch.tensor(reward)
            target = reward 
        else: 
            next_state = torch.FloatTensor((next_state)) 
            next_state_values = dqn.target_net(next_state) 
            next_state_values = torch.FloatTensor(next_state_values)
            target = reward + gamma * torch.max(next_state_values)

        expected_state_action_vals.append(target) 
        q_values = dqn.policy_net(torch.FloatTensor(state))
        q_values = q_values.view(2, 1)
        state_action_val = q_values[action]
        state_action_vals.append(state_action_val)

    expected_state_action_vals = torch.FloatTensor(expected_state_action_vals).requires_grad_(True)
    state_action_vals = torch.FloatTensor(state_action_vals) 
    expected_state_action_vals.requires_grad = True  
    state_action_vals.requires_grad = True  
            
    # compute Huber loss
    loss = F.smooth_l1_loss(state_action_vals, state_action_vals) 
    ## gradient update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss 
""" 
def compute_loss(memory, model, agent, optimizer):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # print("State ", batch.state)
    state = list(batch.state)
    next_state = list(batch.next_state)

    # print("STATE: ", len(state))
    # print("NEXT_STATE: ", len(next_state))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)  
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, torch.Tensor(next_state))), dtype=torch.bool)
    non_final_next_states = torch.cat([torch.Tensor(s) for s in next_state
                                                if s is not None])
    state_batch = torch.cat(state)
    # print("Action ", len(batch.action)) 
    action_batch = []
    for s in batch.action:
        action_batch.append(s)
    
    action_batch = torch.Tensor(action_batch)
    #print("Action ", action_batch.size()) 
    # print("Reward ", batch.reward)  
    reward_batch = torch.cat([torch.from_numpy(np.array(s)).view(1, 1) for s in batch.reward
                                                if s is not None])

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net 
    #print("Obs:", state_batch.size()) 
    #print(action_batch.size()) 
    
    
    # print("State Batch  ", state_batch.view(BATCH_SIZE, 4) )
    # print("Model        ", model(state_batch))
    # print("Action Batch", action_batch)

    ix = torch.LongTensor([[action_batch[i] for i in range(len(action_batch))]])
    # print("IX", ix)

    state_action_values = torch.gather(model(state_batch), 1, ix)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = agent.target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * .99) + reward_batch

    # Compute Huber loss
    loss = F.mse_loss(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def preprocess_state(state):
    state = torch.Tensor(state).view(1, 4)
    return state

def train(dqn, episodes, optimizer, target_update, gamma, env, memory, render):

    '''
    Params: dqn, episodes, optimizer, target_update, gamma, env, memory, render
    '''

    running_reward = 0  
    rewards_list = [] 
    loss_list = []
    
    log_interval = 10
    for ep in range(episodes):  
        state = env.reset()
        score = 0 
        done = False  
        counter = 0
        
        while not done: 
            if (render):
                env.render()
            
            state = preprocess_state(state)
            action = dqn.select_action(state) 
            # print("Action", action)
            next_state, reward, done, _ = env.step(action) 
            memory.storage.append((state, action, next_state, reward, done))
            score += reward 
            rewards_list.append(reward)
            state = next_state 
            
            counter += 1
        
            # print("Counter {}".format(counter)) 
        #compute running_reward here
        running_reward = running_reward * (1 - 1/log_interval) + reward * (1/log_interval) 
        loss = compute_loss(memory, dqn.policy_net, dqn, optimizer)
        loss_list.append(loss)
        
        if (ep % target_update == 0): 
            dqn.target_net.load_state_dict(dqn.policy_net.state_dict()) 

        # do some printing here
        print("Episode: [{}/{}], Score: {}".format(ep, episodes, score)) 
        
    return rewards_list
    
#testing purposes 
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#print(torch.cuda.get_device_name(0))
 
np.random.seed(30)
torch.random.manual_seed(30)

env_name = "CartPole-v0"
env = gym.make(env_name)
env.reset() 
action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]

dqn = DQNAgent(state_dim,  action_dim) 

render = False 
gamma = .99
replay = Replay_Buffer(1000)
episodes = 1000 
optimizer = optim.RMSprop(dqn.policy_net.parameters())
# optimizer = torch.optim.Adam(dqn.policy_net.parameters(), lr=1e-4)

train(dqn, episodes, optimizer, 10, gamma, env, replay, render) 