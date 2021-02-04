import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#loss that we could save
LOSS = []

#parameters
numParticles = 1
phi = 0

save_dir = "data/"
weight_save_file = "weights.pkl" #None to not save, or file name to save
weight_load_file = None #None to not load a file

#weight_load_file = weight_save_file


env_name = 'simple-v0'

if env_name in gym.envs.registration.registry.env_specs:
    del gym.envs.registration.registry.env_specs[env_name]

env = gym.make('microbots:' + env_name, 
               numParticles = numParticles, phi = phi, 
               stateBounds = [-10.0, 10.0, -5.0, 5.0],
               dwellTime = 1.0, maxSteps = 25)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Experience = namedtuple('Experience', 
                        ('state', 'action', 'next_state', 'reward'))



class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    @staticmethod
    def get_current(policy_net, states, actions):
        #return the predicted q values for the state-action pairs passed in
        return torch.squeeze(policy_net(states).gather(dim=1,index=actions.unsqueeze(-1)))
    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1) \
        .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values
    
    


class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        """saves a transition"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    


class DQN(nn.Module):

    def __init__(self, numParticles):
        super(DQN, self).__init__()
        #input fully connected layer (4 * numParticle states)
        self.fc1 = nn.Linear(in_features=4*numParticles, out_features=24)   
        #hidden layer
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        #output layer (outputs value of 4 actions)
        self.out = nn.Linear(in_features=32, out_features=4)

    # takes the current state as an input, yields the output vars
    def forward(self, x):
        x = F.relu(self.fc1(x)) #input -> hidden 1 as ReLU
        x = F.relu(self.fc2(x)) #hidden 1 -> output as ReLU
        x = self.out(x)
        return x
        
        
        
        
        
# Start of the actual code that we use to train

env.reset() #reset the environment (place a new particle)


#Learninig hyperparameters
BATCH_SIZE = 128 #how many memories to randomly sample
GAMMA = 0.999   #discount factor
EPS_START = 0.9 #start probability of random action
EPS_END = 0.05  #final probability of random action
EPS_DECAY = 200 #(exponential) rate of decay in epsilon
TARGET_UPDATE = 10 #how often to update the target network
LEARNING_RATE = 0.5 #learning rate of the optimizer

memory = ReplayMemory(10000)


policy_net = DQN(numParticles).to(device)
target_net = DQN(numParticles).to(device)


if weight_load_file != None:
    policy_net = torch.load(save_dir + weight_load_file)
    policy_net.eval() # eval "to set dropout and batch normalization leyers"


target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(params = policy_net.parameters(), lr=LEARNING_RATE)


steps_done = 0
### This defines a handful of functions, proceed further
#define the epsilon-greedy action selection function
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            #get the action which maximizes our predicted reward
            return policy_net( torch.from_numpy(state) ).argmax()
    else:
        return torch.tensor(random.randint(0, 3), device=device, dtype=torch.long)
        

        

def extract_tensors(experiences):
    #convert a batch of experiences to an experience of batches
    batch = Experience(*zip(*experiences))
    
        
    t1 = torch.squeeze(torch.tensor( np.stack(batch.state) ) )
    t2 = torch.squeeze(torch.stack( batch.action ) )
    t3 = torch.squeeze(torch.tensor( np.stack(batch.reward) ) )
    t4 = torch.squeeze(torch.tensor( np.stack(batch.next_state) ) )

    return (t1,t2,t3,t4)

    
def optimize_model():
    global LOSS #loss storage var
    if len(memory) < BATCH_SIZE: #check if we can sample memory yet
        return
    experiences = memory.sample(BATCH_SIZE)
    states, actions, rewards, next_states = extract_tensors(experiences)
    
    
    current_q_values = QValues.get_current(policy_net, states, actions)
    next_q_values = QValues.get_next(target_net, next_states)
    
   
    target_q_values = (next_q_values * GAMMA) + rewards

    
    loss = F.mse_loss(current_q_values, target_q_values)      
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print loss value (hints at when we've converged)
    LOSS.append(loss.item())
    
    
### change this to 'live' for plots or 'file' to save a CSV
#render_mode = 'None' --- shows nothing
#render_mode = 'live' --- shows plot
render_mode = 'None'
num_episodes = 10
episode_durations = [] #store how long episodes are lasting
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    env.render(mode=render_mode)
    state = env._next_observation()
    for t in count():
        # Select and perform an action
        action = select_action(state)
        
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        next_state = env._next_observation()

        # Store the experience in memory
        if not done:
            memory.push(state, action, next_state, reward)
            optimize_model()

        state = next_state

        env.render(mode=render_mode)

        # Perform one step of the optimization (on the target network)
        if done:
            episode_durations.append(t)
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
                print(i_episode / num_episodes * 100, "%")
            break




if weight_save_file != None:
    torch.save(policy_net, save_dir + weight_save_file)


#if in live mode, the process blocks until the window is closed
print("The End")
env.close()



with open(save_dir + 'loss.csv', 'w') as f:
    for item in LOSS:
        f.write("%s\n" % item)
     

#show the loss plot at the very end
fig = plt.figure()
fig.suptitle("Loss")
plt.plot(LOSS)
plt.show(block=True)        

        
        
        
        
        
        
        
        
        
        
        