import random
import math 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from datetime import datetime

start_time = datetime.now()
#-----------------declaration of variables------------------------

SIZE = 8 
NUM_EPISODES = 1500
MOVE_PENALTY = -10
WALL_PENALTY = -300
REVISIT_PENALTY = -50
ARRIVE_REWARD = 200
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 5000
LR = 0.015
GAMMA = 0.8
BATCH_SIZE = 32

WALL_LOCATION = [(1,4), (2,4), (3,4), (4,4), (4,3), (4,2), (4,1)]
DESTINATION = (6,5)
#------------------------------------------------------------------


class DQNAgent:

    def __init__(self):
        self.model = nn.Sequential( 
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        ) 
        self.optimizer = optim.Adam(self.model.parameters(), LR) # neural network

        self.steps_done = 0 # for epsilon decay
        self.memory = deque(maxlen=10000)

        self.target_model = nn.Sequential( 
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        ) 
        self.step_count = 0
 

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state,
                            action,
                            torch.FloatTensor([reward]),
                            torch.FloatTensor([next_state]))) # store related data to memory

    

    def act(self, state):
        self.steps_done += 1
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY) # epsilon exponential decay
        if ep == NUM_EPISODES: # on the last episode
            eps_threshold = 0 # only use existing Q-network for the final result
        if random.random() > eps_threshold: # if epsilon is larger
            return self.model(state).data.max(1)[1].view(1, 1) # use existing Q-network
        else:
            return torch.LongTensor([[random.randrange(4)]]) # if not go random

    

    def learn(self):

        if len(self.memory) < BATCH_SIZE: # learn only if 
            return

        batch = random.sample(self.memory, BATCH_SIZE) 
        states, actions, rewards, next_states = zip(*batch)
        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)
    
       

        current_q = self.model(states).gather(1, actions)

        # Regular DQN
        Qsa_prime_target_values = self.target_model(next_states).detach()
        Qsa_prime_targets = Qsa_prime_target_values.max(1)[0]  
        expected_q = rewards + (GAMMA * Qsa_prime_targets) # main equation of Q-learning

        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step_count += 1
        
        if (self.step_count) > 49 :
            self.target_update(self.model, self.target_model)
            self.step_count = 0

    def target_update(self, model, target_model):

        for target_param, local_param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(local_param.data)        

class Maze:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.visited_cells = []

    def __str__(self):
        return f"{self.x},{self.y}"

    def action(self, act): # 4 actions, coordinate change for each action
        if act == 0:
            self.move(x = 0,y = -1)
        elif act == 1:
            self.move(x = 1, y = 0)
        elif act == 2:
            self.move(x = 0, y = 1)
        elif act == 3:
            self.move(x = -1, y = 0)


    def move(self,x,y):
        self.x += x
        self.y += y # coordinates after action

        if self.x < 0:
            self.x = 0 
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1 # attempts to move outside the grid doesn't count; thus result in a revisit penalty


    def get_reward(self, x, y):
        if (x,y) == DESTINATION:
            if ep == NUM_EPISODES: # on the last episode
                self.visited_cells.append(DESTINATION)
                print(self.visited_cells) # print list of visited cells as final route
            return ARRIVE_REWARD
        elif (x,y) in WALL_LOCATION:
            return WALL_PENALTY
        elif (x,y) in self.visited_cells:
            return REVISIT_PENALTY
        else:
            return MOVE_PENALTY

    def reset(self):
        self.x = 0
        self.y = 0
        self.visited_cells = []



agent = DQNAgent()


for ep in range(1, NUM_EPISODES+1):

    connect = Maze(1,1) # starting point is (1,1)
    episode_reward = 0

    while True:
        connect.visited_cells.append((connect.x,connect.y))
        state = torch.Tensor([[connect.x, connect.y]])
        act = agent.act(state) # which action is best
        connect.action(act) # take action
        reward = connect.get_reward(connect.x, connect.y) # get reward
        next_state = (connect.x, connect.y) 
        agent.memorize(state, act, reward, next_state) # store data to memory
        agent.learn()
        state = next_state # one step forward
        episode_reward += reward

        if reward == ARRIVE_REWARD: # end 
            if ep % 10 == 0:
                print(str(ep) + " : " + str(episode_reward))                
            break

        if episode_reward < -300000: # if one episode takes too long then force stop then proceed to next
            print("fail, " + str(ep) + " : " + str(episode_reward))
            break    

end_time = datetime.now()
elapsed_time = end_time - start_time
print("Elapsed time : " + str(elapsed_time))

f = open("D:/Archive/Yonsei/23-1/종설/log.txt",'a')
data = "Episodes: %d, Reward: %s, Route: %s, Time: %s \n" % (NUM_EPISODES, str(episode_reward), connect.visited_cells, str(elapsed_time))
f.write(data)
f.close()