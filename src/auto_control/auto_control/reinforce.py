

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

class PolicyNetWork(nn.Module):

    def __init__(self,obs_space_dim:int,action_space_dim:int):
        super().__init__()
        
        hidden_space_1 = 32
        hidden_space_2 = 64

        # 共享网络参数
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dim,hidden_space_1),
            nn.Tanh(),
            nn.Linear(hidden_space_1,hidden_space_2),
            nn.Tanh()
        )

        # 均值
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space_2,action_space_dim)
        )

        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space_2,action_space_dim)
        )
    
    def forward(self,x):
        shared_features = self.shared_net(x.float())
        action_means = self.policy_mean_net(shared_features)
        # 确保标准差是一个正数
        action_stddevs = torch.log(1 + torch.exp(self.policy_stddev_net(shared_features)))

        return action_means,action_stddevs


class Reinforce:
    def __init__(self, obs_space_dims: int, action_space_dims: int):

        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.9  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = PolicyNetWork(obs_space_dims, action_space_dims)
        self.net.load_state_dict(torch.load("checkpoint.pth",weights_only=True))

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
    
    def sample_action(self,state):
        state = torch.tensor(state)
        action_means, action_stddevs = self.net(state)

        actions = []
        log_probs = []

        for mean, stddev in zip(action_means, action_stddevs):
            distrib = Normal(mean + self.eps, stddev + self.eps)
            action = distrib.sample()
            log_prob = distrib.log_prob(action)

            # Append sampled action and its log probability
            actions.append(action)
            log_probs.append(log_prob)
        
        actions = np.array([action.item() for action in actions])
        # print("actions:%.3f" %(actions[0]))
        # 概率之积->log之和
        self.probs.append(torch.sum(torch.stack(log_probs)))

        return actions
    
    def update(self):
        running_g = 0
        gs = []
        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []