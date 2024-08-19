import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import torch.nn.functional as F

class PolicyNet(nn.Module):

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
        action_stddevs = F.softplus(self.policy_stddev_net(shared_features))
        

        return action_means,action_stddevs

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    


class ActorCritic:
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        self.actor_lr = 1e-4
        self.critic_lr = 1e-2

        self.gamma = 0.98  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards
        self.states = []
        self.next_states = []

        self.actor_net = PolicyNet(obs_space_dims, action_space_dims)
        # self.actor_net.load_state_dict(torch.load("actor_latest.pth",weights_only=True))

        self.critic_net = ValueNet(obs_space_dims, hidden_dim=128)
        # self.critic_net.load_state_dict(torch.load("critic_latest.pth",weights_only=True))

        self.actor_optimizer = torch.optim.AdamW(self.actor_net.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.critic_net.parameters(), lr=self.critic_lr)
    
    def sample_action(self,state):
        state = torch.tensor(state)
        action_means, action_stddevs = self.actor_net(state)

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
        states = torch.tensor(np.array(self.states),dtype=torch.float)
        next_states = torch.tensor(np.array(self.next_states),dtype=torch.float)
        rewards = torch.tensor(np.array(self.rewards),dtype=torch.float).view(-1, 1)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic_net(next_states)
        td_delta = td_target - self.critic_net(states)  # 时序差分误差

        actor_loss = 0
        for log_prob, delta in zip(self.probs, td_delta.detach()):
            actor_loss += log_prob * delta * (-1)
        actor_loss /= len(self.probs)
        
        # 均方误差损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic_net(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
    
        self.probs = []
        self.rewards = []
        self.states = []
        self.next_states = []