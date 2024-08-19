
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()

        self.action_bound = action_bound

        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            F.relu()
        )
        self.mu_net = nn.Sequential(
            nn.Linear(hidden_dim, action_dim)
        )
        self.std_net = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            F.softplus()
        )

    def forward(self, x):
        shared_features = self.shared_net(x)
        mu = self.mu_net(shared_features)
        std = self.std_net(shared_features)
        #
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob
    
class QValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            F.relu(),
            nn.Linear(hidden_dim, hidden_dim),
            F.relu(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        return self.net(cat)

class SAC:
    def __init__(self,state_dim, hidden_dim, action_dim,action_bound,
                 actor_lr,critic_lr,alpha_lr,
                 target_entropy,tau,gamma):
        # target_entropy:目标熵
        # tau:软更新参数
        # gamma 下一步奖励的折扣

        # 一个策略，两个评论家
        self.actor = PolicyNet(state_dim,hidden_dim,action_dim,action_bound)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        # Q网络
        self.critic_1 = QValueNet(state_dim,hidden_dim,action_dim)
        self.critic_2 = QValueNet(state_dim,hidden_dim,action_dim)

        # 目标Q网络
        self.tar_critic_1 = QValueNet(state_dim,hidden_dim,action_dim)
        self.tar_critic_2 = QValueNet(state_dim,hidden_dim,action_dim)

        self.tar_critic_1.load_state_dict(self.critic_1.state_dict())
        self.tar_critic_2.load_state_dict(self.critic_2.state_dict())

        # 优化器
        self.critic_1_optimizer = torch.optim.Adam(self.tar_critic_1.parameters(),lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.tar_critic_2.parameters(),lr=critic_lr)

        # alpha是熵正则项的系数，控制熵的重要程度
        alpha = 0.01
        self.log_alpha = torch.tensor(np.log(alpha),dtype=torch.float32)
        self.log_alpha.requires_grad = True

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr=alpha_lr)

        self.target_entropy = target_entropy
        self.tau = tau
        self.gamma = gamma
    
    def take_action(self,state):
        state = torch.tensor(np.array(state),dtype=torch.float32)
        actions = self.actor(state)[0]

        actions = np.array([action.item() for action in actions])
        return actions
    
    def calc_target(self,rewards,next_states):
        next_actions,log_prob = self.actor(next_states)
        # 熵H
        entropy = -log_prob
        # 由目标Q网络打分
        # 下一个状态，下一个动作
        q1_value = self.tar_critic_1(next_states,next_actions)
        q2_value = self.tar_critic_2(next_states,next_actions)

        # 状态价值 = 动作价值+H
        next_value = torch.min(q1_value,q2_value) + self.log_alpha.exp() * entropy

        # 时序差分结算
        td_target = rewards + self.gamma * next_value
        return td_target
    
    # 更新目标网络
    def soft_update(self):
        for param_tar,param in zip(self.tar_critic_1.parameters(),self.critic_1.parameters()):
            param_tar.data.copy_(param.data*self.tau + param_tar.data*(1.0-self.tau))

        for param_tar,param in zip(self.tar_critic_2.parameters(),self.critic_2.parameters()):
            param_tar.data.copy_(param.data*self.tau + param_tar.data*(1.0-self.tau))

    
    def update(self,obs_dict):
        states = torch.tensor(obs_dict['states'],dtype=torch.float)
        next_states = torch.tensor(obs_dict['next_states'],dtype=torch.float)
        actions = torch.tensor(obs_dict['actions'],dtype=torch.float).view(-1,1)
        rewards = torch.tensor(obs_dict['rewards'],dtype=torch.float).view(-1,1)

        # 更新Q网络

        # Q网络的损失函数 网络输出和目标值的均方误差
        td_target = self.calc_target(rewards,next_states)
        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states,actions),td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states,actions),td_target.detach()))

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()


        # 更新策略
        new_actions,log_prob = self.actor(states)
        # H
        entropy = -log_prob
        # 评价当前状态下采取的动作
        q1_value = self.critic_1(states,new_actions)
        q2_value = self.critic_2(states,new_actions)

        actor_loss = torch.mean(self.log_alpha.exp()*log_prob - torch.min(q1_value,q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update()
