from .trackCar_env import TrackCarEnv
# from .reinforce import Reinforce
from .Actor_Critic import ActorCritic
from .SAC import SAC

import rclpy
import numpy as np
from time import sleep

import torch

def rl_net_test(env):
    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    hidden_dim = 64
    gamma = 0.99
    tau = 0.005  # 软更新参数
    target_entropy = -0.1
    # 角速度限制值 1 rad/s
    action_bound = 0.5

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 状态空间5 动作空间1
    agent = SAC(3,hidden_dim,1,action_bound,actor_lr,critic_lr,alpha_lr,target_entropy,tau,gamma,device)

    agent.actor.load_state_dict(torch.load("./checkpoint/actor_latest.pth",weights_only=True))

    state = env.reset()
    done = False
    episode_reward = 0
    episode = 1
    print("开始测试...")

    while rclpy.ok():
        action = agent.take_action(state)
        state,reward,done = env.step(action)
        episode_reward += reward
        print("============episode {} reward: {}==========\n\n".format(episode,episode_reward))
        if done:
            state = env.reset()
            episode_reward = 0
            episode += 1

def main(args=None):
    rclpy.init(args=args)
    env = TrackCarEnv()

    try:
        rl_net_test(env)
    except:
        pass
    finally:
        rclpy.shutdown()