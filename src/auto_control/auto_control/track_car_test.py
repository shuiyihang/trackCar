from .trackCar_env import TrackCarEnv
# from .reinforce import Reinforce
from .Actor_Critic import ActorCritic
from .SAC import SAC

import rclpy
import numpy as np

import torch

def rl_net_test(env):
    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    hidden_dim = 128
    gamma = 0.99
    tau = 0.005  # 软更新参数
    target_entropy = -0.1
    # 角速度限制值 1 rad/s
    action_bound = 1

    # 状态空间5 动作空间1
    agent = SAC(5,hidden_dim,1,action_bound,actor_lr,critic_lr,alpha_lr,target_entropy,tau,gamma)

    agent.actor.load_state_dict(torch.load("actor_best.pth",weights_only=True))

    state = env.reset()
    done = False

    while rclpy.ok():
        action = agent.take_action(state)
        state,reward,done = env.step(action)
        if done:
            state = env.reset()

def main(args=None):
    rclpy.init(args=args)
    env = TrackCarEnv()

    try:
        rl_net_test(env)
    except:
        pass
    finally:
        rclpy.shutdown()