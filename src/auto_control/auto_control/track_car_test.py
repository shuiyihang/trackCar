from .trackCar_env import TrackCarEnv
# from .reinforce import Reinforce
from .Actor_Critic import ActorCritic

import rclpy
import numpy as np

import torch

def rl_net_test(env):
    # 状态空间3 动作空间2
    agent = ActorCritic(5,1)
    agent.actor_net.load_state_dict(torch.load("actor_latest.pth",weights_only=True))
    agent.critic_net.load_state_dict(torch.load("critic_latest.pth",weights_only=True))

    state = env.reset()
    done = False

    while rclpy.ok():
        action = agent.sample_action(state)
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