from .trackCar_env import TrackCarEnv
from .reinforce import Reinforce

import rclpy
import numpy as np

import torch

def rl_net_test(env):
    # 状态空间3 动作空间2
    agent = Reinforce(7,2)

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