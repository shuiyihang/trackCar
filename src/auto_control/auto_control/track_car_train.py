# TODO 参考github https://github.com/kartben/navigation2/blob/85a2ac17321b53ddd276eb3f9f4bdbf7be69de13/nav2_experimental/nav2_rl/nav2_turtlebot3_rl/random_crawl_train.py
# 

from .trackCar_env import TrackCarEnv
from .reinforce import Reinforce

import rclpy
import numpy as np

import torch

total_num_episodes = int(1e3)

def train_Model(env:TrackCarEnv,agent:Reinforce):
    rewards_list = []
    
    print("模型训练开始...")
    for episode in range(total_num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done and rclpy.ok():
            action = agent.sample_action(state)
            state,reward,done = env.step(action)
            agent.rewards.append(reward)
            episode_reward += reward
        
        agent.update()
        rewards_list.append(episode_reward)
        if episode % 10 == 0:
            print("episode: %d avg reward %.3f" %(episode/10,np.mean(rewards_list[-10:])))
    
    # 保存模型
    print("save module to reinforce_track_car")
    torch.save(agent.net.state_dict(),'reinforce_track_car.pth')

def main(args=None):
    rclpy.init(args=args)
    env = TrackCarEnv()

    # 状态空间3 动作空间2
    agent = Reinforce(5,1)

    try:
        train_Model(env,agent)
    except:
        print("save module to checkpoint")
        torch.save(agent.net.state_dict(),'checkpoint.pth')
    finally:
        rclpy.shutdown()
        env.cleanup()
        