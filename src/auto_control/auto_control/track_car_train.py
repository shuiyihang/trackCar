# TODO 参考github https://github.com/kartben/navigation2/blob/85a2ac17321b53ddd276eb3f9f4bdbf7be69de13/nav2_experimental/nav2_rl/nav2_turtlebot3_rl/random_crawl_train.py
# 

from .trackCar_env import TrackCarEnv
from .reinforce import Reinforce
from .Actor_Critic import ActorCritic

import rclpy
import numpy as np
import torch
import matplotlib.pyplot as plt

total_num_episodes = int(1e3)
rewards_list = []
best_reward = 0

def train_Model(env:TrackCarEnv,agent:Reinforce):
    global rewards_list, best_reward, total_num_episodes
    print("模型训练开始...")
    for episode in range(total_num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done and rclpy.ok():
        
            action = agent.sample_action(state)
            next_state,reward,done = env.step(action)

            # 记录奖励
            agent.rewards.append(reward)
            # 记录状态
            agent.states.append(state)
            agent.next_states.append(next_state)

            episode_reward += reward
            state = next_state
        
        agent.update()
        rewards_list.append(episode_reward)

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.actor_net.state_dict(),'actor_best.pth')
            torch.save(agent.critic_net.state_dict(),'critic_best.pth')
        if episode % 10 == 0:
            print("episode: %d avg reward %.3f" %(episode/10,np.mean(rewards_list[-10:])))
    
    # 保存模型
    print("save module")
    torch.save(agent.actor_net.state_dict(),'actor_latest.pth')
    torch.save(agent.critic_net.state_dict(),'critic_latest.pth')

def main(args=None):
    rclpy.init(args=args)
    env = TrackCarEnv()

    # 状态空间3 动作空间2
    agent = ActorCritic(5,1)

    try:
        train_Model(env,agent)
    except:
        print("save module to checkpoint")
        torch.save(agent.actor_net.state_dict(),'actor_latest.pth')
        torch.save(agent.critic_net.state_dict(),'critic_latest.pth')
    finally:
        episodes_list = list(range(len(rewards_list)))
        plt.plot(episodes_list, rewards_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('REINFORCE on {}'.format("ros2_trackCar"))
        plt.show()
        
        rclpy.shutdown()
        env.cleanup()

                