# TODO 参考github https://github.com/kartben/navigation2/blob/85a2ac17321b53ddd276eb3f9f4bdbf7be69de13/nav2_experimental/nav2_rl/nav2_turtlebot3_rl/random_crawl_train.py
# 

from .trackCar_env import TrackCarEnv
from .reinforce import Reinforce
from .Actor_Critic import ActorCritic
from .SAC import SAC
import collections
import random


import rclpy
import numpy as np
import torch
import matplotlib.pyplot as plt
import time


import logging

rewards_list = []
best_reward = 0

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)



def train_Model(env:TrackCarEnv,agent:SAC,replay_buffer:ReplayBuffer,minimal_size=1000,batch_size=64,num_episodes=500):
    global rewards_list, best_reward
    print("模型启动...")
    real_train = False

    prefix_best = 'checkpoint/' + 'actor_best'
    prefix_latest = 'checkpoint/' + 'actor_latest'

    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename='app.log',
        filemode='w'
    )

    logger = logging.getLogger('train_log')


    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done and rclpy.ok():
            action = agent.take_action(state)
            next_state,reward,done = env.step(action)

            replay_buffer.add(state,action,reward,next_state,done)

            episode_reward += reward
            state = next_state
            if replay_buffer.size() > minimal_size:
                if real_train == False:
                    real_train = True
                    print("正式训练开始...")
                
                b_s,b_a,b_r,b_ns,b_d = replay_buffer.sample(batch_size)
                obs_dict = {'states':b_s,'next_states':b_ns,'actions':b_a,'rewards':b_r,'dones':b_d}
                agent.update(obs_dict)
        
        logger.info("============last total reward: {}==========\n\n".format(episode_reward))
        rewards_list.append(episode_reward)

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.actor.state_dict(),prefix_best + '.pth')
        if episode % 10 == 0:
            print("episode: %d avg reward %.3f" %(episode/10,np.mean(rewards_list[-10:])))
    
    # 保存模型
    print("save module")
    torch.save(agent.actor.state_dict(),time.strftime(prefix_latest + '_%m_%d_%H:%M.pth'))

def main(args=None):
    rclpy.init(args=args)
    env = TrackCarEnv()
    num_episodes = int(1e3)
    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    hidden_dim = 64
    gamma = 0.99
    tau = 0.005  # 软更新参数
    buffer_size = 100000
    minimal_size = 2000
    batch_size = 128
    target_entropy = -2
    # 角速度限制值 0.5 rad/s
    action_bound = 0.5
    prefix_latest = 'checkpoint/' + 'actor_latest'

    replay_buffer = ReplayBuffer(buffer_size)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 状态空间5 动作空间1
    agent = SAC(3,hidden_dim,1,action_bound,actor_lr,critic_lr,alpha_lr,target_entropy,tau,gamma,device)

    try:
        train_Model(env,agent,replay_buffer,minimal_size,batch_size,num_episodes)
    except:
        print("save module to checkpoint")
        torch.save(agent.actor.state_dict(),time.strftime(prefix_latest + '_%m_%d_%H:%M.pth'))
    finally:
        episodes_list = list(range(len(rewards_list)))
        plt.plot(episodes_list, rewards_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('SAC on {}'.format("ros2_trackCar"))
        plt.show()
        
        rclpy.shutdown()
        env.cleanup()

                