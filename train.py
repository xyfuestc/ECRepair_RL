import gym
import parl
import paddle

import os
import numpy as np
from parl.utils import logger

from model import Model
from algorithm import DQN
from agent import Agent

from replay_memory import ReplayMemory

LEARN_FREQ = 5
MEMORY_SIZE = 20000
MEMORY_WARMUP_SIZE = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.99


def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))
        # 大于200条记录之后，每执行5步（向左向右移动一次），学习一次
        if (len(rpm) > MEMORY_WARMUP_SIZE and (step % LEARN_FREQ == 0)):
            (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)

        if done:
            break

        total_reward += reward
        obs = next_obs

    return step, total_reward


def test_episode(env, agent, render=False):
    eval_reward = []
    step = 0
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break

        eval_reward.append(episode_reward)

    return np.mean(eval_reward)


def main():
    env = gym.make('MountainCar-v0')
    act_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]  # CartPole-v0: (4,)

    rpm = ReplayMemory(MEMORY_SIZE)

    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    algorithm = DQN(model, act_dim=act_dim, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Agent(algorithm, act_dim=act_dim, e_greed=0.1, e_greed_decrement=1e-6)

    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_episode(env, agent, rpm)

    max_episode = 2000

    episode = 0
    # 每跑50个episode，就测试一次并显示动画
    while episode < max_episode:
        for i in range(0, 50):
            total_reward = run_episode(env, agent, rpm)
            episode += 1

        eval_reward = test_episode(env, agent, render=True)
        # logger.info('episode:{}		e_greed:{}		Test reward:{}'.format(episode, agent.e_greed, eval_reward))

    save_path = './mc_model.ckpt'
    agent.save(save_path)


if __name__ == '__main__':
    main()
