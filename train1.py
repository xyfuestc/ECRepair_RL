from agent import Agent
from algorithm import DQN
from parl.utils import logger
from model import Model
from replay_memory import ReplayMemory
import numpy as np
import gym
from ECGym import ECRepairEnv
from proxy import Proxy
import time
MAX_EPISODE = 2000
GAMMA = 0.9
LEARNING_RATE = 0.01
E_GREED_DECREMENT = 1e-8
MAX_RELAY_MEMORY_SIZE = 20000
RPM_WARMUP_SIZE = 200
RPM_BARCH_SIZE = 32


def run_episode(env, proxy, rpm, render=False):
    obs = env.reset()
    cur_episode_reward = 0
    step = 0
    # print('obs: ', obs.reshape(5, 3))
    while True:
        acts = proxy.sample(obs)  # 需要（n-1）个三元组

        # print('origin: ', acts)
        next_obs, reward, done, next_obs_2d = env.step(acts)

        cur_episode_reward += reward
        step += 1

        rpm.append((obs, acts, reward, next_obs, done))
        if len(rpm) > RPM_WARMUP_SIZE and step % 5 == 0:
            (batch_obs, batch_act, batch_reward, batch_next_obs, batch_terminal) = rpm.sample(RPM_BARCH_SIZE)
            proxy.learn(batch_obs, batch_act, batch_reward, batch_next_obs, batch_terminal)

        # if render:
            # logger.info('obs:{}'.format(next_obs))
        if done:
            # logger.info('obs:{}'.format(next_obs_2d))
            break
        # 更新
        obs = next_obs

    return step, cur_episode_reward


def test_episode(env, proxy, Render=False):
    obs = env.reset()
    test_rewards = []
    cur_episode_reward = 0
    step = 0

    while True:
        act = proxy.predict(obs)
        next_obs, reward, done, _ = env.step(act)
        cur_episode_reward += reward

        step += 1

        if Render:
            env.render()

        if done:
            test_rewards.append(cur_episode_reward)
            break

        if (obs == next_obs).all():
            time.sleep(0.5)
            logger.info('我不会了!我走了{}步走到:{}'.format(step, obs))
            return 0

        obs = next_obs

    # 求平均reward
    return np.mean(test_rewards)


def main():
    # 配置N-节点数，K-数据节点数，L-条带数
    n, k, l = 5, 3, 3
    env = ECRepairEnv(n, k, l)  # N = 5, K = 3, L = 3
    #  act_dim = env.action_space.n
    # obs_dim = env.observation_space.shape[0]  # CartPole-v0: (4,)
    models, algorithms, agents = [], [], []
    for i in range(0, n-1):
        models.append(Model(n*l, n*l))
        algorithms.append(DQN(models[i], n, l, gamma=GAMMA, lr=LEARNING_RATE))
        agents.append(Agent(algorithms[i], n, l, e_greed=0.1, e_greed_decrement=E_GREED_DECREMENT))

    rpm = ReplayMemory(n, l, MAX_RELAY_MEMORY_SIZE)
    # 申明一个总代理,负责指挥各个helper节点训练模型,产生动作
    proxy = Proxy(agents, n, l, e_greed=0.1, e_greed_decrement=E_GREED_DECREMENT)
    episode = 0

    while episode < MAX_EPISODE:
        for j in range(0, 50):
            step, reward = run_episode(env, proxy, rpm, False)
            episode += 1

        logger.info('training episode:{}		step:{}		Test reward:{}'.format(episode, step, reward))

        eval_reward = test_episode(env, proxy, Render=True)
        logger.info('episode:{}		e_greed:{}		Test reward:{}'.format(episode, proxy.e_greed, eval_reward))
        proxy.save_models()


if __name__ == '__main__':
    main()
