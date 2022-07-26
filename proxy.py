import numpy as np
import paddle
import os
from parl.utils import logger  # 日志打印工具


class Proxy:
    def __init__(self,
                 agents,
                 n,
                 l,
                 e_greed=0.1,
                 e_greed_decrement=0):

        self.agents = agents
        self.n = n
        self.l = l
        self.global_step = 0
        self.update_target_steps = 200

        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement

        self.load_models()  # 为每一个helper载入之前训练的模型

    # 选择动作
    def sample(self, obs):
        acts = []
        if np.random.uniform(0, 1) < 1.0 - self.e_greed:
            acts = self.predict(obs)
        else:
            targetNodes = []
            for i in range(0, self.n):
                targetNodes.append(i)

            for i in range(0, len(self.agents)):
                j = np.random.choice(targetNodes)
                targetNodes.remove(j)
                di = np.random.choice(self.l)
                acts.append((i, di, j))

        # 随着训练逐步收敛，探索的程度慢慢降低
        # print(acts)
        self.e_greed = max(0.01, self.e_greed - self.e_greed_decrement)
        return acts

    # 利用DQN预测下一动作
    def predict(self, obs):
        acts = []
        for i in range(0, len(self.agents)):
            obs = paddle.to_tensor(obs, dtype='float32')  # CHW
            di, j = self.agents[i].predict(obs)

            acts.append((i, di, j))

        return acts

    # 利用经验数据让各个helperNode学习
    def learn(self, obs, act, reward, next_obs, terminal):

        k = 0
        actions = []    # 分别记录各个节点(0, 1, 2, 3..., N-1)的动作

        for i in range(0, len(self.agents)):
            actions.append([])

        while k < len(act):
            for a in act[k]:
                i, di, j = a
                actions[i].append(j * self.l + di)

            k += 1

        i = 0
        # act = np.expand_dims(actions[i], -1)
        reward = np.expand_dims(reward, -1)
        terminal = np.expand_dims(terminal, -1)

        obs = paddle.to_tensor(obs, dtype='float32')
        # act = paddle.to_tensor(act, dtype='int32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')

        while i < len(self.agents):

            act = np.expand_dims(actions[i], -1)
            act = paddle.to_tensor(act, dtype='int32')
            loss = self.agents[i].learn(obs, act, reward, next_obs, terminal)

            # logger.info('Node {} loss= {}'.format(i, loss))

            i += 1

    # 为每一个helperNode(0,...,n-2)保存DQN模型
    def save_models(self):
        for i in range(0, len(self.agents)):
            save_path = './ec_model_node_' + str(i) + '.ckpt'
            self.agents[i].save(save_path)

    def load_models(self):
        for i in range(0, len(self.agents)):
            save_path = './ec_model_node_' + str(i) + '.ckpt'
            if os.path.exists(save_path):
                self.agents[i].restore(save_path)
