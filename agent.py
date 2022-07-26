import numpy as np
import paddle.fluid as fluid
import parl
import paddle
from parl.utils import logger


class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 n,
                 l,
                 e_greed=0.1,
                 e_greed_decrement=0):
        assert isinstance(n, int)
        assert isinstance(l, int)
        super(Agent, self).__init__(algorithm)
        self.n = n
        self.l = l
        self.global_step = 0
        self.update_target_steps = 200

        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement

    def predict(self, obs):
        obs = paddle.to_tensor(obs, dtype='float32')  # CHW
        return self.alg.predict(obs)

    def learn(self, obs, act, reward, next_obs, terminal):
        # 调用learn_program
        # self.alg.learn(obs, act, reward, next_obs, done)
        # 定期同步target_Q参数
        if self.global_step % self.update_target_steps == 0:
                self.sync_target()

        self.global_step += 1

        # act = np.expand_dims(act, -1)
        # reward = np.expand_dims(reward, -1)
        # terminal = np.expand_dims(terminal, -1)
        #
        # obs = paddle.to_tensor(obs, dtype='float32')
        # act = paddle.to_tensor(act, dtype='int32')
        # reward = paddle.to_tensor(reward, dtype='float32')
        # next_obs = paddle.to_tensor(next_obs, dtype='float32')
        # terminal = paddle.to_tensor(terminal, dtype='float32')

        loss = self.alg.learn(obs, act, reward, next_obs, terminal)
        return loss.numpy()[0]

    def sync_target(self):
        self.alg.sync_target()
