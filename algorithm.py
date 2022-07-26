import copy
import parl
import paddle
import paddle.nn.functional as F

from parl.utils import logger


class DQN(parl.Algorithm):
    def __init__(self, model, n, l, gamma=None, lr=None):
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.gamma = gamma
        self.lr = lr
        self.n = n
        self.l = l
        self.act_dim = n * l

        self.mse_loss = paddle.nn.MSELoss(reduction='mean')
        self.optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=self.model.parameters())

    def predict(self, obs):
        predict_q = self.model.value(obs)
        act = predict_q.argmax().numpy()[0]
        j = int(act / self.l)
        di = int(act % self.l)
        # logger.info('obs:{}		predict_q:{}	act:{}  j:{}    di:{}'.format(obs, predict_q, act, j, di))

        return di, j

    def learn(self, obs, act, reward, next_obs, terminal):
        # next_pred_value = self.target_model.value(next_obs)
        # best_v = layers.reduce_max(next_pred_value, dim=1)
        # best_v.stop_gradient = True 		# 阻止参数传递，因为target_Q不能变
        # done = layers.cast(done, dtype='float32')
        # # target_Q = reward + (1.0 - done) * self.gamma * np.max(self.target_Q[next_obs, :])
        # target_Q = reward + (1.0 - done) * self.gamma * best_v

        # 过滤act,只保留与自身有关的

        # 1.计算target_Ql
        with paddle.no_grad():
            max_v = self.target_model.value(next_obs).max(1, keepdim=True)  # 寻找第2维数据的最大值
            target_Q = reward + (1.0 - terminal) * self.gamma * max_v

        # 2.计算预测Q
        pred_values = self.model.value(obs)
        # 如果最后一个维度==1，去掉；比如，act.shape = (3, 1)， 使用后，变成(3)
        act = paddle.squeeze(act, axis=-1)
        # logger.info('pred_values:{}    action:{}   target_Q:{}'.format(pred_values, act, target_Q))

        action_onehot = F.one_hot(act, num_classes=self.act_dim)
        pred_value = pred_values * action_onehot
        pred_Q = paddle.sum(pred_value, axis=1, keepdim=True)

        # 3.计算loss并更新网络
        loss = self.mse_loss(pred_Q, target_Q)
        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    # 将model的参数拷贝到target_model
    def sync_target(self):
        self.model.sync_weights_to(self.target_model)
