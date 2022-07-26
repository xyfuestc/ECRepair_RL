import parl
import paddle.nn as nn
import paddle.nn.functional as F


class Model(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        hid1_size = 64
        hid2_size = 64
        # hid3_size = 128
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        # self.fc3 = nn.Linear(hd2_size, hid3_size)
        self.fc3 = nn.Linear(hid2_size, act_dim)

    def value(self, obs):
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        # h3 = F.relu(self.fc2(h2))
        Q = self.fc3(h2)
        return Q
