import random
import collections


class ReplayMemory(object):
    def __init__(self, n, l, max_size):
        self.buffer = collections.deque(maxlen = max_size)
        self.n = n
        self.l = l

    def append(self, exp):
        self.buffer.append(exp)

    def pop(self):
        # s = np.zeros(self.n * self.l, np.int32)
        # s_n = np.zeros(self.n * self.l, np.int32)
        # a = np.zeros(3 * (self.n-1), np.int32)
        exp = self.buffer.pop()
        # print(exp)

        s, a, r, s_n, done = exp
        # print(s, a, r, s_n, done)

        return s, a, r, s_n, done

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        for exp in mini_batch:
            s, a, r, s_n, done = exp
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_n)
            done_batch.append(done)
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

        # return np.array(obs_batch).astype('float32'), np.array(action_batch).astype('float32'), \
        #         np.array(reward_batch).astype('float32'), np.array(next_obs_batch).astype('float32'), \
                # np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)
