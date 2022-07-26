import numpy as np
import gym
import copy
from gym.spaces import MultiDiscrete

# system parameter stable


class ECRepairEnv(gym.Env):
    def __init__(self, N, K, L):
        self.action_space = MultiDiscrete(np.ones(N-1) * N * L)
        # self.action_space = gym.spaces.Discrete(N*L)
        self.N = N  # 节点数
        self.L = L  # 块数
        self.K = K  # 数据节点数
        # self.S = np.zeros(N*L).reshape(N, L)  #状态矩阵
        self.S = np.zeros((N, L), dtype=np.int32)
        self.sSet = []
        self.eSet = []

        low = np.zeros(N * L)
        high = np.ones(N * L) * (2 ^ (N - 1) - 1)  # 每个数据块最大值：2^N-1
        self.observation_space = gym.spaces.Box(low, high, dtype=np.int32)
        self.np_random = None
        self._initial()

    def _initial(self):
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._initial()
        i = 0
        # di = 2^i
        # S2 = self.S.reshape(self.N, self.L)
        for i in range(0, self.N - 1):
            self.S[i] = pow(2, i)
            # logger.info(' S[{}] = {} '.format(i, self.S[i]))

        # 替换节点DN数据为空
        self.S[i + 1] = 0
        self.sSet = []  # 已使用起点集
        self.eSet = []  # 已使用终点集

        return self._make_state()

    def getNKL(self):
        return self.N, self.K, self.L

    def _make_state(self):
        return self.S.ravel()  # 返回一维数组

    def getConflictIndexes(self, actList):
        conflictIndexes = []
        self.sSet = []
        self.eSet = []
        for act in actList:
            Di, di, Dj = act
            if Di == Dj:
                conflictIndexes.append(actList.index(act))
            elif (Di not in self.sSet) and (Dj not in self.eSet):
                self.sSet.append(Di)
                self.eSet.append(Dj)
            else:
                conflictIndexes.append(actList.index(act))

        return conflictIndexes

    def isDone(self):
        for v in self.S[self.N - 1]:
            if bin(v).count("1") < self.K:
                return False

        return True

    def render(self):
        print("current state: ", self.S)

    def step(self, act):

        actIDs = []

        for i in range(0, len(act)):
            act_id = act[i]
            actIDs.append((i, act_id % self.L, int(act_id / self.L)))

        print(actIDs)

        actions = copy.deepcopy(actIDs)


        # 冲突过滤
        conflictIndexes = self.getConflictIndexes(actions)
        for i in reversed(conflictIndexes):
            del actions[i]

        # print(action)
        # 拷贝一份状态,避免出现a->b同时b->c,出现c=a+b的情况,因为是同时进行,所以c只与之前的b有关
        s_copy = copy.deepcopy(self.S)

        reward = -1
        # 合并数据
        for action in actions:
            i, di, j = action
            self.S[i, di].astype(int)
            self.S[j, di].astype(int)
            oldValue = self.S[j, di]
            # 按位或
            s_copy[j, di] |= self.S[i, di]
            if self.S[j, di] == s_copy[j, di]:
                reward += -0.01

        self.S = s_copy


        state = self._make_state()
        done = self.isDone()

        if done:
            reward = 0


        # print('step: ', action)

        return state, reward, done, {}


if __name__ == '__main__':

    # np.set_printoptions(precision=3, suppress=True, linewidth=300, formatter={
    #                     'int': '{:06}'.format})
    from stable_baselines3.common.env_checker import check_env

    env = ECRepairEnv(5, 3, 3)
    # It will check your custom environment and output additional warnings if needed
    check_env(env)
    # env = ECRepairEnv(5, 3, 3)  # RS（N=5，K=3），每个节点包含L=3个块
    # obs = env.reset()
    # step = 0
    # mp = np.ones(4) * 15
    # print(mp)
    #
    # multiDiscrete = MultiDiscrete(mp)
    #
    # # state, reward, done, _  = env.step(multiDiscrete.sample())
    # # print(state, reward)
    # done = False
    # info = []
    # reward = 0
    # while not done:
    #     act = multiDiscrete.sample()
    #     obs, r, done, info = env.step(act)
    #     reward += r
    #     step += 1
    #     env.render()
    #
    #
    # print('成功！总共花了{}步, reward={}.'.format(step, reward))
