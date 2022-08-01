import numpy as np
import gym
import copy
from gym.spaces import MultiDiscrete


class CornGame(gym.Env):
    def __init__(self, n=5, k=3, l=3):
        self.action_space = MultiDiscrete(np.ones(n - 1) * n * l)
        # self.action_space = gym.spaces.Discrete(N*L)
        self.n = n  # 节点数
        self.l = l  # 块数
        self.k = k  # 数据节点数
        # self.S = np.zeros(N*L).reshape(N, L)  #状态矩阵
        self.s = np.zeros((n, l), dtype=np.int32)
        self.sSet = []
        self.eSet = []
        self.corn_done_ids = []
        low = np.zeros(n * l)
        high = np.ones(n * l) * (2 ^ (n - 1) - 1)  # 每个数据块最大值：2^N-1
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
        for i in range(0, self.n - 1):
            self.s[i] = pow(2, i)
            # logger.info(' S[{}] = {} '.format(i, self.S[i]))

        # 替换节点DN数据为空
        self.s[i + 1] = 0
        self.sSet = []  # 已使用起点集
        self.eSet = []  # 已使用终点集
        self.corn_done_ids = [] # 初始化已完成的玉米篮子

        return self._make_state()

    def getNKL(self):
        return self.n, self.k, self.l

    def _make_state(self):
        return self.s.ravel()  # 返回一维数组

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
        return True if len(self.corn_done_ids) == self.l else False

    def col_contain_k_corns(self, col):
        col_values = self.s[:,col]
        for v in col_values:
            if bin(v).count("1") >= self.k:
                return True
        return False

    def render(self):
        print("current state: ", self.s)

    def step(self, act):

        actIDs = []

        for i in range(0, len(act)):
            act_id = act[i]
            actIDs.append((i, act_id % self.l, int(act_id / self.l)))

        actions = copy.deepcopy(actIDs)


        # 冲突过滤
        conflictIndexes = self.getConflictIndexes(actions)
        for i in reversed(conflictIndexes):
            del actions[i]

        reward = -1.0

        print(actions)
        # 拷贝一份状态,避免出现a->b同时b->c,出现c=a+b的情况,因为是同时进行,所以c只与之前的b有关
        new_s = copy.deepcopy(self.s)

        # 合并数据
        for action in actions:
            i, di, j = action
            self.s[i, di].astype(int)
            self.s[j, di].astype(int)
            # 按位或
            new_s[j, di] |= self.s[i, di]

            # 已经完成了的di列,就没必要继续操作了
            if di in self.corn_done_ids:
                reward += -1000
            # 如果di列已经出现了k合并,目的地不是n-1,则受罚
            if self.col_contain_k_corns(di) and j != self.n - 1:
                reward += -1000

            # 如果这一步让di列的玉米满了,则奖励1分
            if j == self.n - 1 and new_s[j, di] != self.s[j, di] \
                    and bin(new_s[j, di]).count("1") >= self.k \
                    and di not in self.corn_done_ids:
                reward += 1
                self.corn_done_ids.append(di)
            # 如果这一步操作无效果,则扣分
            if new_s[j, di] == self.s[j, di]:
                reward += -1000



        if len(actions) == self.n - 1:
            reward += 1

        self.s = new_s


        state = self._make_state()
        done = self.isDone()

        if done:
            reward = 100


        # print('step: ', action)

        return state, reward, done, {}


if __name__ == '__main__':

    # np.set_printoptions(precision=3, suppress=True, linewidth=300, formatter={
    #                     'int': '{:06}'.format})
    from stable_baselines3.common.env_checker import check_env

    env = ECRepairEnv(5, 3, 3)
    # It will check your custom environment and output additional warnings if needed
    # check_env(env)
    # env = ECRepairEnv(5, 3, 3)  # RS（N=5，K=3），每个节点包含L=3个块
    # obs = env.reset()
    step = 0
    mp = np.ones(4) * 15
    # print(mp)
    #
    multiDiscrete = MultiDiscrete(mp)
    #
    # # state, reward, done, _  = env.step(multiDiscrete.sample())
    # # print(state, reward)
    done = False
    info = []
    reward = 0
    # act =[]
    # act = [(0, 2, 4), (1, 0, 4), (2, 0, 1), (3, 2, 4)]
    while not done:
        act = multiDiscrete.sample()
        # act.append(())
        obs, r, done, info = env.step(act)
        reward += r
        step += 1
        env.render()


    print('成功！总共花了{}步, reward={}.'.format(step, reward))
