import numpy as np
M=[]

def main():

    n, k, l = 14, 12, 1000
    uploads, downloads = [], []
    for i in range(0, n-1):  # 第n-1个节点为替换节点
        uploads.append((i, 0))
    for i in range(0, n):
        downloads.append((i, 0))

    # print('uploads: {}, downloads:{}.'.format(uploads, downloads))

    t = Tasks()
    for j in range(0, l):
        uploads = sorted(uploads, key=lambda d:d[1])
        source_kids = select_k_sources(uploads, k)
        downloads = sorted(downloads, key=lambda d:d[1])
        target_id = select_target(downloads)
        t.add_tasks(source_kids, target_id)

    print('uploads: {}, downloads:{}, tasks:{}.'.format(uploads, downloads, t.tasks))



    # set_A, set_B = ['A', 'B', 'C', 'D'], ['E', 'F', 'G', 'H']

    set_A = [i for i in range(0, n-1)]   #[0,1,2,...,n-2]
    set_B = [i for i in range(0, n)]     #[0,1,2,...,n-1]

    i = 0
    # download_count = np.zeros(n-1, dtype=np.int32)
    replace_download_count = 0
    while not t.empty():
        print('第{}轮:'.format(i))
        edges = np.zeros((n, n), dtype=np.int8)

        for u in set_A:
            for v in set_B:
                if t.get_task((u, v)) >= 0:
                    edges[u, v] = 1

        # edges[0, 4] = 1
        # edges[1, 4] = 1
        # edges[2, 4] = 1
        # edges[3, 4] = 1

        print('edges: {}'.format(edges))

        cx = [-1 for _ in range(0, n-1)]
        cy = [-1 for _ in range(0, n)]

        visited = [0 for _ in range(0, n)]

        dh = DFS_hungary(set_A, set_B, edges, cx, cy, visited)
        dh.max_match()
        dh.print_max_paths()
        cur_tasks = dh.turn_to_tasks()
        t.do_tasks(cur_tasks)

        # for v in dh.cx:
        # 替换节点作为下载节点
        if dh.cy[n-1] != -1:
            replace_download_count += 1

        i += 1

    # for count in download_count:
    # 每个接收节点要单独传输给替换节点
    i += l - int(replace_download_count / k)

    print('cr_repairBoost总共用时{}t.'.format(i))

def select_k_sources(uploads, k):
    k_keys = []
    i = 0
    while i < k:
        key, v = uploads[i]
        uploads[i] = (key, v+1)
        k_keys.append(key)
        i += 1

    return k_keys

def select_target(downloads):
    k, v = downloads[0]
    downloads[0] = (k, v + 1)
    return k

class Tasks():
    def __init__(self):
        self.tasks = []

    def get_task(self, task):
        task_s, task_t = task
        for i in range(0, len(self.tasks)):
            s, t, w = self.tasks[i]
            if task_s == s and task_t == t and w > 0:
                return i
        return -1

    def empty(self):
        for _,_,w in self.tasks:
            if w > 0:
                return False
        return True

    def add_tasks(self, source_kids, target_id):
        for source_id in source_kids:
            if source_id != target_id:
                i = self.get_task((source_id, target_id))
                # 任务已存在
                if i > 0:
                    _, _, w = self.tasks[i]
                    self.tasks[i] = (source_id, target_id, w+1)
                # 新任务
                else:
                    self.tasks.append((source_id, target_id, 1))

    def do_tasks(self, tasks):
        for task in tasks:
            i = self.get_task(task)
            u, v = task
            assert i >= 0
            _, _, w = self.tasks[i]
            self.tasks[i] = (u, v, w-1)


class DFS_hungary():

    # 参数初始化
    def __init__(self, set_A, set_B, edge, cx, cy, visited):
        self.set_A, self.set_B = set_A, set_B  # 顶点集合
        self.edge = edge  # 顶点是否连边
        self.cx, self.cy = cx, cy  # 顶点是否匹配
        self.visited = visited  # 顶点是否被访问
        self.M = []  # 匹配
        self.res = 0  # 匹配数

    # 遍历顶点A集合，得到最大匹配
    def max_match(self):
        for i in self.set_A:
            if self.cx[i] == -1:  # 未匹配
                for key in self.set_B:  # 将visited置0表示未访问过
                    self.visited[key] = 0
                self.res += self.path(i)
                # print('i', i, 'M',self.M)

    # 增广路置换获得更大的匹配
    def path(self, u):
        for v in self.set_B:
            if self.edge[u,v] and (not self.visited[v]):  # 如果可连且未被访问过
                self.visited[v] = 1 # 访问该顶点
                if self.cy[v] == -1:  # 如果未匹配， 则建立匹配
                    self.cx[u], self.cy[v] = v, u
                    self.M.append((u, v))
                    return 1
                else:
                    if len(self.M) and (self.cy[v], v) in self.M:
                        self.M.remove((self.cy[v], v))  # 如果匹配则删除之前的匹配
                    if self.path(self.cy[v]):  # 递归调用
                        self.cx[u], self.cy[v] = v, u
                        self.M.append((u, v))
                        return 1
            # print('v', v, 'M', self.M)

        return 0

    def print_max_paths(self):
        print('最大匹配数:{}'.format(self.res))
        for u,v in enumerate(self.cx):
            if v != -1:
                print('({}, {})'.format(u, v))

    def turn_to_tasks(self):
        tasks = []
        for u,v in enumerate(self.cx):
            if v != -1:
                tasks.append((u, v))
        return tasks

if __name__ == '__main__':
    main()
