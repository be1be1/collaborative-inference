from linklist import Linklist


class ModelStruction:
    '''
    定义了对模型结构改造的类，按分支输出链路结构
    '''
    def __init__(self, nodes):
        for k, v in nodes.items():
            if not v['args']:
                # 计算题图的头节点
                self.head = k
                break
        # 维护推理顺序
        self.seq = 0
        # 储存转化后的分支划分结果
        self.blocks = list()
        # 保存已处理的节点
        self.temp = set()
        # 传入的节点
        self.nodes = nodes
        # 计算图关键点（分支、合并点）
        self.key_points = None

    # 关键点标记
    def branch_point(self):
        '''
        对计算图中分支合并的点做标记，0表示分支点，1表示合并点， 2 表示既是分支点也是合并点
        :return: 包含所有关键点的字典 -> Dict
        '''
        points = {}
        for i, v in self.nodes.items():
            # 分支/合并
            if v['users'] > 1 and len(v['args']) > 1:
                points[i] = 2

            # 分支点
            if v['users'] > 1 and len(v['args']) == 1:
                points[i] = 0

            # 合并点
            if v['users'] == 1 and len(v['args']) > 1:
                points[i] = 1
        keys = list(points.keys())
        while 'cat' not in keys[-1]:
            del points[keys.pop()]

        nodes_keys = list(self.nodes.keys())

        del_nodes = nodes_keys[nodes_keys.index(keys[-1]) + 1:]

        # 去除计算图末尾节点, 方便后续将末端代码打包到一起
        for del_node in del_nodes:
            del self.nodes[del_node]

        self.key_points = points

    def bb(self, head):
        '''
        从分支点开始的传播方式
        :param head: 开始节点
        :return: 结束节点 -> str
        '''
        end, _end = None, None
        self.seq += 1
        for i, v in self.nodes.items():
            if head in v['args']:
                end, link = self.nodes_forward(i, delete=True)
                if self.key_points[end] != 0:
                    self.blocks.append({self.seq: link})
                else:  # 分支嵌套分支的情况
                    flag = 0
                    dic = dict()
                    dic[self.seq] = []
                    dic[self.seq].append({flag: link})
                    flag += 1
                    for m, n in self.nodes.items():
                        if end in n['args']:
                            _end, link = self.nodes_forward(m, delete=True)
                            dic[self.seq].append({flag: link})
                    end = _end
                    flag += 1
                    # 此时end为合并点
                    for x, y in self.nodes.items():
                        if end == x:
                            end, link = self.nodes_forward(x, branch_forward=True)
                            dic[self.seq].append({flag: link})
                    self.blocks.append(dic)
        return end

    def nodes_split(self, head):
        '''
        通过传播方式的节点划分方法，遇到关键点则该条链路传播结束
        :param head: 开始节点
        :return: 结束节点 -> str
        '''
        end = None
        if head != self.head and head not in self.key_points.keys():
            raise ValueError("Param Error")

        if head == self.head:
            end, link = self.nodes_forward(head)
            self.blocks.append({self.seq: link})

        elif self.key_points[head] == 0:
            end = self.bb(head)

        elif self.key_points[head] == 2:
            self.seq += 1
            for i, v in self.nodes.items():
                if head == i:
                    end, link = self.nodes_forward(i, double=True)
                    self.blocks.append({self.seq: link})
                    end = self.bb(end)

        elif self.key_points[head] == 1:
            self.seq += 1
            for i, v in self.nodes.items():
                if head == i:
                    end, link = self.nodes_forward(i, branch_forward=True)
                    self.blocks.append({self.seq: link})
        return end

    def refactor(self):
        '''
        全图传播
        :return: None
        '''
        end = self.nodes_split(self.head)
        while True:
            try:
                end = self.nodes_split(end)
                print(end)
                if end not in self.temp:
                    self.temp.add(end)
                else:
                    print("???")
                    break
            except:
                print("--over--")
                break

    # 结点传播
    def nodes_forward(self, start, delete=False, double=False, branch_forward=False):
        '''
        :return: None
        '''
        linklist = Linklist()
        # 添加链表头
        linklist.add(start)
        if double:
            return start, linklist
        # 开始向下传播
        if branch_forward:
            for i, v in self.nodes.items():
                if start in v['args']:
                    if i not in self.key_points:
                        start = i
                        linklist.add(start)
                        break
                    else:
                        if self.key_points[i] == 0:
                            start = i
                            linklist.add(start)
                            break
        while True:
            for i, v in self.nodes.items():
                if start in v['args'] and start not in self.key_points:
                    start = i
                    linklist.add(start)
                    break
            else:
                break
            flag = True
            for i, v in self.nodes.items():
                if i == start:
                    if v['users'] > 1:
                        flag = False
                    if len(v['args']) > 1:
                        if delete:
                            linklist.remove(start)
                        flag = False
            if not flag:
                break
        return start, linklist

    def get_blocks(self):
        '''
        获取所有的模型分支链路
        :return: List
        '''
        # 标记关键点
        self.branch_point()

        # 执行节点划分方法
        self.refactor()
        # print(self.key_points["cat_2"])
        # print(self.temp)
        return self.blocks