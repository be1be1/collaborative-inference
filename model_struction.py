from dnn_models.resnet import *
from dnn_models.GoogLeNet import *
from dnn_models.alexnet import *
from dnn_models.darknet_53 import *
from dnn_models.inception_v3 import *
from dnn_models.inceptionv4 import *
# from dnn_models.karate_club_net import *
from dnn_models.resnet import *
# from dnn_models.simple_gcn import *
from dnn_models.mynet import *
import torch
from torch.fx import symbolic_trace
import re
import pandas as pd
import inspect


class Node:
    def __init__(self, name):
        self.name = name
        self.next = None


class Linklist:
    def __init__(self):
        self.head = None

    def is_empty(self):
        return self.head == None

    def length(self):
        '''
        链表长度
        :return: count -> Int
        '''
        # cur游标， 用来移动遍历节点
        cur = self.head
        # count记录数量
        count = 1
        while cur != None:
            count += 1
            cur = cur.next
        return count

    def travel(self):
        '''
        遍历整个链表
        :return: None
        '''
        cur = self.head
        while cur != None:
            print(cur.name)
            cur = cur.next

    def add(self, item):
        '''
        链表尾部添加元素
        '''
        node = Node(item)
        if self.is_empty():
            self.head = node
        else:
            cur = self.head
            while cur.next != None:
                cur = cur.next
            cur.next = node

    def remove(self, item):
        """删除节点"""
        cur = self.head

        # 如果删除的节点是头节点
        if cur.name == item:
            self.head = cur.next
        else:
            # 找到删除节点的前驱节点
            while cur.next.name != item:
                cur = cur.next
            cur.next = cur.next.next


class ModelStruction:
    def __init__(self, nodes):
        for node in nodes:
            if not node['args']:
                # 计算题图的头节点
                self.head = node['name']
        self.blocks = []
        self.nodes = nodes

        # 执行节点划分方法
        self.nodes_split()

    # 收集分支点
    def branch_point(self):
        points = []
        for i in self.nodes:
            if i['users'] > 1 or len(i['args']) > 1:
                points.append(i['name'])
        return points

    # 按分支打包节点
    def nodes_split(self):
        order = 0
        self.blocks.append((order, self.nodes_forward(self.head)))
        for n in self.branch_point():
            order += 1
            for i in self.nodes:
                if n in i['args']:
                    self.blocks.append((order, self.nodes_forward(i['name'], delete=True)))
            for i in self.nodes:
                if n == i['name'] and len(i['args']) > 1:
                    self.blocks.append((order, self.nodes_forward(i['name'])))

    # 结点传播
    def nodes_forward(self, start, delete=False):
        '''

        :return: None
        '''
        linklist = Linklist()
        # 添加链表头
        linklist.add(start)

        # 开始向下传播
        while True:
            for i in self.nodes:
                if start in i['args'] and start not in self.branch_point():
                    start = i['name']
                    linklist.add(start)
                    break
            flag = True
            for i in self.nodes:
                if i['name'] == start:
                    if i['users'] > 1:
                        flag = False
                    if len(i['args']) > 1:
                        if delete:
                            linklist.remove(start)
                        flag = False
            if not flag:
                break
        return linklist

    def get_blocks(self):
        return self.blocks


def get_nodes_and_code(model):
    '''
    :param model: 实例化出来的网络模型
    :return: List
        example: [{'name': 'conv1_conv', 'operator': 'conv', 'users': 1, 'args': []}
                  {'name': 'conv1_relu', 'operator': 'relu', 'users': 1, 'args': ['conv1_conv']}
                 ]
    '''

    # 通过torch.fx生成图
    gm = symbolic_trace(model)

    # code按行存为列表形式
    code = str(gm.code).split("\n")
    code = [i.strip() for i in code[4:-2]]

    # 节点结果存为列表形式
    df = str(gm.graph).split("\n")[2:-1]
    nodes = []

    pattern = r"   %(.+?) : \[#"
    pattern1 = r"args = \((.+?)\), kwa"

    # 解析图生成约定的输出格式
    for s in df:
        dic = dict()
        name = re.findall(pattern=pattern, string=s)[0]
        dic['name'] = name
        dic['operator'] = name.split("_")[-1]
        dic['users'] = int(s[s.find("users=") + 6: s.find("] =")])
        args = re.findall(pattern=pattern1, string=s)[0]
        if args.startswith("["):
            args = args[args.find("[") + 1: args.rfind("]")]
        else:
            args = args[:-1]
        args = args.split(",")
        args = [i.replace("%", "").strip() for i in args]

        # 计算图的头节点args置空
        if args[0] == "x":
            args = []
        dic['args'] = args
        nodes.append(dic)
    return nodes, code


# 根据name匹配对应代码
def mapper(name, code):
    for c in code:
        if c.split("=")[0].strip() == name:
            return c


# 按分支打包代码
def unpack_code(linklist, code):
    branch = []
    cur = linklist.head
    while cur != None:
        branch.append(mapper(cur.name, code))
        cur = cur.next
    return branch


# 拼接代码的逻辑
def splice_code(branch, branch_name, add_remote=False):
    param = branch[0][branch[0].rfind("(") + 1: branch[0].rfind(")")]

    # 聚合行（cat）
    if "[" in param:
        param = param[param.find("[") + 1: param.rfind("]")]
    s = " "*4 + "def " + branch_name + f"(self, {param}):"

    # 是否打入标签
    if add_remote:
        s = " "*4 + "@ray.remote\n" + s

    # 逐行添加
    for i in branch:
        s += "\n" + " " * 8 + i

    # 添加return行
    return_value = branch[-1].split("=")[0].strip()
    s += "\n" + " " * 8 + "return "
    s += return_value + "\n"
    return s


# 生成用于替换的forward方法
def generate_forward(blocks, codes):
    # 转化为DataFrame格式
    df = pd.DataFrame(blocks, columns=['order', 'linklist'])

    # 进行聚合操作
    count = df.groupby("order")["linklist"].count()

    # reform_codes用于收集forward调用的分支方法，s_forward为forward主方法
    reform_codes, s_forward = [], " " * 4 + "def forward(self,x):\n"

    for i in count.index:
        temp = df[df['order'] == i]

        # 对于串行块的操作
        if count[i] == 1:
            # 获取当前链路源码
            branch = unpack_code(temp['linklist'].values[0], codes)
            # 生成def格式
            s = splice_code(branch, branch_name=f"b{i}")
            if i == 0:  # 起始位置
                s_forward += " " * 8 + f"x = self.b{i}(x)\n"
            else:
                s_group = [f"ray.get({i})" for i in s_group]
                s_forward += " " * 8 + f"x = self.b{i}(%s)\n"%(','.join(s_group))
            reform_codes.append(s)
            reform_codes.append("\n")

        # 对于并行块的操作
        else:
            # 需要聚合的分支
            s_group = []
            # 并行分支遍历操作
            for idx, ll in enumerate(temp.iloc[:, 1].values):
                branch = unpack_code(ll, codes)

                # 并行块“add_remote=True”打入ray标签
                s = splice_code(branch, branch_name=f"b{i}_{idx}", add_remote=True)
                s_forward += " " * 8 + f"b{i}_{idx} = self.b{i}_{idx}.remote(self, x)\n"
                s_group.append(f"b{i}_{idx}")
                reform_codes.append(s)
                reform_codes.append("\n")
    s_forward += " " * 8 + "return x\n"
    reform_codes.append(s_forward)
    return reform_codes


# 重构forward方法
def modify_foward(generate_codes, model_source_code):
    # 注释源码中的forward方法
    model_source_code = model_source_code.split("\n")
    for i, v in enumerate(model_source_code):
        if v.startswith("    def forward(self"):
            start = i
        if v.startswith("        return"):
            end = i
            break
    model_source_code = ["#" + model_source_code[i] if i >= start and i <= end else model_source_code[i] for i in range(len(model_source_code))]
    model_source_code = "\n".join(model_source_code) + "\n"

    # 注释源码后添加我们改造后的代码
    for fun in generate_codes:
        model_source_code += fun

    return model_source_code


# 替换源码的model定义类
def replace_source_code(new_code, source_path):
    # import ray
    new_code = "import ray\n" + new_code
    with open(source_path, "a", encoding="utf-8") as f:
        f.write(new_code)


def main(model, source_path):
    # # 提取网络的模型类名称
    # model_name = source_path.split("/")[-1].split(".")[0]

    # 以str形式提取model定义源码
    model_source_code = inspect.getsource(model)

    model = model()

    # torch.fx方法获取网络层信息和自动生成的代码
    print("获取网络层信息...")
    nodes, codes = get_nodes_and_code(model)

    # 模型层分块打包
    print("网络层分块打包...")
    model_struction = ModelStruction(nodes)
    blocks = model_struction.get_blocks()

    # 需要插入源码的code
    print("重构代码生成...")
    generate_codes = generate_forward(blocks, codes)

    print("注释源码中的forward方法，添加重构代码...")
    new_code = modify_foward(generate_codes, model_source_code)
    print(new_code)
    print("替换源码...")
    replace_source_code(new_code, source_path)

    print(f"success, modify source code -> {source_path}")
    # for i in generate_codes:
    #     print(i)


if __name__ == '__main__':
    # 源码路径
    source_path = "dnn_models/GoogLeNet.py"
    model = GoogLeNet
    main(model, source_path)







