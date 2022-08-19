from torch.fx import symbolic_trace
import re
import pandas as pd


class SourceCodeUpdate:
    def __init__(self, source_path, model):
        self.source_path = source_path
        self.model = model
        self.codes = None

    def get_nodes_and_code(self):
        '''
        :param model: 实例化出来的网络模型
        :return: List
            example: [{'name': 'conv1_conv', 'operator': 'conv', 'users': 1, 'args': []}
                      {'name': 'conv1_relu', 'operator': 'relu', 'users': 1, 'args': ['conv1_conv']}
                     ]
        '''

        # 通过torch.fx生成图
        gm = symbolic_trace(self.model)

        # code按行存为列表形式
        code = str(gm.code).split("\n")
        codes = [i.strip() for i in code if "=" in i]

        # 节点结果存为列表形式
        df = str(gm.graph).split("\n")[2:-1]
        nodes = {}

        pattern = r"   %(.+?) : \[#"
        pattern1 = r"args = \((.+?)\), kwa"

        # 解析图生成约定的输出格式
        for s in df:
            num_user = int(s[s.find("users=") + 6: s.find("] =")])
            if num_user == 0:
                continue
            dic = dict()
            name = re.findall(pattern=pattern, string=s)[0]
            dic['name'] = name
            dic['operator'] = name.split("_")[-1]
            dic['users'] = num_user
            args = re.findall(pattern=pattern1, string=s)[0]
            if args.startswith("["):
                args = args[args.find("[") + 1: args.rfind("]")]
            elif args.startswith("("):
                args = args[args.find("(") + 1: args.rfind(")")]
            else:
                args = args[:-1]
            args = args.split(",")
            args = [i.replace("%", "").strip() for i in args]

            # 计算图的头节点args置空
            # if args[0] == "x" or args[0] =="input_1":
            #     args = []
            args = [a for a in args if a != ""]
            dic['args'] = args
            nodes[dic['name']] = dic
        names = [i['name'] for i in nodes.values()]
        for k, v in nodes.items():
            if len(v['args']) == 1 and v['args'][0] not in names:
                nodes[k]['args'] = []
                break
        # 对code进一步处理
        code = [c for c in codes if c.split("=")[0].strip() in list(nodes.keys())]
        self.codes = code
        return nodes

    # 根据name匹配对应代码
    def mapper(self, name):
        res = ""
        for co in self.codes:
            if co.split("=")[0].strip() == name:
                res = co
                self.codes.remove(res)
                print(f"len: {len(self.codes)}")
                break
        return res

    # TODO unpack最终逻辑
    # 按分支打包代码
    def unpack_code(self, linklist):
        branch = []
        cur = linklist.head
        print("-------head----------")
        print(cur.name)
        while cur != None:
            branch.append(self.mapper(cur.name))
            cur = cur.next
            if cur != None:
                print(cur.name)
        print("-------over----------")
        return branch

    # 拼接代码的逻辑
    def splice_code(self, branch, branch_name, add_remote=False):
        param = branch[0][branch[0].rfind("(") + 1: branch[0].rfind(")")]
        # print("param:", param)
        # 聚合行（cat）
        if "[" in param:
            param = param[param.find("[") + 1: param.rfind("]")]

        if ")" in param:
            param = param[:param.find(")")]
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

    def get_df(self, blocks):
        blocks = [(list(i.keys())[0], list(i.values())[0]) for i in blocks]
        # 转化为DataFrame格式
        df = pd.DataFrame(blocks, columns=['order', 'linklist'])

        # 进行聚合操作
        count = df.groupby("order")["linklist"].count()
        return df, count

    # 生成用于替换的forward方法
    def generate_forward(self, blocks, outermost_layer=True):
        # print(self.codes)
        # 获取格式转化结果
        df, count = self.get_df(blocks)
        # reform_codes用于收集forward调用的分支方法，s_forward为forward主方法
        reform_codes, s_group, s_forward = [], [], " " * 4 + "def forward(self, x):\n"
        for i in count.index:
            temp = df[df['order'] == i]
            # 对于串行块的操作
            if count[i] == 1:
                # 获取当前链路源码,每行为一个元素，以list存储
                branch = self.unpack_code(temp['linklist'].values[0])
                # 生成def格式
                branch_name = f"b{i}"
                # 生成一个完整分支链路的代码
                s = self.splice_code(branch, branch_name=branch_name)
                if i == 0:  # 起始位置
                    s_forward += " " * 8 + f"x = self.b{i}(x)\n"
                else:
                    s_group = [f"ray.get({i})" for i in s_group]
                    s_forward += " " * 8 + f"x = self.b{i}(%s)\n" % (', '.join(s_group))
                reform_codes.append(s)
                reform_codes.append("\n")

            # 对于并行块的操作
            else:
                # 需要聚合的分支
                s_group = []
                # 并行分支遍历操作
                for idx, ll in enumerate(temp.iloc[:, 1].values):
                    if type(ll) == list:
                        inner_codes = self.generate_forward(ll, outermost_layer=False)
                        s = " "*4 + "@ray.remote\n" + " " * 4 + f"def b{i}_{idx}(self, x):\n"
                        cat_params, index, f = [], 0, 999999
                        for cod in inner_codes:
                            for line in cod.split("\n"):
                                cur = " " * 4 + line
                                index += 1
                                if line.strip() == "":
                                    continue
                                if "forward" in cur:
                                    f = index
                                if "def" in cur:
                                    cur = cur.replace("self, ", "")
                                if cur.strip() == "@ray.remote":
                                    continue
                                if "self." in cur and index > f:
                                    cur = cur.replace("self.", "")
                                if ".remote" in cur:
                                    cur = cur.replace("self, ", "").replace(".remote", "").replace(" " * 12, " " * 12 + "_")
                                    cat_params.append(cur.split("=")[0].strip())
                                if "ray.get" in cur:
                                    cur = cur.replace(cur[cur.find("(") + 1: cur.rfind(")")], ','.join(cat_params))
                                s += cur + "\n"
                        s += " " * 8 + "return forward(x)"
                    else:
                        branch = self.unpack_code(ll)
                        # 并行块“add_remote=True”打入ray标签
                        s = self.splice_code(branch, branch_name=f"b{i}_{idx}", add_remote=True)
                    s_forward += " " * 8 + f"b{i}_{idx} = self.b{i}_{idx}.remote(self, x)\n"
                    s_group.append(f"b{i}_{idx}")
                    reform_codes.append(s)
                    reform_codes.append("\n")
        if outermost_layer:
            if len(self.codes) > 0:
                tail = self.splice_code(self.codes, branch_name="tail")
                reform_codes.append(tail)
                s_f = " " * 8 + f"x = self.tail(x)\n"
                s_forward += s_f
        s_forward += " " * 8 + "return x\n"
        reform_codes.append("\n")
        reform_codes.append(s_forward)
        return reform_codes

    # 重构forward方法
    def modify_foward(self, generate_codes, model_source_code):
        # 注释源码中的forward方法
        model_source_code = model_source_code.split("\n")
        start, end = -1, -1
        for i, v in enumerate(model_source_code):
            if v.startswith("    def forward(self"):
                start = i
            if v.startswith("        return"):
                end = i
                if start != -1 and end > start:
                    break
        model_source_code = ["#" + model_source_code[i] if i >= start and i <= end else model_source_code[i] for i in range(len(model_source_code))]
        model_source_code = "\n".join(model_source_code) + "\n"

        # 注释源码后追加改造后的代码
        for fun in generate_codes:
            model_source_code += fun

        return model_source_code

    # 替换源码的model定义类
    def replace_source_code(self, new_code):
        # import ray
        new_code = "import ray\n" + new_code
        with open(self.source_path, "a", encoding="utf-8") as f:
            f.write(new_code)

