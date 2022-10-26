from SourceCodeRefactor.dnn_models.GoogLeNet import *
from SourceCodeRefactor.dnn_models.yolo import *
from SourceCodeRefactor.dnn_models.transformer import *
from SourceCodeRefactor.dnn_models.inceptionv4 import *
from torch.fx import symbolic_trace
import inspect
import numpy as np
import re
import torch
import xlwt
import csv
from SourceCodeRefactor.source_code_rewriting import SourceCodeUpdate


# 修改源码，删除原来的模型定义类
def change_source_code(source_path, model_class_name):
    with open(source_path, "r", encoding="utf-8") as f:
        source_str = f.read() + "\n"
    model_source_code = source_str.split("\n")
    start, end = -1, -1
    for i, v in enumerate(model_source_code):
        if v.startswith(f"class {model_class_name}("):
            start = i
            continue
        if v.startswith("class "):
            end = i
        if start != -1 and end > start:
            break
    else:
        end = len(model_source_code) - 1
    model_source_code = ["" if i >= start and i < end else model_source_code[i] for i in range(len(model_source_code))]
    model_source_code = [i for i in model_source_code if i != ""]
    model_source_code = "\n".join(model_source_code) + "\n"
    return model_source_code


# 重构forward方法
def modify_foward(generate_codes, model_source_code):
    '''
    返回改造后的完整的模型定义类源码
    :param generate_codes: 重生成的代码
    :param model_source_code: 模型定义类的源码
    :return: 更改后的forward部分 -> str
    '''
    # 注释源码中的forward方法
    model_source_code = model_source_code.split("\n")
    start, end = -1, -1
    for i, v in enumerate(model_source_code):
        if v.startswith("    def forward(self"):
            start = i
            # print(start)
        if v.startswith("    def"):
            end = i
        if start != -1 and end > start:
            break
    else:
        end = len(model_source_code) - 1
    model_source_code = ["" if i >= start and i < end else model_source_code[i] for i in range(len(model_source_code))]
    model_source_code = [i for i in model_source_code if i != ""]
    model_source_code = "\n".join(model_source_code) + "\n"
    # 注释源码后追加改造后的代码
    for fun in generate_codes:
        model_source_code += fun
    return model_source_code


# 获取每个节点的中间数据
def get_mi_vars(model, input_shape):
    '''
    :param model: dnn模型
    :param input_shape: 模型的输入
    :return:
    '''
    model_source_code = inspect.getsource(model)  # 模型定义的源码
    model_source_code = model_source_code.replace(f"{model_class_name}, self", "")
    source_str = change_source_code(source_path, model_class_name)  # 对模型源码字符串定义类做删除操作
    model = model()
    gm = symbolic_trace(model)
    codes = str(gm.code).split("\n")  # 重生成forward部分
    f_h, f_r = "", ""
    for c in codes:  # 首尾去除，保留每个算子的调用部分
        if "def forward" in c:
            f_h = c.strip()
        if "return" in c:
            f_r = c.strip()

    # 每个节点的调用代码
    call_codes = [i for i in codes if i and "def forward" not in i and "return " not in i]

    # 前向传播代码插桩操作
    total = len(call_codes) * 2 - 1
    for i in range(1, total, 2):
        attr_name = call_codes[i-1].split("=")[0].strip()
        call_codes.insert(i, f"    middle_variables['{attr_name}'] = {attr_name}")

    # 重新插桩后的forward方法
    op_forward = []
    op_forward.append(" " * 4 + f_h + "\n")
    for c in call_codes:
        if c.strip() != "":
            op_forward.append(" " * 8 + c.strip() + "\n")
    op_forward.append(" " * 8 + f_r + "\n")
    code_str = source_str + modify_foward(op_forward, model_source_code)
    op_test_code = \
f'''model = {model_class_name}()
model.eval()
global middle_variables
middle_variables = dict()
dis = []
for idx, size in enumerate({input_shape}):
    exec("dump_input%s=torch.randn(size=size)" % idx)
    dis.append("dump_input%s"% idx)
exec(f"dump_output=model.forward(%s)"% ','.join(dis))
'''
    code_str += op_test_code
    c = compile(code_str, "", mode="exec")
    exec(c)  # 执行插桩后的代码
    mi_vars = globals()["middle_variables"]  # 获取中间结果
    return mi_vars


class OperatorStatistics:
    def __init__(self, scu, nodes, model_source_code, model_class_name, mi_vars):
        self.nodes = nodes  # 节点信息
        self.scu = scu
        self.codes = scu.codes  # 单算子调用代码集
        self.model_class_name = model_class_name
        self.model_source_code = model_source_code.replace(f"{model_class_name}, self", "")  # 更改继承方式
        self.source_str = change_source_code(source_path, model_class_name)  # 对模型源码字符串定义类做删除操作
        self.mi_vars = mi_vars  # 每个节点输出的中间数据
        self.test_node()  # 拆分测试

    def nodenext(self):
        """
        添加节点的顺延节点信息
        :return: None
        """
        for node in self.nodes.keys():
            succ = []
            for k, v in self.nodes.items():
                if node in v['args']:
                    succ.append(k)
            self.nodes[node]['next'] = succ

    def curr_node_result(self, input_vars, return_var, call_code, sizes):
        """
        :param size: 当前节点输入shape
        :param in_args: forward函数入参
        :param ret_args: forward函数出参
        :return: 当前节点输出shape, cpu运行时间, gpu运行时间
        """
        print("in_args: ", input_vars)
        print("call_code: ", call_code)
        print("ret_args: ", return_var)

        # 生成单算子的forward方法
        op_forward = []
        op_forward.append(" " * 4 + f"def forward(self, {input_vars}):" + "\n")
        op_forward.append(" " * 8 + call_code + "\n")
        op_forward.append(" " * 8 + f"return {return_var}" + "\n")

        code_str = self.source_str
        code_str += self.scu.modify_foward(op_forward, self.model_source_code)
        code_str += "import re\n"
        op_test_code = \
f'''model = {self.model_class_name}()
model.eval()
device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
dis = []
def randbool(size):
    return torch.randint(2,size) == torch.randint(2, size)
for idx, size in enumerate({sizes}):
    if size[1] == "int" or size[1] == "float":
        exec("dump_input%s=size[0]" % idx)
    else:
        if size[1] == "torch.bool":
            exec("dump_input%s=randbool(size[0]).to(device)" % idx)
        elif size[1] == "torch.float32":
            exec("dump_input%s=torch.randn(size=size[0]).to(device)" % idx)
        else:
            raise TypeError("unexpected type")
    dis.append("dump_input%s"% idx)
global cpu_times, gpu_times, outshape
cpu_times, gpu_times = [], []
for i in range(10):
    with torch.autograd.profiler.profile(enabled=True,
                                         use_cuda=True,
                                         record_shapes=False,
                                         use_kineto=True,
                                         use_cpu=True) as prof:
        exec(f"dump_output=model.forward(%s)"% ','.join(dis))
    if hasattr(dump_output, "shape"):
        outshape = dump_output.shape
    else:
        outshape = dump_output
    pattern = r'time total: (.+?s)'
    result = re.findall(pattern=pattern,
                          string=prof.table())
    if len(result) == 2:
        cpu_time, gpu_time = result[0], result[1]
    elif len(result) == 1:
        cpu_time, gpu_time = result[0], "0ms"
    elif len(result) == 0:
        cpu_time, gpu_time = "0ms", "0ms"
    else:
        raise ValueError("Error in parsing operator test result")
    cpu_time = cpu_time if "us" not in cpu_time else str(
            float(cpu_time.replace("us", "")) / 1000) + "ms"
    gpu_time = gpu_time if "us" not in gpu_time else str(
        float(gpu_time.replace("us", "")) / 1000) + "ms"
    print(cpu_time, gpu_time)
    cpu_times.append(cpu_time)
    gpu_times.append(gpu_time)
cpu_times.pop(0) # 第一次测的值不准确需要去掉
gpu_times.pop(0)
'''
        code_str += op_test_code
        # print(code_str)
        c = compile(code_str, "", mode="exec")
        exec(c)  # 执行算子性能测量代码
        cpu_times = [float(i.replace("ms", "")) for i in globals()["cpu_times"]]
        gpu_times = [float(i.replace("ms", "")) for i in globals()["gpu_times"]]
        cpu_time, gpu_time = str(round(np.mean(cpu_times), 3)) + "ms", str(round(np.mean(gpu_times), 3)) + "ms"
        return globals()["outshape"], cpu_time, gpu_time

    # 获取每个节点的入参、出参及调用代码
    def get_args(self, curr):
        # 根据节点名称匹配调用代码
        for c in self.codes:
            if c.split("=")[0].strip() == curr:
                res = c
                break
        else:
            raise ValueError("The current node operator call code is not matched")

        # 正则匹配
        pattern = r"\((.+?)\)"
        if "getattr" in res or "torch." in res or ("self." in res and "(" in res):
            in_args = re.findall(pattern=pattern, string=res)[-1]
            in_args = [i.strip().replace("(", "") for i in in_args.split(",") if not i.strip().isdigit()]
            input_vars = [i for i in in_args if "=" not in i]
            input_vars = ",".join(input_vars)
            if "[" in input_vars and "]" in input_vars:
                input_vars = re.findall(pattern=r"\[(.+?)\]", string=input_vars)[0]
        elif "**" in res or "+" in res or "/" in res:
            if "**" in res:
                pattern1 = r"= (.+?) \*{2} (.+?);"
            elif "+" in res:
                pattern1 = r"= (.+?) \+ (.+?);"
            else:
                pattern1 = r"= (.+?) / (.+?);"
            in_args = re.findall(pattern=pattern1, string=res)[0]
            in_args = [i for i in in_args if i in self.nodes.keys()]
            input_vars = ",".join(in_args)
        elif "self." in res:
            input_vars = ""
        else:
            pattern2 = r"= (.+?)\..+?\((.+?)\)"
            in_args = re.findall(pattern=pattern2, string=res)[0]
            in_args = ",".join(in_args)
            in_args = [i.strip().replace("(", "") for i in in_args.split(",") if not i.strip().isdigit() and i !="-1"]
            input_vars = ",".join(in_args)

        return_var = res.split("=")[0].strip()
        return input_vars, return_var, res

    # 匹配此节点的输入大小
    def mid_vars_mapper(self, input_vars):
        if not input_vars:  # 根节点无输入
            return []
        sizes = []
        for var in input_vars.split(","):
            # print("var: ", var)
            # print(type(self.mi_vars['size']))
            if isinstance(self.mi_vars[var], int):  # 输入为int类型
                sizes.append((self.mi_vars[var], "int"))
            elif isinstance(self.mi_vars[var], float):  # 输入为float类型
                sizes.append((self.mi_vars[var], "float"))
            else:  # 输入为Tensor
                if str(self.mi_vars[var].dtype) == "torch.float32":  # 浮点张量
                    sizes.append((self.mi_vars[var].shape, "torch.float32"))
                elif str(self.mi_vars[var].dtype) == "torch.bool":  # 布尔张量
                    sizes.append((self.mi_vars[var].shape, "torch.bool"))
                else:
                    raise TypeError(f"unexpected type {str(self.mi_vars[var].dtype)}")
        return sizes

    # 每个算子拆分测试
    def test_node(self):
        for k, v in self.nodes.items(): # 遍历每个节点进行测试
            if k in ["src", "tgt"]:  # 两个输入
                self.nodes[k]['time'] = "0.0ms"
                self.nodes[k]['size'] = (64, 16, 512)
            else:
                # 获取该节点的父子节点和调用代码
                input_vars, return_var, call_code = self.get_args(k)
                sizes = self.mid_vars_mapper(input_vars)  # 该节点的输入数据

                # 单节点测试，获取单条结果
                out_shape, cpu_time, gpu_time = self.curr_node_result(input_vars, return_var, call_code, sizes)
                self.nodes[k]['size'] = list(out_shape) if isinstance(out_shape, tuple) else [out_shape]
                self.nodes[k]['time'] = gpu_time

                # 添加顺延节点信息
                for n in input_vars.split(","):
                    if not n:
                        continue

                    # 写入每个节点的子节点信息
                    if "next" not in self.nodes[n.strip()].keys():
                        self.nodes[n.strip()]['next'] = []
                    self.nodes[n.strip()]['next'].append(return_var)


# csv格式存储结果
def convert_csv(data, save_path):
    '''
    :param data: 包含测量结果的节点信息
    :param save_path: csv储存路径
    :return:
    '''
    save_list = []  # 收集测试数据
    for n, v in data.items():
        if "next" not in v.keys():
            v['next'] = []  # 标记末端节点
        edges = ",".join([n + "_2_" + i for i in v['next']])  # 拼接边信息
        dir_succ = ",".join(v['next'])  # 拼接顺延节点信息
        transfer_size = 1
        for i in tuple(v['size']):
            transfer_size *= i
        transfer_size = round((transfer_size * 4) / 1024, 3)  # 计算传输量
        dic = {
            "task": n,
            "c_task": edges,
            "dir_succ": dir_succ,
            "dir_succ_add": (edges + "," + dir_succ) if dir_succ else edges,
            "p": float(v['time'].replace("ms", "")),
            "shape": str(v['size']),
            "transfer_size": transfer_size
        }
        save_list.append(dic)
    header = list(save_list[0].keys())  # csv表头
    # 结果存为csv格式
    with open(save_path, 'w', encoding='utf-8-sig', newline='') as file:
        csv_file = csv.DictWriter(file, header)
        csv_file.writeheader()
        csv_file.writerows(save_list)


# Excel格式存储结果
def convert_excel(data, save_path):
    '''
    :param data: 包含测量结果的节点信息
    :param save_path: excel储存路径
    :return: None
    '''
    statistical_results = []  # 收集测试数据
    for n, v in data.items():
        dic = {}
        dic['task'] = n
        edges = [n + "_2_" + i for i in v['next']]
        dic['c_task'] = ",".join(edges)
        dic['dir_succ'] = ",".join(v['next'])
        # dic['dir_succ_add'] = (dic['dir_succ'] + ", " + dic['c_task']) if dic['dir_succ'] else dic['c_task']
        dic['dir_succ_add'] = (dic['c_task'] + "," + dic['dir_succ']) if dic['dir_succ'] else dic['c_task']
        dic['p0'] = float(v['time'].replace("ms", ""))
        dic['shape'] = str(tuple(v['size']))
        transfer_size = 1  # 初始化
        for i in tuple(v['size']):
            transfer_size *= i
        dic['transfer_size'] = (transfer_size * 4) // 1024
        statistical_results.append(dic)
    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    sheet = book.add_sheet('data0', cell_overwrite_ok=True)
    sheet.write(0, 0, "task")
    sheet.write(0, 1, "c_task")
    sheet.write(0, 2, "dir_succ")
    sheet.write(0, 3, "dir_succ_add")
    sheet.write(0, 4, "p0")
    sheet.write(0, 5, "shape")
    sheet.write(0, 6, "transfer_size")
    for row, con in enumerate(statistical_results):
        column = 0
        for _, v in con.items():
            sheet.write(row + 1, column, v)
            column += 1
    book.save(save_path)


def main(source_path, model_class_name, model, input_shape, save_path):
    model_source_code = inspect.getsource(model)  # 获取定义模型源码部分
    # model = model("SourceCodeRefactor/dnn_models/yolov5x.yaml")
    mi_vars = get_mi_vars(model, input_shape)  # 获取各节点输出（所有中间结果）
    mi_vars['src'] = mi_vars['tgt'] = torch.rand(64, 16, 512) # 输入
    model = model()
    scu = SourceCodeUpdate(source_path, model)
    nodes = scu.get_nodes_and_code()  # 解析节点信息
    OpS = OperatorStatistics(scu, nodes, model_source_code, model_class_name, mi_vars)

    for i, v in OpS.nodes.items():  # 打印结果
        print(i, v)

    convert_csv(OpS.nodes, save_path)  # 保存结果


if __name__ == '__main__':
    source_path = "SourceCodeRefactor/dnn_models/transformer.py"  # 模型源码路径
    model_class_name = "Transformer"  # 模型定义类名
    model = Transformer
    input_shape = [(64, 16, 512), (64, 16, 512)]  # 输入维度
    save_path = "./transformer.csv"  # 结果保存路径
    main(source_path, model_class_name, model, input_shape, save_path)