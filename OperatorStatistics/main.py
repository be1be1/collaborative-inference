from SourceCodeRefactor.dnn_models.GoogLeNet import *
import inspect
import numpy as np
import re
import torch
import xlwt
from SourceCodeRefactor.model_representation import ModelStruction
from SourceCodeRefactor.source_code_rewriting import SourceCodeUpdate


class OperatorStatistics:
    def __init__(self, scu, nodes, model_source_code, model_class_name, input_shape):
        self.nodes = nodes  # 节点信息
        self.scu = scu
        self.codes = scu.codes  # 单算子调用代码集
        self.model_class_name = model_class_name
        self.model_source_code = model_source_code.replace(f"{model_class_name}, self", "")  # 更改继承方式
        self.source_str = change_source_code(source_path, model_class_name)  # 对模型源码字符串定义类做删除操作
        self.input_shape = input_shape  # 模型输入维度
        self.nodenext()
        for k, v in nodes.items():
            if not v['args']:
                self.head = k
            if not v['next']:
                self.tail = k
        self.split_calculation()

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

    def curr_node_result(self, sizes, in_args, ret_args, call_code):
        """
        :param size: 当前节点输入shape
        :param in_args: forward函数入参
        :param ret_args: forward函数出参
        :return: 当前节点输出shape, cpu运行时间, gpu运行时间
        """
        print("size: ", sizes)
        print("in_args: ", in_args)
        print("ret_args: ", ret_args)
        op_forward = []
        op_forward.append(" " * 4 + f"def forward(self, {in_args}):" + "\n")
        op_forward.append(" " * 8 + call_code + "\n")
        op_forward.append(" " * 8 + f"return {ret_args}" + "\n")

        code_str = "import re\n"
        code_str += self.source_str
        code_str += self.scu.modify_foward(op_forward, self.model_source_code)
        op_test_code = \
f'''model = {self.model_class_name}()
model.eval()
device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
dis = []
for idx, size in enumerate({sizes}):
    exec("dump_input%s=torch.randn(size=size).to(device)" % idx)
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
    outshape = dump_output.shape
    pattern = r'time total: (.+?s)'
    result = re.findall(pattern=pattern,
                          string=prof.table())
    if len(result) == 2:
        cpu_time, gpu_time = result[0], result[1]
    elif len(result) == 1:
        cpu_time, gpu_time = result[0], "0ms"
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
        print(code_str)
        GoogLeNet = None
        c = compile(code_str, "", mode="exec")
        exec(c)
        # print(globals()["cpu_times"])
        cpu_times = [float(i.replace("ms", "")) for i in globals()["cpu_times"]]
        gpu_times = [float(i.replace("ms", "")) for i in globals()["gpu_times"]]
        cpu_time, gpu_time = str(round(np.mean(cpu_times), 3)) + "ms", str(round(np.mean(gpu_times), 3)) + "ms"
        return globals()["outshape"], cpu_time, gpu_time

    def get_args(self, curr):
        for c in self.codes:
            if c.split("=")[0].strip() == curr:
                res = c
                break
        else:
            raise ValueError("The current node operator call code is not matched")
        pattern = r"\((.+?)\)"
        in_args = re.findall(pattern=pattern, string=res)[-1]
        in_args = [i.strip() for i in in_args.split(",") if not i.strip().isdigit()]
        in_args = ",".join(in_args)
        if "[" in in_args and "]" in in_args:
            in_args = re.findall(pattern=r"\[(.+?)\]", string=in_args)[0]
        return in_args, res

    def split_calculation(self, nexts=None):
        '''
        单算子拆分计算，获取测试信息
        :return:
        '''
        if nexts != None and len(nexts) == 0:
            return
        flag = True
        for k, v in self.nodes.items():
            if "size" in v.keys():
                flag = False
        if flag:
            in_args, call_code = self.get_args(self.head)
            out_shape, cpu_time, gpu_time = self.curr_node_result([self.input_shape], in_args, self.head, call_code)
            self.nodes[self.head]['size'] = out_shape
            self.nodes[self.head]['time'] = gpu_time
            nexts = self.nodes[self.head]['next']
            self.split_calculation(nexts)
        else:
            for next_node in nexts:
                args = self.nodes[next_node]['args']
                sizes = []
                for arg in args:
                    if "size" in self.nodes[arg].keys():
                        sizes.append(self.nodes[arg]['size'])
                if len(sizes) != len(args):
                    continue
                in_args, call_code = self.get_args(next_node)
                out_shape, cpu_time, gpu_time = self.curr_node_result(sizes, in_args, next_node, call_code)
                self.nodes[next_node]['size'] = out_shape
                self.nodes[next_node]['time'] = gpu_time
                nexts = self.nodes[next_node]['next']
                self.split_calculation(nexts)


def convert_excel(data, save_path, head):
    '''
    数据格式转化为excel形式储存
    :param data: 节点和节点相关信息
    :param save_path: excel储存路径
    :return: None
    '''
    statistical_results = []  # 收集测试数据
    dic0 = dict()
    dic0['task'] = "input"
    dic0['c_task'] = ""
    dic0['dir_succ'] = head
    dic0['dir_succ_add'] = head
    dic0['p0'] = 0
    statistical_results.append(dic0)
    for n, v in data.items():
        dic = {}
        dic['task'] = n
        edges = [n + "_2_" + i for i in v['next']]
        dic['c_task'] = ",".join(edges)
        dic['dir_succ'] = ",".join(v['next'])
        # dic['dir_succ_add'] = (dic['dir_succ'] + ", " + dic['c_task']) if dic['dir_succ'] else dic['c_task']
        dic['dir_succ_add'] = (dic['dir_succ'] + ", " + dic['c_task']) if dic['dir_succ'] else dic['c_task']
        dic['p0'] = float(v['time'].replace("ms", ""))
        dic['shape'] = str(tuple(v['size']))
        transfer_size = 1  # 初始化
        for i in tuple(v['size']):
            transfer_size *= i
        dic['transfer_size'] = (transfer_size * 32) // 1024
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


def main(source_path, model_class_name, model, input_shape, save_path):
    model_source_code = inspect.getsource(model)  # 模型定义的源码
    model = model()
    scu = SourceCodeUpdate(source_path, model)
    nodes = scu.get_nodes_and_code()
    OpS = OperatorStatistics(scu, nodes, model_source_code, model_class_name, input_shape)
    for i, v in OpS.nodes.items():
        print(i, v)
    # exit()
    convert_excel(OpS.nodes, save_path, OpS.head)


if __name__ == '__main__':
    source_path = "SourceCodeRefactor/dnn_models/GoogLeNet.py"
    model_class_name = "GoogLeNet"
    model = GoogLeNet
    input_shape = (1, 1, 96, 96)
    save_path = "./GoogLeNet.xls"
    main(source_path, model_class_name, model, input_shape, save_path)
