import torch
import re
import json
import random
import os
import csv
from sklearn.model_selection import ParameterGrid


# torch.nn调用的算子性能测试方法
def op_profile(size, device, k, param_str, model):
    dump_input = torch.randn(size=size).to(device)
    res_dic = {}
    res_dic['operator'] = k
    res_dic['kwargs'] = param_str
    res_dic['input_shape'] = str(size)
    s = f"operator: {k}" + "|" + f"kwargs: ({param_str})" + "|" + f"input_shape: {str(size)}"
    with torch.autograd.profiler.profile(enabled=True,
                                         use_cuda=True,
                                         record_shapes=False,
                                         use_kineto=True,
                                         use_cpu=True) as prof:
        model(dump_input)
    # print(prof.table())
    return analysis(prof, s, res_dic)


def main(ops, csv_path):
    for k, v in ops.items():
        # 所有的参数组合
        ParameterGrid(v['kwargs'])
        param_combinations = list(ParameterGrid(v['kwargs']))
        # 对于无法实例化的算子
        if k in ["cat", "Linear", "BatchNorm2d", "add", "MultiheadAttention", "AdaptiveAvgPool2d"]:
            for comb in param_combinations:
                res_dic = {}
                if k == "AdaptiveAvgPool2d":
                    tensors_size = [3]
                    tensors_size.extend(random.sample(v['tensors_shape'], comb['tensor_dim']-1))
                    exec("output_size=tuple(tensors_size)")
                    param_str = "(output_size)"
                    call_str = v['transfer'] + param_str
                    model = eval(call_str)
                    device = torch.device('cuda', 0)
                    model.eval()
                    model.to(device)
                    for bz in v['input_shape']['batch_size']:
                        for w_h in v['input_shape']['w_h']:
                            for in_ch in v['input_shape']['in_channels']:
                                size = (bz, in_ch, w_h, w_h)
                                save2csv(op_profile(size, device, k, param_str, model), csv_path)
                    continue
                elif k == "cat":
                    tensors_size = random.sample(v['tensors_shape'], comb['tensor_dim'])
                    for idx in range(comb['tensor_count']):
                        exec(f"t{idx+1}=torch.randn(size=tensors_size, device='cuda')")
                    ts = ",".join([f"t{i}" for i in range(1, comb['tensor_count']+1)])
                    param_str = f"(tensors=[{ts}],dim={comb['op_dim']})"
                    call_str = v['transfer'] + param_str
                    res_dic['input_shape'] = f"{str(tensors_size)}x{comb['tensor_count']}"
                    s = f"operator: {k}" + "|" + f"kwargs: ({param_str})" + "|" + f"input_shape: {str(tensors_size)}x{comb['tensor_count']}"
                elif k == "add":
                    tensors_size = random.sample(v['tensors_shape'], comb['tensor_dim'])
                    exec("input=torch.randn(size=tensors_size, device='cuda')")
                    exec("other=torch.randn(size=tensors_size, device='cuda')")
                    exec("alpha=random.randint(1,100)")
                    param_str = "(input, other, alpha=alpha)"
                    call_str = v['transfer'] + param_str
                    res_dic['input_shape'] = ""
                    s = f"operator: {k}" + "|" + f"kwargs: ({param_str})" + "|" + f"input_shape:  "
                elif k in ["Linear", "BatchNorm2d", "MultiheadAttention"]:
                    param_list = []
                    for p, val in comb.items():
                        param_list.append(p + "=" + str(val))
                    param_str = ",".join(param_list)
                    call_str = v['transfer'] + f'({param_str})'
                    res_dic['input_shape'] = ""
                    s = f"operator: {k}" + "|" + f"kwargs: ({param_str})" + "|" + f"input_shape:  "
                res_dic['operator'] = k
                res_dic['kwargs'] = param_str
                with torch.autograd.profiler.profile(enabled=True,
                                                     use_cuda=True,
                                                     record_shapes=False,
                                                     use_kineto=True,
                                                     use_cpu=True) as prof:
                    eval(call_str)
                res_dic = analysis(prof, s, res_dic)
                save2csv(res_dic, csv_path)
            continue
        if k in ['add']:
            tensors_size = random.sample(v['tensors_shape'], 4)
            t = torch.randn(size=tensors_size, device="cuda")

            continue

        for comb in param_combinations:
            # 算子参数 😊
            param_list = []
            for p, val in comb.items():
                param_list.append(p + "=" + str(val))
            # 参数str 😒
            param_str = ",".join(param_list)
            if "nearest" in param_str:
                param_str = param_str.replace("align_corners =True,", "").replace("align_corners =False,", "")
            model = eval(v['transfer'] + f'({param_str})')

            device = torch.device('cuda', 0)
            model.eval()
            model.to(device)
            for bz in v['input_shape']['batch_size']:
                for w_h in v['input_shape']['w_h']:
                    if k in ["Conv2d", "ConvTranspose2d"]:
                        size = (bz, comb['in_channels'], w_h, w_h)
                        save2csv(op_profile(size, device, k, param_str, model), csv_path)
                    elif k in ["ReLU", "LeakyReLU", "SiLU", "MaxPool2d",
                               "AvgPool2d", "Dropout", "Flatten", "Upsample"]:
                        for in_ch in v['input_shape']['in_channels']:
                            size = (bz, in_ch, w_h, w_h)
                            save2csv(op_profile(size, device, k, param_str, model), csv_path)


# 解析测试结果
def analysis(prof, s, res_dic):
    result = prof.table().split("\n")[-3:-1]

    pattern = r'total: (.+?s)'
    cpu_time = re.findall(pattern=pattern,
                          string=result[0] if "CPU" in result[0] else result[1])[0]
    s += "|" + f"cpu_time: {cpu_time}"
    if "CUDA" in result[1]:
        gpu_time = re.findall(pattern=pattern,
                              string=result[1])[0]
    else:
        gpu_time = "0ms"
    s += "|" + f"gpu_time: {gpu_time}"
    print(s)
    res_dic['cpu_time'] = cpu_time if "us" not in cpu_time else str(
        float(cpu_time.replace("us", "")) / 1000) + "ms"
    res_dic['gpu_time'] = gpu_time if "us" not in gpu_time else str(
        float(gpu_time.replace("us", "")) / 1000) + "ms"
    return res_dic


# 存为csv格式
def save2csv(dic, path):
    headers = ["operator", "kwargs", "input_shape", "cpu_time", "gpu_time"]
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not os.path.getsize(path):
            writer.writeheader()
        writer.writerow(dic)


if __name__ == '__main__':
    with open("config.json", "r") as f:
        ops = json.load(f)
    csv_path = "/home/zjlab/wangds/OperatorStatistics/operator_statistics.csv"
    main(ops, csv_path=csv_path)
    # with open("/home/zjlab/wangds/OperatorStatistics/operator_statistics.txt", "w") as file:
    #     file.writelines(res)


