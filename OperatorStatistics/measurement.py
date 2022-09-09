import OPS
import os
import torch
import time
import re
import json
import sklearn
from sklearn.model_selection import ParameterGrid


def op_profile(size, device, k, param_str, model):
    dump_input = torch.randn(size=size).to(device)
    s = f"operator: {k}" + "|" + f"kwargs: ({param_str})" + "|" + f"input_shape: {str(size)}"
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, use_kineto=True,
                                         use_cpu=True) as prof:
        outputs = model(dump_input)
    result = prof.table().split("\n")[-3:-1]
    pattern = r'total: (.+?s)'
    cpu_time = re.findall(pattern=pattern, string=result[0])[0]
    s += "|" + f"cpu_time: {cpu_time}"
    gpu_time = re.findall(pattern=pattern, string=result[1])[0]
    s += "|" + f"gpu_time: {gpu_time}"
    print(s)
    s += "\n"
    return s


def get_result(ops):
    for k, v in ops.items():
        operator = k
        # 所有的参数组合
        param_combinations = list(ParameterGrid(v['kwargs']))
        for comb in param_combinations:
            param_list = []
            for p, val in comb.items():
                param_list.append(p + "=" + str(val))
            param_str = ",".join(param_list)
            model = eval(v['transfer'] + f'({param_str})')
            device = torch.device('cuda', 0)
            model.eval()
            model.to(device)
            for bz in v['input_shape']['batch_size']:
                for w_h in v['input_shape']['w_h']:
                    if k == "Conv2d":
                        size = (bz, comb['in_channels'], w_h, w_h)
                        statistical_results.append(op_profile(size, device, k, param_str, model))
                    elif k == "ReLU":
                        for in_ch in v['input_shape']['in_channels']:
                            size = (bz, in_ch, w_h, w_h)
                            statistical_results.append(op_profile(size, device, k, param_str, model))
    return statistical_results


if __name__ == '__main__':
    statistical_results = []
    with open("config.json", "r") as f:
        ops = json.load(f)
    res = get_result(ops)
    with open("./算子统计.txt", "w") as file:
        file.writelines(res)


