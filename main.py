from dnn_models.darknet_53 import *
from dnn_models.GoogLeNet import *
from dnn_models.inceptionv4 import *
from dnn_models.inception_v3 import *
# from dnn_models.karate_club_net import *
from dnn_models.resnet import *
# from dnn_models.simple_gcn import *
from dnn_models.mynet import *
import torch
import inspect
from model_representation import ModelStruction
from source_code_rewriting import SourceCodeUpdate


def main(model, source_path):
    # 以str形式提取model定义源码
    model_source_code = inspect.getsource(model)

    model = model()
    model.eval()

    # torch.fx方法获取网络层信息和自动生成的代码
    print("获取网络层信息...")
    scu = SourceCodeUpdate(source_path, model)
    nodes = scu.get_nodes_and_code()
    # for code in codes:
    #     print(code)
    # for i, v in nodes.items():
    #     print(i, v)

    # 模型层分块打包
    print("网络层分块打包...")
    model_struction = ModelStruction(nodes)
    blocks = model_struction.get_blocks()
    # for i in blocks:
    #     print("block: ", i)
    # 需要插入的源码
    print("重构代码生成...")
    generate_codes = scu.generate_forward(blocks)
    print("注释源码中的forward方法，添加重构代码...")
    new_code = scu.modify_foward(generate_codes, model_source_code)
    print("替换源码...")
    scu.replace_source_code(new_code)
    print(f"success, modify source code -> {source_path}")


if __name__ == '__main__':
    # 源码路径
    source_path = "dnn_models/inception_v3.py"
    model = Inception3
    main(model, source_path)


