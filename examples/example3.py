import ray
import torch
from torch.autograd import Variable
from dnn_models.google import *
import time
import warnings
# import sys
# sys.path.append("/data/wangds/model_restruction")

warnings.filterwarnings("ignore")
ray.init(address="auto")
x = torch.randn(1, 1, 96, 96)
inception = GoogLeNet()
t1 = time.time()
for i in range(1000):
    y = inception.forward(x)
t2 = time.time() - t1
print(f"time： {t2}s")