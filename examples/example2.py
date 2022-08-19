from dnn_models.inceptionv4 import *
from dnn_models.GoogLeNet import *
import time, ray
import warnings
import sys
import torch
sys.path.append("/data/wangds/SourceCodeRefactor")

warnings.filterwarnings("ignore")
ray.init(address="auto")
# ray.init()

x = torch.randn(1, 3, 299, 299)
# x = torch.randn(1, 1, 96, 96)

t1 = time.time()
model = InceptionV4()
y = model.forward(x)
print(y.shape)
t2 = time.time() - t1
print(f"time： {t2}s")

# y = ray.get(y)
# print(y.shape)
# print(f"time： {t2}s")


