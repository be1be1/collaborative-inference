import argparse
import torch
from GoogLeNet_6345 import *
import time

model = GoogLeNet()
x = torch.randn(size=(1,1,96,96))
t1 = time.time()
for i in range(16):
    y = model.forward(x)
    print("current frame: %s %s"% (type(ray.get(y)), ray.get(y).shape))
t2 = time.time() - t1    
print(f"inference result: {ray.get(y).shape}")
print(f"time： {t2}s")
