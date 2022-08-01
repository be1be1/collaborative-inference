from dnn_models.GoogLeNet import *
import ray
ray.init()
inputs = torch.randn(1, 1, 96, 96)
model = GoogLeNet()
y = model(Variable(inputs))
print(y.shape)