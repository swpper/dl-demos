import time

import trtpy
import torch

from model import ConvLSTMNet


# path = ...

# model = torch.load(path)   # ?



model = ConvLSTMNet(
    input_channels=1, 
    hidden_channels=7, 
    kernel_size=(3, 3), 
    n_obs=5, 
    n_pred=20
).eval()



trt_model = trtpy.from_torch(
    model,
    dummy_input=torch.randn(1, 3, 224, 224),
    max_batch_size=16,
    onnx_save_file="model.onnx",
    engine_save_file="engine.trtmodel",
)

t0 = time.time()
trt_out = trt_model(torch.randn(16, 3, 224, 224))
t1 = time.time()
print("Inference time:", t1 - t0, 's')
print(trt_out.shape)