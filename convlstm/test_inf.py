import time
import argparse

import numpy as np
import torch
import onnxruntime as ort

from model import ConvLSTMNet


N_OBS = 5
N_PER = 20
H = 64
W = 64
C = 1




# path = ...

# model = torch.load(path)   # ?





# 加载优化前的模型
model = ConvLSTMNet(
    input_channels=1, 
    hidden_channels=7, 
    kernel_size=(3, 3), 
    n_obs=5, 
    n_pred=20
).eval()


# 加载onnx模型
sess_ort = ort.InferenceSession(
    './model.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    # providers=['CPUExecutionProvider']
)
input_name = sess_ort.get_inputs()[0].name
output_name = sess_ort.get_outputs()[0].name
# print(f'input_name: {input_name}, output_name: {output_name}')



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batch_size', type=int, default=30, help='batch size')
    args = argparser.parse_args()

    batch_size = args.batch_size
    print("Batch size: ", batch_size)


    # 创建输入数据
    input_data = np.random.rand(batch_size, N_OBS, H, W, C).astype(np.float32)

    # 使用优化前的模型进行推理，并记录时间
    start_time = time.time()
    output_data_before_optimization = model(input_data)
    end_time = time.time()
    print("Inference time with model before optimization: ", end_time - start_time, type(output_data_before_optimization))

    # 使用优化后的模型进行推理，并记录时间
    # start_time = time.time()
    # output_data_after_optimization = model_after_optimization(input_data)
    # end_time = time.time()
    # print("Inference time with model after optimization: ", end_time - start_time, type(output_data_after_optimization))
    
    # 使用onnx模型进行推理，并记录时间
    start_time = time.time()
    onnx_output = sess_ort.run([output_name], {input_name: input_data})[0]
    end_time = time.time()
    print("Inference time with onnx model: ", end_time - start_time, type(onnx_output))

