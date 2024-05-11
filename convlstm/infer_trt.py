'''
https://github.com/NVIDIA/TensorRT/blob/release/10.0/quickstart/IntroNotebooks/4.%20Using%20PyTorch%20through%20ONNX.ipynb
'''
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 1. 加载TRT引擎
with open("model.trt", "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# 2. 创建执行上下文
context = engine.create_execution_context()

BATCH_SIZE = 16

input_batch = np.random.random(size=(40, 5, 1, 64, 64), dtype=np.float16)
output = np.empty([BATCH_SIZE, 1000], dtype = target_dtype) 

# allocate device memory
d_input = cuda.mem_alloc(1 * input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()


# 3. 分配输入和输出内存
inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    if engine.binding_is_input(binding):
        inputs.append((host_mem, device_mem))
    else:
        outputs.append((host_mem, device_mem))

# 4. 执行推理
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # 将输入数据复制到GPU
    [cuda.memcpy_htod_async(inp[1], inp[0], stream) for inp in inputs]
    # 执行模型
    context.execute_async_v2(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # 将输出数据复制回CPU
    [cuda.memcpy_dtoh_async(out[0], out[1], stream) for out in outputs]
    # 等待流完成
    stream.synchronize()

'''
def predict(batch): # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()
    
    return output
'''

# 5. 准备输入数据
# （这里需要根据模型的输入尺寸和类型来填充输入数据）

# 6. 调用推理函数
outputs = do_inference(context, bindings, inputs, outputs, stream)

# 7. 处理输出数据
# （这里需要根据模型的输出尺寸和类型来解析输出数据）

print(outputs)

# 8. 释放资源
context.pop()
