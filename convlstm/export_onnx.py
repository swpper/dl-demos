

import torch
import torch.nn as nn
import torch.nn.functional as F


from model import ConvLSTMNet


model = ConvLSTMNet(
    input_channels=1, 
    hidden_channels=7, 
    kernel_size=(3, 3), 
    n_obs=5, 
    n_pred=20
).eval()

encoder_tensor = torch.randn(15, 5, 1, 64, 64)
input_names = ['encoder_input']
output_names = ['output']

torch.onnx.export(
    model,
    encoder_tensor,
    'model.onnx',
    input_names=input_names,
    output_names=output_names,
    opset_version=15,
    do_constant_folding=True,
    dynamic_axes={
        'encoder_input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)



