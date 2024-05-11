import os

import torch
import torch.nn as nn
import numpy as np
from convlstm import ConvLSTMCell, ConvLSTM


import torch.nn as nn


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        '''
        Args:
            x: (samples, time, channels, rows, cols)
        '''
        if len(x.size()) != 5:
            raise NotImplementedError('Input tensor should be 5D')
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)

        # outputs = []
        # for t in range(x.size(1)):
        #     xt = x[:, t, :, :, :]
        #     o = self.module(xt)
        #     outputs.append(o.unsqueeze(1))
        # outputs = torch.cat(outputs, dim=1)
        # return outputs

        batch_size, time_steps, features = x.size()[0], x.size()[1], tuple(x.size()[2:])
        x = x.view(-1, *features)
        outputs = self.module(x)
        outputs = outputs.view(batch_size, time_steps, *outputs.size()[1:])
        return outputs
        

class ConvLSTMNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, n_obs=5, n_pred=20):
        super(ConvLSTMNet, self).__init__()
        self.n_obs = n_obs
        self.n_pred = n_pred
        # encoder
        self.en_convlstm = ConvLSTM(
            input_channels, [hidden_channels, hidden_channels, hidden_channels],
            kernel_size, 3, True, return_all_layers=True
        )
        # decoder
        self.de_convlstm4 = ConvLSTM(input_channels, hidden_channels, kernel_size, 1, True)
        self.de_convlstm5 = ConvLSTM(hidden_channels, hidden_channels, kernel_size, 1, True)
        self.de_convlstm6 = ConvLSTM(hidden_channels, hidden_channels, kernel_size, 1, True)
        self.fc = TimeDistributed(nn.Conv2d(
            hidden_channels, 1, kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            # padding_mode='circular'
        ))
    
    def forward(self, encoder_input, decoder_input=None, n_pred=20):
        '''
        Args:
            encoder_input: (samples, time, channels, rows, cols)
            decoder_input: (samples, time, channels, rows, cols)
            n_pred: int, number of predictions to make in inference mode
        '''
        if decoder_input is not None:
            # Training mode
            # Teacher forcing: Feed the target as the next input
            encoder_output_list, [[h1, c1], [h2, c2], [h3, c3]] = self.en_convlstm(encoder_input)
            decoder_output_list4, [h4, c4] = self.de_convlstm4(decoder_input, [[h1, c1]])
            decoder_output_list5, [h5, c5] = self.de_convlstm5(decoder_output_list4, [[h2, c2]])
            decoder_output_list6, [h6, c6] = self.de_convlstm6(decoder_output_list5, [[h3, c3]])
            decoder_output = self.fc(decoder_output_list6)
            return decoder_output
        else:
            # Inference mode
            # Without teacher forcing: use its own predictions as the next input
            encoder_output_list, [[h1, c1], [h2, c2], [h3, c3]] = self.en_convlstm(encoder_input)
            for _ in range(n_pred):
                decoder_output_list4, [h4, c4] = self.de_convlstm4(encoder_input[:, -1:, :, :, :], [[h1, c1]])
                decoder_output_list5, [h5, c5] = self.de_convlstm5(decoder_output_list4, [[h2, c2]])
                decoder_output_list6, [h6, c6] = self.de_convlstm6(decoder_output_list5, [[h3, c3]])
                decoder_output = self.fc(decoder_output_list6)

                encoder_input = torch.cat((encoder_input, decoder_output), axis=1)
                h1, c1 = [h4, c4]
                h2, c2 = [h5, c5]
                h3, c3 = [h6, c6]
            
            return encoder_input[:, self.n_obs: , :, :, :]
    
        
    

if __name__ == '__main__':
    pass
    