import unittest

import torch
import torch.nn as nn

from convlstm import ConvLSTM, ConvLSTMCell
from model import ConvLSTMNet, TimeDistributed


class TestModel(unittest.TestCase):
    def test_convlstm(self):
        model = ConvLSTMNet(input_channels=1, hidden_channels=7, kernel_size=(3, 3))
        print(model)
        encoder_input = torch.randn(1, 5, 1, 32, 32)
        decoder_input = torch.randn(1, 1, 1, 32, 32)
        y = model(encoder_input, decoder_input)
        print(y.shape)
        y = model(encoder_input)
        print(y.shape)


    def test_time_distributed(self):
        # fc = nn.Linear(10, 5)
        fc = nn.Conv2d(10, 5, 3)
        td_fc = TimeDistributed(fc)

        input_data = torch.randn(5, 3, 10, 32, 32)
        output_data = td_fc(input_data)

        print(output_data.size())


if __name__ == '__main__':
    unittest.main()