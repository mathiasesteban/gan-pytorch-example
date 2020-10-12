import torch.nn as nn


class Generator(nn.Module):

    """ Class that defines the the Generator Neural Network """

    def __init__(self, inp, out):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
                nn.Linear(inp, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, out),
                nn.Tanh()
        )

    def forward(self, x):
        x = self.net(x)
        return x
