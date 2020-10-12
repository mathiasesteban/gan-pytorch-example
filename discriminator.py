import torch.nn as nn


class Discriminator(nn.Module):

    """ Class that defines the the Discriminator Neural Network """

    def __init__(self, inp, out):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(inp, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        return x
