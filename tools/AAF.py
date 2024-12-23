from torch import nn
import torch


class AAF(nn.Module):
    def __init__(self, channel, reduction=16, pool="avg"):
        super(AAF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if pool == "max":
            self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel / reduction), bias=False),
            nn.ReLU(inplace=True),  # inplace=True sometimes slightly decrease the memory usage
            # nn.Sigmoid(),
            nn.Linear(int(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1).to(device)
        return x * y.expand_as(x)
    
    
