from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
def silu(input):
    return input * torch.sigmoid(input)

class SiLU(nn.Module):

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return silu(input)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class VGG(nn.Module):
    def __init__(self, vgg_name, act_fn='relu'):
        super(VGG, self).__init__()

        if act_fn == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act_fn == 'relu':
            self.act = torch.nn.ReLU()
        elif act_fn == 'tanh':
            self.act = torch.nn.Tanh()
        elif act_fn == 'silu':
            self.act = SiLU()
        elif self.act_fn == 'elu':
            self.act = torch.nn.ELU()

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           self.act
                           ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)