# -*- coding: utf-8 -*-
from torch import nn
import torch.nn.functional as F
import torch
import numpy


class Discriminator(nn.Module):
    def __init__(self, args=None):
        super(Discriminator, self).__init__()
        self.args = args

        self.FC1 = torch.nn.Linear(in_features=10, out_features=30)
        self.FC2 = torch.nn.Linear(in_features=30, out_features=10)
        self.FC3 = torch.nn.Linear(in_features=10, out_features=2)

    def forward(self, X):
        out = torch.nn.functional.relu(self.FC1(X))
        out = torch.nn.functional.relu(self.FC2(out))
        out = self.FC3(out)
        return out
