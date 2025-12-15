# -*- coding: utf-8 -*-
import torch

from model.base_model import BaseModel
from model.cifar.vgg import Vgg16_net
"""
Bottom model architecture for CIFAR, Resnet20
"""


class VGGCinic(BaseModel, Vgg16_net):
    def __init__(self, param_dict):
        BaseModel.__init__(
            self,
            dataset=param_dict['dataset'],
            type='resnet',
            role=param_dict['role'],
            param_dict=param_dict
        )
        Vgg16_net.__init__(
            self,
            param_dict=param_dict
        )
        if self.cuda:
            self.to('cuda')
        self.is_debug = False

        self.output_dim = param_dict['output_dim']

        self.init_optim(param_dict)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.float()
        if self.cuda:
            x = x.cuda()
        return Vgg16_net.forward(self, x=x)

    def predict(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.float()
        if self.cuda:
            x = x.cuda()
        return Vgg16_net.forward(self, x=x)


def vgg_cinic(**kwargs):
    model = VGGCinic(**kwargs)
    return model
