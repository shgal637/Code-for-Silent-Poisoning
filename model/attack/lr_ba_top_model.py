# -*- coding: utf-8 -*-
from torch import nn
from torch.nn import init

from model.base_model import BaseModel

"""
Model architecture of the inference head, FCN-1
"""


class LrBaTopModel(BaseModel):
    def __init__(self, param_dict, ema=False):
        super(LrBaTopModel, self).__init__(
            dataset=param_dict['dataset'],
            type='fcn',
            role='top',
            param_dict=param_dict
        )
        self.param_dict = param_dict
        if self.param_dict['model_type'] == 'FCN':
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(param_dict['input_dim']),
                nn.ReLU(),
                nn.Linear(in_features=param_dict['input_dim'], out_features=32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(in_features=32, out_features=param_dict['output_dim'])
            )
        else:
            if self.param_dict['model_type'] == 'Resnet-1':
                if self.dataset == 'CIFAR10':
                    features_num = 128
                else:
                    features_num = 512
                self.conv1 = nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=0, bias=False)
                self.bn1 = nn.BatchNorm2d(16)
                self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0, bias=False)
                self.bn2 = nn.BatchNorm1d(320)
                self.linear1 = nn.Linear(in_features=320, out_features=features_num)
                self.linear2 = nn.Linear(in_features=features_num, out_features=param_dict['output_dim'])
            elif self.param_dict['model_type'] == 'Resnet-2-0':
                kernel_size = (3, 3)
                self.in_planes = 128
                self.conv1 = nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=1, bias=False)  # bs,128,8,4
                self.bn1 = nn.BatchNorm2d(128)
                self.conv2 = nn.Conv2d(128, 256, kernel_size=kernel_size, stride=2, padding=1, bias=False)  # bs,256,4,2
                self.bn2 = nn.BatchNorm2d(256)
                self.linear = nn.Linear(256, param_dict['output_dim'])

        self.is_debug = False
        self.input_dim = param_dict['input_dim']

        for layer in self.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                init.ones_(layer.weight)

        if ema:
            for param in self.parameters():
                param.detach_()

        if param_dict['cuda']:
            self.to('cuda')

    def forward(self, x):
        if self.param_dict['model_type'] == 'FCN':
            out = self.classifier(x)
        else:
            if self.param_dict['model_type'] == 'Resnet-1':
                out = nn.functional.relu(self.bn1(self.conv1(x)))
                out = nn.functional.relu(self.bn1(self.conv2(out)))
                out = out.view(out.size(0), -1)
                out = self.linear1(self.bn2(nn.functional.relu(out)))
                out = self.linear2(nn.functional.relu(out))
            elif self.param_dict['model_type'] == 'Resnet-2-0':
                out = nn.functional.relu(self.bn1(self.conv1(x)))
                out = nn.functional.relu(self.bn2(self.conv2(out)))
                out = nn.functional.avg_pool2d(out, out.size()[2:])
                out = out.view(out.size(0), -1)
                out = self.linear(out)
        return out


def lr_ba_top_model(**kwargs):
    model = LrBaTopModel(**kwargs)
    return model
