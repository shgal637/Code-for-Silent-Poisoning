# -*- coding: utf-8 -*-

from torch import nn

from model.base_model import BaseModel
from model.cifar.resnet import BasicBlock
import torch

"""
Top model architecture for CIFAR, FCN-4
"""


class TopModelCifar(BaseModel):
    def __init__(self, param_dict, block, num_blocks, kernel_size, num_classes):
        super(TopModelCifar, self).__init__(
            dataset=param_dict['dataset'],
            type='resnet',
            role='top',
            param_dict=param_dict
        )
        BaseModel.__init__(
            self,
            dataset=param_dict['dataset'],
            type='resnet',
            role='top',
            param_dict=param_dict
        )
        self.param_dict = param_dict
        if param_dict['dataset'] == 'cifar10':
            if self.param_dict['model_type'] == 'FCN':
                self.classifier = nn.Sequential(
                    nn.BatchNorm1d(param_dict['input_dim']),
                    nn.ReLU(),
                    nn.Linear(in_features=param_dict['input_dim'], out_features=64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Linear(in_features=64, out_features=32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Linear(in_features=32, out_features=param_dict['output_dim'])
                )
            else:
                if self.param_dict['model_type'] == 'Resnet-1':
                    self.in_planes = 64
                    self.conv1 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=1, bias=False)
                    self.bn1 = nn.BatchNorm2d(64)
                    self.layer1 = self._make_layer(block, 64, num_blocks[0], kernel_size, stride=1)
                    self.layer2 = self._make_layer(block, 128, num_blocks[1], kernel_size, stride=2)
                    self.linear = nn.Linear(128, param_dict['output_dim'])
                else:
                    if self.param_dict['model_type'] == 'Resnet-2-0':
                        self.in_planes = 128
                        # self.conv1 = nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding=1, bias=False)
                        self.conv1 = nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=1, bias=False)
                        self.bn1 = nn.BatchNorm2d(128)
                        self.layer1 = self._make_layer(block, 128, num_blocks[0], kernel_size, stride=1)
                        self.layer2 = self._make_layer(block, 256, num_blocks[1], kernel_size, stride=2)
                        self.linear = nn.Linear(256, param_dict['output_dim'])
                    else:
                        self.in_planes = 64
                        self.layer3 = self._make_layer(block, 64, num_blocks[0], kernel_size, stride=1)
                        self.layer1 = self._make_layer(block, 128, num_blocks[0], kernel_size, stride=2)
                        self.layer2 = self._make_layer(block, 256, num_blocks[1], kernel_size, stride=2)
                        self.linear = nn.Linear(256, param_dict['output_dim'])
        else:
            if self.param_dict['model_type'] == 'FCN':
                self.classifier = nn.Sequential(
                    nn.BatchNorm1d(param_dict['input_dim']),
                    nn.ReLU(),
                    nn.Linear(in_features=param_dict['input_dim'], out_features=200),
                    nn.BatchNorm1d(200),
                    nn.ReLU(),
                    nn.Linear(in_features=200, out_features=100),
                    nn.BatchNorm1d(100),
                    nn.ReLU(),
                    nn.Linear(in_features=100, out_features=100),
                    nn.BatchNorm1d(100),
                    nn.ReLU(),
                    nn.Linear(in_features=100, out_features=param_dict['output_dim'])
                )
            else:
                if self.param_dict['model_type'] == 'Resnet-1':
                    self.in_planes = 64
                    self.conv1 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=1, bias=False)   # bs,64,16,16
                    self.bn1 = nn.BatchNorm2d(64)
                    self.layer1 = self._make_layer(block, 128, num_blocks[0], kernel_size, stride=2)  # bs,128,8,8
                    self.layer2 = self._make_layer(block, 256, num_blocks[1], kernel_size, stride=2)  # bs,256,4,4
                    self.linear = nn.Linear(256, param_dict['output_dim'])
                elif self.param_dict['model_type'] == 'Resnet-2-0':
                    self.in_planes = 128
                    self.conv1 = nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, padding=1, bias=False)  # bs,128,8,8
                    self.bn1 = nn.BatchNorm2d(128)
                    self.layer1 = self._make_layer(block, 128, num_blocks[0], kernel_size, stride=1)  # bs,128,8,8
                    self.layer2 = self._make_layer(block, 256, num_blocks[1], kernel_size, stride=2)  # bs,256,4,4
                    self.linear = nn.Linear(256, param_dict['output_dim'])

        self.init_optim(param_dict)

    def _make_layer(self, block, planes, num_blocks, kernel_size, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.param_dict['model_type'] == 'FCN':
            out = self.classifier(x)
        else:
            if self.param_dict['model_type'] != 'Resnet-2-1':
                out = nn.functional.relu(self.bn1(self.conv1(x)))
            else:
                out = self.layer3(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = nn.functional.avg_pool2d(out, out.size()[2:])
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out

class TopModelFcnCifar(TopModelCifar, BaseModel):
    def __init__(self, param_dict):
        BaseModel.__init__(
            self,
            dataset=param_dict['dataset'],
            type='resnet',
            role='top',
            param_dict=param_dict
        )
        TopModelCifar.__init__(
            self,
            block=BasicBlock,
            num_blocks=[3, 3, 3],
            kernel_size=(3, 3),
            num_classes=param_dict['output_dim'],
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
        return TopModelCifar.forward(self, x=x)

    def predict(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.float()
        if self.cuda:
            x = x.cuda()
        return TopModelCifar.forward(self, x=x)

def top_model_fcn_cifar(**kwargs):
    model = TopModelFcnCifar(**kwargs)
    return model
