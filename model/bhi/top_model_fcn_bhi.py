# -*- coding: utf-8 -*-

from torch import nn
import torch
from model.base_model import BaseModel
from model.cifar.resnet import BasicBlock
"""
Top model architecture for BHI, FCN-4
"""


class TopModelBhi(BaseModel):
    def __init__(self, param_dict, block, num_blocks, kernel_size, num_classes):
        super(TopModelBhi, self).__init__(
            dataset=param_dict['dataset'],
            type='fcn',
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
        if self.param_dict['model_type'] == 'FCN':
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(param_dict['input_dim']),
                nn.ReLU(),
                nn.Linear(in_features=param_dict['input_dim'], out_features=10),
                nn.BatchNorm1d(10),
                nn.ReLU(),
                nn.Linear(in_features=10, out_features=10),
                nn.BatchNorm1d(10),
                nn.ReLU(),
                nn.Linear(in_features=10, out_features=10),
                nn.BatchNorm1d(10),
                nn.ReLU(),
                nn.Linear(in_features=10, out_features=param_dict['output_dim'])
            )
        else:
            if self.param_dict['model_type'] == 'Resnet-1':
                self.in_planes = 32
                self.conv1 = nn.Conv2d(32, 32, kernel_size=kernel_size, stride=1, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(32)  # bs,32,25,50
                self.layer1 = self._make_layer(block, 32, num_blocks[0], kernel_size, stride=1)  # bs,32,25,50
                self.layer2 = self._make_layer(block, 64, num_blocks[1], kernel_size, stride=2)  # bs,64,13,25
                self.linear = nn.Linear(64, param_dict['output_dim'])

        self.is_debug = False
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
            out = nn.functional.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = nn.functional.avg_pool2d(out, out.size()[2:])
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out

class TopModelFcnBhi(TopModelBhi, BaseModel):
    def __init__(self, param_dict):
        BaseModel.__init__(
            self,
            dataset=param_dict['dataset'],
            type='resnet',
            role='top',
            param_dict=param_dict
        )
        TopModelBhi.__init__(
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
        return TopModelBhi.forward(self, x=x)

    def predict(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.float()
        if self.cuda:
            x = x.cuda()
        return TopModelBhi.forward(self, x=x)

def top_model_fcn_bhi(**kwargs):
    model = TopModelFcnBhi(**kwargs)
    return model
