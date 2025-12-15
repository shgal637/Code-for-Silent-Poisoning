import copy
import logging
import os

import torch
from torch import nn, optim
from torch.nn import init

from common.constants import CHECKPOINT_PATH
from model.my_optimizer import MaliciousSGD


"""
Base class of all model architecture
"""


class BaseModel(nn.Module):
    def __init__(self, dataset, type, role, param_dict):
        nn.Module.__init__(self)
        self.dataset = dataset
        self.type = type
        self.role = role
        self.cuda = param_dict['cuda'] 
        self.learning_rate = param_dict['lr']

        self.optimizer = None
        self.scheduler = None
        self.param_dict = param_dict


    def init_optim(self, param_dict, init_weight=True):
        """
        initialize optimizer

        :param dict param_dict: parameters of optimizer
        :param bool init_weight: whether to initialize model weights
        """
        parameters = self.parameters()
        if self.type == 'bert':
            parameters = [{"params": self.bert.parameters(), "lr": 5e-6},
                          {"params": self.linear.parameters(), "lr": 5e-4}]

        # default use SGD except that param_dict requires using Adam
        if 'optim' in param_dict and param_dict['optim'] == 'adam':
            self.optimizer = optim.Adam(parameters, lr=param_dict['lr'])
        else:
            if 'mal_optim' in param_dict and param_dict['mal_optim']:
                self.optimizer = MaliciousSGD(parameters,momentum=param_dict['momentum'],
                                       weight_decay=param_dict['wd'],
                                       lr=param_dict['lr'],
                                        gamma_lr_scale_up=param_dict['s_r_amplify_ratio'])
            else:
                self.optimizer = optim.SGD(parameters,momentum=param_dict['momentum'],
                                       weight_decay=param_dict['wd'],
                                       lr=param_dict['lr'])

        # use MultiStepLR scheduler only param_dict contains stone
        if 'stone' in param_dict:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=param_dict['stone'],
                                                                  gamma=param_dict['gamma'])
            # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=5, verbose=False,
            #                                            threshold=0.0001, threshold_mode='abs', cooldown=0, min_lr=1e-5,
            #                                            eps=1e-05)

        if init_weight:
            # initialize model weights except for bert
            if self.type != 'bert':
                for m in self.modules():
                    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                        init.kaiming_normal_(m.weight)


    def save(self, name=None, id=None, time=None):
        """
        save model to local file located in CHECKPOINT_PATH/"dataset"

        :param str name: name of local file, use default name if not provided
        :param int id: id of passive party
        """
        path = '{}/{}/{}'.format(CHECKPOINT_PATH, self.dataset, time)
        if not os.path.exists(path):
            os.makedirs(path)
        if name is None:
            if id is None:
                filepath = '{}/{}_{}'.format(path, self.role, self.type)  # for active party
            else:
                filepath = '{}/{}_{}_{}'.format(path, self.role, self.type, id)  # for passive party
        else:
            filepath = '{}/{}'.format(path, name)
        torch.save(self.state_dict(), filepath)

    def load(self, name=None, id=None, time=None):
        """
        load model from local file located in CHECKPOINT_PATH/"dataset"
        :param name: name of local file, use default name if not provided
        :param id: id of passive party
        :return: bool, whether to load successfully or not
        """
        path = '{}/{}/{}'.format(CHECKPOINT_PATH, self.dataset, time)
        if name is None:
            if id is None:
                filepath = '{}/{}_{}'.format(path, self.role, self.type)  # for active party
            else:
                filepath = '{}/{}_{}_{}'.format(path, self.role, self.type, id)  # for passive party
        else:
            filepath = '{}/{}'.format(path, name)
        if os.path.isfile(filepath):
            checkpoint = torch.load(filepath)
            self.load_state_dict(checkpoint)
            return True
        else:
            return False

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.float()
        if self.cuda:
            x = x.cuda()
        return self.classifier(x)

    def predict(self, x):
        return self.forward(x)

    def backward_(self):
        """
        backward using gradients of optimizer
        """
        # self.optimizer.zero_grad()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def backward(self, x, y, grads, epoch=0):
        """
        backward using given grads on y or on x if y is None

        :param x: backward on x, only works if y is None
        :param y: backward on y
        :param grads: gradients for backward
        :param epoch: current epoch
        """
        # self.optimizer.zero_grad()

        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if y is not None:
            y.backward(gradient=grads)
        else:
            x = x.float()
            if self.cuda:
                x = x.cuda()
            output = self.forward(x)
            output.backward(gradient=grads)

        self.optimizer.step()
        self.optimizer.zero_grad()

    def my_backward(self, loss):
        '''
        the old backward
        Parameters
        ----------
        loss

        Returns
        -------

        '''
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
