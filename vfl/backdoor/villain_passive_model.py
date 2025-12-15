# -*- coding: utf-8 -*-
import logging
import math
import random
import numpy as np

import torch

from vfl.party_models import VFLPassiveModel
from vfl.defense.norm_clip import norm_clip
from model.DiscriminativeNetwork import Discriminator
from common.constants import CHECKPOINT_PATH
import os
"""
Malicious passive party for gradient-replacement backdoor
"""

class Vallain_PassiveModel(VFLPassiveModel):

    def __init__(self, bottom_model, amplify_ratio=1, args=None):
        self.target_grad = None
        self.target_indices = None
        self.amplify_ratio = amplify_ratio
        self.components = None
        self.is_debug = False
        self.pair_set = dict()
        self.target_gradients = dict()
        self.backdoor_X = dict()
        self.args = args
        super().__init__(bottom_model, args=args)

        self.attack = False
        self.pattern_lr = torch.tensor(self.args['pattern_lr'])
        self.feature_pattern = None
        self.mask = None
        self.drop_out = True
        self.drop_out_rate = 0.75
        self.max_norms = None
        self.shifting = True
        self.up_bound = 1.2
        self.down_bound = 0.6
        self.adversary = self.args['adversary'] - 1

        if self.args['cuda']:
            self.pattern_lr = self.pattern_lr.cuda()

    def set_train(self):
        """
        set train mode
        """
        self.bottom_model.train()

    def set_eval(self):
        """
        set eval mode
        """
        self.bottom_model.eval()

    def save_data(self, name=None, pattern=None):
        """
        save the pixel pattern or feature pattern
        """
        path = '{}/{}/{}'.format(CHECKPOINT_PATH, self.args['dataset'], self.args['file_time'])
        if not os.path.exists(path):
            os.makedirs(path)
        name = self.args['trigger'] + '_pattern.pt'
        filepath = '{}/{}'.format(path, name)
        if 'feature' in self.args['trigger']:
            torch.save(self.feature_pattern, filepath)
        else:
            torch.save(pattern, filepath)

    def load_data(self, name=None):
        """
        save the pixel pattern or feature pattern
        """
        path = '{}/{}/{}'.format(CHECKPOINT_PATH, self.args['dataset'], self.args['load_time'])
        name = self.args['trigger'] + '-' + str(self.args['epsilon']) + '_pattern.pt'
        filepath = '{}/{}'.format(path, name)
        if os.path.isfile(filepath):
            pattern = torch.load(filepath)
            if pattern is None:
                print(filepath)
                raise ValueError('load pattern error!')
            if 'feature' in self.args['trigger']:
                self.feature_pattern = pattern
            return pattern
        else:
            raise ValueError("load data error, wrong filepath")

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.norm = 0
        self.avg_target_features = None
        self.avg_target_numbers = 0
        self.avg_target_norms = []
        self.dis_rate = 0

    def set_backdoor_indices(self, target_indices, train_loader):
        self.target_indices = target_indices
        self.SR_train_loader = train_loader
        self.target_sample = []
        self.all_target_sample = []
        # self.old_target_sample = []
        for batch_idx, (X, Y_batch, old_imgb, indices) in enumerate(self.SR_train_loader):
            target_indices = []
            all_target_indices = []
            for i in range(len(indices)):
                if indices[i].item() in self.target_indices:
                    target_indices.append(i)
                if Y_batch[i].item() == self.args['backdoor_label']:
                    all_target_indices.append(i)
            if len(target_indices) != 0 or len(all_target_indices) != 0:
                if self.args['dataset'] != 'bhi' and self.args['n_passive_party'] < 2:
                    _, Xb_batch = X
                    if self.args['cuda']:
                        Xb_batch = Xb_batch.cuda()
                        old_imgb = old_imgb.cuda()
                else:
                    if self.args['cuda']:
                        X = X.cuda()
                        old_imgb = old_imgb.cuda()
                    Xb_batch = X[:, self.adversary + 1:self.adversary + 2].squeeze(1)
                for i in target_indices:
                    self.target_sample.append(old_imgb[i])
                for i in all_target_indices:
                    self.all_target_sample.append(old_imgb[i])
        self.target_sample = torch.stack(self.target_sample)
        self.all_target_sample = torch.stack(self.all_target_sample)
        if self.args['cuda']:
            self.target_sample = self.target_sample.cuda()
            self.all_target_sample = self.all_target_sample.cuda()

    def set_batch(self, X, indices):
        self.X = X.clone().detach()
        self.indices = indices
        self.X.requires_grad = True


    def receive_gradients(self, gradients):
        temp_grad = []
        original_gradients = gradients.clone()
        gradients = gradients.clone()
        gradients = self.amplify_ratio * gradients

        self.common_grad = gradients
        # backwards
        self._fit(self.X, self.components)
        return

    def send_components(self):
        result = self._forward_computation(self.X)  # [bs,3,50,50]
        self.components = result
        send_result = result.clone()
        old_send_result = result.clone()

        for index, i in enumerate(self.indices):
            if self.target_indices is not None and i.item() in self.target_indices:
                if self.feature_pattern is not None:
                    self.mask = torch.full_like(send_result[index], 1)
                    if self.drop_out:
                        self.mask = torch.nn.functional.dropout(self.mask, p=self.drop_out_rate, training=True)
                        self.mask = self.mask * (1 - self.drop_out_rate)
                    if self.shifting:
                        num = random.random() * (self.up_bound - self.down_bound) + self.down_bound
                        send_result[index] += self.feature_pattern * self.mask * num
                    else:
                        send_result[index] += self.feature_pattern * self.mask
                    if self.epoch == self.args['target_epochs'] - 1:
                        self.bottom_model.eval()
                        with torch.no_grad():
                            self.mini_distance = None
                            avg_norm = 0
                            H_features = self.bottom_model.forward(self.all_target_sample)
                            for k in range(len(H_features)):
                                avg_norm += torch.norm(H_features[k], p=2)
                                difference = torch.norm(H_features[k] - send_result[index], p=2)
                                if self.mini_distance is None or difference < self.mini_distance:
                                    self.mini_distance = difference
                                    mini_feature = H_features[k]
                            self.norm += self.mini_distance
                            avg_norm = avg_norm / len(H_features)
                            self.dis_rate += self.mini_distance / torch.norm(mini_feature, p=2)
                        self.bottom_model.train()

        return send_result

    def predict(self, X, is_attack=False):
        result = self._forward_computation(X)
        send_results = result.clone()
        if is_attack and self.feature_pattern is not None:
            for i in range(len(send_results)):
                send_results[i] += self.feature_pattern

        return send_results
