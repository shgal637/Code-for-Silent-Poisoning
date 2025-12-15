# !/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
from model.base_model import BaseModel
from common.utils import accuracy

class FP():
    """
    Fine Pruning Defense is described in the paper 'Fine-Pruning'_ by KangLiu. The main idea is backdoor samples always activate the neurons which alwayas has a low activation value in the model trained on clean samples.

    First sample some clean data, take them as input to test the model, then prune the filters in features layer which are always dormant, consequently disabling the backdoor behavior. Finally, finetune the model to eliminate the threat of backdoor attack.

    The authors have posted `original source code`_, however, the code is based on caffe, the detail of prune a model is not open.

    Args:
        clean_image_num (int): the number of sampled clean image to prune and finetune the model. Default: 50.
        prune_ratio (float): the ratio of neurons to prune. Default: 0.02.
        # finetune_epoch (int): the epoch of finetuning. Default: 10.


    .. _Fine Pruning:
        https://arxiv.org/pdf/1805.12185


    .. _original source code:
        https://github.com/kangliucn/Fine-pruning-defense

    .. _related code:
        https://github.com/jacobgil/pytorch-pruning
        https://github.com/eeric/channel_prune


    """

    def __init__(self, args, vfl, prune_ratio=0.95, finetune_epoch=10, max_allowed_acc_drop=0.2,valid_loader=None):
        super(FP, self).__init__()
        self.args = args
        self.prune_ratio = prune_ratio
        self.finetune_epoch = finetune_epoch
        self.max_allowed_acc_drop = max_allowed_acc_drop
        self.vfl = vfl

        for name, module in reversed(list(self.vfl.active_party.top_model.named_modules())):
            if isinstance(module, nn.Conv2d):
                last_conv = module
                self.prune_layer: str = name
                break
        else:
            raise Exception('There is no Conv2d in model.')

        length = last_conv.out_channels
        self.prune_num = int(length * self.prune_ratio)
        self.valid_loader = valid_loader
        self.test_loader = valid_loader

        self.container = []

    def detect(self):
        self.ori_clean_acc = self.test(test_loader=self.test_loader)
        self.prune()

    def test(self, test_loader=None):
        y_predict = []
        y_true = []
        top_k = 1
        if self.args['dataset'] == 'cifar100':
            top_k = 5

        with torch.no_grad():
            self.vfl.set_eval()
            for batch_idx, (X, targets, old_imgb, indices) in enumerate(test_loader):
                party_X_test_dict = dict()
                if self.args['dataset'] != 'bhi' and self.args['n_passive_party'] < 2:
                    active_X_inputs, Xb_inputs = X
                    if self.args['dataset'] == 'yahoo':
                        active_X_inputs = active_X_inputs.long()
                        Xb_inputs = Xb_inputs.long()
                        targets = targets[0].long()
                    if self.args['cuda']:
                        active_X_inputs = active_X_inputs.cuda()
                        Xb_inputs = Xb_inputs.cuda()
                        targets = targets.cuda()
                    party_X_test_dict[0] = Xb_inputs
                else:
                    if self.args['cuda']:
                        X = X.cuda()
                        targets = targets.cuda()
                    active_X_inputs = X[:, 0:1].squeeze(1)
                    for i in range(self.args['n_passive_party']):
                        party_X_test_dict[i] = X[:, i+1:i+2].squeeze(1)
                y_true += targets.data.tolist()
                y_prob_preds = self.vfl.predict(active_X_inputs, party_X_test_dict, type=None)
                y_predict += y_prob_preds.tolist()
        acc = accuracy(y_true, y_predict, top_k=top_k, num_classes=self.args['num_classes'], dataset=self.args['dataset'], is_attack=False)
        return acc

    def prune(self):

        for name, module in list(self.vfl.active_party.top_model.named_modules()):
            if isinstance(module, nn.Linear):
                self.last_conv: nn.Linear = prune.identity(module, 'weight')
                break

        # length = self.last_conv.weight.shape[1]
        # print('length :', length)  # 256

        mask: torch.Tensor = self.last_conv.weight_mask


        assert self.prune_num >= self.finetune_epoch, "prune_ratio too small!"
        self.prune_step(mask, prune_num=max(self.prune_num - self.finetune_epoch, 0))
        acc = self.test(test_loader=self.test_loader)
        print('final test acc: ', acc)

    def forward_hook(self, module, input, output):
        self.container.append(input[0])

    @torch.no_grad()
    def prune_step(self, mask: torch.Tensor, prune_num: int = 1):
        self.container = []
        if prune_num <= 0: return
        hook = self.last_conv.register_forward_hook(self.forward_hook)

        for batch_idx, (X, targets, old_imgb, indices) in enumerate(self.valid_loader):
            party_X_test_dict = dict()
            if self.args['dataset'] != 'bhi' and self.args['n_passive_party'] < 2:
                active_X_inputs, Xb_inputs = X
                if self.args['dataset'] == 'yahoo':
                    active_X_inputs = active_X_inputs.long()
                    Xb_inputs = Xb_inputs.long()
                    targets = targets[0].long()
                if self.args['cuda']:
                    active_X_inputs = active_X_inputs.cuda()
                    Xb_inputs = Xb_inputs.cuda()
                    targets = targets.cuda()
                party_X_test_dict[0] = Xb_inputs
            else:
                if self.args['cuda']:
                    X = X.cuda()
                    targets = targets.cuda()
                active_X_inputs = X[:, 0:1].squeeze(1)
                for i in range(self.args['n_passive_party']):
                    party_X_test_dict[i] = X[:, i + 1:i + 2].squeeze(1)

            comp_list = []
            # passive parties send latent representations
            for id in self.vfl.party_ids:
                comp_list.append(self.vfl.party_dict[id].predict(party_X_test_dict[id], is_attack=False))

            bottom_y = self.vfl.active_party.bottom_model.forward(active_X_inputs)
            grad_comp_list = [bottom_y] + comp_list
            if self.args['aggregate'] == 'Concate':
                temp = torch.cat(grad_comp_list, -1)
            elif self.args['aggregate'] == 'Add':
                temp = grad_comp_list[0]
                for comp in grad_comp_list[1:]:
                    temp = temp + comp
            else:
                temp = grad_comp_list[0]
                for comp in grad_comp_list[1:]:
                    temp = temp + comp
                temp = temp / len(grad_comp_list)

            self.vfl.active_party.top_model.forward(temp)

        hook.remove()
        feats_list = torch.cat(self.container).mean(dim=0)  # [10] [256]
        idx_rank = feats_list.argsort()
        counter = 0
        for idx in idx_rank:
            if mask[:, idx].norm(p=1) > 1e-6:
                mask[:, idx] = 0.0
                counter += 1
                # print(f'[{counter}/{prune_num}] Pruned channel id {idx}/{len(idx_rank)}')
                if counter >= min(prune_num, len(idx_rank)):
                    break


    def prune_step2(self, mask: torch.Tensor, prune_num: int = 1):
        self.container = []
        if prune_num <= 0: return
        hook = self.last_conv.register_forward_hook(self.forward_hook)

        for batch_idx, (X, targets, old_imgb, indices) in enumerate(self.valid_loader):
            party_X_test_dict = dict()
            if self.args['dataset'] != 'bhi' and self.args['n_passive_party'] < 2:
                active_X_inputs, Xb_inputs = X
                if self.args['dataset'] == 'yahoo':
                    active_X_inputs = active_X_inputs.long()
                    Xb_inputs = Xb_inputs.long()
                if self.args['cuda']:
                    active_X_inputs = active_X_inputs.cuda()
                    Xb_inputs = Xb_inputs.cuda()
                party_X_test_dict[0] = Xb_inputs
            else:
                if self.args['cuda']:
                    X = X.cuda()
                active_X_inputs = X[:, 0:1].squeeze(1)
                for i in range(self.args['n_passive_party']):
                    party_X_test_dict[i] = X[:, i + 1:i + 2].squeeze(1)
            comp_list = []
            # passive parties send latent representations
            for id in self.vfl.party_ids:
                comp_list.append(self.vfl.party_dict[id].predict(party_X_test_dict[id], is_attack=False))

            bottom_y = self.vfl.active_party.bottom_model.forward(active_X_inputs)
            grad_comp_list = [bottom_y] + comp_list
            if self.args['aggregate'] == 'Concate':
                temp = torch.cat(grad_comp_list, -1)
            elif self.args['aggregate'] == 'Add':
                temp = grad_comp_list[0]
                for comp in grad_comp_list[1:]:
                    temp = temp + comp

            self.vfl.active_party.top_model.forward(temp)

        hook.remove()
        feats_list = torch.cat(self.container).mean(dim=[0, 2, 3]) 
        idx_rank = feats_list.argsort()
        num_channels = len(feats_list)
        prunned_channels = int(num_channels * prune_num)
        mask = torch.ones(num_channels).cuda()
        for element in idx_rank[:prunned_channels]:
            mask[element] = 0
        mask = mask.reshape(1, -1, 1, 1)
