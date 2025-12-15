import copy
import logging
import random

import numpy
import numpy as np
import torch

from common.utils import accuracy, print_running_time
from vfl.backdoor.baseline_backdoor import baseline_backdoor
from vfl.backdoor.lr_ba_backdoor import lr_ba_backdoor, lr_ba_backdoor_for_representation, poison_predict, finetune_bottom_model
from vfl.vfl import VFL
import csv
from vfl.defense.fine_pruning import FP

use_wandb = False
# wandb
import wandb
import os
os.environ["WANDB_DISABLE_CODE"]="true"

# from torchsummary import summary

class VFLFixture(object):

    def __init__(self, vfl: VFL, args):
        self.vfl = vfl
        self.dataset = args['dataset']
        self.args = args
        # test wandb
        if use_wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                # project='debug' + self.args['backdoor'],
                project='sr_ba_'+self.args['trigger'],
                name=self.args['file_time'],
                # track hyperparameters and run metadata
                config=self.args
            )
        self.target_sample = None
        self.adversary = self.args['adversary']-1  # the index in passive parties

    def set_local_dataset(self, train_loader=None):
        # for villain and splitNN
        local_datasets = {}
        local_datasets[-1] = []
        labels = []
        all_indices = []
        for i in range(self.args['n_passive_party']):
            local_datasets[i] = []
        for batch_idx, (X, Y_batch, old_imgb, indices) in enumerate(train_loader):
            if self.args['n_passive_party'] < 2:
                active_X_batch, Xb_batch = X
                if self.args['cuda']:
                    active_X_batch = active_X_batch.cuda()  # batch,3,32,16
                    Xb_batch = Xb_batch.cuda()
                    Y_batch = Y_batch.cuda()
                    indices = indices.cuda()
                local_datasets[-1].append(active_X_batch)
                local_datasets[0].append(Xb_batch)
            else:
                if self.args['cuda']:
                    X = X.cuda()
                    Y_batch = Y_batch.cuda()
                    indices = indices.cuda()
                active_X_batch = X[:, 0:1].squeeze(1)
                local_datasets[-1].append(active_X_batch)
                for i in range(self.args['n_passive_party']):
                    Xb_batch = X[:, i + 1:i + 2].squeeze(1)
                    local_datasets[i].append(Xb_batch)
            labels.append(Y_batch)
            all_indices.append(indices)
        self.vfl.party_dict[self.adversary].local_dataset = local_datasets[self.adversary]
        self.vfl.party_dict[self.adversary].labels = labels
        self.vfl.party_dict[self.adversary].indices = all_indices

    def load_test(self, train_loader, test_loader, backdoor_test_loader=None, top_k=1):
        self.vfl.load()
        self.vfl.active_party.set_status('test')
        self.vfl.active_party.epoch = -1
        pattern = None
        if self.args['backdoor'] == 'villain':
            pattern = self.vfl.party_dict[self.adversary].load_data()
        if self.args['backdoor'] == 'sr_ba':
            if self.args['idea'] == 2:
                self.vfl.party_dict[self.adversary].load_data_idea2()
            else:
                pattern = self.vfl.party_dict[self.adversary].load_data()
            if self.args['trigger'] == 'pixel':
                if self.args['trigger_add']:
                    backdoor_test_loader.dataset.pixel_pattern = pattern
                else:
                    backdoor_test_loader.dataset.pattern_mask = pattern
        if self.args['backdoor'] == 'TECB':
            pattern = self.vfl.party_dict[self.adversary].load_data()
            backdoor_test_loader.dataset.pixel_pattern = pattern

        train_acc, train_ARR = self.predict(train_loader, num_classes=self.args['num_classes'],
                                            dataset=self.args['dataset'], top_k=top_k,
                                            n_passive_party=self.args['n_passive_party'], type='train')

        if self.args['Teco']:
            if self.args['defense_threshold'] == 0:
                mads = np.asarray(self.vfl.active_party.thresholds)
                mads_sorted = sorted(mads, reverse=True)
                index_percent = int(self.args['poison_rate'] * len(mads_sorted))
                self.vfl.active_party.threshold = mads_sorted[index_percent]
            else:
                self.vfl.active_party.threshold = self.args['defense_threshold']
            print("self.vfl.active_party.threshold: ", self.vfl.active_party.threshold)

        self.vfl.active_party.epoch = 0
        if self.args['outlier_detection']:
            self.vfl.active_party.MAD_function()
        test_acc, test_ARR = self.predict(test_loader, num_classes=self.args['num_classes'],
                                          dataset=self.args['dataset'], top_k=top_k,
                                          n_passive_party=self.args['n_passive_party'], type='test')
        self.vfl.active_party.set_status('backdoor_test')
        backdoor_acc, _ = self.predict(backdoor_test_loader, num_classes=self.args['num_classes'],
                                       dataset=self.args['dataset'], top_k=top_k,
                                       n_passive_party=self.args['n_passive_party'], type='attack')
        self.vfl.active_party.set_status('test')
        print('train_acc: {}, train_ARR: {}, test_acc: {}, test_ARR:{}, backdoor_acc:{}'.format(train_acc, train_ARR,
                                                                                                test_acc, test_ARR,
                                                                                                backdoor_acc))
        if self.args['outlier_detection']:
            print('detection FP: {}, TP: {}'.format(self.vfl.active_party.Detection_FP / len(test_loader.dataset), self.vfl.active_party.Detection_TP/len(backdoor_test_loader.dataset)))
            print('benign_MAD: {}, backdoor_MAD: {}'.format(sum(self.vfl.active_party.benign_MAD)/len(self.vfl.active_party.benign_MAD),
                                                            sum(self.vfl.active_party.backdoor_MAD)/len(self.vfl.active_party.backdoor_MAD)))

        if self.args['Teco']:
            if len(self.vfl.active_party.attack_mads) != 0:
                print('attack_recon_loss: {}, len{}'.format(sum(self.vfl.active_party.attack_mads)/len(self.vfl.active_party.attack_mads), len(self.vfl.active_party.attack_mads)))
            if len(self.vfl.active_party.clean_mads) != 0:
                print('normal_recon_loss: {}, len{}'.format(sum(self.vfl.active_party.clean_mads)/len(self.vfl.active_party.clean_mads), len(self.vfl.active_party.clean_mads)))
            if self.vfl.active_party.attack_true_detection + self.vfl.active_party.attack_false_detection > 0:
                print('Teco DA: attack_TP: {}/{} {}%; clean_TP: {}/{} {}%'.format(
                    self.vfl.active_party.attack_true_detection, self.vfl.active_party.attack_false_detection,
                    self.vfl.active_party.attack_true_detection / (self.vfl.active_party.attack_true_detection
                                                               + self.vfl.active_party.attack_false_detection) * 100,
                    self.vfl.active_party.clean_true_detection, self.vfl.active_party.clean_false_detection,
                    self.vfl.active_party.clean_true_detection
                    / (self.vfl.active_party.clean_true_detection + self.vfl.active_party.clean_false_detection) * 100))

        if self.args['FP']:
            # print(self.vfl.active_party.top_model)
            fine_tune = FP(args=self.args, vfl=self.vfl, valid_loader=test_loader, prune_ratio=self.args['FP'])
            fine_tune.detect()
            self.vfl = fine_tune.vfl
            train_acc, train_ARR = self.predict(train_loader, num_classes=self.args['num_classes'],
                                                dataset=self.args['dataset'], top_k=top_k,
                                                n_passive_party=self.args['n_passive_party'], type='train')
            test_acc, test_ARR = self.predict(test_loader, num_classes=self.args['num_classes'],
                                              dataset=self.args['dataset'], top_k=top_k,
                                              n_passive_party=self.args['n_passive_party'], type='test')
            self.vfl.active_party.set_status('backdoor_test')
            backdoor_acc, _ = self.predict(backdoor_test_loader, num_classes=self.args['num_classes'],
                                           dataset=self.args['dataset'], top_k=top_k,
                                           n_passive_party=self.args['n_passive_party'], type='attack')
            self.vfl.active_party.set_status('test')
            print('after FP:')
            print(
                'train_acc: {}, train_ARR: {}, test_acc: {}, test_ARR:{}, backdoor_acc:{}'.format(train_acc, train_ARR,
                                                                                                  test_acc, test_ARR,
                                                                                                  backdoor_acc))

    def fit(self, train_loader=None, test_loader=None, backdoor_test_loader=None, title=''):
        """
        VFL training

        :param train_loader: loader of train dataset
        :param test_loader: loader of normal test dataset
        :param backdoor_test_loader: loader of backdoor test dataset
        :param title: attack type
        """
        top_k = 1
        if self.dataset == 'cifar100':
            top_k = 5
        # load VFL
        if self.args['load_model']:
            self.load_test(train_loader, test_loader, backdoor_test_loader, top_k)
        # train VFL
        else:
            self.set_local_dataset(train_loader)

            # record start time before federated training if evaluating execution time
            if self.args['time']:
                start_time = print_running_time(None, None)
            for ep in range(self.args['target_epochs']):
                if self.args['backdoor'] == 'TECB':
                    # prepare datas for different phase
                    if ep < self.vfl.party_dict[self.adversary].CBP_end_epoch:  # CBP phase
                        pass
                    elif ep >= self.vfl.party_dict[self.adversary].CBP_end_epoch and ep < self.vfl.party_dict[self.adversary].TGA_start_epoch:
                        # set backdoor_indices to none
                        train_loader.dataset.backdoor_indices = None
                    else:  # TGA phase
                        # self.vfl.party_dict[self.adversary].backdoor_indices = train_loader.dataset.TGA_backdoor_indices
                        # set backdoor_indices into none_target samples
                        # train_loader.dataset.backdoor_indices = train_loader.dataset.TGA_backdoor_indices
                        train_loader.dataset.backdoor_indices = None

                self.vfl.active_party.set_status('train')
                loss_list = []
                self.vfl.set_train()
                self.vfl.set_current_epoch(ep)

                for batch_idx, (X, Y_batch, old_imgb, indices) in enumerate(train_loader):
                    party_X_train_batch_dict = dict()
                    if self.args['dataset'] != 'bhi' and self.args['n_passive_party'] < 2:
                        active_X_batch, Xb_batch = X
                        if self.args['dataset'] == 'yahoo':
                            active_X_batch = active_X_batch.long()
                            Xb_batch = Xb_batch.long()
                            Y_batch = Y_batch[0].long()
                        if self.args['cuda']:
                            active_X_batch = active_X_batch.cuda()
                            Xb_batch = Xb_batch.cuda()
                            Y_batch = Y_batch.cuda()
                            indices = indices.cuda()
                            old_imgb = old_imgb.cuda()
                        if self.args['debug'] and batch_idx == 0:
                            print('fit data is on ', active_X_batch.device)
                        party_X_train_batch_dict[0] = Xb_batch
                        if self.args['backdoor'] == 'sr_ba':
                            self.vfl.party_dict[self.adversary].original_X = old_imgb
                            # self.vfl.party_dict[self.adversary].original_Y = Y_batch
                    else:
                        if self.args['cuda']:
                            X = X.cuda()
                            Y_batch = Y_batch.cuda()
                            indices = indices.cuda()
                        active_X_batch = X[:, 0:1].squeeze(1)  # bs,N_party,3,50,50
                        for i in range(self.args['n_passive_party']):
                            party_X_train_batch_dict[i] = X[:, i+1:i+2].squeeze(1)  # bs,3,50,50
                        if self.args['debug'] and batch_idx == 0:
                            print('fit data is on ', active_X_batch.device)
                        if self.args['backdoor'] == 'sr_ba':
                            self.vfl.party_dict[self.adversary].original_X = old_imgb

                    if self.args['backdoor'] == 'TECB' and ep >= self.vfl.party_dict[self.adversary].TGA_start_epoch:
                        candidate = list(
                            set(indices.tolist()) - set(list(self.vfl.party_dict[self.adversary].target_indices)))
                        num = len(
                            list(set(indices.tolist()) & set(list(self.vfl.party_dict[self.adversary].target_indices))))
                        self.vfl.party_dict[self.adversary].backdoor_indices = random.sample(candidate, num)
                        for sample_temp_index in range(len(X)):
                            if indices[sample_temp_index] in self.vfl.party_dict[self.adversary].backdoor_indices:
                                if self.args['cuda']:
                                    party_X_train_batch_dict[self.adversary][sample_temp_index] += train_loader.dataset.pixel_pattern.detach().clone().cuda()
                                else:
                                    party_X_train_batch_dict[self.adversary][sample_temp_index] += train_loader.dataset.pixel_pattern.detach().clone()

                    loss, grad_list = self.vfl.fit(active_X_batch, Y_batch, party_X_train_batch_dict, indices)
                    if self.args['debug']:
                        print(batch_idx,  'learn done!')

                    # dynamic pixel trigger
                    if self.args['trigger'] == 'pixel' and self.args['generation'] and self.args['backdoor'] == 'sr_ba':
                        grad_pattern = 0  # [tensor(3,32,16)]
                        for grad in grad_list:
                            if grad is not None:
                                grad_pattern += grad
                        if not isinstance(grad_pattern, int):
                            grad_pattern = grad_pattern.clone().detach().cpu()
                            if self.args['trigger_add']:
                                pattern = train_loader.dataset.pixel_pattern.clone().detach()
                                pattern.requires_grad = True
                                pattern_norm = torch.norm(pattern, 2)
                                pattern_norm.backward()
                                grad_pattern = grad_pattern + pattern.grad.data.detach() * self.args['local_feature_gamma']
                                pixel_pattern = train_loader.dataset.pixel_pattern - grad_pattern * self.vfl.party_dict[self.adversary].pattern_lr
                                train_loader.dataset.pixel_pattern = pixel_pattern
                                # pixel_pattern = train_loader.dataset.pixel_pattern - grad_pattern * self.args['pattern_lr']
                                # train_loader.dataset.pixel_pattern = torch.clamp(pixel_pattern, min=-1*self.args['epsilon'], max=self.args['epsilon'])
                            else:
                                x_top = train_loader.dataset.location[0]
                                x_bot = train_loader.dataset.location[1]
                                y_top = train_loader.dataset.location[2]
                                y_bot = train_loader.dataset.location[3]
                                grad_pattern = grad_pattern[:, x_top:x_bot, y_top:y_bot]
                                # grad_pattern = torch.sign(grad_pattern)
                                # ours optim trigger
                                pattern_mask_old = train_loader.dataset.pattern_mask
                                pattern_mask = pattern_mask_old - grad_pattern * self.vfl.party_dict[self.adversary].pattern_lr
                                pattern_mask = torch.clamp(pattern_mask, min=0, max=1)
                                mask = 1 * (pattern_mask_old == -10)
                                input_shape = pattern_mask_old.shape
                                full_image = torch.zeros(input_shape)
                                full_image.fill_(-10)
                                train_loader.dataset.pattern_mask = (1 - mask) * pattern_mask + mask * full_image
                        if self.args['debug']:
                            print(batch_idx, 'generation done!')

                    # TECB
                    if self.args['backdoor'] == 'TECB' and ep < self.vfl.party_dict[self.adversary].CBP_end_epoch:
                        grad_pattern = 0  # [tensor(3,32,16)]
                        for grad in grad_list:
                            if grad is not None:
                                grad_pattern += grad
                        if not isinstance(grad_pattern, int):
                            grad_pattern = grad_pattern.clone().detach().cpu()
                            grad_pattern = grad_pattern.sign()
                            pixel_pattern = train_loader.dataset.pixel_pattern - grad_pattern * self.vfl.party_dict[self.adversary].pattern_lr
                            pixel_pattern = torch.clamp(pixel_pattern, -self.args['epsilon'], self.args['epsilon'])
                            train_loader.dataset.pixel_pattern = pixel_pattern

                    loss_list.append(loss)

                if self.args['debug']:
                    print('learning and trigger generation done!')
                # adjust learning rate
                self.vfl.scheduler_step()

                # not evaluate main-task performance if evaluating execution time
                if not self.args['time']:
                    # normal stealthy detection
                    if ep == self.args['target_epochs'] - 1 and self.vfl.active_party.status == 'train':
                        self.vfl.active_party.MAD_function()

                    # compute main-task accuracy
                    self.vfl.active_party.set_status('test')
                    ave_loss = np.sum(loss_list)/len(train_loader.dataset)
                    train_acc, train_ARR = self.predict(train_loader, num_classes=self.args['num_classes'],
                                       dataset=self.args['dataset'], top_k=top_k,
                                       n_passive_party=self.args['n_passive_party'], type='train')
                    test_acc, test_ARR = self.predict(test_loader, num_classes=self.args['num_classes'],
                                       dataset=self.args['dataset'], top_k=top_k,
                                       n_passive_party=self.args['n_passive_party'], type='test')
                    # self.vfl.scheduler_step(test_acc)
                    if self.args['debug']:
                        print('train test acc done!')
                    # compute backdoor task accuracy
                    backdoor_acc = 0
                    if backdoor_test_loader is not None and self.args['backdoor'] != 'lr_ba':
                        if self.args['trigger'] == 'pixel' and self.args['backdoor'] == 'sr_ba':
                            backdoor_test_loader.dataset.pixel_pattern = train_loader.dataset.pixel_pattern
                            backdoor_test_loader.dataset.pattern_mask = train_loader.dataset.pattern_mask
                        if ep == self.args['target_epochs'] - 1:
                            if self.args['backdoor'] == 'sr_ba' and self.args['trigger'] == 'pixel':
                                logging.info('backdoor_test_loader.dataset.pixel_pattern: {0}'.format(backdoor_test_loader.dataset.pixel_pattern))
                                logging.info('backdoor_test_loader.dataset.pattern_mask: {0}'.format(backdoor_test_loader.dataset.pattern_mask))

                        if self.args['backdoor'] == 'TECB':
                            backdoor_test_loader.dataset.pixel_pattern = train_loader.dataset.pixel_pattern
                        if ep == self.args['target_epochs'] - 1 and self.args['backdoor'] == 'TECB':
                            logging.info('backdoor_test_loader.dataset.pixel_pattern: {0}'.format(backdoor_test_loader.dataset.pixel_pattern))

                        self.vfl.active_party.set_status('backdoor_test')
                        backdoor_acc, _ = self.predict(backdoor_test_loader, num_classes=self.args['num_classes'],
                                                    dataset=self.args['dataset'], top_k=top_k, n_passive_party=self.args['n_passive_party'], type='attack')
                        if ep == self.args['target_epochs'] - 1 and self.vfl.active_party.status == 'backdoor_test':
                            for client in range(self.args['n_passive_party']):
                                self.vfl.active_party.test_norm[client] = self.vfl.active_party.test_norm[client] / self.args['backdoor_test_size']
                                self.vfl.active_party.test_dis_rate[client] = self.vfl.active_party.test_dis_rate[client] / self.args['backdoor_test_size']
                                try:
                                    self.vfl.active_party.test_norm[client] = self.vfl.active_party.test_norm[client].item()
                                    self.vfl.active_party.test_dis_rate[client] = self.vfl.active_party.test_dis_rate[client].item()
                                except:
                                    pass
                                logging.info('test_norm {}: {}'.format(client, self.vfl.active_party.test_norm[client]))
                                logging.info('test_dis_rate {}: {}'.format(client, self.vfl.active_party.test_dis_rate[client]))

                                self.vfl.active_party.test_norm2[client] = self.vfl.active_party.test_norm2[client] / self.args['backdoor_test_size']
                                self.vfl.active_party.test_dis_rate2[client] = self.vfl.active_party.test_dis_rate2[client] / self.args['backdoor_test_size']
                                try:
                                    self.vfl.active_party.test_norm2[client] = self.vfl.active_party.test_norm2[client].item()
                                    self.vfl.active_party.test_dis_rate2[client] = self.vfl.active_party.test_dis_rate2[client].item()
                                except:
                                    pass
                                logging.info('test_norm2 {}: {}'.format(client, self.vfl.active_party.test_norm2[client]))
                                logging.info('test_dis_rate2 {}: {}'.format(client, self.vfl.active_party.test_dis_rate2[client]))
                                logging.info('backdoor task error num : {}, {}'.format(self.vfl.active_party.ERROR_NUM,
                                                                                         self.vfl.active_party.ERROR_NUM / self.args['backdoor_test_size']))
                                logging.info('backdoor task DIS_NUM : {}, {}'.format(self.vfl.active_party.DIS_NUM,
                                                                                         self.vfl.active_party.DIS_NUM / self.args['backdoor_test_size']))

                        self.vfl.active_party.set_status('test')

                        if self.args['debug']:
                            print('backdoor acc done!')
                        if backdoor_acc == None:
                            backdoor_acc = 0
                        if train_acc == None:
                            train_acc = 0
                        if test_acc == None:
                            test_acc = 0
                        logging.info("--- {} epoch: {}, train loss: {}, train_acc: {}%, test acc: {}%, "
                                     "backdoor acc: {}%".format(title, ep,ave_loss, train_acc * 100,test_acc * 100, backdoor_acc * 100))

                        different_norm = 0
                        normal_norm = 0
                        dis_rate = 0
                        normal_target_norm = 0
                        mean_norms = 0
                        std_norms = 0

                        if self.args['backdoor'] == 'no':
                            with torch.no_grad():
                                self.vfl.set_eval()
                                self.target_sample = []
                                features = []
                                for batch_idx, (X, Y_batch, old_imgb, indices) in enumerate(train_loader):
                                    target_indices = []
                                    for i in range(len(indices)):
                                        if Y_batch[i] == self.args['backdoor_label']:
                                            target_indices.append(i)
                                    if self.args['dataset'] != 'bhi' and self.args['n_passive_party'] < 2:
                                        _, Xb_batch = X
                                        if self.args['cuda']:
                                            Xb_batch = Xb_batch.cuda()
                                    else:
                                        if self.args['cuda']:
                                            X = X.cuda()
                                        Xb_batch = X[:, self.adversary+1:self.adversary+2].squeeze(1)
                                    H_features = self.vfl.party_dict[self.adversary].bottom_model.forward(Xb_batch) 
                                    for i in target_indices:
                                        self.target_sample.append(H_features[i])
                                self.target_sample = torch.stack(self.target_sample)

                                norms = 0
                                if self.args['model_type'] != 'FCN':
                                    self.target_sample = torch.reshape(self.target_sample, (len(self.target_sample), -1))
                                for target_feature in self.target_sample:
                                    norms += torch.norm(target_feature, 2)
                                norms = norms / len(self.target_sample)
                                normal_target_norm = norms

                        # similar to usenix 2023, set epsilon by the var
                        if self.args['backdoor'] == 'villain' and (ep == self.args['backdoor_epochs']):
                            with torch.no_grad():
                                self.vfl.set_eval()
                                self.target_sample = []
                                features = []
                                for batch_idx, (X, Y_batch, old_imgb, indices) in enumerate(train_loader):
                                    target_indices = []
                                    for i in range(len(indices)):
                                        if Y_batch[i] == self.args['backdoor_label']:
                                            target_indices.append(i)
                                    if self.args['dataset'] != 'bhi' and self.args['n_passive_party'] < 2:
                                        _, Xb_batch = X
                                        if self.args['cuda']:
                                            Xb_batch = Xb_batch.cuda()
                                    else:
                                        if self.args['cuda']:
                                            X = X.cuda()
                                        Xb_batch = X[:, self.adversary+1:self.adversary+2].squeeze(1)
                                    H_features = self.vfl.party_dict[self.adversary].bottom_model.forward(Xb_batch) 
                                    for i in target_indices:
                                        self.target_sample.append(H_features[i])
                                    if len(features) == 0:
                                        features = H_features
                                    else:
                                        features = torch.cat((features,H_features), dim=0) 
                                self.target_sample = torch.stack(self.target_sample)

                                norms = 0
                                if self.args['model_type'] != 'FCN':
                                    self.target_sample = torch.reshape(self.target_sample, (self.target_sample.size()[0], -1))
                                    features = torch.reshape(features, (features.size()[0], -1))
                                for target_feature in self.target_sample:
                                    norms += torch.norm(target_feature, 2)
                                norms = norms / len(self.target_sample)
                                normal_target_norm = norms
                                mean_norms = torch.mean(features, dim=0)
                                std_norms = torch.std(features, dim=0)
                                stride_pattern = [0] * len(features[0])
                                temp = []
                                for i in range(self.args['m_dimension'] // 2):
                                    if i % 2 == 0:
                                        temp = temp + [-1,-1]
                                    else:
                                        temp = temp + [1,1]
                                if len(temp) < self.args['m_dimension']:
                                    temp = temp.append(temp[-1] * -1)
                                sorted, indices = torch.sort(std_norms, descending=True)
                                for i in range(self.args['m_dimension']):
                                    stride_pattern[indices[i]] = temp[i]
                                stride_pattern = torch.tensor(stride_pattern)
                                if self.args['model_type'] != 'FCN':
                                    std_norms = torch.reshape(std_norms, H_features.size()[1:])
                                    stride_pattern = torch.reshape(stride_pattern, H_features.size()[1:])
                                if self.args['cuda']:
                                    stride_pattern = stride_pattern.cuda()
                                    std_norms = std_norms.cuda()
                                self.vfl.party_dict[self.adversary].feature_pattern = std_norms * self.args['epsilon'] * stride_pattern
                                if self.args['cuda']:
                                    self.vfl.party_dict[self.adversary].feature_pattern = self.vfl.party_dict[self.adversary].feature_pattern.cuda()
                                logging.info(self.vfl.party_dict[self.adversary].feature_pattern)

                        if self.args['backdoor'] == 'villain':
                            self.vfl.party_dict[self.adversary].dis_rate = self.vfl.party_dict[self.adversary].dis_rate / self.args['backdoor_train_size']
                            self.vfl.party_dict[self.adversary].norm = self.vfl.party_dict[self.adversary].norm / self.args['backdoor_train_size']
                            dis_rate = (self.vfl.party_dict[self.adversary].dis_rate)
                            different_norm = (self.vfl.party_dict[self.adversary].norm)
                            if ep > self.args['backdoor_epochs'] and dis_rate!= 0:
                                dis_rate = dis_rate.item()
                                different_norm = different_norm.item()
                            with torch.no_grad():
                                H_features = self.vfl.party_dict[self.adversary].bottom_model.forward(self.vfl.party_dict[self.adversary].target_sample)
                                for i in range(len(H_features)):
                                    normal_target_norm += torch.norm(H_features[i], 2)
                                normal_target_norm = normal_target_norm / len(H_features)

                        if self.args['backdoor'] == 'sr_ba' and self.args['idea'] != 2:
                            if self.vfl.party_dict[self.adversary].dis_rate != 0:
                                self.vfl.party_dict[self.adversary].dis_rate = self.vfl.party_dict[self.adversary].dis_rate / self.args['backdoor_train_size']
                                self.vfl.party_dict[self.adversary].norm = self.vfl.party_dict[self.adversary].norm / self.args['backdoor_train_size']
                                try:
                                    dis_rate = self.vfl.party_dict[self.adversary].dis_rate.item()
                                    different_norm = self.vfl.party_dict[self.adversary].norm.item()
                                except:
                                    dis_rate = self.vfl.party_dict[self.adversary].dis_rate
                                    different_norm = self.vfl.party_dict[self.adversary].norm

                        if self.args['backdoor'] == 'splitNN':
                            if self.vfl.party_dict[self.adversary].dis_rate != 0:
                                self.vfl.party_dict[self.adversary].dis_rate = self.vfl.party_dict[self.adversary].dis_rate / self.args['backdoor_train_size']
                                self.vfl.party_dict[self.adversary].norm = self.vfl.party_dict[self.adversary].norm / self.args['backdoor_train_size']
                                try:
                                    dis_rate = self.vfl.party_dict[self.adversary].dis_rate.item()
                                    different_norm = self.vfl.party_dict[self.adversary].norm.item()
                                except:
                                    dis_rate = self.vfl.party_dict[self.adversary].dis_rate
                                    different_norm = self.vfl.party_dict[self.adversary].norm
                        if self.args['backdoor'] == 'TECB':
                            if self.vfl.party_dict[self.adversary].dis_rate != 0:
                                try:
                                    dis_rate = self.vfl.party_dict[self.adversary].dis_rate.item()
                                    different_norm = self.vfl.party_dict[self.adversary].norm.item()
                                except:
                                    dis_rate = self.vfl.party_dict[self.adversary].dis_rate
                                    different_norm = self.vfl.party_dict[self.adversary].norm

                    else:
                        logging.info("--- {} epoch: {}, train loss: {}, train_acc: {}%, test acc: {}%".format(title, ep, ave_loss, train_acc * 100, test_acc * 100))

                    if self.args['outlier_detection'] and self.args['backdoor'] != 'lr_ba':
                        detecte_Precision = 0
                        detecte_Recall = 0
                        benign_MAD = 0
                        backdoor_MAD_mean = 0
                        backdoor_MAD_var = 0
                        if ep > self.args['backdoor_epochs'] and (self.args['backdoor'] == 'villain' or self.args['backdoor'] == 'sr_ba'):
                            if self.vfl.active_party.attack_indices:
                                detecte_TP = 0
                                for indices in self.vfl.active_party.attack_indices:
                                    if indices.item() in self.vfl.party_dict[self.adversary].target_indices:
                                        detecte_TP += 1
                                detecte_Recall = detecte_TP / len(self.vfl.party_dict[self.adversary].target_indices)
                                detecte_Precision = detecte_TP / len(self.vfl.active_party.attack_indices)
                            # print('backdoor_MAD: ', self.vfl.active_party.backdoor_MAD[:10])
                            # print('benign MAD: ', self.vfl.active_party.benign_MAD)
                            if len(self.vfl.active_party.backdoor_MAD) > 0:
                                benign_MAD = self.vfl.active_party.benign_MAD / (len(train_loader.dataset) - self.args['backdoor_train_size'])
                                backdoor_MAD_cpu = torch.tensor(self.vfl.active_party.backdoor_MAD, device='cpu')
                                backdoor_MAD_mean = torch.mean(backdoor_MAD_cpu)
                                backdoor_MAD_var = torch.var(backdoor_MAD_cpu)

                    if self.args['embedding_detection']:
                        detecte_Precision = 0
                        detecte_Recall = 0
                        detecte_TP = 0
                        if ep > self.args['backdoor_epochs'] and (self.args['backdoor'] == 'villain' or self.args['backdoor'] == 'sr_ba'):
                            if self.vfl.active_party.attack_indices:
                                detecte_TP = 0
                                for indices in self.vfl.active_party.attack_indices:
                                    if indices.item() in self.vfl.party_dict[self.adversary].target_indices:
                                        detecte_TP += 1

                    if use_wandb and self.args['backdoor'] != 'lr_ba':
                        if self.args['backdoor'] == 'sr_ba' and not self.args['trigger_add']:
                            if self.args['outlier_detection']:
                                wandb.log({'loss': ave_loss, "train acc": train_acc, 'test acc': test_acc, 'backdoor_acc': backdoor_acc,
                                   'pattern_norm': self.vfl.party_dict[self.adversary].norm, 'ARR': test_ARR, 'dis_rate':self.vfl.party_dict[self.adversary].dis_rate,
                                    'D_P': detecte_Precision, 'D_R': detecte_Recall,'benign_MAD': benign_MAD, 'backddor_MAD_mean': backdoor_MAD_mean,
                                    'backddor_MAD_var': backdoor_MAD_var})
                            else:
                                wandb.log({'loss': ave_loss, "train acc": train_acc, 'test acc': test_acc,'backdoor_acc': backdoor_acc,
                                           'pattern_norm': self.vfl.party_dict[self.adversary].norm, 'ARR': test_ARR,'dis_rate': self.vfl.party_dict[self.adversary].dis_rate
                                })
                        elif (self.args['backdoor'] == 'sr_ba' and self.args['trigger_add']) or self.args['backdoor'] == 'villain':  # ADD
                            if self.args['outlier_detection']:
                                wandb.log({'loss': ave_loss, "train acc": train_acc, 'test acc': test_acc,'backdoor_acc': backdoor_acc,
                                           'pattern_norm': self.vfl.party_dict[self.adversary].norm, 'ARR': test_ARR,
                                           'normal_norm': normal_target_norm, 'dis_rate':self.vfl.party_dict[self.adversary].dis_rate,
                                           'D_P':detecte_Precision, 'D_R':detecte_Recall,
                                           'benign_MAD':benign_MAD, 'backddor_MAD_mean':backdoor_MAD_mean,'backddor_MAD_var':backdoor_MAD_var})
                            elif self.args['embedding_detection']:
                                wandb.log({'loss': ave_loss, "train acc": train_acc, 'test acc': test_acc,
                                           'backdoor_acc': backdoor_acc, 'pattern_norm': self.vfl.party_dict[self.adversary].norm,
                                           'ARR': test_ARR, 'dis_rate': self.vfl.party_dict[self.adversary].dis_rate,
                                           'TP': detecte_TP})
                            else:
                                wandb.log({'loss': ave_loss, "train acc": train_acc, 'test acc': test_acc,
                                           'backdoor_acc': backdoor_acc,'pattern_norm': self.vfl.party_dict[self.adversary].norm, 'ARR': test_ARR,
                                           'normal_norm': normal_target_norm,'dis_rate': self.vfl.party_dict[self.adversary].dis_rate})
                        else:
                            wandb.log({'loss': ave_loss, "train acc": train_acc, 'test acc': test_acc,
                                   'backdoor_acc': backdoor_acc, 'ARR': test_ARR, 'normal_target_norm':normal_target_norm, 'mean_norms':mean_norms, 'std_norms':std_norms})

                if ep == self.args['target_epochs'] - 1 and self.args['backdoor'] == 'sr_ba' and self.args['idea'] == 2 and self.args['trigger'] == 'feature':
                    self.vfl.party_dict[self.adversary].save_data_idea2()

                # write in csv
                if ep == self.args['target_epochs'] - 1 and self.args['backdoor'] != 'lr_ba':
                    csv_filename = '../' + f"{self.args['dataset']}_saved_framework.csv"
                    with open(csv_filename, 'a+', newline='') as f:
                        writer = csv.writer(f)
                        defense = 'None'
                        if 'noisy_gradients' in self.args and self.args['noisy_gradients']:
                            defense = 'lap-noise-' + str(self.args['noise_scale'])
                        if 'gradient_compression' in self.args and self.args['gradient_compression']:
                            defense = 'gc-' + str(self.args['gc_percent'])
                        if 'max_norm' in self.args and self.args['max_norm']:
                            defense = 'max_norm'
                        if 'ABL' in self.args and self.args['ABL']:
                            defense = 'ABL-' + str(self.args['isolation_ratio']) + '-' + str(self.args['gradient_ascent_type']) \
                                      + '-' + str(self.args['flooding_gamma']) + '-' + str(self.args['t_epochs'])
                        h = 0
                        if 'mal_optim' in self.args:
                            h = int(self.args['mal_optim'])
                        backdoor = 'None'
                        if self.args['backdoor'] is not None:
                            backdoor = self.args['backdoor']
                        if self.args['trigger_add']:
                            self.args['trigger'] = self.args['trigger'] + '-' + str(self.args['epsilon'])
                        tricks = 'None'
                        if self.args['dropout'] != 0:
                            tricks = 'dropout-' + str(self.args['dropout'])

                        writer.writerow([str(self.args['file_time']), self.args['dataset'], backdoor, defense, tricks, int(self.args['active_top_trainable']),self.args['model_type'],
                                         self.args['aggregate'], self.args['trigger'], int(self.args['trigger_add']), int(self.args['generation']), self.args['pattern_lr'], self.args['sr_feature_amplify_ratio'], h, self.args['s_r_amplify_ratio'],
                                         self.args['target_epochs'], str(self.args['passive_bottom_stone']), self.args['target_batch_size'],
                                         self.args['passive_bottom_lr'], self.args['passive_bottom_gamma'], self.args['target_train_size'],self.args['m_dimension'],
                                         self.args['poison_rate'], self.args['backdoor_test_size'], self.args['local_feature_gamma'],
                                         train_acc * 100, test_acc * 100, backdoor_acc * 100, self.args['n_passive_party'], different_norm, dis_rate,
                                         self.vfl.active_party.test_norm[self.adversary], self.vfl.active_party.test_dis_rate[self.adversary],
                                         self.vfl.active_party.test_norm2[self.adversary], self.vfl.active_party.test_dis_rate2[self.adversary]])
                    f.close()
            # print execution time of federated training
            if self.args['time']:
                start_time = print_running_time('federated training', start_time)
            # save VFL
            if self.args['save_model']:
                self.vfl.save()
            # save data
            if self.args['backdoor'] == 'sr_ba' and self.args['save_model']:
                if 'pixel' in self.args['trigger']:
                    if self.args['trigger_add']:
                        self.vfl.party_dict[self.adversary].save_data(pattern=train_loader.dataset.pixel_pattern)
                    else:
                        self.vfl.party_dict[self.adversary].save_data(pattern=train_loader.dataset.pattern_mask)
                else:
                    self.vfl.party_dict[self.adversary].save_data()
            if self.args['backdoor'] == 'villain' and self.args['save_model']:
                self.vfl.party_dict[self.adversary].save_data()

            if self.args['backdoor'] == 'TECB' and self.args['save_model']:
                self.vfl.party_dict[self.adversary].save_data(pattern=train_loader.dataset.pixel_pattern)

            # wandb test
            if use_wandb:
                wandb.finish()

    def sr_ba_attack(self, train_loader, test_loader, backdoor_train_loader, backdoor_test_loader, backdoor_indices):
        top_k = 1
        if self.dataset == 'cifar100':
            top_k = 5

        # compute main and backdoor task accuracy before conducting LR-BA
        logging.info("--- before SR-BA finetune: main test acc: {}".
                     format(self.predict(test_loader, num_classes=self.args['num_classes'],
                                         dataset=self.args['dataset'], top_k=top_k,
                                         n_passive_party=self.args['n_passive_party'])[0]))
        logging.info("--- before SR-BA finetune: backdoor test acc: {}".
                     format(self.predict(backdoor_test_loader, num_classes=self.args['num_classes'],
                                         dataset=self.args['dataset'], top_k=top_k,
                                         n_passive_party=self.args['n_passive_party'])[0]))

        backdoor_output_list = [self.vfl.party_dict[self.adversary].feature_pattern.unsqueeze(dim=0)] # [(1,10)] or [(1,32,16,16)]
        logging.info("--- SR-BA backdoor start")
        acc_list = poison_predict(
            self.vfl,
            backdoor_output_list, backdoor_test_loader, self.args['dataset'], self.args,
            top_k=top_k,
            num_classes=self.args['num_classes']
        )
        logging.info("--- debug SR-BA attack acc: {0}%".format(acc_list[0]*100))
        logging.info("--- finetune bottom model")
        # fine-tune to get the malicious bottom model
        new_bottom_model = finetune_bottom_model(old_bottom_model=self.vfl.party_dict[self.adversary].bottom_model,
                                                      backdoor_output_list=backdoor_output_list,
                                                      train_loader=backdoor_train_loader,
                                                      backdoor_indices=backdoor_indices,
                                                      args=self.args)

        # compute main and backdoor task accuracy after conducting LR-BA if not evaluating execution time
        if not self.args['time']:
            for i, bottom_model in enumerate(new_bottom_model):
                self.vfl.party_dict[self.adversary].bottom_model = bottom_model
                train_acc, _ = self.predict(train_loader, num_classes=self.args['num_classes'],dataset=self.args['dataset'], top_k=top_k,n_passive_party=self.args['n_passive_party'])
                logging.info("--- after SR-BA backdoor: main train acc: {}%".format(train_acc * 100))
                test_acc, test_ARR = self.predict(test_loader, num_classes=self.args['num_classes'], dataset=self.args['dataset'], top_k=top_k, n_passive_party=self.args['n_passive_party'])
                logging.info("--- after SR-BA backdoor: main test acc: {}%  test ARR: {}%".format(test_acc * 100, test_ARR * 100))
                backdoor_acc, _ = self.predict(backdoor_test_loader, num_classes=self.args['num_classes'],dataset=self.args['dataset'], top_k=top_k,
                                                 n_passive_party=self.args['n_passive_party'])
                logging.info("--- after SR-BA backdoor: backdoor test acc: {}%".format(backdoor_acc * 100))

            csv_filename = '../' + f"{self.args['dataset']}_saved_framework.csv"
            with open(csv_filename, 'a+', newline='') as f:
                writer = csv.writer(f)
                defense = 'None'
                if 'noisy_gradients' in self.args and self.args['noisy_gradients']:
                    defense = 'lap-noise-' + str(self.args['noise_scale'])
                if 'gradient_compression' in self.args and self.args['gradient_compression']:
                    defense = 'gc-' + str(self.args['gc_percent'])
                if 'max-norm' in self.args and self.args['max-norm']:
                    defense = 'max-norm'
                h = 0
                if 'mal_optim' in self.args:
                    h = int(self.args['mal_optim'])
                backdoor = 'None'
                if self.args['backdoor'] is not None:
                    backdoor = self.args['backdoor']
                writer.writerow(
                    [str(self.args['file_time']), self.args['dataset'], backdoor, defense, int(self.args['active_top_trainable']),self.args['model_type'],
                     self.args['aggregate'], self.args['trigger'],int(self.args['trigger_add']), int(self.args['generation']), self.args['pattern_lr'],
                     h, self.args['s_r_amplify_ratio'],
                     self.args['target_epochs'], str(self.args['passive_bottom_stone']), self.args['target_batch_size'],
                     self.args['passive_bottom_lr'], self.args['passive_bottom_gamma'], self.args['target_train_size'],
                     self.args['backdoor_train_size'], self.args['backdoor_test_size'], train_acc * 100,
                     test_acc * 100, backdoor_acc * 100, self.args['n_passive_party']])
            f.close()
        return

    def get_normal_representation_for_backdoor_label(self, test_loader):
        """
        collect normal representations output by the attacker's clean bottom model on normal inputs

        :param test_loader: testing dataset for main task
        :return: normal representations
        """
        attacker_id = 0
        self.vfl.set_eval()
        target_X, target_Y = None, None
        for batch_idx, (X, Y_batch, indices) in enumerate(test_loader):
            party_X_train_batch_dict = dict()
            if self.args['dataset'] != 'bhi' and self.args['n_passive_party'] < 2:
                active_X_batch, Xb_batch = X
                if self.args['dataset'] == 'yahoo':
                    Xb_batch = Xb_batch.long()
                    Y_batch = Y_batch[0].long()
                party_X_train_batch_dict[0] = Xb_batch
            else:
                for i in range(self.args['n_passive_party']):
                    party_X_train_batch_dict[i] = X[:, i + 1:i + 2].squeeze(1)

            Y = Y_batch.numpy()
            target_indices = np.where(Y == self.args['backdoor_label'])[0]
            target_X = torch.cat([target_X, party_X_train_batch_dict[attacker_id][target_indices]], dim=0) \
                if target_X is not None else party_X_train_batch_dict[attacker_id][target_indices]
            if target_X.shape[0] >= 100:
                break
        result = self.vfl.party_dict[attacker_id].predict(target_X[:100])
        return result


    def lr_ba_attack(self, train_loader, test_loader,
                   backdoor_train_loader, backdoor_test_loader, backdoor_indices,
                   labeled_loader, unlabeled_loader, lr_ba_top_model=None,
                   add_from_unlabeled_tag=False, get_error_features_tag=True, random_tag=False):
        """
        conduct LR-BA attack, happens after VFL training

        :param train_loader: loader of normal train dataset
        :param test_loader: loader of normal test dataset
        :param backdoor_train_loader: loader of backdoor train dataset
        :param backdoor_test_loader: loader of backdoor test dataset
        :param backdoor_indices: indices of backdoor samples in normal train dataset
        :param labeled_loader: loader of labeled samples in normal train dataset
        :param unlabeled_loader: loader of unlabeled samples in normal train dataset
        :param lr_ba_top_model: inference head, not training if provided
        :param add_from_unlabeled_tag: whether to initialize from unlabeled samples
        :param get_error_features_tag: invalid
        :param random_tag: whether to randomly initialize backdoor latent representation
        :return: inference head
        """
        top_k = 1
        if self.dataset == 'cifar100':
            top_k = 5

        # compute main and backdoor task accuracy before conducting LR-BA
        logging.info("--- before LR-BA backdoor: main test acc: {}".
                     format(self.predict(test_loader, num_classes=self.args['num_classes'],
                                         dataset=self.args['dataset'], top_k=top_k,
                                         n_passive_party=self.args['n_passive_party'])))
        # self.vfl.active_party.set_status('backdoor_test')
        logging.info("--- before LR-BA backdoor: backdoor test acc: {}".
                     format(self.predict(backdoor_test_loader, num_classes=self.args['num_classes'],
                                         dataset=self.args['dataset'], top_k=top_k,
                                         n_passive_party=self.args['n_passive_party'],
                                         type='attack')))
        # self.vfl.active_party.set_status('test')


        logging.info("--- LR-BA backdoor start")
        self.vfl.active_party.backdoor_indice = backdoor_indices

        # conduct LR-BA attack
        bottom_model_list, lr_ba_top_model, backdoor_output_list = lr_ba_backdoor(
            train_loader=train_loader,
            test_loader=test_loader,
            backdoor_indices=backdoor_indices,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            vfl=self.vfl,
            args=self.args,
            backdoor_train_loader=backdoor_train_loader,
            backdoor_test_loader=backdoor_test_loader,
            lr_ba_top_model=lr_ba_top_model,
            add_from_unlabeled_tag=add_from_unlabeled_tag,
            get_error_features_tag=get_error_features_tag,
            random_tag=random_tag
        )


        # compute main and backdoor task accuracy after conducting LR-BA if not evaluating execution time
        if not self.args['time']:
            for i, bottom_model in enumerate(bottom_model_list):
                self.vfl.party_dict[self.adversary].bottom_model = bottom_model
                if isinstance(self.args['lr_ba_generate_epochs'], list):
                    logging.info("--- LR-BA generate epochs: {}".format(self.args['lr_ba_generate_epochs'][i]))
                else:
                    logging.info("--- LR-BA generate epochs: {}".format(self.args['lr_ba_generate_epochs']))

                train_acc, _ = self.predict(train_loader, num_classes=self.args['num_classes'],dataset=self.args['dataset'], top_k=top_k,n_passive_party=self.args['n_passive_party'])
                logging.info("--- after LR-BA backdoor: main train acc: {}%".format(train_acc * 100))
                test_acc, _ = self.predict(test_loader, num_classes=self.args['num_classes'], dataset=self.args['dataset'], top_k=top_k, n_passive_party=self.args['n_passive_party'])
                logging.info("--- after LR-BA backdoor: main test acc: {}%".format(test_acc * 100))

                if self.args['outlier_detection'] and self.args['backdoor'] == 'lr_ba':
                    # self.vfl.active_party.MAD_function()
                    self.vfl.active_party.attack_indices = []

                self.vfl.active_party.set_status('backdoor_test')
                self.vfl.active_party.epoch = self.args['target_epochs'] - 1
                backdoor_acc, _ = self.predict(backdoor_test_loader, num_classes=self.args['num_classes'],dataset=self.args['dataset'], top_k=top_k,
                                                 n_passive_party=self.args['n_passive_party'],type='attack')
                self.vfl.active_party.epoch = 0
                self.vfl.active_party.set_status('test')

                with torch.no_grad():
                    self.vfl.set_eval()
                    all_target_sample = []
                    norm_type = 2
                    for batch_idx, (X, Y_batch, old_imgb, indices) in enumerate(train_loader):
                        target_indices = []
                        for i in range(len(indices)):
                            if Y_batch[i].item() == self.args['backdoor_label']:
                                target_indices.append(i)
                        if len(target_indices) != 0:
                            if self.args['dataset'] != 'bhi' and self.args['n_passive_party'] < 2:
                                _, Xb_batch = X
                                if self.args['cuda']:
                                    Xb_batch = Xb_batch.cuda()
                            else:
                                if self.args['cuda']:
                                    X = X.cuda()
                                Xb_batch = X[:, self.adversary + 1:self.adversary + 2].squeeze(1)
                            for i in target_indices:
                                all_target_sample.append(Xb_batch[i])
                    all_target_sample = torch.stack(all_target_sample)
                    if self.args['cuda']:
                        all_target_sample = all_target_sample.cuda()
                    H_features = self.vfl.party_dict[self.adversary].bottom_model.forward(all_target_sample)
                    mini_feature = None
                    mini_distance = None
                    for index in range(len(H_features)):
                        difference = torch.norm(H_features[index] - backdoor_output_list[0], norm_type)
                        if mini_distance is None or difference < mini_distance:
                            mini_distance = difference
                            mini_feature = H_features[index]
                    different_norm = mini_distance.item()
                    mini_feature_norm = torch.norm(mini_feature, p=norm_type)
                    dis_rate = (mini_distance / mini_feature_norm).item()

                for client in range(self.args['n_passive_party']):
                    self.vfl.active_party.test_norm[client] = self.vfl.active_party.test_norm[client].item() / self.args['backdoor_test_size']
                    self.vfl.active_party.test_dis_rate[client] = self.vfl.active_party.test_dis_rate[client].item() / self.args['backdoor_test_size']
                    logging.info('test_norm {}: {}'.format(client, self.vfl.active_party.test_norm[client]))
                    logging.info('test_dis_rate {}: {}'.format(client, self.vfl.active_party.test_dis_rate[client]))

                    self.vfl.active_party.test_norm2[client] = self.vfl.active_party.test_norm2[client].item() / self.args['backdoor_test_size']
                    self.vfl.active_party.test_dis_rate2[client] = self.vfl.active_party.test_dis_rate2[client].item() / self.args['backdoor_test_size']
                    logging.info('test_norm2 {}: {}'.format(client, self.vfl.active_party.test_norm2[client]))
                    logging.info('test_dis_rate2 {}: {}'.format(client, self.vfl.active_party.test_dis_rate2[client]))

                if self.args['outlier_detection'] and self.args['backdoor'] == 'lr_ba':
                    backddor_MAD_mean = 0
                    backddor_MAD_var = 0
                    logging.info(self.vfl.active_party.attack_indices)
                    if len(self.vfl.active_party.backdoor_MAD) > 0:
                        backdoor_MAD_cpu = torch.tensor(self.vfl.active_party.backdoor_MAD, device='cpu')
                        backddor_MAD_mean = torch.mean(backdoor_MAD_cpu)
                        backddor_MAD_var = torch.var(backdoor_MAD_cpu)
                    logging.info('--- outlier detection TP: {}/{}={}%'.format(len(self.vfl.active_party.attack_indices), self.args['backdoor_test_size'], len(self.vfl.active_party.attack_indices)/self.args['backdoor_test_size']*100))
                    logging.info('MAD mean: {};   MAD var: {}'.format(backddor_MAD_mean, backddor_MAD_var))

                logging.info("--- after LR-BA backdoor: backdoor test acc: {}%".format(backdoor_acc * 100))

            csv_filename = '../' + f"{self.args['dataset']}_saved_framework.csv"
            with open(csv_filename, 'a+', newline='') as f:
                writer = csv.writer(f)
                defense = 'None'
                if 'noisy_gradients' in self.args and self.args['noisy_gradients']:
                    defense = 'lap-noise-' + str(self.args['noise_scale'])
                if 'gradient_compression' in self.args and self.args['gradient_compression']:
                    defense = 'gc-' + str(self.args['gc_percent'])
                if 'max-norm' in self.args and self.args['max-norm']:
                    defense = 'max-norm'
                h = 0
                if 'mal_optim' in self.args:
                    h = int(self.args['mal_optim'])
                backdoor = 'None'
                if self.args['backdoor'] is not None:
                    backdoor = self.args['backdoor']
                tricks = 'None'
                if self.args['dropout'] != 0:
                    tricks = 'dropout-' + str(self.args['dropout'])
                writer.writerow(
                    [str(self.args['file_time']), self.args['dataset'], backdoor, defense, tricks, int(self.args['active_top_trainable']),self.args['model_type'],
                     self.args['aggregate'], self.args['trigger'],int(self.args['trigger_add']), int(self.args['generation']), self.args['pattern_lr'],
                     self.args['sr_feature_amplify_ratio'], h, self.args['s_r_amplify_ratio'],
                     self.args['target_epochs'], str(self.args['passive_bottom_stone']), self.args['target_batch_size'],
                     self.args['passive_bottom_lr'], self.args['passive_bottom_gamma'], self.args['target_train_size'],self.args['m_dimension'],
                    self.args['poison_rate'], self.args['backdoor_test_size'], train_acc * 100,
                     test_acc * 100, backdoor_acc * 100, self.args['n_passive_party'],different_norm, dis_rate,
                     self.vfl.active_party.test_norm[self.adversary], self.vfl.active_party.test_dis_rate[self.adversary],
                     self.vfl.active_party.test_norm2[self.adversary], self.vfl.active_party.test_dis_rate2[self.adversary]])

            f.close()
        return lr_ba_top_model

    def baseline_attack(self, train_loader, test_loader,
                        backdoor_train_loader, backdoor_test_loader, backdoor_indices,
                        labeled_loader, unlabeled_loader, lr_ba_top_model=None):

        top_k = 1
        if self.dataset == 'cifar100':
            top_k = 5

        # compute main and backdoor task accuracy before conducting baseline attack
        logging.info("--- before baseline backdoor: main test acc: {}".
                     format(self.predict(test_loader, num_classes=self.args['num_classes'],
                                         dataset=self.args['dataset'], top_k=top_k,
                                         n_passive_party=self.args['n_passive_party'])))
        logging.info("--- before baseline backdoor: backdoor test acc: {}".
                     format(self.predict(backdoor_test_loader, num_classes=self.args['num_classes'],
                                         dataset=self.args['dataset'], top_k=top_k,
                                         n_passive_party=self.args['n_passive_party'],
                                         type='attack')))

        logging.info("--- baseline backdoor start")

        # conduct LR-BA attack
        bottom_model = baseline_backdoor(
            train_loader=train_loader,
            test_loader=test_loader,
            backdoor_indices=backdoor_indices,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            vfl=self.vfl,
            args=self.args,
            backdoor_train_loader=backdoor_train_loader,
            backdoor_test_loader=backdoor_test_loader,
            lr_ba_top_model=lr_ba_top_model
        )

        # compute main and backdoor task accuracy after conducting LR-BA
        self.vfl.party_dict[self.adversary].bottom_model = bottom_model
        logging.info("--- after baseline backdoor: main test acc: {}".
                     format(self.predict(test_loader, num_classes=self.args['num_classes'],
                                         dataset=self.args['dataset'], top_k=top_k,
                                         n_passive_party=self.args['n_passive_party'])))
        logging.info("--- after baseline backdoor: backdoor test acc: {}".
                     format(self.predict(backdoor_test_loader, num_classes=self.args['num_classes'],
                                         dataset=self.args['dataset'], top_k=top_k,
                                         n_passive_party=self.args['n_passive_party'],
                                         type='attack')))
        return

    def predict(self, test_loader, num_classes, dataset, top_k=1, n_passive_party=2, type=None):
        """
        compute accuracy of VFL system on test dataset

        :param test_loader: loader of test dataset
        :param num_classes: number of dataset classes
        :param dataset: dataset name
        :param top_k: top-k accuracy
        :param n_passive_party: number of passive parties
        :param is_attack: whether to compute attack accuracy
        :return: accuracy
        """
        y_predict = []
        y_true = []

        # avg_features = {}

        with torch.no_grad():
            self.vfl.set_eval()
            # if self.args['backdoor'] == 'sr_ba' and self.args['trigger'] == 'pixel':
            #     self.vfl.party_dict[self.adversary].norm = 0
            #     self.vfl.party_dict[self.adversary].dis_rate = 0

            for batch_idx, (X, targets, old_imgb, indices) in enumerate(test_loader):
                party_X_test_dict = dict()
                if dataset != 'bhi' and self.args['n_passive_party'] < 2:
                    active_X_inputs, Xb_inputs = X
                    if self.args['dataset'] == 'yahoo':
                        active_X_inputs = active_X_inputs.long()
                        Xb_inputs = Xb_inputs.long()
                        targets = targets[0].long()
                    if self.args['cuda']:
                        active_X_inputs = active_X_inputs.cuda()
                        Xb_inputs = Xb_inputs.cuda()
                        targets = targets.cuda()
                        indices = indices.cuda()
                        old_imgb = old_imgb.cuda()
                    if self.args['debug'] and batch_idx == 0:
                        print('predict data is on ', active_X_inputs.device)
                    party_X_test_dict[0] = Xb_inputs
                else:
                    if self.args['cuda']:
                        X = X.cuda()
                        targets = targets.cuda()
                        indices = indices.cuda()
                    active_X_inputs = X[:, 0:1].squeeze(1)
                    for i in range(n_passive_party):
                        party_X_test_dict[i] = X[:, i+1:i+2].squeeze(1)
                    if self.args['debug'] and batch_idx == 0:
                        print('fit data is on ', active_X_inputs.device)
                y_true += targets.data.tolist()
                if self.args['backdoor'] == 'sr_ba':  # for evaluate
                    self.vfl.party_dict[self.adversary].original_X = old_imgb
                    self.vfl.party_dict[self.adversary].original_Y = targets
                # For TECB
                if self.args['backdoor'] == 'TECB':  # for evaluate
                    self.vfl.party_dict[self.adversary].original_Y = targets
                if type == 'train':
                    self.vfl.active_party.y = targets
                self.vfl.active_party.indices = indices

                y_prob_preds = self.vfl.predict(active_X_inputs, party_X_test_dict, type=type)
                y_predict += y_prob_preds.tolist()

        is_attack = False
        if type == 'attack':
            is_attack = True

        acc = accuracy(y_true, y_predict, top_k=top_k, num_classes=num_classes, dataset=dataset, is_attack=is_attack)
        ARR = 0
        if type != 'attack':
            for i in range(len(y_true)):
                if y_true[i] == self.args['backdoor_label']:
                    indices = sorted(range(len(y_predict[i])), key=lambda k: y_predict[i][k], reverse=True)
                    if self.args['backdoor_label'] in indices[:top_k]:
                        ARR += 1
            ARR = ARR / y_true.count(self.args['backdoor_label'])

        if self.args['backdoor'] == 'TECB' and is_attack:
            if self.vfl.party_dict[self.adversary].norm != 0:
                self.vfl.party_dict[self.adversary].norm = (self.vfl.party_dict[self.adversary].norm / len(test_loader.dataset.data)).item()
                self.vfl.party_dict[self.adversary].dis_rate = (self.vfl.party_dict[self.adversary].dis_rate / len(test_loader.dataset.data)).item()

        if self.args['debug']:
            print('predict done!')
        return acc, ARR


    def lr_ba_attack_for_representation(self, train_loader, test_loader,
                                      labeled_loader, unlabeled_loader, lr_ba_top_model=None,
                                      add_from_unlabeled_tag=True, get_error_features_tag=True, random_tag=False):
        """
        collect backdoor latent representation generated by LR-BA

        :param train_loader: loader of normal train dataset
        :param test_loader: loader of normal test dataset
        :param labeled_loader: loader of labeled samples in normal train dataset
        :param unlabeled_loader: loader of unlabeled samples in normal train dataset
        :param lr_ba_top_model: inference head, not training if provided
        :param add_from_unlabeled_tag: whether to initialize from unlabeled samples
        :param get_error_features_tag: invalid
        :param random_tag: whether to randomly initialize backdoor latent representation
        :return: inference head and backdoor latent representation
        """
        logging.info("--- LR-BA backdoor start")

        # conduct LR-BA attack
        lr_ba_top_model, representation = lr_ba_backdoor_for_representation(
            train_loader=train_loader,
            test_loader=test_loader,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            vfl=self.vfl,
            args=self.args,
            lr_ba_top_model=lr_ba_top_model,
            add_from_unlabeled_tag=add_from_unlabeled_tag,
            get_error_features_tag=get_error_features_tag,
            random_tag=random_tag
        )
        return lr_ba_top_model, representation[0]