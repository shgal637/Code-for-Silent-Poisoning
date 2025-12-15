# -*- coding: utf-8 -*-
import logging
import math
import random
import numpy as np
import torch
from vfl.party_models import VFLPassiveModel
from vfl.defense.norm_clip import norm_clip
from common.constants import CHECKPOINT_PATH
import os
import statistics
"""
Malicious passive party for our backdoor
"""

class SR_BA_PassiveModel(VFLPassiveModel):

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
        self.pattern_lr = self.args['pattern_lr']
        self.feature_pattern = None
        self.mask = None
        if self.args['trigger'] == 'feature' and self.args['generation'] is False and self.args['model_type'] == 'FCN':
            self.feature_pattern = torch.tensor([0.]*10)
            if self.args['cuda']:
                self.feature_pattern = self.feature_pattern.cuda()

        if self.args['dropout'] == 0:
            self.drop_out = False
        else:
            self.drop_out = True
            self.drop_out_rate = self.args['dropout']
        self.shifting = False
        self.up_bound = 1.2
        self.down_bound = 0.6

        self.adversary = self.args['adversary'] - 1
        self.alpha = 0.5
        self.feature_pattern_dict = {}
        self.original_Y = None
        self.original_X = None

        self.temp_grad_record_acc = None
        self.temp_grad_record_error = None
        self.grad_record = None
        self.MAD = None
        self.FP = 0
        self.TP = 0
        # self.dis_threhold = 0.1
        # self.rms_lr = 0
        # self.max_lr = self.args['pattern_lr']

        # self.Discriminator = Discriminator()
        # self.D_optimizer = torch.optim.SGD(self.Discriminator.parameters(), momentum=0.9, weight_decay=0.0005, lr=0.01)

            # self.Discriminator = self.Discriminator.cuda()

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
            if 'feature' in self.args['trigger']:
                self.feature_pattern = pattern
            return pattern
        else:
            name = self.args['trigger'] + '_pattern.pt'
            filepath = '{}/{}'.format(path, name)
            if os.path.isfile(filepath):
                pattern = torch.load(filepath)
                if 'feature' in self.args['trigger']:
                    self.feature_pattern = pattern
                return pattern
            else:
                raise ValueError("load data error, wrong filepath")

    def save_data_idea2(self):
        """
        save the pixel pattern or feature pattern
        """
        path = '{}/{}/{}'.format(CHECKPOINT_PATH, self.args['dataset'], self.args['file_time'])
        if not os.path.exists(path):
            os.makedirs(path)
        for class_name in self.feature_pattern_dict.keys():
            name = 'idea2_pattern_' + str(class_name) + '.pt'
            filepath = '{}/{}'.format(path, name)
            torch.save(self.feature_pattern_dict[class_name], filepath)

    def load_data_idea2(self):
        """
        save the pixel pattern or feature pattern
        """
        path = '{}/{}/{}'.format(CHECKPOINT_PATH, self.args['dataset'], self.args['load_time'])
        for class_name in range(self.args['num_classes']):
            name = 'idea2_pattern_' + str(class_name) + '.pt'
            filepath = '{}/{}'.format(path, name)
            pattern = torch.load(filepath)
            self.feature_pattern_dict[class_name] = pattern

    def grad_MAD_function(self):
        if len(self.target_indices) > int(self.args['backdoor_train_size'] * 0.95):
            from sklearn.cluster import KMeans
            norms = []
            indices = []
            for indice, grad_norm in self.grad_record.items():
                indices.append(indice)
                norms.append(grad_norm)
            norms = torch.tensor(norms, device='cpu')
            norms = np.array(norms).reshape(-1, 1)
            logging.info(norms)
            keans_model = KMeans(n_clusters=2, max_iter=50, n_init=10)
            keans_model.fit(norms)
            labels = keans_model.labels_
            zero_num = len(np.where(labels == 0)[0])
            one_num = len(labels) - zero_num
            if zero_num > one_num:
                acc_label = 0
            else:
                acc_label = 1
            error_index = np.where(labels != acc_label)[0]
            error_index = sorted(error_index, key=lambda x: norms[x], reverse=True)  # down

            relation_index = error_index
            friends_relation = {}
            for index in range(len(norms)):
                if index != 0 and index != len(norms) - 1:
                    friends_relation[index] = (norms[index] / norms[index-1] + norms[index] / norms[index+1]) / 2
                elif index == 0:
                    friends_relation[index] = norms[index] / norms[index+1]
                else:
                    friends_relation[index] = norms[index] / norms[index-1]
            relation_index = sorted(relation_index, key=lambda x: friends_relation[x], reverse=True)  # down
            total_scores = {}
            for index in error_index:
                total_scores[index] = np.where(error_index == index)[0] + np.where(relation_index == index)[0]
            total_scores = sorted(total_scores.items(), key=lambda x: x[1], reverse=False)

            error_index = total_scores

            for index, score in error_index[:int(self.args['backdoor_train_size'] * 0.05)]:
                if indices[index] in self.temp_grad_record_acc:
                    self.FP += 1
                else:
                    self.TP += 1
                self.target_indices = np.delete(self.target_indices, np.where(self.target_indices == indices[index].item()))

    def set_epoch(self, epoch):

        self.epoch = epoch
        self.original_X = None
        self.norm = 0
        self.dis_rate = 0

        self.temp_grad_record_acc = []
        self.temp_grad_record_error = []
        self.grad_record = {}

    def set_backdoor_indices(self, target_indices, train_loader):
        self.target_indices = target_indices
        self.SR_train_loader = train_loader
        self.target_sample = []
        self.all_target_sample = []
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
                    Xb_batch = X[:, self.adversary+1:self.adversary+2].squeeze(1)
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

    def set_labeled_indice(self, indices):
        if indices is None:
            return
        self.labeled_indices = indices
        all_labeled_indices = []
        for value in self.labeled_indices.values():
            all_labeled_indices.extend(value)
        self.labeled_sample = {}
        for batch_idx, (X, Y_batch, old_imgb, indices) in enumerate(self.SR_train_loader):
            current_labeled_indices = []
            for i in range(len(indices)):
                if indices[i].item() in all_labeled_indices:
                    current_labeled_indices.append(i)
            if len(current_labeled_indices) != 0:
                if self.args['dataset'] != 'bhi' and self.args['n_passive_party'] < 2:
                    _, Xb_batch = X
                    if self.args['cuda']:
                        Xb_batch = Xb_batch.cuda()
                        old_imgb = old_imgb.cuda()
                else:
                    if self.args['cuda']:
                        X = X.cuda()
                        old_imgb = old_imgb.cuda()
                    Xb_batch = X[:, self.adversary+1:self.adversary+2].squeeze(1)
                for i in current_labeled_indices:
                    if Y_batch[i] in self.labeled_sample.keys():
                        self.labeled_sample[Y_batch[i].item()].append(old_imgb[i])
                    else:
                        self.labeled_sample[Y_batch[i].item()] = [old_imgb[i]]
        if self.args['cuda']:
            device = 'cuda'
        else:
            device = 'cpu'
        for key in self.labeled_sample.keys():
            self.labeled_sample[key] = torch.stack(self.labeled_sample[key]).to(device)  # [100,3,32,16]


    def receive_gradients(self, gradients):
        # ours sr_ba
        temp_grad = []
        original_gradients = gradients.clone()
        gradients = gradients.clone()

        temp_grad_dict = {}

        # get the target gradient of samples labeled backdoor class
        for index, i in enumerate(self.indices):
            if self.args['trigger'] == 'feature' and not self.args['trigger_add'] and self.args['idea'] == 2:
                if self.labeled_indices is not None:
                    label = None
                    for num_class in self.labeled_indices.keys():
                        if i.item() in self.labeled_indices[num_class]:
                            label = num_class
                    if label is not None:  # labeled data
                        if label not in temp_grad_dict.keys():
                            temp_grad_dict[label] = [index]
                        else:
                            temp_grad_dict[label].append(index)
            else:
                if self.target_indices is not None and i.item() in self.target_indices:
                    temp_grad.append(index)
                    if self.args['trigger'] == 'pixel':
                        gradients[index] = self.amplify_ratio * gradients[index]
                    elif self.args['trigger'] == 'feature' and self.args['trigger_add'] and self.args['idea'] == 1:
                        gradients[index] = self.alpha * gradients[index]

        self.common_grad = gradients
        # backwards
        self._fit(self.X, self.components)
        # self.backward()

        pixel_grad = 0
        if self.args['trigger'] == 'pixel':
            if temp_grad:
                for index in temp_grad:
                    pixel_grad += self.X.grad.data[index]
                pixel_grad = pixel_grad / len(temp_grad)
        feature_grad = 0
        if self.args['trigger'] == 'feature' and self.args['generation']:
            for index in temp_grad:
                if self.drop_out:
                    feature_grad = feature_grad + original_gradients[index] * self.mask[index] * self.amplify_ratio
                else:
                    feature_grad += original_gradients[index] * self.amplify_ratio
            if len(temp_grad) != 0:
                if self.drop_out:
                    self.mask = torch.stack(self.mask, dim=0)
                    num = torch.sum(self.mask[temp_grad], dim=0)
                    feature_grad = feature_grad / (num + 1e-6)
                else:
                    feature_grad = feature_grad / len(temp_grad)

        if not isinstance(feature_grad, int) and self.args['trigger'] == 'feature' and self.feature_pattern is not None and self.args['generation']:
            if self.args['trigger_add']:
                norm_type = 2
                # add loss function
                backdoor_output = self.feature_pattern.clone().detach()
                backdoor_output.requires_grad = True
                distance = torch.norm(backdoor_output, p=norm_type)
                distance.backward()
                backdoor_output_grad = backdoor_output.grad.data.detach()
                backdoor_output_grad = norm_clip(backdoor_output_grad, max_norm=torch.norm(feature_grad, p=2), norm_type=2)
                feature_grad = feature_grad + backdoor_output_grad * 0.5

                self.feature_pattern = self.feature_pattern - feature_grad * self.pattern_lr

            if not self.args['trigger_add'] and self.args['idea'] != 2:
                self.mini_distance = None
                mini_feature = None
                norm_type = 2
                self.bottom_model.eval()
                with torch.no_grad():
                    avg = 0
                    H_features = self.bottom_model.forward(self.target_sample)
                    for index in range(len(H_features)):
                        avg += torch.norm(H_features[index], p=norm_type)
                        difference = torch.norm(H_features[index] - self.feature_pattern, p=norm_type)
                        if self.mini_distance is None or difference < self.mini_distance:
                            self.mini_distance = difference
                            mini_feature = H_features[index]
                    avg = avg / len(H_features)
                backdoor_output = self.feature_pattern.clone().detach()
                backdoor_output.requires_grad = True
                mini_feature = mini_feature.clone().detach()

                distance = torch.norm(backdoor_output - mini_feature, p=norm_type)
                distance.backward()
                backdoor_output_grad = backdoor_output.grad.data.detach()
                feature_grad = feature_grad + backdoor_output_grad * self.args['local_feature_gamma']

                self.feature_pattern = self.feature_pattern - feature_grad * self.pattern_lr
                self.bottom_model.train()

        if not self.args['trigger_add'] and self.args['idea'] == 2:
            if temp_grad_dict:
                self.bottom_model.eval()
                for num_class in temp_grad_dict.keys():
                    feature_grad_dict = 0
                    for index in temp_grad_dict[num_class]:
                        feature_grad_dict += self.amplify_ratio * original_gradients[index]  # , 10
                    feature_grad_dict /= len(temp_grad_dict[num_class])

                    self.mini_distance = None
                    mini_feature = None
                    norm_type = 2
                    with torch.no_grad():
                        H_features = self.bottom_model.forward(self.labeled_sample[num_class])
                        for index in range(len(H_features)):
                            difference = torch.norm(H_features[index] - self.feature_pattern_dict[num_class], p=norm_type)
                            if self.mini_distance is None or difference < self.mini_distance:
                                self.mini_distance = difference
                                mini_feature = H_features[index]
                    backdoor_output = self.feature_pattern_dict[num_class].clone().detach()
                    backdoor_output.requires_grad = True
                    mini_feature = mini_feature.clone().detach()

                    distance = torch.norm(backdoor_output - mini_feature, p=norm_type)
                    distance.backward()
                    backdoor_output_grad = backdoor_output.grad.data.detach()
                    feature_grad_dict = feature_grad_dict + backdoor_output_grad
                    self.feature_pattern_dict[num_class] = self.feature_pattern_dict[num_class] - feature_grad_dict * self.pattern_lr
                self.bottom_model.train()

        return pixel_grad

    def send_components(self):
        result = self._forward_computation(self.X)
        self.components = result
        send_result = result.clone()

        self.mask = []
        if self.args['trigger'] == 'feature' and self.args['idea'] != 2:
            for index, i in enumerate(self.indices):
                current_mask = torch.full_like(send_result[index], 1)
                if self.target_indices is not None and i.item() in self.target_indices:
                    if self.args['generation']:
                        if self.feature_pattern is not None:
                            if self.drop_out:
                                current_mask = torch.nn.functional.dropout(current_mask, p=self.drop_out_rate, training=True)
                                current_mask = current_mask * (1 - self.drop_out_rate)
                            if self.args['trigger_add']:
                                if self.shifting:
                                    num = random.random() * (self.up_bound - self.down_bound) + self.down_bound
                                    send_result[index] += self.feature_pattern * current_mask * num
                                else:
                                    if self.args['idea'] == 1:
                                        send_result[index] = send_result[index] * self.alpha + self.feature_pattern * current_mask
                                    else:
                                        send_result[index] += self.feature_pattern * current_mask
                            else:
                                if self.drop_out:
                                    with torch.no_grad():
                                        mini_distance = None
                                        mini_feature = None
                                        H_features = self.bottom_model.forward(self.target_sample)
                                        for k in range(len(H_features)):
                                            difference = (H_features[k] - self.feature_pattern) * current_mask
                                            difference = torch.norm(difference, p=2)
                                            if mini_distance is None or difference < mini_distance:
                                                mini_distance = difference
                                                mini_feature = H_features[k]
                                    send_result[index] = self.feature_pattern * current_mask + mini_feature * (1 - current_mask)
                                else:
                                    send_result[index] = self.feature_pattern
                        else:
                            if self.args['trigger_add']:
                                self.feature_pattern = torch.rand(send_result[index].shape)
                            else:
                                self.feature_pattern = send_result[index]
                            if self.args['cuda']:
                                self.feature_pattern = self.feature_pattern.cuda()
                    elif self.args['trigger_add']:
                        if self.feature_pattern is None:
                                self.feature_pattern = torch.full_like(send_result[index], 0)
                                if self.args['cuda']:
                                    self.feature_pattern = self.feature_pattern.cuda()
                        if self.shifting:
                            num = random.random() * (self.up_bound - self.down_bound) + self.down_bound
                            send_result[index] += self.feature_pattern * num
                        else:
                            send_result[index] += self.feature_pattern

                    # distance evaluation
                    if self.epoch == self.args['target_epochs'] - 1:
                        self.bottom_model.eval()
                        with torch.no_grad():
                            self.mini_distance = None
                            H_features = self.bottom_model.forward(self.all_target_sample)
                            avg_norm = 0
                            mini_feature = None
                            for k in range(len(H_features)):
                                avg_norm += torch.norm(H_features[k], p=2)
                                difference = torch.norm(H_features[k] - send_result[index], p=2)
                                if self.mini_distance is None or difference < self.mini_distance:
                                    self.mini_distance = difference
                                    mini_feature = H_features[k]
                            self.norm += self.mini_distance
                            avg_norm = avg_norm / len(H_features)
                            # self.dis_rate = self.dis_rate + self.mini_distance / avg_norm
                            self.dis_rate += self.mini_distance / torch.norm(mini_feature, p=2)
                        self.bottom_model.train()

                self.mask.append(current_mask)

        if self.args['trigger'] == 'pixel':
            for index, i in enumerate(self.indices):
                if self.target_indices is not None and i.item() in self.target_indices:
                    if self.epoch == self.args['target_epochs'] - 1:
                        self.bottom_model.eval()
                        with torch.no_grad():
                            self.mini_distance = None
                            H_features = self.bottom_model.forward(self.all_target_sample)
                            avg_norm = 0
                            mini_feature = None
                            for k in range(len(H_features)):
                                avg_norm += torch.norm(H_features[k], p=2)
                                difference = torch.norm(H_features[k] - send_result[index], p=2)
                                if self.mini_distance is None or difference < self.mini_distance:
                                    self.mini_distance = difference
                                    mini_feature = H_features[k]
                            self.norm += self.mini_distance
                            avg_norm = avg_norm / len(H_features)
                            self.dis_rate = self.dis_rate + self.mini_distance / torch.norm(mini_feature, p=2)
                        self.bottom_model.train()

        # test idea-2
        if self.args['trigger'] == 'feature' and not self.args['trigger_add'] and self.args['idea'] == 2:
            for index, i in enumerate(self.indices):
                if self.labeled_indices is not None:
                    label = None
                    for num_class in self.labeled_indices.keys():
                        if i.item() in self.labeled_indices[num_class]:
                            label = num_class
                    if label is not None:  # labeled data
                        if label in self.feature_pattern_dict.keys():
                            send_result[index] = self.feature_pattern_dict[label]
                        else:
                            self.feature_pattern_dict[label] = send_result[index]

        return send_result

    def random_components(self, shape):
        variance = 1e-6
        result = torch.randn(shape)
        result = math.sqrt(variance)*result
        return result

    def predict(self, X, is_attack=False):
        result = self._forward_computation(X)
        send_results = result.clone()
        alpha = 1
        beta = 1
        sigmma = 0

        if (not is_attack) and self.args['trigger'] == 'feature' and self.args['idea'] == 2:
            for i in range(len(send_results)):
                self.h1 = torch.norm(send_results[i], 2)
                self.h3 = torch.norm(self.feature_pattern_dict[self.original_Y[i].item()], 2)
                send_results[i] = alpha * send_results[i] - beta * self.feature_pattern_dict[self.original_Y[i].item()]
                self.h2 = torch.norm(send_results[i], 2)

        if is_attack and self.args['trigger'] == 'feature':
            if self.args['generation']:
                if self.args['trigger_add']:
                    for i in range(len(send_results)):
                        if self.args['idea'] == 1:
                            send_results[i] = send_results[i] * self.alpha + self.feature_pattern
                        else:
                            send_results[i] += self.feature_pattern
                else:
                    if self.args['idea'] != 2:
                        for i in range(len(send_results)):
                            if self.args['random_test']:
                                noise = norm_clip(send_results[i], max_norm=torch.norm(self.feature_pattern)*self.args['test_norm'], norm_type=2)
                                send_results[i] = self.feature_pattern + noise
                            else:
                                send_results[i] = self.feature_pattern
                    else:
                        for i in range(len(send_results)):
                            send_results[i] = alpha * send_results[i] - beta * self.feature_pattern_dict[self.original_Y[i].item()] + sigmma * self.feature_pattern_dict[self.args['backdoor_label']]
            else:
                if self.args['trigger_add']:
                    for i in range(len(send_results)):
                        send_results[i] += self.feature_pattern
                else:
                    H_features = self.bottom_model.forward(self.target_sample)
                    H_features = torch.sum(H_features, dim=0) / len(self.target_sample)
                    for i in range(len(send_results)):
                        send_results[i] = H_features

        return send_results
