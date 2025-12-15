import logging

import numpy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import wandb
import copy
from vfl.defense.gradient_compression import GradientCompression
from vfl.defense.noisy_gradients import laplace_noise, new_laplace_noise
from vfl.defense.norm_clip import norm_clip
from vfl.defense.Max_norm import gradient_masking, new_gradient_masking

import statistics
mode = 0

class VFLActiveModel(object):
    """
    VFL active party
    """
    def __init__(self, bottom_model, args, top_model=None):
        super(VFLActiveModel, self).__init__()
        self.bottom_model = bottom_model
        self.is_debug = False

        self.classifier_criterion = nn.CrossEntropyLoss()
        self.parties_grad_component_list = []  # latent representations from passive parties
        self.X = None
        self.y = None
        self.bottom_y = None  # latent representation from local bottom model
        self.top_grads = None  # gradients of local bottom model
        self.parties_grad_list = []  # gradients for passive parties
        self.epoch = None  # current train epoch
        self.indices = None  # indices of current train samples

        self.top_model = top_model
        self.top_trainable = True if self.top_model is not None else False

        self.args = args.copy()

        if self.args['cuda']:
            self.classifier_criterion.cuda()

        # outlier detection
        self.m_median = {}
        self.m_MAD = {}
        self.record_last = {}
        self.record_last_indice = {}
        self.median = {}
        self.avg = {}
        self.MAD = {}
        for j in range(self.args['n_passive_party']):
            self.m_median[j] = {}
            self.m_MAD[j] = {}
            self.record_last[j] = {}
            self.median[j] = {}
            self.avg[j] = {}
            self.MAD[j] = {}
            self.record_last_indice[j] = {}
            for i in range(self.args['num_classes']):
                self.m_median[j][i] = None
                self.m_MAD[j][i] = None
                self.record_last[j][i] = None
                self.record_last_indice[j][i] = None
                self.median[j][i] = None
                self.avg[j][i] = None
                self.MAD[j][i] = None
        self.attack_indices = []
        self.attack_index = []
        self.backdoor_MAD = []
        self.benign_MAD = []

        self.embedding_record_current = None
        self.embedding_record_current_y = None
        self.embedding_record_last = None
        self.embedding_record_last_y = None

        self.status = None
        self.test_norm = {}
        self.test_dis_rate = {}
        self.test_norm2 = {}
        self.test_dis_rate2 = {}
        for j in range(self.args['n_passive_party']):
            self.test_norm[j] = 0
            self.test_dis_rate[j] = 0
            self.test_norm2[j] = 0
            self.test_dis_rate2[j] = 0

        self.backdoor_y_test_true = None
        self.backdoor_indice = None

        self.ERROR_NUM = 0
        self.DIS_NUM = 0
        self.loss_record = []
        self.Detection_FP = 0
        self.Detection_TP = 0

    def set_indices(self, indices):
        self.indices = indices

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.args['outlier_detection'] and self.args['backdoor'] != 'lr_ba':
            if self.epoch > self.args['backdoor_epochs']:
                self.MAD_function()
                self.attack_indices = []
                self.attack_index = []
                self.backdoor_MAD = []
                self.benign_MAD = 0
        if self.args['embedding_detection'] and self.args['backdoor'] != 'lr_ba':
            self.embedding_record_last = self.embedding_record_current
            self.embedding_record_current = None
            self.embedding_record_last_y = self.embedding_record_current_y
            self.embedding_record_current_y = None
            self.attack_indices = []
            self.attack_index = []
        # clean
        if self.status == 'train':
            for j in range(self.args['n_passive_party']):
                self.record_last[j] = {}
                self.record_last_indice[j] = {}
                for i in range(self.args['num_classes']):
                    self.record_last[j][i] = None
                    self.record_last_indice[j][i] = None

        if self.args['ABL'] and self.epoch == self.args['t_epochs']:
            losses_idx = sorted(self.loss_record, key=lambda x:x[1])   #[indice, loss] ascend
            perm = losses_idx[0: int(len(losses_idx) * self.args['isolation_ratio'])]
            DA = 0
            for i in perm:
                self.attack_indices.append(i[0].item())
                if i[0].item() in self.backdoor_indice:
                    DA += 1
            logging.info('LBA DA : {}/{}={}%'.format(DA, len(perm), DA/len(perm)*100))

    def set_batch(self, X, y):
        self.X = X
        self.y = y

    def set_status(self, status):
        if status in ['train', 'test', 'backdoor_test', 'none']:
            self.status = status
            if status == 'backdoor_test':
                for client in range(self.args['n_passive_party']):
                    self.test_norm[client] = 0
                    self.test_dis_rate[client] = 0
                    self.test_norm2[client] = 0
                    self.test_dis_rate2[client] = 0
        else:
            print('ERROR status to set!!!')
            import sys
            sys.exit()

    def MAD_function(self):
        if 'load_time' in self.args and self.args['load_time']:
            client = self.args['adversary'] - 1
            if mode == 0:
                for label in self.record_last[client]:
                    records = self.record_last[client][label].clone().detach()  # [k,32,8,4]
                    norms = []
                    for record in records:
                        norms.append(torch.norm(record, p=2).item())
                    self.median[client][label] = statistics.median(norms)
                    self.avg[client][label] = statistics.mean(norms)
                    diff_norms = [abs(norm - self.median[client][label]) for norm in norms]
                    self.MAD[client][label] = statistics.median(diff_norms) 
                    self.MAD[client][label] = self.MAD[client][label] * 1.4826

                    self.m_median[client][label] = torch.median(records, dim=0).values 
                    for i in range(len(records)):
                        records[i] = torch.abs(records[i] - self.m_median[client][label])
                    self.m_MAD[client][label] = torch.median(records, dim=0).values
                    self.m_MAD[client][label] = self.m_MAD[client][label] * 1.4826
            if mode == 1:
                for label in self.record_last[client]:
                    norms = []
                    records = self.record_last[client][label].clone().detach() 
                    H_features = records
                    for index in range(len(records)):
                        sample = records[index]
                        mini_distance = None
                        for temp_k in range(len(H_features)):
                            if temp_k != index:
                                difference = torch.norm(H_features[temp_k] - sample, p=2)
                                if mini_distance is None or difference < mini_distance:
                                    mini_distance = difference
                        norm = mini_distance
                        norms.append(norm.item())
                    self.median[client][label] = statistics.median(norms)
                    self.avg[client][label] = statistics.mean(norms)
                    diff_norms = [abs(norm - self.median[client][label]) for norm in norms]
                    self.MAD[client][label] = statistics.median(diff_norms)  # constant
                    self.MAD[client][label] = self.MAD[client][label] * 1.4826

                    self.m_median[client][label] = torch.median(records, dim=0).values  # [32,8,4]
                    for i in range(len(records)):
                        records[i] = torch.abs(records[i] - self.m_median[client][label])
                    self.m_MAD[client][label] = torch.median(records, dim=0).values  # [32,8,4]
                    self.m_MAD[client][label] = self.m_MAD[client][label] * 1.4826
                    print('median[{}][{}]: {}, avg: {}'.format(client, label, self.median[client][label], self.avg[client][label]))
        else:
            for client in self.record_last.keys():
                for label in self.record_last[client]:
                    records = self.record_last[client][label].clone().detach()  # [k,32,8,4]
                    norms = []
                    for record in records:
                        norms.append(torch.norm(record, p=2).item())
                    self.median[client][label] = statistics.median(norms)
                    self.avg[client][label] = statistics.mean(norms)
                    diff_norms = [abs(norm - self.median[client][label]) for norm in norms]
                    self.MAD[client][label] = statistics.median(diff_norms)  # constant
                    self.MAD[client][label] = self.MAD[client][label] * 1.4826

                    self.m_median[client][label] = torch.median(records, dim=0).values  # [32,8,4]
                    for i in range(len(records)):
                        records[i] = torch.abs(records[i] - self.m_median[client][label])
                    self.m_MAD[client][label] = torch.median(records, dim=0).values  # [32,8,4]
                    self.m_MAD[client][label] = self.m_MAD[client][label] * 1.4826

    def outlier_record(self, component_list, outlier_indice):
        if 'load_time' in self.args and self.args['load_time']:
            client = self.args['adversary'] - 1
            for index in range(len(component_list[client])): # [bs,]
                if outlier_indice and outlier_indice[client] and index in outlier_indice[client]:
                    continue
                else:
                    label = self.y[index].item()
                    if self.record_last[client][label] is None:
                        self.record_last[client][label] = torch.unsqueeze(component_list[client][index], dim=0)  # [1,32,8,4]
                        self.record_last_indice[client][label] = torch.unsqueeze(self.indices[index], dim=0)
                    else:
                        self.record_last[client][label] = torch.cat([self.record_last[client][label], torch.unsqueeze(component_list[client][index], dim=0)], dim=0) # [k,32,8,4]
                        self.record_last_indice[client][label] = torch.cat([self.record_last_indice[client][label], torch.unsqueeze(self.indices[index], dim=0)])
        else:
            for client in range(len(component_list)):
                for index in range(len(component_list[client])):  # [bs,]
                    if outlier_indice and outlier_indice[client] and index in outlier_indice[client]:
                        continue
                    else:
                        label = self.y[index].item()
                        if self.record_last[client][label] is None:
                            self.record_last[client][label] = torch.unsqueeze(component_list[client][index],
                                                                              dim=0)  # [1,32,8,4]
                            self.record_last_indice[client][label] = torch.unsqueeze(self.indices[index], dim=0)
                        else:
                            self.record_last[client][label] = torch.cat(
                                [self.record_last[client][label], torch.unsqueeze(component_list[client][index], dim=0)],
                                dim=0)  # [k,32,8,4]
                            self.record_last_indice[client][label] = torch.cat(
                                [self.record_last_indice[client][label], torch.unsqueeze(self.indices[index], dim=0)])
        return

    def _fit(self, X, y):
        """
        compute gradients, and update local bottom model and top model

        :param X: features of active party
        :param y: labels
        """
        if self.status != 'train':
            print('try to train the model in none-train status!!!')
            import sys
            sys.exit()

        # get local latent representation
        self.bottom_y = self.bottom_model.forward(X)
        self.K_U = self.bottom_y.detach().requires_grad_()

        # compute gradients based on labels, including gradients for passive parties
        self._compute_common_gradient_and_loss(y)

        # update parameters of local bottom model and top model
        self._update_models(X, y)

    def predict(self, X, component_list, type):
        """
        get the final prediction

        :param X: feature of active party
        :param component_list: latent representations from passive parties
        :return: prediction label
        """

        if 'load_time' in self.args and self.args['load_time'] and self.status == 'test' and self.epoch == -1:
            outlier_indice = {}
            self.outlier_record(component_list, outlier_indice)

        # get local latent representation
        U = self.bottom_model.forward(X)

        # clip latent representation if using clip defense
        if 'norm_clip' in self.args and self.args['norm_clip']:
            for component in component_list:
                for value in component:
                    norm_clip(value, max_norm=self.args['clip_threshold'])
            for temp in U:
                norm_clip(temp, max_norm=self.args['clip_threshold'])


        # sum up latent representation in VFL without model splitting
        if not self.top_trainable:
            for comp in component_list:
                U = U + comp
        # use top model to predict in VFL with model splitting
        else:
            if self.args['aggregate'] == 'Concate':
                temp = torch.cat([U] + component_list, -1)
            elif self.args['aggregate'] == 'Add':
                temp = U
                for comp in component_list:
                    temp = temp + comp
            elif self.args['aggregate'] == 'Mean':
                temp = U
                for comp in component_list:
                    temp = temp + comp
                temp = temp / (len(component_list)+1)
            U = self.top_model.forward(temp)

        # ABL compute loss
        self.y = self.y.long().to(U.device)
        if self.args['ABL'] and self.epoch == self.args['t_epochs'] - 1 and type == 'train':
            losses = nn.functional.cross_entropy(U, self.y, reduction='none')
            for i in range(len(U)):
                self.loss_record.append([self.indices[i], losses[i]])

        result = F.softmax(U, dim=1)

        # detection
        if self.args['outlier_detection'] and self.args['backdoor'] != 'lr_ba' and self.median[0][0] is not None:
            # self.MAD_function()
            self.attack_index = []
            # for client in range(len(component_list)):
            client = self.args['adversary'] - 1
            for index in range(len(component_list[client])):  # [bs,]
                # label = self.y[index].item()
                label = torch.argmax(result[index]).item()
                if mode == 0:
                    MAD_score = abs(torch.norm(component_list[client][index], p=2) - self.median[client][label]) / self.MAD[client][label]
                if mode == 1:
                    records = self.record_last[client][label].clone().detach()  # [k,32,8,4]
                    H_features = records
                    mini_distance = None
                    sample = component_list[client][index]
                    for temp_k in range(len(H_features)):
                        difference = torch.norm(H_features[temp_k] - sample, p=2)
                        if mini_distance is None or difference < mini_distance:
                            mini_distance = difference
                    MAD_score = abs(mini_distance - self.median[client][label]) / self.MAD[client][label]

                if type == 'test':
                    self.benign_MAD.append(MAD_score)
                elif type == 'attack':
                    self.backdoor_MAD.append(MAD_score)
                if MAD_score > self.args['outlier_detection']:
                    # some method to mitigate the attack
                    self.attack_indices.append(self.indices[index])
                    self.attack_index.append(index)
            if type == 'test':
                self.Detection_FP += len(self.attack_index)
            elif type =='attack':
                self.Detection_TP += len(self.attack_index)

        # normal stealthy detection
        if self.epoch is not None and self.epoch == self.args['target_epochs'] - 1 and self.status == 'backdoor_test' and self.args['backdoor'] != 'no':
            for client in range(len(component_list)):  # [n, bs, features]
                for index in range(len(component_list[client])):  # [bs,]
                    label = torch.argmax(result[index]).item()
                    if label != self.args['backdoor_label']:
                        self.ERROR_NUM += 1
                    mini_distance = None
                    mini_indice = None
                    for temp_i in range(len(self.record_last[client][label])):
                        record = self.record_last[client][label][temp_i]
                        distance = torch.norm(component_list[client][index] - record, 2)
                        if mini_distance is None or distance < mini_distance:
                            mini_distance = distance
                            mini_indice = self.record_last_indice[client][label][temp_i]
                    if mini_indice.item() not in self.backdoor_indice:
                        self.DIS_NUM += 1
                    self.test_norm[client] += mini_distance
                    self.test_dis_rate[client] += mini_distance / self.avg[client][label]
                    true_label = self.backdoor_y_test_true[self.indices[index]]
                    mini_distance = None
                    for record in self.record_last[client][true_label]:
                        distance = torch.norm(component_list[client][index] - record, 2)
                        if mini_distance is None or distance < mini_distance:
                            mini_distance = distance
                    self.test_norm2[client] += mini_distance
                    self.test_dis_rate2[client] += mini_distance / self.avg[client][true_label]

        return result

    def receive_components(self, component_list):
        """
        receive latent representations from passive parties

        :param component_list: latent representations from passive parties
        """
        for party_component in component_list:
            self.parties_grad_component_list.append(party_component.detach().requires_grad_())

        self.attack_index = []
        if self.args['outlier_detection'] and self.args['backdoor'] != 'lr_ba':
            outlier_indice = {}
            if self.epoch > self.args['backdoor_epochs']:
                for client in range(len(component_list)):
                    outlier_indice[client] = []
                    for index in range(len(component_list[client])):  # [bs,]
                        label = self.y[index].item()
                        MAD_score = abs(torch.norm(component_list[client][index], p=2) - self.median[client][label]) / self.MAD[client][label]
                        if self.indices[index].item() in self.backdoor_indice:
                            self.backdoor_MAD.append(MAD_score)
                        else:
                            self.benign_MAD += MAD_score
                        if MAD_score > 3:
                            # some method to mitigate the attack
                            outlier_indice[client].append(index)
                            if self.indices[index] not in self.attack_indices:
                                self.attack_indices.append(self.indices[index])
                                self.attack_index.append(index)

            if self.epoch > self.args['backdoor_epochs'] - 1:
                self.outlier_record(component_list, outlier_indice)
        else:
            if self.epoch == self.args['target_epochs'] - 1:
                outlier_indice = {}
                self.outlier_record(component_list, outlier_indice)

        if self.args['embedding_detection'] and self.args['backdoor'] != 'lr_ba':
            outlier_indice = {}
            if self.epoch > self.args['backdoor_epochs']:
                for client in range(len(component_list)):
                    outlier_indice[client] = []
                    for index in range(len(component_list[client])): 
                        if self.indices[index].item() in self.backdoor_indice:
                            true_label = self.y[index].item()
                            mini_distance = None
                            mini_label = None
                            for record_index in range(len(self.embedding_record_last)):
                                distance = torch.norm(self.embedding_record_last[record_index][client] - component_list[client][index], 2)
                                if mini_distance is None or distance < mini_distance:
                                    mini_distance = distance
                                    mini_label = self.embedding_record_last_y[record_index].item()
                            if mini_label != true_label:
                                outlier_indice[client].append(index)
                                self.attack_indices.append(self.indices[index])
            if self.epoch > self.args['backdoor_epochs'] - 1:
                temp = 0
                for client in range(len(component_list)):
                    if outlier_indice and outlier_indice[client]:
                        temp = 1
                if temp == 1:
                    for index in range(len(component_list[0])):
                        temp = 1
                        for client in range(len(component_list)):
                            if (outlier_indice and outlier_indice[client] and index in outlier_indice[client]):
                                temp = 0
                        if temp:
                            embedding = []
                            for client in range(len(component_list)):
                                embedding.append(component_list[client][index])

                            if self.embedding_record_current is None:
                                self.embedding_record_current = torch.stack(embedding,0).unsqueeze(0)
                                self.embedding_record_current_y = torch.tensor(self.y[index]).unsqueeze(0)
                            else:
                                self.embedding_record_current = torch.cat((self.embedding_record_current, torch.stack(embedding, 0).unsqueeze(0)),0)  # 50000,3,10
                                self.embedding_record_current_y = torch.cat((self.embedding_record_current_y, torch.tensor(self.y[index]).unsqueeze(0)),0)  # 50000,1
                else:
                    if self.embedding_record_current is None:
                        self.embedding_record_current = torch.stack(component_list,0).permute(1,0,2)
                        self.embedding_record_current_y = self.y
                    else:
                        self.embedding_record_current = torch.cat((self.embedding_record_current, torch.stack(component_list,0).permute(1,0,2)), 0)  # 50000,3,10
                        self.embedding_record_current_y = torch.cat((self.embedding_record_current_y, self.y), 0)  # 50000,1


    def fit(self):
        """
        backward
        """
        self.parties_grad_list = []
        self._fit(self.X, self.y)
        self.parties_grad_component_list = []

    def _compute_common_gradient_and_loss(self, y):
        """
        compute loss and gradients, including gradients for passive parties

        :param y: label
        """
        # compute prediction
        U = self.K_U

        grad_comp_list = [self.K_U] + self.parties_grad_component_list
        if not self.top_trainable:
            for grad_comp in self.parties_grad_component_list:
                U = U + grad_comp
        else:
            if self.args['aggregate'] == 'Concate':
                temp = torch.cat(grad_comp_list, -1)
            elif self.args['aggregate'] == 'Add':
                temp = grad_comp_list[0]
                for comp in grad_comp_list[1:]:
                    temp = temp + comp
            elif self.args['aggregate'] == 'Mean':
                temp = grad_comp_list[0]
                for comp in grad_comp_list[1:]:
                    temp = temp + comp
                temp = temp / len(grad_comp_list)
            U = self.top_model.forward(temp)

        # compute loss
        y = y.long().to(U.device)
        if not self.args['ABL'] or self.epoch < self.args['t_epochs']:
            class_loss = self.classifier_criterion(U, y)
        else:
            class_loss = nn.functional.cross_entropy(U, y, reduction='none')  # [bs]


        if self.args['ABL']:
            if self.epoch < self.args['t_epochs']:
                if self.args['gradient_ascent_type'] == 'LGA':
                    # add Local Gradient Ascent(LGA) loss
                    class_loss = torch.sign(class_loss - self.args['flooding_gamma']) * class_loss
                elif self.args['gradient_ascent_type'] == 'Flooding':
                    # add flooding loss
                    class_loss = (class_loss - self.args['flooding_gamma']).abs() + self.args['flooding_gamma']
            else:
                for i in range(len(self.indices)):
                    if self.indices[i].item() in self.attack_indices:
                        class_loss[i] = -1 * class_loss[i]
                class_loss = torch.mean(class_loss)

        # compute gradients
        if self.top_trainable:
            class_loss.backward(retain_graph=True)
            grad_list = [temp.grad for temp in grad_comp_list]
        else:
            grad_list = torch.autograd.grad(outputs=class_loss, inputs=grad_comp_list)
        # save gradients of local bottom model
        self.top_grads = grad_list[0]
        # save gradients for passive parties
        for index in range(0, len(self.parties_grad_component_list)):
            parties_grad = grad_list[index+1]
            self.parties_grad_list.append(parties_grad)

        # add noise to gradients for passive parties if using noisy gradient defense
        if 'noisy_gradients' in self.args and self.args['noisy_gradients']:
            self.top_grads = laplace_noise(self.top_grads, self.args['noise_scale'])
            # self.top_grads = new_laplace_noise(self.top_grads, self.args['noise_scale'])
            for i, party_grad in enumerate(self.parties_grad_list):
                self.parties_grad_list[i] = laplace_noise(party_grad, self.args['noise_scale'])
                # self.parties_grad_list[i] = new_laplace_noise(party_grad, self.args['noise_scale'])
        # compress gradient for all parties if using gradient compression defense
        elif 'gradient_compression' in self.args and self.args['gradient_compression']:
            gc = GradientCompression(gc_percent=self.args['gc_percent'])
            self.top_grads = gc.compression(self.top_grads)
            for i in range(0, len(self.parties_grad_list)):
                self.parties_grad_list[i] = gc.compression(self.parties_grad_list[i])
        elif 'max_norm' in self.args and self.args['max_norm']:
            self.top_grads = gradient_masking(self.top_grads)  # [batch, 10] or [bs,,,,]
            for i in range(0, len(self.parties_grad_list)):
                self.parties_grad_list[i] = gradient_masking(self.parties_grad_list[i])
            # self.top_grads = new_gradient_masking(self.top_grads)  # [batch, 10] or [bs,,,,]
            # for i in range(0, len(self.parties_grad_list)):
            #     self.parties_grad_list[i] = new_gradient_masking(self.parties_grad_list[i])

        self.loss = class_loss.item()*self.K_U.shape[0]

    def send_gradients(self):
        """
        send gradients to passive parties
        """
        return self.parties_grad_list

    def _update_models(self, X, y):
        """
        update parameters of local bottom model and top model

        :param X: features of active party
        :param y: invalid
        """
        self.bottom_model.backward(X, self.bottom_y, self.top_grads)
        # update parameters of top model
        if self.top_trainable:
            self.top_model.backward_()

    def get_loss(self):
        return self.loss

    def save(self):
        """
        save model to local file
        """
        if self.top_trainable:
            self.top_model.save(time=self.args['file_time'])
        self.bottom_model.save(time=self.args['file_time'])

    def load(self):
        """
        load model from local file
        """
        if self.top_trainable:
            self.top_model.load(time=self.args['load_time'])
        self.bottom_model.load(time=self.args['load_time'])

    def set_train(self):
        """
        set train mode
        """
        if self.top_trainable:
            self.top_model.train()
        self.bottom_model.train()

    def set_eval(self):
        """
        set eval mode
        """
        if self.top_trainable:
            self.top_model.eval()
        self.bottom_model.eval()

    def scheduler_step(self):
        """
        adjust learning rate during training
        """
        if self.top_trainable and self.top_model.scheduler is not None:
            self.top_model.scheduler.step()
        if self.bottom_model.scheduler is not None:
            self.bottom_model.scheduler.step()

    def set_args(self, args):
        self.args = args

    def zero_grad(self):
        """
        clear gradients
        """
        if self.top_trainable:
            self.top_model.zero_grad()
        self.bottom_model.zero_grad()


class VFLPassiveModel(object):
    """
    VFL passive party
    """
    def __init__(self, bottom_model, id=None, args=None):
        super(VFLPassiveModel, self).__init__()
        self.bottom_model = bottom_model
        self.is_debug = False
        self.common_grad = None  # gradients
        # self.partial_common_grad = None
        self.X = None
        self.indices = None
        self.epoch = None
        self.y = None
        self.id = id  # id of passive party
        self.args = args

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_batch(self, X, indices):
        self.X = X
        self.indices = indices

    def _forward_computation(self, X, model=None):
        """
        forward

        :param X: features of passive party
        :param model: invalid
        :return: latent representation of passive party
        """
        if model is None:
            A_U = self.bottom_model.forward(X)
        else:
            A_U = model.forward(X)
        self.y = A_U
        return A_U

    def _fit(self, X, y):
        """
        backward

        :param X: features of passive party
        :param y: latent representation of passive party
        """
        self.bottom_model.backward(X, y, self.common_grad, self.epoch)
        return

    def receive_gradients(self, gradients):
        """
        receive gradients from active party and update parameters of local bottom model

        :param gradients: gradients from active party
        """
        self.common_grad = gradients
        self._fit(self.X, self.y)

    def send_components(self):
        """
        send latent representation to active party
        """
        result = self._forward_computation(self.X)
        return result

    def predict(self, X, is_attack=False):
        return self._forward_computation(X)

    def save(self):
        """
        save model to local file
        """
        self.bottom_model.save(id=self.id, time=self.args['file_time'])

    def load(self, load_attack=False):
        """
        load model from local file

        :param load_attack: invalid
        """
        if load_attack:
            self.bottom_model.load(name='attack', time=self.args['load_time'])
        else:
            self.bottom_model.load(id=self.id, time=self.args['load_time'])

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

    def scheduler_step(self):
        """
        adjust learning rate during training
        """
        if self.bottom_model.scheduler is not None:
            self.bottom_model.scheduler.step()

    def zero_grad(self):
        """
        clear gradients
        """
        self.bottom_model.zero_grad()
