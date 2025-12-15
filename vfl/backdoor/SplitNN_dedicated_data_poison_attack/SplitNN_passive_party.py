import torch
from vfl.party_models import VFLPassiveModel
from common.constants import CHECKPOINT_PATH
import os
import numpy as np
from vfl.backdoor.SplitNN_dedicated_data_poison_attack.cal_centers import filter_dim, cal_target_center, search_vec

"""
Malicious passive party for SplitNN_dedicated_data_poison_attack  backdoor
"""

class SplitNN_poison_PassiveModel(VFLPassiveModel):
    def __init__(self, bottom_model, args=None, id=None):
        VFLPassiveModel.__init__(self, bottom_model, id, args)
        self.components = None
        self.is_debug = False
        self.args = args

        self.adversary = self.args['adversary'] - 1
        self.local_dataset = None
        self.labels = None
        self.indices = None

        self.feature_pattern = None 
        self.clean_epoch = 50 
        self.update_epoch = 30
        if self.args['dataset'] == 'bhi':
            self.clean_epoch = 3
            self.update_epoch = 5
        self.steal_num = self.args['backdoor_train_size'] 
        self.target_num = 50 
        self.steal_samples = None
        self.steal_ids = None

    def set_epoch(self, epoch):
        if epoch == 0:
            self.set_eval()
            self.steal_dataset()
            self.set_train()
        self.epoch = epoch
        if epoch >= self.clean_epoch and (epoch - self.clean_epoch) % self.update_epoch == 0:
            self.set_eval()
            self.feature_pattern = self.design_vec()
            self.set_train()
        self.norm = 0
        self.dis_rate = 0

    def steal_dataset(self):
        with torch.no_grad():
            self.set_eval()
            self.steal_samples = []
            self.steal_ids = []
            self.all_target_sample = []  
            for batch_index in range(len(self.local_dataset)): 
                if len(self.steal_samples) == self.steal_num:
                    break
                Xb_batch = self.local_dataset[batch_index]
                Y_batch = self.labels[batch_index]
                indice = self.indices[batch_index]
                for i in range(len(Xb_batch)):
                    if Y_batch[i] == self.args['backdoor_label']:
                        self.all_target_sample.append(Xb_batch[i])
                        if len(self.steal_samples) < self.steal_num:
                            self.steal_samples.append(Xb_batch[i])
                            self.steal_ids.append(indice[i].item())  
            self.steal_samples = torch.stack(self.steal_samples, dim=0) 
            self.all_target_sample = torch.stack(self.all_target_sample)
            if self.args['cuda']:
                self.all_target_sample = self.all_target_sample.cuda()

    def design_vec(self):
        target_clean_vecs = self.generate_target_clean_vecs() 
        dim = filter_dim(target_clean_vecs, self.target_num)
        center = cal_target_center(target_clean_vecs[dim].copy(), kernel_bandwidth=1000)  # [1,4096]
        target_vec = search_vec(center, target_clean_vecs)  # [1,4096]
        target_vec = target_vec.reshape(self.bottom_model(self.steal_samples[0].unsqueeze(0)).shape[1:])  # 128,8,4
        target_vec = torch.tensor(target_vec)
        if self.args['cuda']:
            target_vec = target_vec.cuda()
        return target_vec

    def generate_target_clean_vecs(self):
        vecs = []
        for sample in self.steal_samples:
            embedding = self.bottom_model(sample.unsqueeze(0))
            vecs.append(embedding.detach().cpu().numpy())
        vecs = np.array(vecs)  
        shape = vecs.shape 
        vecs = vecs.reshape((shape[0], -1))
        return vecs

    def send_components(self):
        result = self._forward_computation(self.X)
        self.components = result
        send_result = result.clone()

        if self.epoch >= self.clean_epoch:
            for index, i in enumerate(self.indices):
                if self.steal_ids is not None and i.item() in self.steal_ids: 
                    if self.feature_pattern is not None:
                        send_result[index] = self.feature_pattern
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
                                self.dis_rate += self.mini_distance / torch.norm(mini_feature, p=2)
                            self.bottom_model.train()
        return send_result

    def predict(self, X, is_attack=False):
        result = self._forward_computation(X)
        send_results = result
        if is_attack and self.feature_pattern is not None:
            send_results = self.feature_pattern.unsqueeze(0).repeat(len(X), 1, 1, 1)
        return send_results

    def save(self):
        """
        save model to local file
        """
        self.bottom_model.save(id=self.id, time=self.args['file_time'])
        self.save_data()

    def load(self, load_attack=False):
        """
        load model from local file

        :param load_attack: invalid
        """
        self.bottom_model.load(id=self.id, time=self.args['load_time'])
        self.load_data()

    def save_data(self, name=None, pattern=None):
        """
        save the pixel pattern or feature pattern
        """
        path = '{}/{}/{}'.format(CHECKPOINT_PATH, self.args['dataset'], self.args['file_time'])
        if not os.path.exists(path):
            os.makedirs(path)
        if name is None:
            name = ''
        name += 'vec_pattern.pt'
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
        if name is None:
            name = ''
        name += 'vec_pattern.pt'
        filepath = '{}/{}'.format(path, name)
        if os.path.isfile(filepath):
            pattern = torch.load(filepath)
            if 'feature' in self.args['trigger']:
                self.feature_pattern = pattern
            return pattern
        else:
            raise ValueError("load data error, wrong filepath")