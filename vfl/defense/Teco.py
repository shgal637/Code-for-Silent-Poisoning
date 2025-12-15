# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from vfl.party_models import VFLActiveModel
from imagecorruptions import corrupt
import random

class Teco_active_party(VFLActiveModel):
    def __init__(self, bottom_model, args=None, top_model=None):
        super(Teco_active_party, self).__init__(bottom_model, args=args, top_model=top_model)

        self.bottom_model = bottom_model
        self.is_debug = False

        self.classifier_criterion = nn.CrossEntropyLoss()
        self.parties_grad_component_list = []
        self.X = None
        self.y = None
        self.bottom_y = None
        self.top_grads = None
        self.parties_grad_list = []
        self.epoch = None
        self.indices = None
        self.top_model = top_model
        self.top_trainable = True if self.top_model is not None else False
        self.args = args.copy()
        if self.args['cuda']:
            self.classifier_criterion.cuda()
        # for defense
        self.threshold = 0
        self.thresholds = []
        self.corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'motion_blur',
                     'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
                     'jpeg_compression']
        self.attack_indices = {}
        self.attack_true_detection = 0
        self.attack_false_detection = 0
        self.clean_true_detection = 0
        self.clean_false_detection = 0
        self.teco_labels = []
        self.mads = []
        self.attack_mads = []
        self.clean_mads = []

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.attack_true_detection = 0
        self.attack_false_detection = 0
        self.clean_true_detection = 0
        self.clean_false_detection = 0

    def dg(self, image, cor_type, severity):
        '''
        Parameters
        ----------
        image: embeddings, 128, 64
        cor_type: corruption
        severity: int
        Returns: the corrupted embedding, shape same as image
        -------
        '''
        a = 0
        b = 255
        image = image.detach().cpu().numpy()
        min_val = np.min(image)
        max_val = np.max(image)
        normalized_tensor = a + ((image - min_val) * (b - a) / (max_val - min_val))
        normalized_tensor = normalized_tensor.astype(np.uint8)
        corrupt_image = corrupt(normalized_tensor, corruption_name=cor_type, severity=severity)
        restored_tensor = min_val + ((corrupt_image - a) * (max_val - min_val) / (b - a))
        restored_tensor = torch.tensor(restored_tensor)
        if self.args['cuda']:
            restored_tensor = restored_tensor.cuda()
        return restored_tensor

    def predict(self, X, component_list, type):
        """
        get the final prediction

        :param X: feature of active party
        :param component_list: latent representations from passive parties
        :param type：test or attack
        :return: prediction label
        """
        # get local latent representation
        U = self.bottom_model.forward(X)

        # sum up latent representation in VFL without model splitting
        if not self.top_trainable:
            raise ValueError('not support none-trainable active party!')

        if len(component_list) > 1:
            raise ValueError('not support more than two party!')

        # use top model to predict in VFL with model splitting
        else:
            if self.args['aggregate'] == 'Concate':
                temp = torch.cat([U] + component_list, -1)
                U = self.top_model.forward(temp)
                clean_result = F.softmax(U, dim=1)
            else:
                raise ValueError('not support sum or mean aggregation!')

            defense_result = clean_result.clone()

            if type == 'attack' or type == 'test' or type == 'train':
                filled_input = temp.clone()
                filled_input = filled_input.reshape(filled_input.size(0), filled_input.size(1), filled_input.size(2) * filled_input.size(3))
                filled_input = filled_input.unsqueeze(-1).repeat(1, 1, 1, 3) 

                record_dict = {}
                pre_labels = torch.max(clean_result, dim=1)[1]
                for j in range(len(pre_labels)):
                    save_name = str(j)
                    record_dict[save_name] = {}
                    record_dict[save_name]['original'] = [pre_labels[j].item()]
                for name in self.corruptions:
                    for severity in range(1, 6):
                        corrupted_input = filled_input.clone()
                        for i in range(len(filled_input)):
                            corrupted_input[i] = self.dg(filled_input[i], cor_type=name, severity=severity)
                        corrupted_input = corrupted_input[:,:,:,0] 
                        corrupted_input = corrupted_input.reshape(temp.size(0), temp.size(1), temp.size(2), temp.size(3))

                        outputs = self.top_model.forward(corrupted_input)
                        outputs = F.softmax(outputs, dim=1) 
                        pre_labels = torch.max(outputs, dim=1)[1] 
                        for j in range(len(pre_labels)):
                            save_name = str(j)
                            if name not in record_dict[save_name].keys():
                                record_dict[save_name][name] = []
                                record_dict[save_name][name].append(record_dict[save_name]['original'][0])
                            record_dict[save_name][name].append(pre_labels[j].item())

                total_images = 0
                mads = []
                images = list(record_dict.keys())
                keys = list(record_dict[images[0]].keys())
                total_images += len(images)
                for image_index in range(len(images)):
                    img = images[image_index]
                    indexs = []
                    img_preds = record_dict[img]
                    for corruption in keys[1:]:
                        flag = 0
                        for i in range(6):
                            if int(img_preds[corruption][i]) != int(img_preds[corruption][0]):
                                index = i
                                flag = 1
                                indexs.append(index)
                                break
                        if flag == 0:
                            indexs.append(6)
                    indexs = np.asarray(indexs)
                    mad = np.std(indexs)
                    mads.append(mad)
                    if type == 'attack':
                        self.attack_mads.append(mad)
                        if mad > self.threshold:
                            ori_label = record_dict[img]['original'][0]
                            categories = list(range(self.args['num_classes']))
                            categories.remove(ori_label)
                            another_class = random.choice(categories)
                            _, probabilities = self.generate_logits(another_class, self.args['num_classes'])
                            defense_result[int(img)] = probabilities
                    elif type == 'test':
                        self.clean_mads.append(mad)
                        if mad > self.threshold:
                            ori_label = record_dict[img]['original'][0]
                            categories = list(range(self.args['num_classes']))
                            categories.remove(ori_label)
                            another_class = random.choice(categories)
                            _, probabilities = self.generate_logits(another_class, self.args['num_classes'])
                            defense_result[int(img)] = probabilities
                    else:
                        self.thresholds.append(mad)


                if type == 'attack' or type == 'test':
                    mads = np.asarray(mads)
                    pred = np.where(mads > self.threshold, 1, 0)

                    if type == 'attack':
                        self.attack_true_detection = self.attack_true_detection + np.sum(pred)
                        self.attack_false_detection = self.attack_false_detection + len(pred) - np.sum(pred)
                    else:
                        self.clean_true_detection = self.clean_true_detection + len(pred) - np.sum(pred)
                        self.clean_false_detection = self.clean_false_detection + np.sum(pred)

        return defense_result

    def generate_logits(self, label, num_classes):
        logits = np.full(num_classes, 0.1)
        logits[label] = 2
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(exp_logits)
        probabilities = torch.tensor(probabilities,dtype=torch.float)
        if self.args['cuda']:
            probabilities = probabilities.cuda()
        return logits, probabilities
