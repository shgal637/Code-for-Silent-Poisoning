# -*- coding: utf-8 -*-
from typing import Any, Callable, Optional, Tuple

import cv2
import torch
from torchvision.datasets import VisionDataset

from datasets.common import add_pixel_pattern_backdoor, add_pixel_pattern_backdoor_original
import numpy as np
from PIL import Image
import random
"""
define multiple image dataset, support multiple parties, used for BHI
"""

class MultiImageDataset(VisionDataset):
    def __init__(
            self,
            X, y,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            backdoor_indices=None,
            party_num=2,
            trigger=None,
            trigger_add=False,
            source_indices=None,  # none_target for sr_ba
            adversary=1
    ) -> None:

        super(MultiImageDataset, self).__init__(root="", transform=transform,
                                                target_transform=target_transform)
        self.data: Any = []  # X
        self.targets = []  # Y

        self.data = X
        self.targets = y

        self.backdoor_indices = backdoor_indices  # backdoor indices of dataset
        self.party_num = party_num  # parties number
        self.source_indices = source_indices

        # pattern_mask: torch.Tensor = torch.tensor([[1, -10] * 10] * 20)
        pattern_mask: torch.Tensor = torch.tensor([
            [1., 0., 1.],
            [-10., 1., -10.],
            [-10., -10., 0.],
            [-10., 1., -10.],
            [1., 0., 1.]
        ])
        pattern_mask = pattern_mask.unsqueeze(0)
        self.pattern_mask = pattern_mask.repeat(3, 1, 1)
        x_top = 3
        y_top = 3
        x_bot = x_top + self.pattern_mask.shape[1]
        y_bot = y_top + self.pattern_mask.shape[2]
        self.location = [x_top, x_bot, y_top, y_bot]

        if backdoor_indices is not None and source_indices is not None:
            self.indice_map = dict(zip(backdoor_indices, source_indices))
        else:
            self.indice_map = None

        self.trigger = trigger
        if self.trigger is None:
            self.trigger = 'pixel'
        self.trigger_add = trigger_add
        self.pixel_pattern = None
        self.attacker = adversary

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img_groups, target = self.data[index], self.targets[index]  # img_groups[N_party,50,50,3]
        if len(img_groups) == self.party_num:
            images_list = []
            old_image = 0
            # split images into parties
            for img_id in range(self.party_num):
                img_path = img_groups[img_id]
                image = img_path
                if self.transform is not None:
                    image = self.transform(image)
                # add trigger if index is in backdoor indices, only for the first passive party
                if img_id == self.attacker:
                    old_image = image
                    if self.trigger == 'pixel':
                        if self.indice_map is not None and index in self.indice_map.keys():
                            source_indice = self.indice_map[index]
                            source_img_groups = self.data[source_indice]
                            source_img_path = source_img_groups[1]
                            source_image = source_img_path
                            if self.transform is not None:
                                source_image = self.transform(source_image)
                            image = source_image
                            # old_image = image

                    if self.trigger == 'pixel':
                        if self.backdoor_indices is not None and index in self.backdoor_indices:
                            image = add_pixel_pattern_backdoor(image, pattern_tensor=self.pattern_mask, location=self.location)
                            # image1 = add_pixel_pattern_backdoor_original(image)
                images_list.append(image)
        else:
            img, target = self.data[index], self.targets[index]
            if type(img) is np.str_:
                img = Image.open(img)
            else:
                img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)

            images_list = self.split_func(img)

            old_image = images_list[self.attacker]
            # sample replace
            if self.trigger == 'pixel':
                if self.indice_map is not None and index in self.indice_map.keys():
                    source_indice = self.indice_map[index]
                    source_img = self.data[source_indice]
                    if self.transform is not None:
                        source_img = self.transform(source_img)
                    source_img_list = self.split_func(source_img)
                    source_image = source_img_list[self.attacker]
                    images_list[self.attacker] = source_image

            if self.trigger == 'pixel':
                if self.backdoor_indices is not None and index in self.backdoor_indices:
                    if self.trigger_add:
                        if self.pixel_pattern is None:
                            self.pixel_pattern = torch.randn(images_list[self.attacker].shape)
                            self.pixel_pattern = self.pixel_pattern / 2
                        images_list[self.attacker] = images_list[self.attacker] + self.pixel_pattern
                    else:
                        images_list[self.attacker] = add_pixel_pattern_backdoor(images_list[self.attacker], self.pattern_mask, self.location)

        images = torch.stack(tuple(image for image in images_list), 0)  # 3,3,50,50
        return images, target, old_image

    def __len__(self) -> int:
        return len(self.data)

    def split_func(self, img):
        if self.party_num == 4:
            # split image into halves vertically for parties
            img_a, img_b1, img_b2, img_b3 = img[:, :16, :16], img[:, :16, 16:], img[:, 16:, :16], img[:, 16:, 16:]
            images_list = [img_a, img_b1, img_b2, img_b3]
        elif self.party_num == 8:
            img_a, img_b1, img_b2, img_b3, img_b4, img_b5, img_b6, img_b7 = img[:, :16, :8], img[:, :16, 8:16], \
                img[:, :16, 16:24], img[:, :16, 24:], img[:, 16:, :8], img[:, 16:, 8:16], img[:, 16:, 16:24], img[:,16:, 24:]
            images_list = [img_a, img_b1, img_b2, img_b3, img_b4, img_b5, img_b6, img_b7]
        elif self.party_num == 6:
            img_a, img_b1, img_b2, img_b3, img_b4, img_b5 = img[:, :16, :11], img[:, :16, 11:22], img[:, :16, 22:], \
                img[:, 16:, :11], img[:, 16:, 11:22], img[:, 16:, 22:]
            images_list = [img_a, img_b1, img_b2, img_b3, img_b4, img_b5]
        else:
            images_list = []
            length = 32 // self.party_num + 1
            for i in range(self.party_num):
                end = min((i + 1) * length, 32)
                images_list.append(img[:, :, i * length:end])
        return images_list
