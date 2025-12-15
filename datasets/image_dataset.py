# -*- coding: utf-8 -*-
import random
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
import torch
from datasets.common import add_pixel_pattern_backdoor

"""
define image dataset, only support two parties, used for CIFAR and CINIC
"""


class ImageDataset(VisionDataset):

    def __init__(
            self,
            X, y,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            backdoor_indices=None,   # target for ar_ba
            half=None,
            trigger=None,
            trigger_add=False,
            source_indices=None  # none_target for sr_ba
    ) -> None:

        super(ImageDataset, self).__init__(root="", transform=transform,
                                           target_transform=target_transform)
        self.data: Any = []  # X
        self.targets = []  # Y

        self.data = X
        self.targets = y

        self.backdoor_indices = backdoor_indices  # backdoor indices of dataset
        self.source_indices = source_indices

        if backdoor_indices is not None and source_indices is not None:
            self.indice_map = dict(zip(backdoor_indices, source_indices))
        else:
            self.indice_map = None

        self.half = half  # vertical halves to split
        self.trigger = trigger
        if self.trigger is None:
            self.trigger = 'pixel'
        self.trigger_add = trigger_add
        # self.pixel_pattern = torch.zeros(3, 32, 16)
        self.pixel_pattern = torch.randn(3,32,16)
        self.pixel_pattern = self.pixel_pattern / 2

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

    def __getitem__(self, index: int) -> Tuple[Tuple[Any, Any], Any, Any]:
        img, target = self.data[index], self.targets[index]
        if type(img) is np.str_:
            img = Image.open(img)
        else:
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        # split image into halves vertically for parties
        img_a, img_b = img[:, :, :self.half], img[:, :, self.half:]

        if self.target_transform is not None:
            target = self.target_transform(target)

        old_imgb = img_b

        if self.trigger == 'pixel' or self.trigger_add:
            if self.indice_map is not None and index in self.indice_map.keys():
                source_indice = self.indice_map[index]
                source_img = self.data[source_indice]
                if type(img) is np.str_:
                    source_img = Image.open(source_img)
                else:
                    source_img = Image.fromarray(source_img)
                if self.transform is not None:
                    source_img = self.transform(source_img)
                img_b = source_img[:, :, self.half:]

        if self.trigger == 'pixel':
            if self.backdoor_indices is not None and index in self.backdoor_indices:
                if self.trigger_add:
                    img_b = img_b + self.pixel_pattern
                else:
                    img_b = add_pixel_pattern_backdoor(img_b, self.pattern_mask, self.location)

        return (img_a, img_b), target, old_imgb

    def __len__(self) -> int:
        return len(self.data)
