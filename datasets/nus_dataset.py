# -*- coding: utf-8 -*-
from typing import Any, Callable, Optional, Tuple

import numpy as np
from torch.utils.data import TensorDataset
from torch import Tensor

"""
define text dataset, only support two parties, used for NUS-WIDE
"""


class NUS_dataset(TensorDataset):
    def __init__(
            self,
            Xa, Xb, y,
            backdoor_indices=None,
            trigger=None
    ) -> None:

        super(NUS_dataset, self).__init__(Xa, Xb, y)

        self.backdoor_indices = backdoor_indices  # backdoor indices of dataset
        self.trigger = trigger
        if self.trigger == None:
            self.trigger = 'pixel'

    def __getitem__(self, index: int) -> Tuple[Tuple[Any, Any], Any]:
        X_a, X_b, target = self.tensors[index]

        # add trigger if index is in backdoor indices
        if self.trigger == 'pixel':
            if self.backdoor_indices is not None and index in self.backdoor_indices:
                X_b[-1] = 1

        return (X_a, X_b), target

