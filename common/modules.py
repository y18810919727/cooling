#!/usr/bin/python
# -*- coding:utf8 -*-
import torch
from torch import Tensor


class MSELoss_nan(torch.nn.MSELoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        indices = torch.logical_and(~torch.isnan(input), ~torch.isnan(target))
        return super().forward(input[indices], target[indices])

