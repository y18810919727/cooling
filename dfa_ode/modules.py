#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn


class MLPCell(nn.Module):
    def __init__(self, k_in, k_state, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k_in+k_state, 2*k_state, bias=bias),
            nn.Tanh(),
            nn.Linear(2*k_state, k_state, bias=bias)
        )

    def forward(self, xt, ht):
        return self.net(torch.cat([xt, ht], dim=-1))


class ODEMergeCell(nn.Module):
    def __init__(self, k_in, k_out, k_state, num_layers, Ly, hidden_size=32, ode_2order=False,
                 name='default', y_type=None, cell='gru'):
        super().__init__()
        self.k_in = k_in
        self.k_out = k_out
        self.k_state = k_state
        self.name = name
        self.ode_2order = ode_2order
        y_type = [0 if x == 'd' else x for x in y_type]
        y_type = [1 if x == 's' else x for x in y_type]
        y_type = [2 if x == 'n' else x for x in y_type]
        self.y_type = y_type
        self.Ly = Ly
        if self.ode_2order:
            assert k_state % 2 == 0, 'ode_2order=True mode requires the k_state is even.'

        if cell == 'gru':   #目前跑的都是gru cell
            self.cell = nn.GRUCell(k_state+k_in, k_state+k_in)
        elif cell == 'mlp':
            self.cell = MLPCell(k_state, k_state, bias=True)

    def forward(self, state, xt, dt):

        yt, ht = state[..., :self.k_out], state[..., self.k_out:]

        if self.ode_2order:
            q = ht.size(-1)/2
            v, s = ht[:, :q], ht[:, -q:]
            dv = self.derivate_correct(v, self.cell(ht, xt), factor=1)
            ds = self.derivate_correct(s, v, factor=1)
            ht_dt = torch.cat([dv, ds], dim=-1)
        else:
            ht_dt = self.derivate_correct(ht, self.cell(xt, ht), factor=1)

        ht = ht + ht_dt * dt
        yt = self.update_y(yt, ht, dt)
        return torch.cat([yt, ht], dim=-1)

    def update_y(self, yt, ht, dt):
        Ly_out = self.Ly(ht)
        y_type = torch.LongTensor(self.y_type).to(yt.device)
        # nyt = torch.zeors_like(Ly_out)
        nyt = torch.clone(yt)
        nyt[..., y_type == 0] = Ly_out[..., y_type == 0]
        nyt[..., y_type == 1] += (Ly_out[..., y_type == 1] - yt[..., y_type == 1]) * dt
        nyt[..., y_type == 2] += Ly_out[..., y_type == 2] * dt
        return nyt

    def derivate_correct(self, ht, ht_dt, factor=1.0):
        return (ht_dt - ht) * factor


class Classification(nn.Module):
    def __init__(self, input_size, hidden, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, input_size, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden, bias=False),
            nn.Sigmoid(),
            nn.Linear(hidden, 1, bias=True),
        )

    def forward(self, ht, xt):
        return self.net(
            torch.cat([ht, xt], dim=-1)
        )
