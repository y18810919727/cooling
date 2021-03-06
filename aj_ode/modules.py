#!/usr/bin/python
# -*- coding:utf8 -*-
import torchdiffeq
import torch
from torch import nn



def my_odeint(x, h0, cell, t,scheme=None,adjoint=None, **kwargs):
    odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
    my_Func = ODE_Func(x=x, cell=cell)
    ht = odeint(my_Func,h0,t,rtol=0.001,
                   atol=0.001,method=scheme)
    return ht[1]

class ODE_Func(nn.Module):
    def __init__(self,x,cell):
        super(ODE_Func, self).__init__()
        self.cell = cell
        self.x = x
    def forward(self, t, h):
        return self.derivate_correct(h, self.cell(self.x,h),1)

    def derivate_correct(self, ht, ht_dt, factor=1.0):
        return (ht_dt - ht) * factor


class ODE_Rnn_Func(nn.Module):
    def __init__(self,k_h):
        super(ODE_Rnn_Func, self).__init__()
        self.func = nn.Sequential(
            nn.Linear(k_h, 2 * k_h),
            nn.Tanh(),
            nn.Linear(2 * k_h, k_h)
        )
    def forward(self, t, h):
        return self.func(h)

class ODE_RNN(nn.Module):
    def __init__(self, k_in, k_out, k_h, k_expand_in,k_t,num_layers, Ly,scheme, hidden_size=32, ode_2order=False,
                 name='default', y_type=None, cell='gru'):
        super().__init__()
        self.k_in = k_in
        self.k_out = k_out
        self.k_h = k_h
        self.name = name
        self.ode_2order = ode_2order
        if cell == 'gru':
            self.cell = nn.GRUCell(k_expand_in+k_in,  k_h)
        y_type = [0 if x == 'd' else x for x in y_type]
        y_type = [0 if x == 's' else x for x in y_type]
        y_type = [0 if x == 'n' else x for x in y_type]
        self.y_type = y_type
        self.Ly = Ly
        self.scheme = scheme
        self.func = ODE_Rnn_Func(k_h)


    def forward(self, state, xt,dt,ti):

        yt, ht = state[..., :self.k_out], state[..., self.k_out:]
        t = torch.tensor([0.0, 0.1]).cuda()
        ht_ = torchdiffeq.odeint_adjoint(self.func, ht, t, rtol=0.001,
                                        atol=0.001, method=self.scheme)[1]
        ht = self.cell(xt,ht_)
        yt = self.update_y(yt, ht)
        return torch.cat([yt, ht], dim=-1)

    def update_y(self, yt, ht):
        Ly_out = self.Ly(ht)
        y_type = torch.LongTensor(self.y_type).to(yt.device)
        nyt = torch.clone(yt)
        nyt[..., y_type == 0] = Ly_out[..., y_type == 0]
        return nyt


class ODEOneCell(nn.Module):
    def __init__(self, k_in, k_out, k_h, k_expand_in, k_t, num_layers, Ly,scheme,hidden_size=32, ode_2order=False,
                 name='default', y_type=None, cell='gru'):
        super().__init__()
        self.k_in = k_in
        self.k_out = k_out
        self.k_h = k_h
        self.name = name
        self.ode_2order = ode_2order
        y_type = [0 if x == 'd' else x for x in y_type]
        y_type = [1 if x == 's' else x for x in y_type]
        y_type = [2 if x == 'n' else x for x in y_type]
        self.y_type = y_type
        self.Ly = Ly
        self.scheme = scheme
        self.cell = nn.GRUCell(k_expand_in+k_in, k_h)
    def forward(self, state, xt, dt):
        yt, ht = state[..., :self.k_out], state[..., self.k_out:]
        t = torch.tensor([0.0, 0.1]).cuda()
        ht = my_odeint(xt, ht, self.cell, t,scheme=self.scheme,adjoint=True)
        yt = self.update_y(yt, ht, dt)
        return torch.cat([yt, ht], dim=-1)
    def update_y(self, yt, ht, dt):
        Ly_out = self.Ly(ht)
        y_type = torch.LongTensor(self.y_type).to(yt.device)
        nyt = torch.clone(yt)
        nyt[..., y_type == 0] = Ly_out[..., y_type == 0]
        nyt[..., y_type == 1] += (Ly_out[..., y_type == 1] - yt[..., y_type == 1]) * dt
        nyt[..., y_type == 2] += Ly_out[..., y_type == 2] * dt
        return nyt

    def derivate_correct(self, ht, ht_dt, factor=1.0):
        return (ht_dt - ht) * factor



class ODEMergeCell(nn.Module):
    def __init__(self, k_in, k_out, k_h, k_expand_in,k_t,num_layers, Ly,scheme, hidden_size=32, ode_2order=False,
                 name='default', y_type=None, cell='gru'):
        super().__init__()
        self.k_in = k_in
        self.k_out = k_out
        self.k_h = k_h
        self.name = name
        self.ode_2order = ode_2order
        y_type = [0 if x == 'd' else x for x in y_type]
        y_type = [1 if x == 's' else x for x in y_type]
        y_type = [2 if x == 'n' else x for x in y_type]
        self.y_type = y_type
        self.Ly = Ly
        self.scheme=scheme
        self.cell = nn.GRUCell(k_expand_in+k_in+k_t, k_h)
    def forward(self, state, xt, dt):
        yt, ht = state[..., :self.k_out], state[..., self.k_out:]

        t = torch.tensor([0.0,0.1]).cuda()
        ht=my_odeint(xt, ht, self.cell, t,scheme=self.scheme, adjoint=False)
        yt = self.update_y(yt, ht, dt)
        return torch.cat([yt, ht], dim=-1)

    def update_y(self, yt, ht, dt):
        Ly_out = self.Ly(ht)
        y_type = torch.LongTensor(self.y_type).to(yt.device)
        nyt = torch.clone(yt)
        nyt[..., y_type == 0] = Ly_out[..., y_type == 0]
        nyt[..., y_type == 1] += (Ly_out[..., y_type == 1] - yt[..., y_type == 1]) * dt
        nyt[..., y_type == 2] += Ly_out[..., y_type == 2] * dt
        return nyt

class Predictor(nn.Module):
    def __init__(self, input_size, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden, bias=False),
            nn.Sigmoid(),
            nn.Linear(hidden, 1, bias=True),
        )

    def forward(self, x_in ,ht):
        return self.net(
            torch.cat([x_in,ht], dim=-1)
        )
