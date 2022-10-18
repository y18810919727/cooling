#!/usr/bin/python
# -*- coding:utf8 -*-
import torchdiffeq
import torch
from torch import nn


class ODE_Func(nn.Module):
    def __init__(self,x,cell):
        super(ODE_Func, self).__init__()
        self.cell = cell
        self.x = x
    def forward(self, t, h):
        return self.derivate_correct(h, self.cell(self.x,h),1) #(4000,20)

    def derivate_correct(self, ht, ht_dt, factor=1.0):
        return (ht_dt - ht) * factor

class VectorField(nn.Module):   #走
    def __init__(self, dX_dt, func,indices,ode_i,flag):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func: As cdeint.
        """
        super(VectorField, self).__init__()
        if not isinstance(func, nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt    #这里是dX_dt
        self.func = func      #这里是对h做处理的函数
        self.indices = indices
        self.ode_i = ode_i
        self.flag = flag
        self.control_gradient = None
    def forward(self, t, h):  #
        # control_gradient is of shape (..., input_channels)
        if(self.flag == 0):
            control_gradient = self.dX_dt(t)[self.indices]  # (4000,2)
            self.control_gradient = control_gradient
            self.flag = 1
        else:
            control_gradient = self.control_gradient
        vector_field = self.func(h)     #(4000,20,2)
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        out = (vector_field @ control_gradient.unsqueeze(-1)).squeeze(-1)#矩阵相乘 (32,8)
        return out


class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, input_channels * hidden_channels*2)
        self.linear2 = torch.nn.Linear(input_channels * hidden_channels*2, input_channels * hidden_channels)

    def forward(self, z):
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimensions, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        return z

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








class CDE(nn.Module):
    def __init__(self, k_in, k_out, k_h, k_expand_in,k_t,num_layers, Ly,scheme, hidden_size=32, ode_2order=False,
                 name='default', y_type=None, cell='gru'):
        super().__init__()
        self.k_in = k_in
        self.k_out = k_out
        self.k_h = k_h
        self.name = name
        self.ode_2order = ode_2order
        # if cell == 'gru':
        #     self.cell = nn.GRUCell(k_expand_in+k_in,  k_h)
        y_type = [0 if x == 'd' else x for x in y_type]
        y_type = [0 if x == 's' else x for x in y_type]
        y_type = [0 if x == 'n' else x for x in y_type]
        self.y_type = y_type
        self.Ly = Ly
        self.scheme = scheme
        self.func = CDEFunc(self.k_in,self.k_h)
        self.cell = nn.GRUCell(k_expand_in + k_in, k_h)
    def forward(self, state, xt,dt,t,dx_dt,indices,ode_i):

        yt, ht = state[..., :self.k_out], state[..., self.k_out:]
        t = torch.tensor([t,t+0.1]).cuda()
        ht = self.cde_odeint(dx_dt, ht, self.func, t, rtol=0.01,
                           atol=0.01,adjoint=True, indices=indices, ode_i=ode_i)
        ht = self.cell(xt, ht)
        yt = self.update_y(yt, ht)

        return torch.cat([yt, ht], dim=-1)
    def cde_odeint(self,dxdt, h0, func, t,rtol=None, atol=None, adjoint=None, indices=None, ode_i=None):
        odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
        my_Func = VectorField(dX_dt=dxdt, func=func, indices=indices, ode_i=ode_i,flag=0)
        #my_Func = self.func
        #z0 = self.initial(spline.evaluate(t[0]))
        ht = odeint(my_Func, y0=h0, t=t, rtol=0.01,
                    atol=0.01, method=self.scheme)[1]  # 这里的func应该包括上dx/dt
        return ht

    def update_y(self, yt, ht):
        Ly_out = self.Ly(ht)
        y_type = torch.LongTensor(self.y_type).to(yt.device)
        nyt = torch.clone(yt)
        nyt[..., y_type == 0] = Ly_out[..., y_type == 0]
        return nyt


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
        ht = self.my_odeint(xt, ht, self.cell, t,scheme=self.scheme,adjoint=True)
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
    def my_odeint(self,x, h0, cell, t, scheme=None, adjoint=None):
        odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
        my_Func = ODE_Func(x=x, cell=cell)
        ht = odeint(my_Func, h0, t, rtol=0.01,
                    atol=0.01, method=scheme)
        return ht[1]

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
    def forward(self, state, xt, dt,t):
        yt, ht = state[..., :self.k_out], state[..., self.k_out:]

        t = torch.tensor([t,t+0.1]).cuda()
        ht=self.my_odeint(xt, ht, self.cell, t,scheme=self.scheme, adjoint=False)
        yt = self.update_y(yt, ht, dt)
        return torch.cat([yt, ht], dim=-1)

    def my_odeint(self,x, h0, cell, t, scheme=None, adjoint=None):
        odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
        my_Func = ODE_Func(x=x, cell=cell)
        ht = odeint(my_Func, h0, t, rtol=0.01,
                    atol=0.01, method=scheme)
        return ht[1]
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
