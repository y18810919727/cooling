#!/usr/bin/python
# -*- coding:utf8 -*-

import torch

from common.modules import MSELoss_nan
from torch import nn
from dfa_ode.odes_stationary import DFA_ODENets

class DFA_MIMO(nn.Module):

    def __init__(self, ode_nums, layers, k_in, k_out, k_h, y_mean, y_std, odes_para,
                 ode_2order=False, transformations=None, state_transformation_predictor=None, Ly_share=False,
                 dropout=0., cell_type='merge'):
        # potential kwargs: meandt, train_scheme, eval_scheme

        super().__init__()
        self.k_in = k_in
        self.k_out = k_out
        self.k_h = k_h
        self.dropout = dropout
        self.criterion = MSELoss_nan()
        self.cell_type = cell_type
        self.expand_input = nn.Sequential(nn.Linear(k_in, k_h), nn.Tanh())
        self.expand_input_out = nn.Sequential(nn.Linear(k_in + k_out, k_h), nn.Tanh())
        self.dfa_odes_forward = DFA_ODENets(ode_nums, layers, k_in, k_out, k_h, y_mean, y_std,#前项预测，输入输入量做预测
                                                  odes_para=odes_para, ode_2order=ode_2order,
                                                  state_transformation_predictor=state_transformation_predictor,
                                                  transformations_rules=transformations, cell_type=cell_type,
                                                  Ly_share=Ly_share)
        self.dfa_odes_posterior = DFA_ODENets(ode_nums, layers, k_in, k_out, k_h, y_mean, y_std,#后验编码，输入输出量做编码
                                                   odes_para=odes_para, ode_2order=ode_2order,
                                                    cell_type=cell_type,
                                                    Ly_share=Ly_share)
    def states_classification(self, states, inputs,t):
        reshape = False
        bs, l = None, None
        if len(states.shape) == 3:
            reshape = True
            bs, l, _ = states.shape
            states = states.reshape(-1, states.shape[-1]).contiguous()
            inputs = inputs.reshape(-1, inputs.shape[-1]).contiguous()
            t = t.reshape(-1, t.shape[-1]).contiguous()
        if self.cell_type == 'merge':
            xt = torch.cat([t,inputs, self.expand_input(inputs)], dim=-1)
        else:
            xt = torch.cat([inputs, self.expand_input(inputs)], dim=-1)

        new_s, new_s_prob, extra_info = self.dfa_odes_forward.state_transform(states, xt,x_in=inputs)
        if reshape:
            new_s_prob = new_s_prob.reshape(bs, l, -1)
            new_s = new_s.reshape(bs, l, -1)
            if 'predicted_stop_cum_time' in extra_info.keys():
                extra_info['predicted_stop_cum_time'] = extra_info['predicted_stop_cum_time'].reshape(bs, l, -1)
                extra_info['real_cum_time'] = extra_info['real_cum_time'].reshape(bs, l, -1)

        return new_s, new_s_prob, extra_info

    def forward_posterior(self, in_x ,inputs, state0=None, dfa_states=None, dt=None, **kwargs):  # potential kwargs: dt
        return self.model_call(
            self.dfa_odes_posterior, self.expand_input_out,in_x, inputs, state0=state0, dfa_states=dfa_states, dt=dt, **kwargs
        )

    def forward_prediction(self,inputs, state0=None, dfa_states=None, dt=None, **kwargs):  # potential kwargs: dt #预测
        return self.model_call(
            self.dfa_odes_forward, self.expand_input,inputs,inputs, state0=state0, dfa_states=dfa_states, dt=dt, **kwargs
        )
    """
    modelcall,给定初始状态，将输入扔进去，更新状态，预测
    """
    def model_call(self, model, expand_cell, in_x,inputs , state0=None, dfa_states=None, dt=None, **kwargs):
        """
        :param model: dfa_odes
        :param expand_cell: The model expanding inputs with size(..., k_in) to high-level feature (..., k_h)
        :param inputs:  (batch_size, seq_len, k_in)
        :param state0: The  initial state of ode nets
        :param dfa_states: Given dfa states in each position
        :param dt:  (batch_size, seq_len, 1)
        :param kwargs:
        :return: outputs (bs, len, k_out), states (bs, len, k_h)
        """

        state = state0
        outputs = []
        states = []
        inputs = expand_cell(inputs) #升到20维  //第一层
        inputs = torch.cat([in_x, inputs], dim=-1)

        if dt is None:
            dt = torch.ones(*inputs.size()[:2], device=inputs.device).unsqueeze(dim=-1) / 10

        for i in range(inputs.size(1)):
            x0 = inputs[:, i, :]
            dt_i = dt[:, i, :]
            in_x_i = in_x[:, i, :]
            dfa_state_i = dfa_states[:, i, :] if dfa_states is not None else None
            output, state = model(state, x0, dt=dt_i, new_s=dfa_state_i,x_in=in_x_i)
            outputs.append(output)
            states.append(state)

        outputs = torch.stack(outputs, dim=1)  # outputs: (batch, seq_len, 2)
        states = torch.stack(states, dim=1)

        return outputs, states

    def select_dfa_states(self, states):
        return self.dfa_odes_forward.select_dfa_states(states)

    def generate_state0(self, inputs, outputs, dfa_states=None, dt=None, last=True):
        """

        :param in_out_seqs: bs, len, k_in+k_out
        :param dt:
        :param last:
        :return:
        """
        #这里的inputs为x
        outputs, states = self.forward_posterior(inputs,torch.cat([inputs, outputs], dim=-1), dfa_states=dfa_states, dt=dt)

        return states[:, -1] if last else states

    def encoding_plus_predict(self, X_tn, dt_tn, history_Y_tn, history_s_tn, history_length, future_s_tn=None):
        """

        :param X_tn: Inputs with shape (bs, la+lb, k_in)    la+lb过去序列的长度和未来序列的长度
        :param dt_tn: Time deltas with shape (bs, la+lb, 1)  时间，也是la+lb的长度
        :param history_Y_tn: shape(bs, la, k_out)      历史的y的长度
        :param history_s_tn: shape(bs, la, 1)          过去的状态标记，手工打的标签
        :param history_length: la
        :param future_s_tn: shape(bs, lb, 1)              未来的(验证集给了
        :return:
        """

        assert history_s_tn.shape[1] == history_length

        history_X_tn, predicted_X_tn = X_tn[:, :history_length], X_tn[:, history_length:] #过去的序列，未来的序列
        history_dt_tn, predicted_dt_tn = dt_tn[:, :history_length], dt_tn[:, history_length:] #过去的时间间隔，未来的时间间隔

        # Y_pred, h_pred = model(X_tn, state0=htrain_pred[:, -1, :], dt=dt_tn)
        Y_pred, h_pred = self.forward_prediction(
            predicted_X_tn,          #x_tn的后一半拿出来
            state0=self.generate_state0(history_X_tn, history_Y_tn, history_s_tn, history_dt_tn),   #初始状态，编码模块，把过去这些东西扔进去做编码
            dfa_states=future_s_tn,        #未来的状态
            dt=predicted_dt_tn)            #未来的时间

        return Y_pred, h_pred              #返回预测结果和预测隐变量的结果
