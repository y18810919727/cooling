#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json
from collections import defaultdict
import pandas as pd
from util import get_Dataset, visualize_prediction, array_operate_with_nan, t2np, SimpleLogger

import torch
import argparse

parser = argparse.ArgumentParser(description='Models for Continuous Stirred Tank dataset')

# model definition
methods = """
set up model
- model:
    GRU (compensated GRU to avoid linear increase of state; has standard GRU as special case for Euler scheme and equidistant data)
    GRUinc (incremental GRU, for baseline only)
- time_aware:
    no: ignore uneven spacing: for GRU use original GRU implementation; ignore 'scheme' variable
    input: use normalized next interval size as extra input feature
    variable: time-aware implementation
"""


parser.add_argument("--low", type=float, default=12, help='The temperature activating the work of cooling')
parser.add_argument("--high", type=float, default=20, help='The temperature stopping the cooling system')
parser.add_argument("--model", type=str, default='test')
parser.add_argument("--interpol", type=str, default='constant', choices=['constant', 'linear'])
parser.add_argument("--bptt", type=int, default=400, help="bptt")
parser.add_argument("--data", type=str, default='test.csv')
parser.add_argument("--save_dir", type=str, default='None')




def main(paras):
    save_dir = paras.save_dir

    log_file = os.path.join(save_dir, '%.2f-%.2f-log.txt' %(paras.low, paras.high))
    logging = SimpleLogger(log_file) #log to file

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(os.path.join('optimization', 'model', paras.model + '.pt'), map_location=device)
    model.dfa_odes_forward.transforms = defaultdict(list)
    model.dfa_odes_forward.add_transform(1, 2, [[0, 'geq', paras.high]])
    model.dfa_odes_forward.add_transform(4, 5, [[0, 'leq', paras.low]])
    data_path = os.path.join('optimization', 'data', paras.data)

    Xtest, Ytest, ttest, stest = [df.to_numpy(dtype=np.float32) for df in get_Dataset(data_path)]

    dttest = np.append(ttest[1:] - ttest[:-1], ttest[1]-ttest[0]).reshape(-1, 1)

    Y_mean, Y_std = array_operate_with_nan(Ytest, np.mean), array_operate_with_nan(Ytest, np.std)
    Ytest = (Ytest - Y_mean) / Y_std
    X_mean, X_std = array_operate_with_nan(Xtest, np.mean), array_operate_with_nan(Xtest, np.std)
    Xtest = (Xtest - X_mean) / X_std

    Xtest_tn = torch.tensor(Xtest, dtype=torch.float).unsqueeze(0)
    Ytest_tn = torch.tensor(Ytest, dtype=torch.float).unsqueeze(0)
    dttest_tn = torch.tensor(dttest, dtype=torch.float).unsqueeze(0)
    stest_tn = torch.tensor(stest, dtype=torch.int).unsqueeze(0)  # (1, Ntrain, 1)

    if torch.cuda.is_available():
        Xtest_tn = Xtest_tn.cuda()
        Ytest_tn = Ytest_tn.cuda()
        dttest_tn = dttest_tn.cuda()
        stest_tn = stest_tn.cuda()

    Ytest_pred, test_state_pred = model.encoding_plus_predict(
        Xtest_tn,  dttest_tn,  Ytest_tn[:, :paras.bptt], stest_tn[:, :paras.bptt], paras.bptt, None)

    test_dfa_state_pred_array = model.select_dfa_states(test_state_pred[0]).int().detach().cpu().numpy()

    visualize_prediction(
        Ytest[paras.bptt:] * Y_std + Y_mean, t2np(Ytest_pred) * Y_std + Y_mean, test_dfa_state_pred_array, save_dir,
        seg_length=2000, dir_name='vis-%.1f-%.1f' % (paras.low, paras.high))

    predicted_power = (t2np(Ytest_pred) * Y_std + Y_mean)[..., -1].reshape(-1)
    predicted_power[predicted_power < 0] = 0
    sum_power = (
            predicted_power *
            dttest_tn[:, paras.bptt:].reshape(-1).detach().cpu().numpy()
    ).sum()
    logging('Low: %.2f, High: %.3f, Predicted Power: %.3f' %(paras.low, paras.high, float(sum_power)))
    return sum_power


if __name__ == '__main__':

    paras = parser.parse_args()
    paras.save_dir = os.path.join('optimization', 'results', paras.model)

    sum_powers = []
    Min = 13
    Max = 19
    step = 0.5

    for low in np.arange(Min, Max, step):
        paras.low = low
        log_file = os.path.join(paras.save_dir, '%.2f-%.2f-log.txt' % (paras.low, paras.high))
        if os.path.exists(log_file):
            with open(log_file) as f:
                s = f.read()
                pos = s.rfind(': ')
                sum_powers.append(float(s[pos+2:]))

        else:
            sum_powers.append(main(paras))
    from matplotlib import pyplot as plt
    plt.plot(np.arange(Min, Max, step), sum_powers)
    plt.xlabel('The set point of lower bound temperature in cooling system')
    plt.ylabel('Power Consumption')
    plt.savefig(os.path.join(paras.save_dir, '%.1f-%.1f-%.1f.png' % (Min, Max, step)))

