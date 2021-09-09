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
import pickle
from matplotlib import pyplot as plt  # 画图
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
parser.add_argument("--model", type=str, default='best_dev_model-Copy1')
parser.add_argument("--interpol", type=str, default='constant', choices=['constant', 'linear'])
parser.add_argument("--bptt", type=int, default=400, help="bptt")
parser.add_argument("--data", type=str, default='test.csv')
parser.add_argument("--save_dir", type=str, default='None')
#数据集
parser.add_argument("--datasets", type=list, default=['Data_train_1_7_1','Data_train_1_8k','Data_train_3_8k','Data_train_4_2k','Data_validate'], help="datasets")



if __name__ == '__main__':

    paras = parser.parse_args()
    paras.save_dir = os.path.join('optimization', 'results', paras.model)
    if not os.path.exists(paras.save_dir):
        os.mkdir(paras.save_dir)
    # 导入入归一化值
    filename = "./optimization/mean_std.pkl"
    file = open(filename, "rb")
    data = pickle.load(file)
    X_mean = data['X_mean']
    X_std = data['X_std']
    Y_mean = data['Y_mean']
    Y_std = data['Y_std']


    for everdata in paras.datasets:

        sum_powers = []  # 遍历从min到max
        Min = 12
        Max = 19
        step = 0.5
        Xtest, Ytest, ttest, stest = [df.to_numpy(dtype=np.float32) for df in
                                      get_Dataset('./mydata/Back_' + everdata + '.csv')]
        dttest = np.append(ttest[1:] - ttest[:-1], ttest[1] - ttest[0]).reshape(-1, 1)
        # Y_mean, Y_std = array_operate_with_nan(Ytest, np.mean), array_operate_with_nan(Ytest, np.std)  # 归一化
        # X_mean, X_std = array_operate_with_nan(Xtest, np.mean), array_operate_with_nan(Xtest, np.std)

        Ytest = (Ytest - Y_mean) / Y_std
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

        save_dir = os.path.join(paras.save_dir,everdata)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        log_file = os.path.join(save_dir, '%s-%.2f-%.2f-log.txt' % (everdata,paras.low, paras.high))
        logging = SimpleLogger(log_file)  # log to file
        logging('dataset: %s, Min: %.2f, Max: %.3f, ' % (everdata,Min, Max))
        for low in np.arange(Min, Max, step):
            #paras.low = low
            #log_file = os.path.join(paras.save_dir, ' %.2f-%.2f-log.txt' % (paras.low, paras.high))

            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            model = torch.load(os.path.join('optimization', 'model', paras.model + '.pt'),
                               map_location=device)  # load模型
            model.dfa_odes_forward.transforms = defaultdict(list)
            model.dfa_odes_forward.add_transform(1, 2, [[0, 'geq', paras.high]])
            model.dfa_odes_forward.add_transform(4, 1, [[0, 'leq',low]])
            Ytest_pred, test_state_pred = model.encoding_plus_predict(  # 模型预测
                Xtest_tn, dttest_tn, Ytest_tn[:, :paras.bptt], stest_tn[:, :paras.bptt], paras.bptt, None)
            test_dfa_state_pred_array = model.select_dfa_states(
                test_state_pred[0]).int().detach().cpu().numpy()  # 把状态也拿出来

            visualize_prediction(  # 可视化
                Ytest[paras.bptt:] * Y_std + Y_mean, t2np(Ytest_pred) * Y_std + Y_mean, test_dfa_state_pred_array,
                Xtest[paras.bptt:, 0] * X_std[0] + X_mean[0], save_dir,
                seg_length=2000, dir_name='vis-%.1f-%.1f' % (low, paras.high))

            predicted_power = (t2np(Ytest_pred) * Y_std + Y_mean)[..., -1].reshape(-1)  # 算power
            predicted_power[predicted_power < 0] = 0
            sum_power = (
                    predicted_power *
                    dttest_tn[:, paras.bptt:].reshape(-1).detach().cpu().numpy()
            ).sum()
            logging('Low: %.2f, High: %.3f, Predicted Power: %.3f' % (low, paras.high, float(sum_power)))

            sum_powers.append(sum_power)


        # if os.path.exists(log_file):
        #     with open(log_file) as f:
        #         s = f.read()
        #         pos = s.rfind(': ')
        #         #sum_powers.append(main(paras))
        #         sum_powers.append(float(s[pos+2:]))
        #
        # else:


        plt.plot(np.arange(Min, Max, step), sum_powers)
        plt.xlabel('The set point of lower bound temperature in cooling system')
        plt.ylabel('Power Consumption')
        plt.savefig(os.path.join(save_dir, '%.1f-%.1f-%.1f.png' % (Min, Max, step)))

