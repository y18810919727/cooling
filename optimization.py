#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json
from collections import defaultdict
import pandas as pd
from util import get_Dataset, visualize_prediction, array_operate_with_nan, t2np, SimpleLogger,visualize_prediction_power,visualize_prediction_power

import torch
import argparse
import pickle
from matplotlib import pyplot as plt
parser = argparse.ArgumentParser(description='Models for Continuous Stirred Tank dataset')
parser.add_argument("--low", type=float, default=12, help='The temperature activating the work of cooling')
parser.add_argument("--high", type=float, default=20, help='The temperature stopping the cooling system')
parser.add_argument("--model", type=str, default='best_dev_model.pt')
parser.add_argument("--bptt", type=int, default=800, help="bptt")
parser.add_argument("--data", type=str, default='test.csv')
parser.add_argument("--save_dir", type=str, default='None')
parser.add_argument("--datasets", type=list, default=['train_P-1.7k','train_P-3.8k','train_P-6.3k'], help="datasets")
parser.add_argument("--datasets2", type=list, default=['heat load-1.7k','heat load-3.8k','heat load-6.3k'], help="datasets")


if __name__ == '__main__':

    paras = parser.parse_args()
    paras.save_dir = os.path.join('results', paras.save_dir)
    save_dir_img = os.path.join(paras.save_dir,'optimization')
    if not os.path.exists(save_dir_img):
        os.mkdir(save_dir_img)
    log_file_all = os.path.join(save_dir_img, 'all.txt' )
    logging_all = SimpleLogger(log_file_all)  # log to file
    filename = "./results/optimization_test/data.pkl"
    file = open(filename, "rb")
    data = pickle.load(file)
    X_mean = data['X_mean']
    X_std = data['X_std']
    Y_mean = data['Y_mean']
    Y_std = data['Y_std']

    sum_powers_all = []
    cops_all = []
    pue_all = []
    for idx,everdata in enumerate(paras.datasets):
        sum_powers = []
        cops = []
        pues = []
        Min = 12
        Max = 18.5
        step = 0.5
        Xtest, Ytest, ttest, stest = [df.to_numpy(dtype=np.float32) for df in
                                      get_Dataset('./data/' + everdata + '.csv')]
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

        save_dir = os.path.join(save_dir_img,everdata)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        log_file = os.path.join(save_dir, '%s-%.2f-log.txt' % (everdata,paras.low))
        logging = SimpleLogger(log_file)  # log to file
        logging('dataset: %s, Min: %.2f, Max: %.3f, ' % (everdata,Min, Max))
        for low in np.arange(Min, Max, step):
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            model = torch.load(os.path.join(paras.save_dir, paras.model),
                               map_location=device)  # load模型
            model.aj_odes_forward.transforms = defaultdict(list)
            model.aj_odes_forward.add_transform(1, 2, [[0, 'geq', paras.high]])
            model.aj_odes_forward.add_transform(4, 1, [[0, 'leq',low]])
            Ytest_pred, test_state_pred = model.encoding_plus_predict(  # 模型预测
                Xtest_tn, dttest_tn, Ytest_tn[:, :paras.bptt], stest_tn[:, :paras.bptt], paras.bptt, None)
            test_aj_state_pred_array = model.select_aj_states(
                test_state_pred[0]).int().detach().cpu().numpy()  # 把状态也拿出来

            visualize_prediction_power(  # 可视化画图专用
                Ytest[paras.bptt:] * Y_std + Y_mean, t2np(Ytest_pred) * Y_std + Y_mean, test_aj_state_pred_array,
                Xtest[paras.bptt:, 0] * X_std[0] + X_mean[0],low
                ,save_dir,
                seg_length=1000, dir_name='vis-%.1f' % (low))

            predicted_power = (t2np(Ytest_pred) * Y_std + Y_mean)[..., -1].reshape(-1)  # 算power
            predicted_power[predicted_power < 0] = 0
            sum_power = (
                predicted_power[:7200] *
                    dttest_tn[:, paras.bptt:paras.bptt+7200].reshape(-1).detach().cpu().numpy()/(3.6*100000)
            ).sum()
            logging('Low: %.2f,  Predicted Power: %.3f' % (low, float(sum_power)))
            sum_powers.append(sum_power)

            #cop
            pcooling_power = (t2np(Ytest_pred) * Y_std + Y_mean)[..., 1].reshape(-1)
            pcooling_power[pcooling_power <0 ] = 0
            sum_pcooling_power =  (
                pcooling_power[:7200] *
                    dttest_tn[:, paras.bptt:paras.bptt+7200].reshape(-1).detach().cpu().numpy()/(3.6*100000)
            ).sum()
            logging('Low: %.2f, sum_pcooling_powerr: %.3f' % (low, float(sum_pcooling_power)))
            cop = round((sum_pcooling_power/sum_power), 2) #百分比
            logging('Low: %.2f, cop: %.3f' % (low, float(cop)))
            cops.append(cop)

            #pue
            pserver = (Xtest*X_std+X_mean)[...,0].reshape(-1)
            pserver_power = (pserver[paras.bptt:paras.bptt+7200] *
                    dttest_tn[:, paras.bptt:paras.bptt+7200].reshape(-1).detach().cpu().numpy()/(3.6*100000)
            ).sum()
            pue = round((1+(sum_power/pserver_power)), 2) #百分比
            logging('Low: %.2f, pue: %.3f' % (low, float(pue)))
            pues.append(pue)

        min_sum_power = min(sum_powers)
        ratio_power_decline = ((sum_powers[0]-min_sum_power)/sum_powers[0])*100
        logging_all('%s ratio_power_decline: %.2f' % (everdata,ratio_power_decline))
        cops_all.append(cops)
        pue_all.append(pues)
        sum_powers_all.append(sum_powers)


    plt.title('all' + " - " + 'Power')
    plt.figure(figsize=(8, 5))
    color = ['blue', 'red', 'green', 'purple']
    line = ['-', '--', ':']
    for i, sum_power in enumerate(sum_powers_all):
        # plt.title('Compare the power consumption at different lower temperature limits', fontsize=20)
        plt.plot(np.arange(Min, Max, step), sum_power, label=paras.datasets2[i], linestyle=line[i], color=color[i])
        min_idx = sum_power.index(min(sum_power))
        plt.plot(Min + (min_idx * step), sum_power[min_idx], marker='x', color=color[i], markersize='20')
        plt.legend(fontsize=18)
    plt.text(13.98, 4.35, '    Optimal\nlower threshold', fontsize=16, color='black')
    plt.arrow(15.75,4.4,-1.1, -0.55,shape='full',head_width=0.1,head_length=0.2,length_includes_head=True,ec ='black')
    plt.arrow(15.75, 4.4, -0.28, 0.8, shape='full', head_width=0.1, head_length=0.2, length_includes_head=True, ec='black')
    plt.arrow(15.75,4.4,0.26, 1.3, shape='full',head_width=0.1,head_length=0.2,length_includes_head=True,ec ='black')

    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.xlabel('$Ti_{\min}(℃)$', fontsize=20)
    plt.ylabel('Energy consumption(kwh)', fontsize=20)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir_img, 'power%.1f-%.1f-%.1f.png' % (Min, Max, step)))
    plt.close()


    # cop
    plt.figure(figsize=(8, 5))
    color = ['blue', 'red', 'green', 'purple']
    line = ['-', '--', ':']
    for i, cops in enumerate(cops_all):
        # plt.title('Compare the power consumption at different lower temperature limits', fontsize=20)
        plt.plot(np.arange(Min, Max, step), cops, label=paras.datasets2[i], linestyle=line[i], color=color[i])
        #plt.legend(fontsize=18)
        plt.legend(fontsize=18)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.xlabel('$Ti_{\min}(℃)$', fontsize=20)
    plt.ylabel('COP', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_img, 'cop%.1f-%.1f-%.1f.png' % (Min, Max, step)))
    plt.close()

    # pue
    plt.figure(figsize=(8, 5))
    color = ['blue', 'red', 'green', 'purple']
    line = ['-', '--', ':']
    for i, pues in enumerate(pue_all):
        # plt.title('Compare the power consumption at different lower temperature limits', fontsize=20)
        plt.plot(np.arange(Min, Max, step), pues, label=paras.datasets2[i], linestyle=line[i], color=color[i])
        min_idx = pues.index(min(pues))
        plt.plot(Min + (min_idx * step), pues[min_idx], marker='x', color=color[i], markersize='20')
        plt.legend(fontsize=18)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.xlabel('$Ti_{\min}(℃)$', fontsize=20)
    plt.ylabel('PUE', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_img, 'pue%.1f-%.1f-%.1f.png' % (Min, Max, step)))
    plt.close()

