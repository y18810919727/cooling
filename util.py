import numpy as np
import os
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from matplotlib.ticker import ScalarFormatter

import time

class SimpleLogger(object):
    def __init__(self, f, header='#logger output'):
        dir = os.path.dirname(f)
        #print('test dir', dir, 'from', f)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(f, 'w',encoding='utf-8') as fID:
            fID.write('%s\n'%header)
        self.f = f

    def __call__(self, *args):
        #standard output
        print(*args)
        #log to file
        try:
            with open(self.f, 'a',encoding='utf-8') as fID:
                fID.write(' '.join(str(a) for a in args)+'\n')
        except:
            print('Warning: could not log to', self.f)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def draw_table(file,integral,error,length,base_dir,dir_name='visualizations'):

    if not os.path.exists(os.path.join(base_dir, dir_name)):
        os.mkdir(os.path.join(base_dir, dir_name))
    plt.figure(figsize=(20, 4))
    vals = integral
    col = []
    for i in range(0,len(integral[0])):
        col.append(f'{length * i} - {length * (i + 1)}')
    row = ["truth", "prediction"]
    plt.subplot(2, 1, 1)
    plt.title(file+" - "+'integral (ws)')
    tab = plt.table(cellText=vals,
                    colLabels=col,
                    rowLabels=row,
                    loc='center',
                    cellLoc='center',
                    rowLoc='center')
    tab.scale(1, 2)
    plt.axis('off')

    vals = error
    col = ["MAE","MAPE(%)"]
    row = ["mean", "std"]
    plt.subplot(2, 1, 2)
    plt.title(file + " - " + 'error')
    tab = plt.table(cellText=vals,
                    colLabels=col,
                    rowLabels=row,
                    loc='center',
                    cellLoc='center',
                    rowLoc='center')
    tab.scale(1, 2)
    plt.axis('off')

    plt.savefig(os.path.join(
        base_dir, dir_name, 'compare.png'
    ))
    plt.close()


def draw_table_all(file,error_all,base_dir):

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    plt.figure(figsize=(20, 4))
    vals = error_all
    col = file
    row = ["mae", "mape"]
    plt.title('error_all')
    tab = plt.table(cellText=vals,
                    colLabels=col,
                    rowLabels=row,
                    loc='center',
                    cellLoc='center',
                    rowLoc='center')
    tab.scale(1, 2)
    plt.axis('off')

    plt.savefig(os.path.join(
        base_dir, 'compare.png'
    ))
    plt.close()

def calculation_ms(truth, prediction,dttrain,length):
    rounds = (int(truth.shape[0]/length))
    integral = []
    integral_tarr = []
    integral_parr = []
    mae_s = []
    mape_s = []
    for i in range(rounds):
        integral_t = (truth[i*length:(i+1)*length] * dttrain.reshape(-1)[i*length:(i+1)*length]*10).sum()
        integral_p = (prediction[i*length:(i+1)*length] * dttrain.reshape(-1)[i*length:(i+1)*length]*10).sum()

        integral_tarr.append(integral_t)
        integral_parr.append(integral_p)
        mae_s.append(abs(integral_t-integral_p))
        mape_s.append(abs((integral_t-integral_p)/integral_t)*100)
    integral.append(integral_tarr)
    integral.append(integral_parr)
    mae = np.mean(mae_s)
    mape = np.mean(mape_s)
    mae_std = np.std(mae_s,ddof=1)
    mape_std = np.std(mape_s,ddof=1)
    error=[[mae,mape],[mae_std,mape_std]]
    return integral,error


def array_operate_with_nan(array, operator):
    assert len(array.shape) == 2
    means = []
    for i in range(array.shape[1]):
        temp_col = array[:, i]
        means.append(operator(temp_col[temp_col == temp_col]))
    return np.array(means, dtype=np.float32)

class TimeRecorder:
    def __init__(self):
        self.infos = {}

    def __call__(self, info, *args, **kwargs):
        class Context:
            def __init__(self, recoder, info):
                self.recoder = recoder
                self.begin_time = None
                self.info = info

            def __enter__(self):
                self.begin_time = time.time()

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.recoder.infos[self.info] = time.time() - self.begin_time

        return Context(self, info)

    def __str__(self):
        return ' '.join(['{}:{:.2f}s'.format(info, t) for info, t in self.infos.items()])


def add_state_label_one(df):
    def is_nan(x):
        return x != x
    pcooling, power_cooling, Ti = df['Pcooling'], df['Power cooling'], df['Ti']
    states = []
    cur_state = 0
    for item, (cooling, power, ti) in enumerate(zip(pcooling, power_cooling, Ti)):
        nxt_i = min(item + 1, len(df) - 1)
        nc, np, nti = pcooling[nxt_i], power_cooling[nxt_i], Ti[nxt_i]
        if is_nan(cooling) or is_nan(power):
            states.append(0)
            continue
        states.append(0)
    ndf = df.copy(deep=True)
    ndf['states'] = states
    return ndf


def add_state_label(df):
    def is_nan(x):
        return x != x
    pcooling, power_cooling, Ti = df['Pcooling'], df['Power cooling'], df['Ti']
    states = []
    cur_state = 0
    for item, (cooling, power, ti) in enumerate(zip(pcooling, power_cooling, Ti)):
        nxt_i = min(item + 1, len(df) - 1)
        nc, np, nti = pcooling[nxt_i], power_cooling[nxt_i], Ti[nxt_i]
        if is_nan(cooling) or is_nan(power):
            states.append(cur_state)
            continue
        if cur_state == 0:
            if 0 <= cooling <= 0:
                cur_state = 1
            elif cooling == 23300:
                cur_state = 4
        elif cur_state == 1:
            #if ti >= 20:
            if np - power > 200 and ti >= 20:
                cur_state = 2
        elif cur_state == 2:
            if power > np and power > 5000:
                cur_state = 3
        elif cur_state == 3:
            if cooling == 23300:
                cur_state = 4
        elif cur_state == 4:
            if cooling <= 17000 and ti <= 13:
            #if cooling == 0:
                cur_state = 1
        states.append(cur_state)
    ndf = df.copy(deep=True)
    ndf['states'] = states
    return ndf


def get_Dataset(path):
    df = pd.read_csv(path)
    df = process_dataset(df)
    return df[['Pserver', 'Tr']], df[['Ti', 'Pcooling', 'Power cooling']], df[['time']], df[['states']]

def get_Dataset_one(path):
    df = pd.read_csv(path)
    df = process_dataset_one(df)
    return df[['Pserver', 'Tr']], df[['Ti', 'Pcooling', 'Power cooling']], df[['time']], df[['states']]

def process_dataset_one(df):

    df = add_state_label_one(df)
    from datetime import datetime
    beg_time_str = df['Time'].iloc[0]
    beg_time = datetime.strptime(beg_time_str[:-3]+beg_time_str[-2:], '%Y-%m-%dT%H:%M:%S%z')
    df['time'] = df['Time'].apply(
        lambda time_str: (datetime.strptime(time_str[:-3]+time_str[-2:], '%Y-%m-%dT%H:%M:%S%z')-beg_time
                          ).total_seconds()/10
    )
    df['delta'] = df['time'][1:] - df['time'][:-1]
    df.interpolate(axis=0, method='linear', limit_direction='both', inplace=True)
    return df

def process_dataset(df):

    df = add_state_label(df)
    from datetime import datetime
    beg_time_str = df['Time'].iloc[0]
    beg_time = datetime.strptime(beg_time_str[:-3]+beg_time_str[-2:], '%Y-%m-%dT%H:%M:%S%z')
    df['time'] = df['Time'].apply(
        lambda time_str: (datetime.strptime(time_str[:-3]+time_str[-2:], '%Y-%m-%dT%H:%M:%S%z')-beg_time
                          ).total_seconds()/10
    )
    df['delta'] = df['time'][1:] - df['time'][:-1]
    df.interpolate(axis=0, method='linear', limit_direction='both', inplace=True)
    return df


class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here

def visualize_prediction(Y_label, Y_pred, s_test, pserver,base_dir, seg_length=500, dir_name='visualizations'):
    assert len(Y_pred) == len(Y_label)
    if not os.path.exists(os.path.join(base_dir, dir_name)):
        os.mkdir(os.path.join(base_dir, dir_name))
    max_state = int(np.max(s_test))
    ID = 0
    for begin in range(0, len(Y_pred), seg_length):
        ID += 1
        plt.figure(figsize=(18, 15))
        y_label_seg = Y_label[begin:min(begin + seg_length, len(Y_label))]
        y_pred_seg = Y_pred[begin:min(begin + seg_length, len(Y_pred))]
        s_test_seg = s_test[begin:min(begin + seg_length, len(Y_pred))]
        #         scatter = plt.scatter(np.arange(begin, begin+len(tdf)), tdf['Power cooling'], c=tdf['states'], s=10)
        X = np.arange(begin, begin + len(y_label_seg))
        outputs_names = ['Inlet temperature(℃)', 'Cooling production(w)', 'Instant cooling power(w)']
        classes = ['unknown', 'Off', 'Start up stage 1', 'Start up stage 2', 'On']

        y_pserver = pserver[begin:min(begin + seg_length, len(Y_label))]

        for i, y_name in enumerate(outputs_names):
            plt.subplot(7, 2, i * 2 + 2)
            y_label = y_label_seg[:, i]
            y_pred = y_pred_seg[:, i]
            for state in range(0,max_state+1):
                indices = (s_test_seg.squeeze(axis=-1) == state)
                scatter = plt.scatter(X[indices], y_pred[indices], label=classes[state], s=5, marker='o')
            plt.xlabel('indexes')
            plt.ylabel(y_name)
            plt.legend()
            plt.subplot(7, 2, i * 2 + 1)

            plt.plot(X, y_label, '-k', label='Time Series')
            plt.xlabel('indexes')
            plt.ylabel(y_name)
            plt.legend()
            plt.subplot(7, 1, i +4)
            y_label = y_label_seg[:, i]
            y_pred = y_pred_seg[:, i]
            plt.plot(X, y_label, '-k', label='Time Series')
            for state in range(1,max_state + 1):
                indices = (s_test_seg.squeeze(axis=-1) == state)
                scatter = plt.scatter(X[indices], y_pred[indices], label= classes[state], s=5, marker='o')
            plt.xlabel('indexes')
            plt.ylabel(y_name)
            plt.legend()

        plt.subplot(7, 1, 7)

        plt.plot(X, y_pserver, '-k', label='Time Series')
        plt.xlabel('indexes')
        plt.ylabel("Pserver")
        plt.legend()

        plt.savefig(os.path.join(
            base_dir, dir_name, '%i-%i-%i.png' % (ID, begin, begin + seg_length)
        ))
        plt.close()
def visualize_prediction_power(Y_label, Y_pred, s_test, pserver,temperature,base_dir, seg_length=500, dir_name='visualizations'):
    assert len(Y_pred) == len(Y_label)
    if not os.path.exists(os.path.join(base_dir, dir_name)):
        os.mkdir(os.path.join(base_dir, dir_name))
    max_state = int(np.max(s_test))
    ID = 0
    for begin in range(0, len(Y_pred), seg_length):
        ID += 1
        plt.figure(figsize=(8, 5))
        y_label_seg = Y_label[begin:min(begin + seg_length, len(Y_label))]
        y_pred_seg = Y_pred[begin:min(begin + seg_length, len(Y_pred))]
        s_test_seg = s_test[begin:min(begin + seg_length, len(Y_pred))]
        #         scatter = plt.scatter(np.arange(begin, begin+len(tdf)), tdf['Power cooling'], c=tdf['states'], s=10)
        X = np.arange(begin, begin + len(y_label_seg))
        outputs_names = ['', '', '']
        #outputs_names = ['Inlet temperature(℃)', 'Cooling production(w)', 'Instant cooling power(w)']
        classes = ['unknown', 'Off', 'Start up stage 1', 'Start up stage 2', 'On']
        ax = plt.subplot(1, 1, 1)
        yfmt = ScalarFormatterForceFormat()
        yfmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(yfmt)
        ax.yaxis.get_offset_text().set_fontsize(19)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
        #plt.title('lower temperature limits-%d°C'%(temperature), fontsize=18)
        y_label = y_label_seg[:, 2]
        y_pred = y_pred_seg[:, 2]
        for state in range(1,max_state+1):
            indices = (s_test_seg.squeeze(axis=-1) == state)
            # scatter = plt.scatter(X[indices], y_pred[indices],s=1,marker='o')
            y_1 = y_pred.copy()
            for id, v in enumerate(indices):
                if v == False:
                    y_1[id] = None
            # print(y_1)
            plt.plot(X, y_1)
        plt.xlabel('Time(s)', fontsize=24)
        plt.ylabel(outputs_names[2],fontsize=24)
        #plt.legend(fontsize=21, loc = 1,labels=['Off', 'Start up stage 1', 'Start up stage 2', 'On'])
        plt.tight_layout()
        plt.savefig(os.path.join(
            base_dir, dir_name, '%i-%i-%i.png' % (ID, begin, begin + seg_length)
        ))
        plt.savefig(os.path.join(
            base_dir, dir_name, '%i-%i-%i.pdf' % (ID, begin, begin + seg_length)
        ))
        plt.savefig(os.path.join(
            base_dir, dir_name, '%i-%i-%i.eps' % (ID, begin, begin + seg_length)
        ),format="eps",dpi=600)
        plt.close()
def t2np(tensor):
    return tensor.squeeze(dim=0).detach().cpu().numpy()

