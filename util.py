import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
plt.switch_backend('agg')

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


def draw_table(file,integral,error,length,base_dir,dir_name='visualizations'):

    if not os.path.exists(os.path.join(base_dir, dir_name)):
        os.mkdir(os.path.join(base_dir, dir_name))
    plt.figure(figsize=(20, 4))
    # 列名
    vals = integral
    col = []
    for i in range(0,len(integral[0])):
        col.append(f'{length * i} - {length * (i + 1)}')
    # 行名
    row = ["truth", "prediction"]
    # 表格里面的具体值
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
    # 行名
    row = ["mean", "std"]
    # 表格里面的具体值
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
    # 列名
    vals = error_all
    col = file
    # 行名
    row = ["mae", "mape"]
    # 表格里面的具体值
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

#算预测powercoling的预测值和真实值的每length个点的平均值，标准差，积分

def calculation_ms(truth, prediction,dttrain,length):
    rounds = (int(truth.shape[0]/length))
    integral = []
    integral_tarr = []
    integral_parr = []
    mae_s = []
    #rmse_s = []
    mape_s = []
    for i in range(rounds):
        integral_t = (truth[i*length:(i+1)*length] * dttrain.reshape(-1)[i*length:(i+1)*length]*10).sum()
        integral_p = (prediction[i*length:(i+1)*length] * dttrain.reshape(-1)[i*length:(i+1)*length]*10).sum()

        integral_tarr.append(integral_t)
        integral_parr.append(integral_p)
        mae_s.append(abs(integral_t-integral_p))
        mape_s.append(abs((integral_t-integral_p)/integral_t)*100)
        #rmse_s.append((integral_t - integral_p) ** 2)
    integral.append(integral_tarr)
    integral.append(integral_parr)
    mae = np.mean(mae_s)
    mape = np.mean(mape_s)
    #rmse = np.sqrt(np.mean(mape_s))
    mae_std = np.std(mae_s,ddof=1)
    mape_std = np.std(mape_s,ddof=1)
    error=[[mae,mape],[mae_std,mape_std]]
    return integral,error



def show_data(t, target, pred, folder, tag ,everdata,msg=''):
    length = min(t.shape[0], target.shape[0], pred.shape[0])
    t, target, pred = [x[-length:] for x in [t, target, pred]]

    plt.clf()
    plt.figure(figsize=(10, 9))
    outputs_names = ['Ti', 'Pcooling', 'Power cooling']
    labels = ['truth','one-ode']
    # linear
    n = target.shape[1]
    for i in range(n):
        plt.subplot(n, 1, i+1)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel(outputs_names[i],fontsize=15)

        plt.plot(t*10, target[:, i], 'g--',label=labels[0])
        plt.plot(t*10, pred[:, i], 'r.',label=labels[1])
        plt.legend(fontsize=15, loc = 1)
        #ax_i.set_ylim(minv - view/10, maxv + view/10)
        if i == 0:
            plt.title(msg,fontsize=20)
    plt.xlabel('indexes', fontsize=15)
    #fig, axs = plt.subplots(6, 1)
    #for i, ax in enumerate(axs):
    #    ax.plot(target[:, i], 'g--', pred[:, i], 'r-')

    plt.savefig("%s/%s-%s.png"%(folder, tag,everdata))
    plt.close('all')


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)




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


def interpolate_tensors_with_nan(tensors):
    raise NotImplementedError
    tr = TimeRecorder()
    from common.interpolate import NaturalCubicSpline, natural_cubic_spline_coeffs
    with tr('linspace'):
        truth_time_steps = torch.linspace(0, 1, tensors.shape[1]).to(tensors.device)
    with tr('cubic spline'):
        coeffs = natural_cubic_spline_coeffs(truth_time_steps, tensors)
        interpolation = NaturalCubicSpline(truth_time_steps, coeffs)
    with tr('interpolation'):
        tensors_nonan = torch.stack([interpolation.evaluate(t) for t in truth_time_steps], dim=1)
    print(tr)
    return tensors_nonan


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
    beg_time_str = df['Time'].iloc[0] #取第0行数据
    beg_time = datetime.strptime(beg_time_str[:-3]+beg_time_str[-2:], '%Y-%m-%dT%H:%M:%S%z')
    df['time'] = df['Time'].apply(  #按照时间按0.1每步化成序列
        lambda time_str: (datetime.strptime(time_str[:-3]+time_str[-2:], '%Y-%m-%dT%H:%M:%S%z')-beg_time
                          ).total_seconds()/10
    )
    df['delta'] = df['time'][1:] - df['time'][:-1]   #为啥按索引对齐运算，以至于结果都等于0
    df.interpolate(axis=0, method='linear', limit_direction='both', inplace=True)  #按列线性插值
    return df

def process_dataset(df):

    df = add_state_label(df)
    from datetime import datetime
    beg_time_str = df['Time'].iloc[0] #取第0行数据
    beg_time = datetime.strptime(beg_time_str[:-3]+beg_time_str[-2:], '%Y-%m-%dT%H:%M:%S%z')
    df['time'] = df['Time'].apply(  #按照时间按0.1每步化成序列
        lambda time_str: (datetime.strptime(time_str[:-3]+time_str[-2:], '%Y-%m-%dT%H:%M:%S%z')-beg_time
                          ).total_seconds()/10
    )
    df['delta'] = df['time'][1:] - df['time'][:-1]   #为啥按索引对齐运算，以至于结果都等于0
    df.interpolate(axis=0, method='linear', limit_direction='both', inplace=True)  #按列线性插值
    return df


def get_mlp_network(layer_sizes, outputs_size):

    modules_list = []
    for i in range(1, len(layer_sizes)):
        modules_list.append(
            nn.Linear(layer_sizes[i - 1], layer_sizes[i])
        )
        modules_list.append(nn.Tanh())
    modules_list.append(
        nn.Linear(layer_sizes[-1], outputs_size)
    )
    return nn.Sequential(*modules_list)


#画功耗误差折线图
def draw_power_error(data,datasets,dir_name='visualizations'):
    plt.figure(figsize=(10, 6))
    #plt.tick_params(labelsize=20)
    #plt.title('power-error', fontsize=18)
    X = [i for i in range(5, 125, 10)]
    plt.xticks(X,fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Time(m)',fontsize=22)
    plt.ylabel("MAPE of energy consumption(%)",fontsize=22)
    X = [i for i in range(5, 125, 5)]
    for id,everdata in enumerate(datasets):
        plt.plot(X, data[id], label=everdata, marker='o')
        #plt.plot(X, data[id], marker='o')
    plt.legend(fontsize=19, loc = 1)
    plt.tight_layout()
    plt.savefig(os.path.join(
        dir_name, 'power_error.png'
    ))
    plt.savefig(os.path.join(
        dir_name, 'power_error.eps'
    ), format="eps", dpi=600)
    plt.close()


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
            # 对于one_ode 这里改成0
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


def visualize_prediction_compare(Y_label, Y_pred, s_test, pserver,base_dir, seg_length=500, dir_name='visualizations'):
    assert len(Y_pred) == len(Y_label)
    if not os.path.exists(os.path.join(base_dir, dir_name)):
        os.mkdir(os.path.join(base_dir, dir_name))
    max_state = int(np.max(s_test))
    ID = 0
    for begin in range(0, len(Y_pred), seg_length):
        ID += 1
        plt.figure(figsize=(20, 9))

        y_label_seg = Y_label[begin:min(begin + seg_length, len(Y_label))]
        y_pred_seg = Y_pred[begin:min(begin + seg_length, len(Y_pred))]
        s_test_seg = s_test[begin:min(begin + seg_length, len(Y_pred))]
        #         scatter = plt.scatter(np.arange(begin, begin+len(tdf)), tdf['Power cooling'], c=tdf['states'], s=10)
        X = np.arange(begin, begin + len(y_label_seg))
        outputs_names = ['Inlet temperature(℃)', 'Cooling production(w)', 'Instant cooling power(w)']
        classes = ['unknown', 'Off', 'Start up stage 1', 'Start up stage 2', 'On']
        for i, y_name in enumerate(outputs_names):

            plt.subplots_adjust(hspace=0.4)  # 调整子图间距
            plt.subplot(3, 2, i * 2 + 2)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            y_label = y_label_seg[:, i]
            y_pred = y_pred_seg[:, i]
            for state in range(1,max_state+1):
                indices = (s_test_seg.squeeze(axis=-1) == state)
                scatter = plt.scatter(X[indices], y_pred[indices], label=classes[state], s=5, marker='o')
            if(i==2):
                plt.xlabel('Time(s)', fontsize=15)
            plt.ylabel(y_name, fontsize=15)
            plt.legend( fontsize=15, loc = 1)
            plt.subplot(3, 2, i * 2 + 1)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.plot(X, y_label, '-k', label='Truth')
            if (i == 2):
                plt.xlabel('Time(s)', fontsize=15)
            plt.ylabel(y_name, fontsize=15)
            plt.legend(fontsize=15, loc = 1)


        plt.savefig(os.path.join(
            base_dir, dir_name, '%i-%i-%i.png' % (ID, begin, begin + seg_length)
        ))
        plt.close()


from matplotlib.ticker import ScalarFormatter

class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format here

'''
中文论文图
'''
def visualize_prediction_one2(Y_label, Y_pred, s_test, pserver,base_dir, seg_length=500, dir_name='visualizations',lablee = None):
    assert len(Y_pred) == len(Y_label)
    if not os.path.exists(os.path.join(base_dir, dir_name)):
        os.mkdir(os.path.join(base_dir, dir_name))
    max_state = int(np.max(s_test))
    ID = 0
    for begin in range(0, len(Y_pred), seg_length):
        ID += 1
        fig = plt.figure(figsize=(7, 9))
        y_label_seg = Y_label[begin:min(begin + seg_length, len(Y_label))]
        y_pred_seg = Y_pred[begin:min(begin + seg_length, len(Y_pred))]
        s_test_seg = s_test[begin:min(begin + seg_length, len(Y_pred))]
        #         scatter = plt.scatter(np.arange(begin, begin+len(tdf)), tdf['Power cooling'], c=tdf['states'], s=10)
        X = np.arange(begin, begin + len(y_label_seg))
        #outputs_names = ['', '', '']
        outputs_names = ['Instant\ncooling power(w)', 'Cooling\nproduction(w)', 'Inlet\ntemperature(℃)']
        classes = ['unknown', 'Off', 'Start up stage 1', 'Start up stage 2', 'On']
        colors = ['m','r','#800080','#FFD700','g']
        for i, y_name in enumerate(outputs_names):
            ax = plt.subplot(3, 1, i+1)
            plt.subplots_adjust(left=0.15,hspace=0.4)  # 调整子图间距
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            y_label = y_label_seg[:, i]
            y_pred = y_pred_seg[:, 2-i]
            if (i!=2):
                yfmt = ScalarFormatterForceFormat()
                yfmt.set_powerlimits((0, 0))
                ax.yaxis.set_major_formatter(yfmt)
                ax.yaxis.get_offset_text().set_fontsize(18)
            #plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y',useLocale=True)
            if lablee == 'rnn':
                for state in range(1,max_state+1):
                    indices = (s_test_seg.squeeze(axis=-1) == state)
                    #scatter = plt.scatter(X[indices], y_pred[indices],s=1,marker='o')
                    y_1 = y_pred.copy()
                    for id, v in enumerate(indices):
                        if v == False:
                            y_1[id] = None
                    #print(y_1)
                    plt.plot(X, y_1,color = colors[state])

            if lablee == 'ours':
                for state in range(1,max_state+1):
                    indices = (s_test_seg.squeeze(axis=-1) == state)
                    #scatter = plt.plot(X, y_pred,'-g')
                    y_1 = y_pred.copy()
                    for i, v in enumerate(indices):
                        if v == False:
                            y_1[i] = None
                    plt.plot(X, y_1,color = colors[state],linewidth = '2.6')
            elif lablee=='truth':
                    #indices = (s_test_seg.squeeze(axis=-1) == 0)
                    plt.plot(X, y_pred, '-k', label='truth',linewidth = '2.6')
            else:
                    indices = (s_test_seg.squeeze(axis=-1) == 0)
                    scatter = plt.plot(X[indices],y_pred[indices], color='#808080',linewidth = '2.6')


            plt.ylabel(y_name, fontsize=18)
        plt.xlabel('Time(s)', fontsize=18)
        fig.align_ylabels()
        plt.tight_layout()
        plt.savefig(os.path.join(
            base_dir, dir_name, '%i-%i-%i.png' % (ID, begin, begin + seg_length)
        ))
        plt.savefig(os.path.join(
            base_dir, dir_name, '%i-%i-%i.eps' % (ID, begin, begin + seg_length)
        ),format="eps",dpi=600)
        plt.close()


def visualize_prediction_one(Y_label, Y_pred, s_test, pserver,base_dir, seg_length=500, dir_name='visualizations',lablee = None):
    assert len(Y_pred) == len(Y_label)
    if not os.path.exists(os.path.join(base_dir, dir_name)):
        os.mkdir(os.path.join(base_dir, dir_name))
    max_state = int(np.max(s_test))
    ID = 0
    for begin in range(0, len(Y_pred), seg_length):
        ID += 1
        fig = plt.figure(figsize=(7, 9))
        y_label_seg = Y_label[begin:min(begin + seg_length, len(Y_label))]
        y_pred_seg = Y_pred[begin:min(begin + seg_length, len(Y_pred))]
        s_test_seg = s_test[begin:min(begin + seg_length, len(Y_pred))]
        #         scatter = plt.scatter(np.arange(begin, begin+len(tdf)), tdf['Power cooling'], c=tdf['states'], s=10)
        X = np.arange(begin, begin + len(y_label_seg))
        #outputs_names = ['', '', '']
        outputs_names = ['Inlet\ntemperature(℃)', 'Cooling\nproduction(w)', 'Instant\ncooling power(w)']
        classes = ['unknown', 'Off', 'Start up stage 1', 'Start up stage 2', 'On']
        for i, y_name in enumerate(outputs_names):

            ax = plt.subplot(3, 1, i+1)
            plt.subplots_adjust(left=0.15,hspace=0.4)  # 调整子图间距
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            y_label = y_label_seg[:, i]
            y_pred = y_pred_seg[:, i]
            if (i!=0):
                yfmt = ScalarFormatterForceFormat()
                yfmt.set_powerlimits((0, 0))
                ax.yaxis.set_major_formatter(yfmt)
                ax.yaxis.get_offset_text().set_fontsize(18)
            #plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y',useLocale=True)
            if lablee == 'rnn':
                for state in range(1,max_state+1):
                    indices = (s_test_seg.squeeze(axis=-1) == state)
                    #scatter = plt.scatter(X[indices], y_pred[indices],s=1,marker='o')
                    y_1 = y_pred.copy()
                    for id, v in enumerate(indices):
                        if v == False:
                            y_1[id] = None
                    #print(y_1)
                    plt.plot(X, y_1)

            if lablee == 'ours':
                for state in range(1,max_state+1):
                    indices = (s_test_seg.squeeze(axis=-1) == state)
                    #scatter = plt.plot(X, y_pred,'-g')
                    y_1 = y_pred.copy()
                    for i, v in enumerate(indices):
                        if v == False:
                            y_1[i] = None
                    plt.plot(X, y_1)
            elif lablee=='truth':
                    #indices = (s_test_seg.squeeze(axis=-1) == 0)
                    plt.plot(X, y_pred, '-k', label='truth')
            else:
                    indices = (s_test_seg.squeeze(axis=-1) == 0)
                    scatter = plt.plot(X[indices],y_pred[indices], '-b')


            plt.ylabel(y_name, fontsize=18)
        plt.xlabel('Time(s)', fontsize=18)
        fig.align_ylabels()
        plt.tight_layout()
        plt.savefig(os.path.join(
            base_dir, dir_name, '%i-%i-%i.png' % (ID, begin, begin + seg_length)
        ))
        plt.savefig(os.path.join(
            base_dir, dir_name, '%i-%i-%i.eps' % (ID, begin, begin + seg_length)
        ),format="eps",dpi=600)
        plt.close()

#中文论文
def visualize_prediction_power2(Y_label, Y_pred, s_test, pserver,temperature,base_dir, seg_length=500, dir_name='visualizations'):
    assert len(Y_pred) == len(Y_label)
    if not os.path.exists(os.path.join(base_dir, dir_name)):
        os.mkdir(os.path.join(base_dir, dir_name))
    max_state = int(np.max(s_test))
    ID = 0
    colors = ['m', 'r', '#800080', '#FFD700', 'g']
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
            if(state == 2 or state==3):
                plt.fill_between(X, y_1,color="skyblue", alpha=0.5)
            plt.plot(X, y_1,color = colors[state])
            #plt.plot(X, y_1)
        plt.xlabel('Time(s)', fontsize=24)
        plt.ylabel(outputs_names[2],fontsize=24)
        #plt.legend(fontsize=21, loc = 1,labels=['Off', 'Start up stage 1', 'Start up stage 2', 'On'])
        plt.tight_layout()
        plt.savefig(os.path.join(
            base_dir, dir_name, '%i-%i-%i.png' % (ID, begin, begin + seg_length)
        ))
        # plt.savefig(os.path.join(
        #     base_dir, dir_name, '%i-%i-%i.pdf' % (ID, begin, begin + seg_length)
        # ))
        # plt.savefig(os.path.join(
        #     base_dir, dir_name, '%i-%i-%i.eps' % (ID, begin, begin + seg_length)
        # ),format="eps",dpi=600)
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





def display_states_confusion_matrix(true, pred, path, labels, print_handle=print):

    true_label = list(map(lambda x: labels[x], true))
    pred_label = list(map(lambda x: labels[x], pred))
    final_labels = [labels[x] for x in set(true)]
    cm = confusion_matrix(
        true_label,
        pred_label,
        labels=final_labels
    )
    print_handle('Confusion matrix: \n', cm)
    disp = ConfusionMatrixDisplay(cm, display_labels=final_labels)
    disp.plot(cmap='Greens')
    plt.savefig('%s.png' % path)


def t2np(tensor):
    return tensor.squeeze(dim=0).detach().cpu().numpy()

