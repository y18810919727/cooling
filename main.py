import os
import sys
import numpy as np
from time import time
from datetime import timezone, timedelta,datetime
import torch
GPU = torch.cuda.is_available()

parent = os.path.dirname(sys.path[0])#os.getcwd())
sys.path.append(parent)
from dfa_ode.model_dfa import DFA_MIMO
from dfa_ode.train import EpochTrainer
from util import SimpleLogger, show_data, init_weights, array_operate_with_nan, get_Dataset, visualize_prediction, t2np, \
    draw_table, draw_table_all, calculation_ms,get_Dataset_one

import pandas as pd


from tensorboard_logger import configure, log_value
import argparse
import os
import pickle
import sys
import traceback
import shutil
import yaml

"""
potentially varying input parameters
"""
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

"""
论文中的代码参数
"""

parser.add_argument("--test", action="store_true", help="Testing model in para.save")
parser.add_argument("--time_aware", type=str, default='variable', choices=['no', 'input', 'variable'], help=methods)
parser.add_argument("--model", type=str, default='GRU', choices=['GRU', 'GRUinc', 'ARNN', 'ARNNinc', 'DFA'])
parser.add_argument("--interpol", type=str, default='constant', choices=['constant', 'linear'])

# data
parser.add_argument("--missing", type=float, default=0.0, help="fraction of missing samples (0.0 or 0.5)")

# model architecture
parser.add_argument("--k_h", type=int, default=20, help="dimension of hidden state")

# in case method == 'variable'
RKchoices = ['Euler', 'Midpoint', 'Kutta3', 'RK4']
parser.add_argument("--scheme", type=str, default='Euler', choices=RKchoices, help='Runge-Kutta training scheme')

# training
parser.add_argument("--batch_size", type=int, default=4000, help="batch size")
parser.add_argument("--visualization_len", type=int, default=2000, help="The length of visualization.")
parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
parser.add_argument("--bptt", type=int, default=800, help="bptt")
parser.add_argument("--dropout", type=float, default=0., help="drop prob")
parser.add_argument("--l2", type=float, default=0., help="L2 regularization")


# admin
parser.add_argument("--save", type=str, default='results', help="experiment logging folder")
parser.add_argument("--eval_epochs", type=int, default=10, help="validation every so many epochs")
parser.add_argument("--seed", type=int, default=None, help="random seed")
parser.add_argument("--powertime", type=int, default=1800, help="powertime")


# ODE_DFA
parser.add_argument("--dfa_yaml", type=str, default='dfa1', help="The setting of dfa ode")
parser.add_argument("--dfa_known", action="store_true",
                    help="The states of dfa for each positions are known in test and evaluation.")

parser.add_argument("--linear_decoder", action="store_true", help="Type of Ly")

# during development
parser.add_argument("--reset", action="store_true", help="reset even if same experiment already finished")
parser.add_argument("--short_encoding", action="store_true", help="Encoding short sequences to generate state0")
parser.add_argument("--debug", action="store_true", help="debug mode, for acceleration")

#数据集['Data_train_1_7_1','Data_train_1_8k','Data_train_3_8k','Data_train_4_2k','Data_validate']
#files = ['P-1.7k.csv','P-1.85k.csv','P-3.8K.csv','P-4.2k.csv','P-6.3k.csv']
parser.add_argument("--datasets", type=list, default=['P-1.7k','P-1.85k','P-3.8k','P-4.2k','P-6.3k'], help="datasets")
parser.add_argument("--describe", type=str, default="allp_alld  ", help="describe")
parser.add_argument("--mymodel", type=str, default='merge', choices=['merge', 'rnn', 'one'])

paras = parser.parse_args()

hard_reset = paras.reset
"""
results文件夹生成目录
"""
# if paras.save already exists and contains log.txt:
# reset if not finished, or if hard_reset
paras.save = os.path.join('results', paras.save) #路径拼接，改变paras.save为'results/tmp'
if paras.test:
    model_test_path = os.path.join(paras.save, 'best_dev_model.pt')
    paras.save = os.path.join(paras.save, 'test')
    if not os.path.exists(paras.save):
        os.mkdir(paras.save)
    paras.eval_epochs = 10
    paras.epochs = 10

log_file = os.path.join(paras.save, 'log.txt')
if os.path.isfile(log_file) and not paras.test:  #判断是否为文件
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        completed = 'Finished' in content
        if 'tmp' not in paras.save and completed and not hard_reset:
            print('Exit; already completed and no hard reset asked.')
            sys.exit()  # do not overwrite folder with current experiment
        else:  # reset folder
            shutil.rmtree(paras.save, ignore_errors=True) #递归地删除文件夹



# setup logging
logging = SimpleLogger(log_file) #log to file
configure(paras.save) #tensorboard logging
logging('Args: {}'.format(paras))



beijing = timezone(timedelta(hours=8))
time_beijing = datetime.utcnow().astimezone(beijing)
aim = paras.describe
describe = "实验时间:{}\n实验目的:{}\n".format(time_beijing,aim)
logging('\n实验描述', '\n{}'.format(describe))

"""
fixed input parameters
"""
"""
目前测试集和验证集是一样的，没有分开
"""
#frac_dev = 15/100
#frac_test = 15/100

GPU = torch.cuda.is_available()
logging('Using GPU?', GPU)

# set random seed for reproducibility
#设置模型的随机数种子
if paras.seed is None:
    paras.seed = np.random.randint(1000000)
torch.manual_seed(paras.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(paras.seed)
np.random.seed(paras.seed)

logging('Random seed', paras.seed)



"""
Load data
"""
"""
分为debug和非debug模式的训练集和测试集文件
"""
datasets = paras.datasets        #datasets=['Data_train_1_7_1','Data_train_1_8k','Data_train_3_8k','Data_train_4_2k','Data_validate']
all_sqe_nums = {}
len_sqe = []
if paras.debug: # Using short dataset for acceleration
    if paras.mymodel == 'one':
        Xtrain, Ytrain, ttrain, strain = [df.to_numpy(dtype=np.float32) for df in get_Dataset_one('./mydata2/train_P-1.7k.csv')]
        Xdev, Ydev, tdev, sdev = [df.to_numpy(dtype=np.float32) for df in get_Dataset_one('./mydata2/P-1.7k.csv')]
    else:
        Xtrain, Ytrain, ttrain, strain = [df.to_numpy(dtype=np.float32) for df in get_Dataset('./mydata2/train_P-1.7k.csv')]
        Xdev, Ydev, tdev, sdev = [df.to_numpy(dtype=np.float32) for df in get_Dataset('./mydata2/P-1.7k.csv')]
    len_sqe.append(0)
    len_sqe.append(int(Xtrain.size / 2))
    all_sqe_nums['train'] =len_sqe

else:
    Xtrain=[]
    Ytrain=[]
    ttrain=[]
    strain=[]
    Xtrain =np.array(Xtrain)
    for everdata in datasets:
        len_sqe = []
        len_sqe.append(int(Xtrain.size/2))
        if paras.mymodel == 'one':
            X, Y, t, s = [df.to_numpy(dtype=np.float32) for df in get_Dataset_one('./mydata2/train_' + everdata + '.csv')]
        else:
            X, Y, t, s = [df.to_numpy(dtype=np.float32) for df in get_Dataset('./mydata2/train_'+everdata+'.csv')]
        Xtrain = np.append(Xtrain,X).reshape(-1, 2)
        Ytrain = np.append(Ytrain,Y).reshape(-1, 3)
        ttrain = np.append(ttrain,t).reshape(-1, 1)
        strain = np.append(strain,s).reshape(-1, 1)
        len_sqe.append(int(Xtrain.size / 2))
        all_sqe_nums[everdata] = len_sqe    #每个数据集的范围

#dt时间间隔
dttrain = np.append(ttrain[1:] - ttrain[:-1], ttrain[1]-ttrain[0]).reshape(-1, 1)  #化为1列,从1到最后减去从0到
dttrain[dttrain < -0.1] = 0.1


k_in = Xtrain.shape[1]  #输入2维
k_out = Ytrain.shape[1] #输出3维


Ntrain = Xtrain.shape[0]  #序列长度

#Ndev = Xdev.shape[0]



#logging('first {} for training, then {} for development and {} for testing'.format(Ntrain, Ndev, Ntest))

"""
evaluation function
RRSE error
"""

"""
评估指标，算rrse
"""
def prediction_error(truth, prediction):
    length = min(truth.shape[0], prediction.shape[0])
    truth, prediction = truth[-length:], prediction[-length:]
    if len(truth.shape) == 1:
        indices = np.logical_and(~np.isnan(truth), ~np.isnan(prediction))
        truth, prediction = truth[indices], prediction[indices]
        se = np.sum((truth - prediction) ** 2, axis=0)  # summed squared error per channel
        rse = se / np.sum((truth - np.mean(truth, axis=0))**2)  # relative squared error
        rrse = np.mean(np.sqrt(rse))  # square root, followed by mean over channels
        return 100 * rrse  # in percentage
    # each shape (sequence, n_outputs)
    # Root Relative Squared Error
    results = [prediction_error(truth[:, i], prediction[:, i]) for i in range(truth.shape[-1])]
    return np.mean(results)



"""
- model:
    GRU (compensated GRU to avoid linear increase of state; has standard GRU as special case for Euler scheme and equidistant data)
    GRUinc (incremental GRU, for baseline only)
"""

#time_aware options
"""
把时间输入到模型的维度中插入进去
"""
"""
if paras.time_aware == 'input':

    # expand X matrices with additional input feature, i.e., normalized duration dt to next sample
    dt_mean, dt_std = np.mean(dttrain), np.std(dttrain)
    dttrain_n = (dttrain - dt_mean) / dt_std
    dtdev_n = (dtdev - dt_mean) / dt_std
    dttest_n = (dttest - dt_mean) / dt_std

    Xtrain = np.concatenate([Xtrain, dttrain_n], axis=1)
    Xdev = np.concatenate([Xdev, dtdev_n], axis=1)
    Xtest = np.concatenate([Xtest, dttest_n], axis=1)
    k_in += 1
"""

#算均值和标准差，神经网络输入之前需要把数据做归一化
Y_mean, Y_std = array_operate_with_nan(Ytrain, np.mean), array_operate_with_nan(Ytrain, np.std)
Ytrain = (Ytrain - Y_mean) / Y_std    #标准化处理

X_mean, X_std = array_operate_with_nan(Xtrain, np.mean), array_operate_with_nan(Xtrain, np.std)
Xtrain = (Xtrain - X_mean) / X_std
dt_mean = np.mean(dttrain)
# if paras.time_aware == 'no' or paras.time_aware == 'input':
#     # in case 'input': variable intervals already in input X;
#     # now set actual time intervals to 1 (else same effect as time_aware == 'variable')
#     dttrain = np.ones(dttrain.shape)
#     dtdev = np.ones(dtdev.shape)



# set model:
if not paras.test:
    # Generating a new model
    import yaml
    fs = open('./dfa_ode/transformations/{}.yaml'.format(paras.dfa_yaml), encoding='UTF-8', mode='r')
    dfa_setting = yaml.load(fs, Loader=yaml.FullLoader)

    model = DFA_MIMO(dfa_setting['ode_nums'], 1, k_in, k_out, paras.k_h, y_mean=Y_mean, y_std=Y_std,
                     odes_para=dfa_setting['odes'],
                     ode_2order=dfa_setting['ode_2order'],
                     transformations=dfa_setting['transformations'],
                     state_transformation_predictor=dfa_setting['predictors'], cell_type=paras.mymodel,
                     Ly_share=dfa_setting['Ly_share'])

    model.apply(init_weights)
else:
    model = torch.load(model_test_path)

#转到cuda
if GPU:
    model = model.cuda()

params = sum([np.prod(p.size()) for p in model.parameters()])  #提取该model所有权重参数总量
logging('\nModel %s (time_aware: %s, scheme %s) with %d trainable parameters' % (paras.model, paras.time_aware, paras.scheme, params))
for n, p in model.named_parameters():
    p_params = np.prod(p.size())
    print('\t%s\t%d (cuda: %s)'%(n, p_params, str(p.is_cuda)))

logging('Architecture: ', model)
log_value('model/params', params, 0)

optimizer = torch.optim.Adam(model.parameters(), lr=paras.lr, weight_decay=paras.l2)  #优化算法


#构建训练器，把训练的代码封装成一个类
trainer = EpochTrainer(model, optimizer, paras.epochs, Xtrain, Ytrain, strain, dttrain,
                       batch_size=paras.batch_size, gpu=GPU, bptt=paras.bptt,all_sqe_nums=all_sqe_nums, save_dir=paras.save,
                       logging=logging, debug=paras.debug,mymodel = paras.mymodel)  #dttrain ignored for all but 'variable' methods
t00 = time()
#参数，每训练10轮做一次评估，
best_dev_error = 1.e5
best_dev_epoch = 1
error_test = -1

max_epochs_no_decrease = 50  # If error in dev does not decrease in long time, the training will be paused early.


try:  # catch error and redirect to logger

    for epoch in range(1, paras.epochs + 1):
        mse_train = trainer(epoch) #训练
        if epoch % paras.eval_epochs == 0:
            with torch.no_grad(): #验证  停止梯度计算
                model.eval()      #测试
                # (1) forecast on train data steps
                """
                拿系统过去的信息，得到初始状态，联合未来的输入做预测
                """
                error_dev_sum = 0
                # corresponding test result:
                for everdata in datasets:
                    if paras.mymodel == 'one':
                        X, Y, t, s = [df.to_numpy(dtype=np.float32) for df in
                                      get_Dataset_one('./mydata2/train_' + everdata + '.csv')]
                    else:
                        X, Y, t, s = [df.to_numpy(dtype=np.float32) for df in
                                      get_Dataset('./mydata2/train_' + everdata + '.csv')]
                    if paras.mymodel == 'one':
                        Xtrain, Ytrain, ttrain, strain = [df.to_numpy(dtype=np.float32) for df in get_Dataset_one('./mydata2/train_'+everdata+'.csv')]
                        Xdev, Ydev, tdev, sdev = [df.to_numpy(dtype=np.float32) for df in get_Dataset_one('./mydata2/'+everdata+'.csv')]  #验证集也改为和测试集一样
                    else:
                        Xtrain, Ytrain, ttrain, strain = [df.to_numpy(dtype=np.float32) for df in
                                                          get_Dataset('./mydata2/train_' + everdata + '.csv')]
                        Xdev, Ydev, tdev, sdev = [df.to_numpy(dtype=np.float32) for df in
                                                  get_Dataset('./mydata2/' + everdata + '.csv')]  # 验证集也改为和测试集一样

                    dttrain = np.append(ttrain[1:] - ttrain[:-1], ttrain[1] - ttrain[0]).reshape(-1,1)  # 化为1列,从1到最后减去从0到
                    dtdev = np.append(tdev[1:] - tdev[:-1], tdev[1] - tdev[0]).reshape(-1, 1)
                    Ytrain, Ydev = [(Y - Y_mean) / Y_std for Y in [Ytrain, Ydev]]  # 标准化处理
                    Xtrain, Xdev = [(X - X_mean) / X_std for X in [Xtrain, Xdev]]

                    Xtrain_tn = torch.tensor(Xtrain, dtype=torch.float).unsqueeze(0)  # (1, Ntrain, k_in)
                    Ytrain_tn = torch.tensor(Ytrain, dtype=torch.float).unsqueeze(0)  # (1, Ntrain, k_out)
                    dttrain_tn = torch.tensor(dttrain, dtype=torch.float).unsqueeze(0)  # (1, Ntrain, 1)
                    strain_tn = torch.tensor(strain, dtype=torch.int).unsqueeze(0)  # (1, Ntrain, 1)
                    Xdev_tn = torch.tensor(Xdev, dtype=torch.float).unsqueeze(0)  # (1, Ndev, k_in)
                    Ydev_tn = torch.tensor(Ydev, dtype=torch.float).unsqueeze(0)  # (1, Ndev, k_out)
                    dtdev_tn = torch.tensor(dtdev, dtype=torch.float).unsqueeze(0)  # (1, Ndev, 1)
                    sdev_tn = torch.tensor(sdev, dtype=torch.int).unsqueeze(0)  # (1, Ntrain, 1)
                    if GPU:
                        Xtrain_tn = Xtrain_tn.cuda()
                        Ytrain_tn = Ytrain_tn.cuda()
                        dttrain_tn = dttrain_tn.cuda()
                        Xdev_tn = Xdev_tn.cuda()
                        Ydev_tn = Ydev_tn.cuda()
                        dtdev_tn = dtdev_tn.cuda()
                        strain_tn, sdev_tn = [s.cuda() for s in [strain_tn, sdev_tn]]

                    Ytrain_pred, htrain_pred = model.encoding_plus_predict(
                        Xtrain_tn,  dttrain_tn,  Ytrain_tn[:, :paras.bptt], strain_tn[:, :paras.bptt], paras.bptt,
                        strain_tn[:, paras.bptt:])
                    error_train = prediction_error(Ytrain[paras.bptt:], t2np(Ytrain_pred))
                    """
                    训练集做个预测，得到指标画个图
                    """

                    if not os.path.exists('{}/predict_seq'.format(paras.save)):
                        os.mkdir('{}/predict_seq'.format(paras.save))
                    train_path = os.path.join(paras.save, 'predict_seq/visualizations-train-{}'.format(best_dev_epoch))
                    if not os.path.exists(train_path):
                        os.mkdir(train_path)

                    visualize_prediction(
                        Ytrain[paras.bptt:] * Y_std + Y_mean, t2np(Ytrain_pred) * Y_std + Y_mean, strain[paras.bptt:],Xtrain[paras.bptt:,0]* X_std[0] + X_mean[0],
                        os.path.join(paras.save, 'predict_seq'),
                        seg_length=paras.visualization_len, dir_name='visualizations-train-%s/%s' % (str(best_dev_epoch), everdata))
                    # integral, error = calculation_ms(Ytrain[paras.bptt:, 2] * Y_std[2] + Y_mean[2],
                    #                              t2np(Ytrain_pred)[:, 2] * Y_std[2] + Y_std[2],dttrain, 1800)
                    #
                    # if (int(len(integral)) != 0):
                    #     draw_table(everdata, integral, error, 1800, os.path.join(paras.save, 'predict_seq'),
                    #                dir_name='visualizations-train-%s/%s' % (str(best_dev_epoch), everdata))
                    # (2) forecast on dev data
                    # Ydev_pred, hdev_pred = model(Xdev_tn, state0=htrain_pred[:, -1, :], dt=dtdev_tn)



                    Ydev_pred, hdev_pred = model.encoding_plus_predict(
                        Xdev_tn,  dtdev_tn,  Ydev_tn[:, :paras.bptt], sdev_tn[:, :paras.bptt], paras.bptt,
                        sdev_tn[:, paras.bptt:])
                    #error_train = prediction_error(Ytrain[paras.bptt:], t2np(Ytrain_pred))
                    """
                    验证集做个预测，算到指标打个日志
                    """
                    mse_dev = model.criterion(Ydev_pred, Ydev_tn[:, paras.bptt:]).item()
                    error_dev = prediction_error(Ydev[paras.bptt:], t2np(Ydev_pred))
                    error_dev_sum += error_dev
                    # report evaluation results
                    # log_value('train/mse', mse_train, epoch)
                    # log_value('train/error', error_train, epoch)
                    # log_value('dev/loss', mse_dev, epoch)
                    # log_value('dev/error', error_dev, epoch)
                    # current_trainresults_path = os.path.join(paras.save, 'current_trainresults')
                    # if not os.path.exists(current_trainresults_path):
                    #     os.mkdir(current_trainresults_path)
                    #
                    # current_devresults_path = os.path.join(paras.save, 'current_devresults')
                    # if not os.path.exists(current_devresults_path):
                    #     os.mkdir(current_devresults_path)
                    #
                    # best_dev_devresults_path = os.path.join(paras.save, 'best_dev_devresults')
                    # if not os.path.exists(best_dev_devresults_path):
                    #     os.mkdir(best_dev_devresults_path)

                    # logging('epoch %04d |%s| loss %.3f (train), %.3f (dev) | error %.3f (train), %.3f (dev) | tt %.2fmin'%
                    #         (epoch,everdata ,mse_train, mse_dev, error_train, error_dev, (time()-t00)/60.))

                    # show_data(ttrain[paras.bptt:], Ytrain[paras.bptt:], t2np(Ytrain_pred), current_trainresults_path, 'current_trainresults',everdata,
                    #                msg='train results (train error %.3f) at iter %d-%s' % (error_train, epoch,everdata))
                    #
                    # show_data(tdev[paras.bptt:], Ydev[paras.bptt:], t2np(Ydev_pred),current_devresults_path, 'current_devresults',everdata,
                    #                 msg='dev results (dev error %.3f) at iter %d-%s' % (error_dev, epoch,everdata))

                    dev_path = os.path.join(paras.save, 'predict_seq/visualizations-dev-{}'.format(epoch))  # 分别创建文件夹
                    if not os.path.exists(dev_path):
                        os.mkdir(dev_path)
                    # np.save('{}/predict_seq/visualizations-dev-{}/{}_dev_result_{}'.format(paras.save, epoch,everdata, epoch),
                    #         np.stack([Ydev[paras.bptt:], t2np(Ydev_pred)]))
                    # 画验证集
                    visualize_prediction(
                        Ydev[paras.bptt:] * Y_std + Y_mean, t2np(Ydev_pred) * Y_std + Y_mean, sdev[paras.bptt:],Xdev[paras.bptt:,0]* X_std[0] + X_mean[0],
                        # stest状态标签
                        os.path.join(paras.save, 'predict_seq'),
                        seg_length=paras.visualization_len, dir_name='visualizations-dev-%s/%s' %( str(epoch), everdata))

                    predict_result = os.path.join(paras.save, 'predict_result/')
                    if not os.path.exists(predict_result):
                        os.mkdir(predict_result)
                    dev_result = os.path.join(predict_result, 'dev/')
                    if not os.path.exists(dev_result):
                        os.mkdir(dev_result)
                    dev_result = os.path.join(dev_result, 'epoch_%s'%(str(epoch)))
                    if not os.path.exists(dev_result):
                        os.mkdir(dev_result)
                    pickle.dump(
                        {'t_tdev': tdev, 'y_target_dev': Ydev, 'y_pred_dev': t2np(Ydev_pred), 'x_dev': Xdev,
                         'sdev': sdev[paras.bptt:],
                         'Y_mean': Y_mean, 'Y_std': Y_std, 'X_mean': X_mean, 'X_std': X_std},
                        open(os.path.join(dev_result, 'datafigs_{}.pkl'.format(everdata)), 'wb'))

                    # make figure of best model on train, dev and test set for debugging
                    # show_data(tdev, Ydev, t2np(Ydev_pred), best_dev_devresults_path, 'best_dev_devresults',everdata,
                    #           msg='dev results-%s' % (error_dev, epoch,everdata))

                #验证集上，如果误差比最好的要小
                #验证集上达到一个最好的效果，拿测试集做预测，看效果
                # update best dev model
                if error_dev_sum < best_dev_error:
                    best_dev_error = error_dev_sum
                    best_dev_epoch = epoch
                    log_value('dev/best_error', best_dev_error, epoch)
                    logging('new best dev error %.3f' % best_dev_error)
                    error_all_mae = []
                    error_all_mape = []
                    error_test_sum = 0

                    for everdata in datasets:

                        if paras.mymodel == 'one':
                            Xtest, Ytest, ttest, stest = [df.to_numpy(dtype=np.float32) for df in
                                                  get_Dataset_one('./mydata2/'+everdata+'.csv')]
                        else:
                            Xtest, Ytest, ttest, stest = [df.to_numpy(dtype=np.float32) for df in
                                                          get_Dataset('./mydata2/' + everdata + '.csv')]
                        dttest = np.append(ttest[1:] - ttest[:-1], ttest[1] - ttest[0]).reshape(-1, 1)
                        # Y_mean, Y_std = array_operate_with_nan(Ytest, np.mean), array_operate_with_nan(Ytest, np.std)  # 归一化
                        # X_mean, X_std = array_operate_with_nan(Xtest, np.mean), array_operate_with_nan(Xtest, np.std)

                        Ytest = (Ytest - Y_mean) / Y_std
                        Xtest = (Xtest - X_mean) / X_std

                        Xtest_tn = torch.tensor(Xtest, dtype=torch.float).unsqueeze(0)
                        Ytest_tn = torch.tensor(Ytest, dtype=torch.float).unsqueeze(0)
                        dttest_tn = torch.tensor(dttest, dtype=torch.float).unsqueeze(0)
                        stest_tn = torch.tensor(stest, dtype=torch.int).unsqueeze(0)  # (1, Ntrain, 1)

                        Xtest_tn = Xtest_tn.cuda()
                        Ytest_tn = Ytest_tn.cuda()
                        dttest_tn = dttest_tn.cuda()

                        stest_tn = stest_tn.cuda()

                        Ytest_pred, test_state_pred = model.encoding_plus_predict(
                            Xtest_tn,  dttest_tn,  Ytest_tn[:, :paras.bptt], stest_tn[:, :paras.bptt], paras.bptt,
                            stest_tn[:, paras.bptt:] if paras.dfa_known else None)
                        # mse_test = model.criterion(Ytest_pred, Ytest_tn[paras.bptt:]).item()
                        #根据预测的结果，看一下模型每个时间点的状态分类
                        test_dfa_state_pred_array = model.select_dfa_states(test_state_pred[0]).int().detach().cpu().numpy()
                        #log_value('test/corresp_error', error_test, epoch)
                        predict_path = os.path.join(paras.save, 'predict_seq/visualizations-test-{}'.format(epoch))
                        if not os.path.exists(predict_path):
                            os.mkdir(predict_path)

                        # #将状态结果保存到目录里面
                        # np.save('{}/predict_seq/visualizations-test-{}/{}_test_result_{}'.format(paras.save,epoch,everdata, epoch),
                        #         np.stack([Ytest[paras.bptt:], t2np(Ytest_pred)]))
                        #画图可视化出来，画测试集
                        visualize_prediction(
                            Ytest[paras.bptt:] * Y_std + Y_mean, t2np(Ytest_pred) * Y_std + Y_mean, test_dfa_state_pred_array,Xtest[paras.bptt:,0]* X_std[0] + X_mean[0],
                            os.path.join(paras.save, 'predict_seq'),
                            seg_length=paras.visualization_len, dir_name='visualizations-test-%s/%s' % (str(best_dev_epoch) , everdata))# 模型自己预测的

                        integral,error = calculation_ms(Ytest[paras.bptt:, 2] * Y_std[2] + Y_mean[2],
                                          t2np(Ytest_pred)[:, 2] * Y_std[2] + Y_mean[2],dttest,paras.powertime)

                        if (int(len(integral[0])) != 0):
                            draw_table(everdata, integral, error, paras.powertime,  os.path.join(paras.save, 'predict_seq'), dir_name='visualizations-test-%s/%s' % (str(best_dev_epoch) , everdata))
                        error_all_mae.append(str(error[0][0])+" ± "+str(error[1][0]))
                        error_all_mape.append(str(error[0][1])+" ± "+str(error[1][1]))
                        # show_data(ttest, Ytest, t2np(Ytest_pred), paras.save+'/predict_seq/visualizations-test-%s'  % (str(best_dev_epoch)),
                        #           'best_dev_testresults', everdata,
                        #           msg='test results (test error %.3f) at iter %d (=best dev) -%s' % (error_test, epoch,everdata))

                        # save model
                        #torch.save(model.state_dict(), os.path.join(paras.save, 'best_dev_model_state_dict.pt'))
                        torch.save(model, os.path.join(paras.save, 'best_dev_model.pt'))  #存最好的模型
                        predict_result = os.path.join(paras.save, 'predict_result/')
                        if not os.path.exists(predict_result):
                            os.mkdir(predict_result)

                        predict_result = os.path.join(predict_result, 'test/')
                        if not os.path.exists(predict_result):
                            os.mkdir(predict_result)
                        pickle.dump({'t_test': ttest, 'y_target_test': Ytest, 'y_pred_test': t2np(Ytest_pred),'x_test':Xtest,'test_dfa_state_pred_array':test_dfa_state_pred_array,
                                     'Y_mean':Y_mean,'Y_std':Y_std,'X_mean':X_mean,'X_std':X_std},
                                    open(os.path.join(predict_result, 'datafigs_{}.pkl'.format(everdata)), 'wb'))

                    error_all=[error_all_mae,error_all_mape]
                    draw_table_all(datasets, error_all , os.path.join(paras.save, 'predict_seq'))
                    #draw_table_all(datasets,test_rres,os.path.join(paras.save, 'predict_seq'))
                elif epoch - best_dev_epoch > max_epochs_no_decrease:
                    logging('Development error did not decrease over %d epochs -- quitting.'%max_epochs_no_decrease)
                    break


    log_value('finished/best_dev_error', best_dev_error, 0)
    log_value('finished/corresp_test_error', error_test_sum, 0)

    logging('Finished: best dev error', best_dev_error,
              'at epoch', best_dev_epoch,
              'with corresp. test error', error_test_sum)

# 大概流程，训练后用验证集验证一下，然后用测试集画图


except:
    var = traceback.format_exc()
    logging(var)
