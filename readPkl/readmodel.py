import os
import sys
import numpy as np
from time import time
import torch
GPU = torch.cuda.is_available()

parent = os.path.dirname(sys.path[0])#os.getcwd())
sys.path.append(parent)
from dfa_ode.model_dfa import DFA_MIMO
from dfa_ode.train import EpochTrainer
from util import SimpleLogger, show_data, init_weights, array_operate_with_nan, get_Dataset, visualize_prediction, t2np, draw_table,draw_table_all


parser = argparse.ArgumentParser(description='Models for Continuous Stirred Tank dataset')

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


#算预测powercoling的预测值和真实值的每2k个点的平均值，标准差，积分

def compare_pc(truth, prediction,length):
    rounds = (int(truth.shape[0]/length))
    integral = []
    integral_tarr = []
    integral_parr = []
    mae_s = []
    #rmse_s = []
    mape_s = []
    for i in range(rounds):
        integral_t,integral_p = np.trapz(truth[i*length:(i+1)*length]) ,np.trapz(prediction[i * length:(i + 1) * length])

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

datasets=['Data_train_1_7_1','Data_train_1_8k','Data_train_3_8k','Data_train_4_2k','Data_validate']
all_sqe_nums = {}
len_sqe = []

Xtrain=[]
Ytrain=[]
ttrain=[]
strain=[]
Xtrain =np.array(Xtrain)
for everdata in datasets:
    len_sqe = []
    len_sqe.append(int(Xtrain.size/2))
    X, Y, t, s = [df.to_numpy(dtype=np.float32) for df in get_Dataset('../mydata/Front_'+everdata+'.csv')]
    Xtrain = np.append(Xtrain,X).reshape(-1, 2)
    Ytrain = np.append(Ytrain,Y).reshape(-1, 3)
    ttrain = np.append(ttrain,t).reshape(-1, 1)
    strain = np.append(strain,s).reshape(-1, 1)
    len_sqe.append(int(Xtrain.size / 2))
    all_sqe_nums[everdata] = len_sqe

    #dt时间间隔
dttrain = np.append(ttrain[1:] - ttrain[:-1], ttrain[1]-ttrain[0]).reshape(-1, 1)  #化为1列,从1到最后减去从0到
dttrain[dttrain < -0.1] = 0.1

model = torch.load(model_test_path)


optimizer = torch.optim.Adam(model.parameters(), lr=paras.lr, weight_decay=paras.l2)  #优化算法

for everdata in datasets:

    Xtrain, Ytrain, ttrain, strain = [df.to_numpy(dtype=np.float32) for df in
                                      get_Dataset('./mydata/Front_' + everdata + '.csv')]
    Xdev, Ydev, tdev, sdev = [df.to_numpy(dtype=np.float32) for df in get_Dataset('./mydata/Back_' + everdata + '.csv')]
    dttrain = np.append(ttrain[1:] - ttrain[:-1], ttrain[1] - ttrain[0]).reshape(-1, 1)  # 化为1列,从1到最后减去从0到
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
        Xtrain_tn, dttrain_tn, Ytrain_tn[:, :paras.bptt], strain_tn[:, :paras.bptt], paras.bptt,
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
        Ytrain[paras.bptt:] * Y_std + Y_mean, t2np(Ytrain_pred) * Y_std + Y_mean, strain[paras.bptt:],
        Xtrain[paras.bptt:, 0] * X_std[0] + X_mean[0],
        os.path.join(paras.save, 'predict_seq'),
        seg_length=paras.visualization_len, dir_name='visualizations-train-%s/%s' % (str(best_dev_epoch), everdata))
    # integral, error = compare_pc(Ytrain[paras.bptt:, 2] * Y_std[2] + Y_mean[2],
    #                              t2np(Ytrain_pred)[:, 2] * Y_std[2] + Y_std[2], 1800)
    #
    # if (int(len(integral)) != 0):
    #     draw_table(everdata, integral, error, 1800, os.path.join(paras.save, 'predict_seq'),
    #                dir_name='visualizations-train-%s/%s' % (str(best_dev_epoch), everdata))
    # (2) forecast on dev data
    # Ydev_pred, hdev_pred = model(Xdev_tn, state0=htrain_pred[:, -1, :], dt=dtdev_tn)

    Ydev_pred, hdev_pred = model.encoding_plus_predict(
        Xdev_tn, dtdev_tn, Ydev_tn[:, :paras.bptt], sdev_tn[:, :paras.bptt], paras.bptt,
        sdev_tn[:, paras.bptt:])
    # error_train = prediction_error(Ytrain[paras.bptt:], t2np(Ytrain_pred))
    """
    验证集做个预测，算到指标打个日志
    """
    mse_dev = model.criterion(Ydev_pred, Ydev_tn[:, paras.bptt:]).item()
    error_dev = prediction_error(Ydev[paras.bptt:], t2np(Ydev_pred))
    error_dev_sum += error_dev
    # report evaluation results
    log_value('train/mse', mse_train, epoch)
    log_value('train/error', error_train, epoch)
    log_value('dev/loss', mse_dev, epoch)
    log_value('dev/error', error_dev, epoch)

    logging('epoch %04d |%s| loss %.3f (train), %.3f (dev) | error %.3f (train), %.3f (dev) | tt %.2fmin' %
            (epoch, everdata, mse_train, mse_dev, error_train, error_dev, (time() - t00) / 60.))
    show_data(ttrain[paras.bptt:], Ytrain[paras.bptt:], t2np(Ytrain_pred), paras.save, 'current_trainresults',
              msg='train results (train error %.3f) at iter %d' % (error_train, epoch))
    show_data(tdev[paras.bptt:], Ydev[paras.bptt:], t2np(Ydev_pred), paras.save, 'current_devresults',
              msg='dev results (dev error %.3f) at iter %d' % (error_dev, epoch))

    dev_path = os.path.join(paras.save, 'predict_seq/visualizations-dev-{}'.format(epoch))  # 分别创建文件夹
    if not os.path.exists(dev_path):
        os.mkdir(dev_path)

    np.save('{}/predict_seq/visualizations-dev-{}/{}_dev_result_{}'.format(paras.save, epoch, everdata, epoch),
            np.stack([Ydev[paras.bptt:], t2np(Ydev_pred)]))
    # 画验证集
    visualize_prediction(
        Ydev[paras.bptt:] * Y_std + Y_mean, t2np(Ydev_pred) * Y_std + Y_mean, sdev[paras.bptt:],
        Xdev[paras.bptt:, 0] * X_std[0] + X_mean[0],
        # stest状态标签
        os.path.join(paras.save, 'predict_seq'),
        seg_length=paras.visualization_len, dir_name='visualizations-dev-%s/%s' % (str(epoch), everdata))

    # make figure of best model on train, dev and test set for debugging
    show_data(tdev, Ydev, t2np(Ydev_pred), paras.save, 'best_dev_devresults',
              msg='dev results (dev error %.3f) at iter %d' % (error_dev, epoch))

# 验证集上，如果误差比最好的要小
# 验证集上达到一个最好的效果，拿测试集做预测，看效果
# update best dev model
if error_dev_sum < best_dev_error:
    best_dev_error = error_dev_sum
    best_dev_epoch = epoch
    log_value('dev/best_error', best_dev_error, epoch)

    logging('new best dev error %.3f' % best_dev_error)
    error_all_mae = []
    error_all_mape = []
    for everdata in datasets:
        Xtest, Ytest, ttest, stest = [df.to_numpy(dtype=np.float32) for df in
                                      get_Dataset('./mydata/Back_' + everdata + '.csv')]
        dttest = np.append(ttest[1:] - ttest[:-1], ttest[1] - ttest[0]).reshape(-1, 1)

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
            Xtest_tn, dttest_tn, Ytest_tn[:, :paras.bptt], stest_tn[:, :paras.bptt], paras.bptt,
            stest_tn[:, paras.bptt:] if paras.dfa_known else None)
        # mse_test = model.criterion(Ytest_pred, Ytest_tn[paras.bptt:]).item()
        error_test = prediction_error(Ytest[paras.bptt:], t2np(Ytest_pred))
        # 根据预测的结果，看一下模型每个时间点的状态分类
        test_dfa_state_pred_array = model.select_dfa_states(test_state_pred[0]).int().detach().cpu().numpy()

        log_value('test/corresp_error', error_test, epoch)

        predict_path = os.path.join(paras.save, 'predict_seq/visualizations-test-{}'.format(epoch))
        if not os.path.exists(predict_path):
            os.mkdir(predict_path)

        # 将状态结果保存到目录里面
        np.save('{}/predict_seq/visualizations-test-{}/{}_test_result_{}'.format(paras.save, epoch, everdata, epoch),
                np.stack([Ytest[paras.bptt:], t2np(Ytest_pred)]))
        # 画图可视化出来，画测试集
        visualize_prediction(
            Ytest[paras.bptt:] * Y_std + Y_mean, t2np(Ytest_pred) * Y_std + Y_mean, test_dfa_state_pred_array,
            Xtest[paras.bptt:, 0] * X_std[0] + X_mean[0],
            os.path.join(paras.save, 'predict_seq'),
            seg_length=paras.visualization_len,
            dir_name='visualizations-test-%s/%s' % (str(best_dev_epoch), everdata))  # 模型自己预测的

        integral, error = compare_pc(Ytest[paras.bptt:, 2] * Y_std[2] + Y_mean[2],
                                     t2np(Ytest_pred)[:, 2] * Y_std[2] + Y_std[2], 1800)

        if (int(len(integral)) != 0):
            draw_table(everdata, integral, error, 1800, os.path.join(paras.save, 'predict_seq'),
                       dir_name='visualizations-test-%s/%s' % (str(best_dev_epoch), everdata))
        error_all_mae.append(str(error[0][0]) + " ± " + str(error[1][0]))
        error_all_mape.append(str(error[0][1]) + " ± " + str(error[1][1]))
        show_data(ttest, Ytest, t2np(Ytest_pred),
                  paras.save + '/predict_seq/visualizations-test-%s' % (str(best_dev_epoch)),
                  'best_dev_testresults_%s_' % everdata,
                  msg='test results (test error %.3f) at iter %d (=best dev)' % (error_test, epoch))

        # save model
        # torch.save(model.state_dict(), os.path.join(paras.save, 'best_dev_model_state_dict.pt'))
        torch.save(model, os.path.join(paras.save, 'best_dev_model.pt'))  # 存最好的模型

        predict_result = os.path.join(paras.save, 'predict_result/')
        if not os.path.exists(predict_result):
            os.mkdir(predict_result)

        pickle.dump({'t_test': ttest, 'y_target_test': Ytest, 'y_pred_test': t2np(Ytest_pred), 'x_test': Xtest,
                     'test_dfa_state_pred_array': test_dfa_state_pred_array, 'Y_mean': Y_mean, 'Y_std': Y_std},
                    open(os.path.join(predict_result, 'datafigs_{}.pkl'.format(everdata)), 'wb'))

    error_all = [error_all_mae, error_all_mape]
    draw_table_all(datasets, error_all, os.path.join(paras.save, 'predict_seq'))

































