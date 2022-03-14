#读取测试结果

import pickle
import os
import argparse
import numpy as np


from util import SimpleLogger, show_data, init_weights, array_operate_with_nan, get_Dataset, visualize_prediction, t2np, draw_table,draw_table_all,calculation_ms,visualize_prediction_compare,draw_power_error,visualize_prediction_one,visualize_prediction_one2
parser = argparse.ArgumentParser(description='Models for Continuous Stirred Tank dataset')
parser.add_argument("--save", type=str, default='test', help="experiment logging folder")
parser.add_argument("--bptt", type=int, default=800, help="bptt")
parser.add_argument("--powertime", type=int, default=1800, help="powertime")
parser.add_argument("--dev_epoch", type=str, default=80, help="experiment logging folder")

parser.add_argument("--test", action="store_true", help="")
parser.add_argument("--dev", action="store_true", help="")
parser.add_argument("--one", action="store_true", help="")
parser.add_argument("--compare", action="store_true", help="")

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
    return results




def rrse(truth, prediction):
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


'''
用来处理保存的数据
1. 测试集曲线对比图
2. 测试集功耗误差对比图，从5分钟到120分钟
3. 测试机的功耗误差对比，不同数据集的图表
4. 测试集的所有数据的误差图
5. 验证集的rrse汇总日志，目前要给定epoch
'''

if __name__ == '__main__':

    paras = parser.parse_args()
    paras.save = os.path.join('../results', paras.save)  # 路径拼接，改变paras.save为'results/tmp'
    # 数据集['Data_train_1_7_1','Data_train_1_8k','Data_train_3_8k','Data_train_4_2k','Data_validate']
    datasets=['P-1.7k','P-3.8k','P-6.3k']
    all_sqe_nums = {}
    len_sqe = []
    result_path = os.path.join(paras.save, 'predict_result')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    error_all_mae = []
    error_all_mape = []
    test_rres = []

    dev_path = os.path.join(result_path, 'dev')
    if not os.path.exists(dev_path):
        os.mkdir(dev_path)

    log_file = os.path.join(dev_path, 'rrse.txt')
    logging = SimpleLogger(log_file)  # log to file
    all_power_error = []
    if paras.test:
        for everdata in datasets:

            #测试集
            test_path = os.path.join(result_path, 'test/compare_img')
            if not os.path.exists(test_path):
                os.mkdir(test_path)
            filename = "predict_result/test/datafigs_"+everdata+".pkl"
            file=open(os.path.join(paras.save, filename),"rb")
            data=pickle.load(file)
            X_mean = data['X_mean']
            X_std = data['X_std']
            Y_mean = data['Y_mean']
            Y_std = data['Y_std']
            Ytest = data['y_target_test']
            y_pred_test = data['y_pred_test']
            Xtest = data['x_test']

            ttest = data['t_test']
            dttest = np.append(ttest[1:] - ttest[:-1], ttest[1] - ttest[0]).reshape(-1, 1)
            test_dfa_state_pred_array = data['test_dfa_state_pred_array']
            error_all_path = os.path.join(result_path, 'test/error_all')
            if not os.path.exists(error_all_path):
                os.mkdir(error_all_path)
            show_data(ttest, Ytest * Y_std + Y_mean, y_pred_test* Y_std + Y_mean, error_all_path,
                      'error_all', everdata,
                      msg='one-ode(%s)' % (everdata))

            visualize_prediction_compare( Ytest[paras.bptt:] * Y_std + Y_mean, y_pred_test * Y_std + Y_mean, test_dfa_state_pred_array,Xtest[paras.bptt:,0]* X_std[0] + X_mean[0],
                                        test_path,
                                        seg_length=2000, dir_name='%s' %(everdata))# 模型自己预测的

            integral, error = calculation_ms(Ytest[paras.bptt:, 2] * Y_std[2] + Y_mean[2],
                                         y_pred_test[:, 2] * Y_std[2] + Y_mean[2], dttest, paras.powertime)
            if (int(len(integral[0])) != 0):
                draw_table(everdata, integral, error,  paras.powertime, test_path,dir_name='%s' %(everdata))


            ever_power_error = []
            for i in range(5,125,5):
                integral, error = calculation_ms(Ytest[paras.bptt:, 2] * Y_std[2] + Y_mean[2],
                               y_pred_test[:, 2] * Y_std[2] + Y_mean[2], dttest, i*60)
                ever_power_error.append(error[0][1])
            all_power_error.append(ever_power_error)
            #print(data)
            file.close()

        draw_power_error(all_power_error,datasets,dir_name=test_path)

    if paras.dev:
        for everdata in datasets:
            #验证集
            filename = "predict_result/dev/datafigs_"+everdata+".pkl"
            file=open(os.path.join(paras.save, filename),"rb")
            data=pickle.load(file)
            X_mean = data['X_mean']
            X_std = data['X_std']
            Y_mean = data['Y_mean']
            Y_std = data['Y_std']
            Ytest = data['y_target_dev']
            y_pred_test = data['y_pred_dev']
            Xtest = data['x_dev']

            ttest = data['t_tdev']
            dttest = np.append(ttest[1:] - ttest[:-1], ttest[1] - ttest[0]).reshape(-1, 1)
            test_dfa_state_pred_array = data['sdev']
            error_test = prediction_error(Ytest[paras.bptt:], y_pred_test)
            logging('dataset: %s, Ti: %.2f,Pcooling: %.2f, Pserver: %.2f, ' % (everdata,error_test[0], error_test[1], error_test[2]))
            file.close()

    # if paras.one:
    #     for everdata in datasets:
    #         #验证集,取dev,one，model
    #         filename = "predict_result/dev/epoch_90/datafigs_"+everdata+".pkl"
    #         file=open(os.path.join(paras.save, filename),"rb")
    #         data=pickle.load(file)
    #         X_mean = data['X_mean']
    #         X_std = data['X_std']
    #         Y_mean = data['Y_mean']
    #         Y_std = data['Y_std']
    #         Ytest = data['y_target_dev']
    #         y_pred_test = data['y_pred_dev']
    #         Xtest = data['x_dev']
    #
    #         ttest = data['t_tdev']
    #         dttest = np.append(ttest[1:] - ttest[:-1], ttest[1] - ttest[0]).reshape(-1, 1)
    #         test_dfa_state_pred_array = data['sdev']
    #         visualize_prediction_one( Ytest[paras.bptt:] * Y_std + Y_mean, y_pred_test * Y_std + Y_mean, test_dfa_state_pred_array,Xtest[paras.bptt:,0]* X_std[0] + X_mean[0],
    #                                     dev_path,
    #                                     seg_length=1000, dir_name='%s' %(everdata))# 模型自己预测的
    #
    #         file.close()




    if paras.compare:
        one = '1008-one-model-alld'
        ode_rnn = '1005_ode_rnn'
        h_ode = '1016_wendu_alls_cooling_no_high'#'1005-allp_alld'
        for everdata in datasets:
            #验证集,取dev,one，model
            one_model_file = '../results/'+one+"/predict_result/dev/epoch_90/datafigs_"+everdata+".pkl"

            ode_rnn_file='../results/'+ode_rnn+"/predict_result/dev/datafigs_"+everdata+".pkl"
            h_ode_file = '../results/'+h_ode+"/predict_result/dev/datafigs_"+everdata+".pkl"
            file1=open(one_model_file,"rb")
            data1=pickle.load(file1)
            X_mean = data1['X_mean']
            X_std = data1['X_std']
            Y_mean = data1['Y_mean']
            Y_std = data1['Y_std']
            Ytest = data1['y_target_dev']
            y_pred_test_one = data1['y_pred_dev']
            Xtest = data1['x_dev']

            ttest = data1['t_tdev']
            dttest = np.append(ttest[1:] - ttest[:-1], ttest[1] - ttest[0]).reshape(-1, 1)
            state_one = data1['sdev']
            file1.close()

            file2 = open(ode_rnn_file, "rb")
            data2=pickle.load(file2)
            y_pred_test_rnn = data2['y_pred_dev']
            state_rnn = data2['sdev']
            file2.close()

            file3 = open(h_ode_file, "rb")
            data3=pickle.load(file3)
            y_pred_test_ours = data3['y_pred_dev']
            state_ours = data3['sdev']
            file3.close()
            visualize_prediction_one2(Ytest[paras.bptt:] * Y_std + Y_mean, Ytest[paras.bptt:] * Y_std + Y_mean, state_one,
                                     Xtest[paras.bptt:, 0] * X_std[0] + X_mean[0],
                                     './truth/',
                                     seg_length=1000, dir_name='%s' % (everdata),lablee = 'truth')  # truth

            visualize_prediction_one2( Ytest[paras.bptt:] * Y_std + Y_mean, y_pred_test_one * Y_std + Y_mean,state_one,Xtest[paras.bptt:,0]* X_std[0] + X_mean[0],
                                        './one/',
                                        seg_length=1000, dir_name='%s' %(everdata),lablee = 'one')# one

            visualize_prediction_one2(Ytest[paras.bptt:] * Y_std + Y_mean, y_pred_test_rnn * Y_std + Y_mean, state_rnn,
                                     Xtest[paras.bptt:, 0] * X_std[0] + X_mean[0],
                                     './rnn/',
                                     seg_length=1000, dir_name='%s' % (everdata),lablee = 'rnn')  # rnn

            visualize_prediction_one2(Ytest[paras.bptt:] * Y_std + Y_mean, y_pred_test_ours * Y_std + Y_mean, state_ours,
                                     Xtest[paras.bptt:, 0] * X_std[0] + X_mean[0],
                                     './ours/',
                                     seg_length=1000, dir_name='%s' % (everdata),lablee = 'ours')  # ours
