#读取测试结果

import pickle
import os
import argparse
import numpy as np


from util import SimpleLogger, show_data, init_weights, array_operate_with_nan, get_Dataset, visualize_prediction, t2np, draw_table,draw_table_all,calculation_ms
parser = argparse.ArgumentParser(description='Models for Continuous Stirred Tank dataset')
parser.add_argument("--save", type=str, default='test', help="experiment logging folder")
parser.add_argument("--bptt", type=int, default=800, help="bptt")
parser.add_argument("--powertime", type=int, default=1800, help="powertime")

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

if __name__ == '__main__':

    paras = parser.parse_args()
    paras.save = os.path.join('../results', paras.save)  # 路径拼接，改变paras.save为'results/tmp'
    datasets=['Data_train_1_7_1','Data_train_1_8k','Data_train_3_8k','Data_train_4_2k','Data_validate']
    all_sqe_nums = {}
    len_sqe = []
    result_path = os.path.join(paras.save, 'predict_test')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    file = open("../results/tmp1/1.pkl", "rb")
    data = pickle.load(file)
    X_mean = data['X_mean']
    X_std = data['X_std']
    dttest = data['dttest']
    error_all_mae = []
    error_all_mape = []
    test_rres = []
    for everdata in datasets:


        filename = "predict_result/datafigs_"+everdata+".pkl"
        file=open(os.path.join(paras.save, filename),"rb")
        data=pickle.load(file)

        Ytest = data['y_target_test']
        y_pred_test = data['y_pred_test']
        Xtest = data['x_test']
        Y_mean = data['Y_mean']
        Y_std = data['Y_std']
        ttest = data['t_test']
        dttest = np.append(ttest[1:] - ttest[:-1], ttest[1] - ttest[0]).reshape(-1, 1)
        test_dfa_state_pred_array = data['test_dfa_state_pred_array']
        visualize_prediction( Ytest[paras.bptt:] * Y_std + Y_mean, y_pred_test * Y_std + Y_mean, test_dfa_state_pred_array,Xtest[paras.bptt:,0]* X_std[0] + X_mean[0],
                                    result_path,
                                    seg_length=2000, dir_name='%s' %(everdata))# 模型自己预测的

        integral, error = calculation_ms(Ytest[paras.bptt:, 2] * Y_std[2] + Y_mean[2],
                                     y_pred_test[:, 2] * Y_std[2] + Y_mean[2], dttest, paras.powertime)
        if (int(len(integral[0])) != 0):
            draw_table(everdata, integral, error,  paras.powertime, result_path,dir_name='%s' %(everdata))
        error_all_mae.append(str(error[0][0]) + " ± " + str(error[1][0]))
        error_all_mape.append(str(error[0][1]) + " ± " + str(error[1][1]))
        error_test = prediction_error(Ytest[paras.bptt:], y_pred_test)
        test_rres.append(error_test.tolist())
        #print(data)
        file.close()

    error_all = [error_all_mae, error_all_mape]
    draw_table_all(datasets, error_all,[test_rres],result_path)