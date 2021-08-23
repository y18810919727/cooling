#读取测试结果

import pickle
file=open("../results/0816_predict_ns_state4/data4figs.pkl","rb")
data=pickle.load(file)
y_target_test = data['y_target_test']
y_pred_test =data['y_pred_test']
# visualize_prediction(
#                             Ytest[paras.bptt:] * Y_std + Y_mean, t2np(Ytest_pred) * Y_std + Y_mean, test_dfa_state_pred_array,Xtest[paras.bptt:,0]* X_std[0] + X_mean[0],
#                             os.path.join(paras.save, 'predict_seq'),
#                             seg_length=paras.visualization_len, dir_name='visualizations-test-%s/%s' % (str(best_dev_epoch) , everdata))# 模型自己预测的
print(data)
file.close()