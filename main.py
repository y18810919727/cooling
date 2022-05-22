import numpy as np
import torch
import argparse
import os
import pickle
import sys
import traceback
import shutil
from time import time
from tensorboard_logger import configure, log_value
from aj_ode.model_aj import AJ_MIMO
from aj_ode.train import EpochTrainer
from util import SimpleLogger, init_weights, array_operate_with_nan, get_Dataset, visualize_prediction, t2np, \
    draw_table, draw_table_all, calculation_ms,get_Dataset_one
from datetime import timezone, timedelta,datetime


GPU = torch.cuda.is_available()
parent = os.path.dirname(sys.path[0])
sys.path.append(parent)
"""
potentially varying input parameters
"""
parser = argparse.ArgumentParser(description='Models for Continuous Stirred Tank dataset')
# model architecture
parser.add_argument("--k_h", type=int, default=20, help="dimension of hidden state")
# in case method == 'variable'
RKchoices = ['euler', 'midpoint', 'rk4']
parser.add_argument("--scheme", type=str, default='euler', choices=RKchoices, help='Runge-Kutta training scheme')
# training
parser.add_argument("--batch_size", type=int, default=4000, help="batch size")
parser.add_argument("--visualization_len", type=int, default=2000, help="The length of visualization.")
parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
parser.add_argument("--bptt", type=int, default=800, help="bptt")
parser.add_argument("--l2", type=float, default=0., help="L2 regularization")
parser.add_argument("--save", type=str, default='results', help="experiment logging folder")
parser.add_argument("--eval_epochs", type=int, default=5, help="validation every so many epochs")
parser.add_argument("--seed", type=int, default=None, help="random seed")
parser.add_argument("--powertime", type=int, default=1800, help="powertime")

# ODE_AJ
parser.add_argument("--aj_yaml", type=str, default='aj_ns', help="The setting of aj ode")
parser.add_argument("--aj_known", action="store_true",
                    help="The states of aj for each positions are known in test and evaluation.")
# during development
parser.add_argument("--reset", action="store_true", help="reset even if same experiment already finished")
parser.add_argument("--debug", action="store_true", help="debug mode, for acceleration")

#files = ['P-1.7k.csv','P-3.8K.csv','P-4.2k.csv','P-6.3k.csv']
parser.add_argument("--datasets_folder", type=str, default='./data')
parser.add_argument("--datasets", type=list, default=['P-1.7k','P-3.8k','P-6.3k'], help="datasets")
parser.add_argument("--mymodel", type=str, default='merge', choices=['merge', 'rnn', 'one'])
paras = parser.parse_args()
hard_reset = paras.reset

datasets_folder = paras.datasets_folder
# if paras.save already exists and contains log.txt:
# reset if not finished, or if hard_reset
paras.save = os.path.join('results', paras.save)
log_file = os.path.join(paras.save, 'log.txt')
if os.path.isfile(log_file):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        completed = 'Finished' in content
        if 'tmp' not in paras.save and completed and not hard_reset:
            print('Exit; already completed and no hard reset asked.')
            sys.exit()  # do not overwrite folder with current experiment
        else:  # reset folder
            shutil.rmtree(paras.save, ignore_errors=True)



# setup logging
logging = SimpleLogger(log_file) #log to file
configure(paras.save) #tensorboard logging
logging('Args: {}'.format(paras))





"""
fixed input parameters
"""
GPU = torch.cuda.is_available()
logging('Using GPU?', GPU)

# set random seed for reproducibility
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

datasets = paras.datasets
all_sqe_nums = {}
len_sqe = []
if paras.debug: # Using short dataset for acceleration
    Xtrain, Ytrain, ttrain, strain = [df.to_numpy(dtype=np.float32) for df in get_Dataset(datasets_folder+'/train_P-1.7k.csv')]
    Xdev, Ydev, tdev, sdev = [df.to_numpy(dtype=np.float32) for df in get_Dataset(datasets_folder+'/P-1.7k.csv')]
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
            X, Y, t, s = [df.to_numpy(dtype=np.float32) for df in get_Dataset_one(datasets_folder+'/train_' + everdata + '.csv')]
        else:
            X, Y, t, s = [df.to_numpy(dtype=np.float32) for df in get_Dataset(datasets_folder+'/train_'+everdata+'.csv')]
        Xtrain = np.append(Xtrain,X).reshape(-1, 2)
        Ytrain = np.append(Ytrain,Y).reshape(-1, 3)
        ttrain = np.append(ttrain,t).reshape(-1, 1)
        strain = np.append(strain,s).reshape(-1, 1)
        len_sqe.append(int(Xtrain.size / 2))
        all_sqe_nums[everdata] = len_sqe


dttrain = np.append(ttrain[1:] - ttrain[:-1], ttrain[1]-ttrain[0]).reshape(-1, 1)
dttrain[dttrain < -0.1] = 0.1


k_in = Xtrain.shape[1]
k_out = Ytrain.shape[1]


Ntrain = Xtrain.shape[0]


"""
evaluation function
RRSE error
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
    results = [prediction_error(truth[:, i], prediction[:, i]) for i in range(truth.shape[-1])]
    return np.mean(results)



Y_mean, Y_std = array_operate_with_nan(Ytrain, np.mean), array_operate_with_nan(Ytrain, np.std)
Ytrain = (Ytrain - Y_mean) / Y_std

X_mean, X_std = array_operate_with_nan(Xtrain, np.mean), array_operate_with_nan(Xtrain, np.std)
Xtrain = (Xtrain - X_mean) / X_std
dt_mean = np.mean(dttrain)




# Generating a new model
import yaml
fs = open('./aj_ode/transformations/{}.yaml'.format(paras.aj_yaml), encoding='UTF-8', mode='r')
aj_setting = yaml.load(fs, Loader=yaml.FullLoader)

model = AJ_MIMO(aj_setting['ode_nums'], 1, k_in, k_out, paras.k_h, y_mean=Y_mean, y_std=Y_std,
                 odes_para=aj_setting['odes'],
                 ode_2order=aj_setting['ode_2order'],
                 transformations=aj_setting['transformations'],
                 state_transformation_predictor=aj_setting['predictors'], cell_type=paras.mymodel,
                 Ly_share=aj_setting['Ly_share'],
                scheme=paras.scheme)

model.apply(init_weights)




if GPU:
    model = model.cuda()

params = sum([np.prod(p.size()) for p in model.parameters()])
logging('\n(scheme %s) with %d trainable parameters' % (paras.scheme, params))
for n, p in model.named_parameters():
    p_params = np.prod(p.size())
    print('\t%s\t%d (cuda: %s)'%(n, p_params, str(p.is_cuda)))

logging('Architecture: ', model)
log_value('model/params', params, 0)

optimizer = torch.optim.Adam(model.parameters(), lr=paras.lr, weight_decay=paras.l2)



trainer = EpochTrainer(model, optimizer, paras.epochs, Xtrain, Ytrain, strain, dttrain,
                       batch_size=paras.batch_size, gpu=GPU, bptt=paras.bptt,all_sqe_nums=all_sqe_nums, save_dir=paras.save,
                       logging=logging, debug=paras.debug,mymodel = paras.mymodel)  #dttrain ignored for all but 'variable' methods
t00 = time()
best_dev_error = 1.e5
best_dev_epoch = 1
error_test = -1

max_epochs_no_decrease = 30  # If error in dev does not decrease in long time, the training will be paused early.


try:  # catch error and redirect to logger

    for epoch in range(1, paras.epochs + 1):
        mse_train = trainer(epoch)
        if epoch % paras.eval_epochs == 0:
            with torch.no_grad():
                model.eval()
                error_dev_sum = 0
                # corresponding test result:
                for everdata in datasets:
                    if paras.mymodel == 'one':
                        X, Y, t, s = [df.to_numpy(dtype=np.float32) for df in
                                      get_Dataset_one(datasets_folder+'/train_' + everdata + '.csv')]
                    else:
                        X, Y, t, s = [df.to_numpy(dtype=np.float32) for df in
                                      get_Dataset(datasets_folder+'/train_' + everdata + '.csv')]
                    if paras.mymodel == 'one':
                        Xtrain, Ytrain, ttrain, strain = [df.to_numpy(dtype=np.float32) for df in get_Dataset_one(datasets_folder+'/train_'+everdata+'.csv')]
                        Xdev, Ydev, tdev, sdev = [df.to_numpy(dtype=np.float32) for df in get_Dataset_one(datasets_folder+'/'+everdata+'.csv')]
                    else:
                        Xtrain, Ytrain, ttrain, strain = [df.to_numpy(dtype=np.float32) for df in
                                                          get_Dataset(datasets_folder+'/train_' + everdata + '.csv')]
                        Xdev, Ydev, tdev, sdev = [df.to_numpy(dtype=np.float32) for df in
                                                  get_Dataset(datasets_folder+'/' + everdata + '.csv')]

                    dttrain = np.append(ttrain[1:] - ttrain[:-1], ttrain[1] - ttrain[0]).reshape(-1,1)
                    dtdev = np.append(tdev[1:] - tdev[:-1], tdev[1] - tdev[0]).reshape(-1, 1)
                    Ytrain, Ydev = [(Y - Y_mean) / Y_std for Y in [Ytrain, Ydev]]
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

                    if not os.path.exists('{}/predict_seq'.format(paras.save)):
                        os.mkdir('{}/predict_seq'.format(paras.save))
                    train_path = os.path.join(paras.save, 'predict_seq/visualizations-train-{}'.format(best_dev_epoch))
                    if not os.path.exists(train_path):
                        os.mkdir(train_path)

                    visualize_prediction(
                        Ytrain[paras.bptt:] * Y_std + Y_mean, t2np(Ytrain_pred) * Y_std + Y_mean, strain[paras.bptt:],Xtrain[paras.bptt:,0]* X_std[0] + X_mean[0],
                        os.path.join(paras.save, 'predict_seq'),
                        seg_length=paras.visualization_len, dir_name='visualizations-train-%s/%s' % (str(best_dev_epoch), everdata))

                    Ydev_pred, hdev_pred = model.encoding_plus_predict(
                        Xdev_tn,  dtdev_tn,  Ydev_tn[:, :paras.bptt], sdev_tn[:, :paras.bptt], paras.bptt,
                        sdev_tn[:, paras.bptt:])
                    mse_dev = model.criterion(Ydev_pred, Ydev_tn[:, paras.bptt:]).item()
                    error_dev = prediction_error(Ydev[paras.bptt:], t2np(Ydev_pred))
                    error_dev_sum += error_dev
                    logging(
                        'epoch %04d |%s| rrse %d '%(epoch, everdata,error_dev))
                    dev_path = os.path.join(paras.save, 'predict_seq/visualizations-dev-{}'.format(epoch))
                    if not os.path.exists(dev_path):
                        os.mkdir(dev_path)
                    visualize_prediction(
                        Ydev[paras.bptt:] * Y_std + Y_mean, t2np(Ydev_pred) * Y_std + Y_mean, sdev[paras.bptt:],Xdev[paras.bptt:,0]* X_std[0] + X_mean[0],
                        os.path.join(paras.save, 'predict_seq'),
                        seg_length=paras.visualization_len, dir_name='visualizations-dev-%s/%s' %( str(epoch), everdata))
                logging('epoch %04d| rrse_all %d ' % (epoch, error_dev_sum))
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
                                                  get_Dataset_one(datasets_folder+'/'+everdata+'.csv')]
                        else:
                            Xtest, Ytest, ttest, stest = [df.to_numpy(dtype=np.float32) for df in
                                                          get_Dataset(datasets_folder+'/' + everdata + '.csv')]
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
                            Xtest_tn,  dttest_tn,  Ytest_tn[:, :paras.bptt], stest_tn[:, :paras.bptt], paras.bptt,
                            stest_tn[:, paras.bptt:] if paras.aj_known else None)
                        test_aj_state_pred_array = model.select_aj_states(test_state_pred[0]).int().detach().cpu().numpy()
                        predict_path = os.path.join(paras.save, 'predict_seq/visualizations-test-{}'.format(epoch))
                        if not os.path.exists(predict_path):
                            os.mkdir(predict_path)
                        visualize_prediction(
                            Ytest[paras.bptt:] * Y_std + Y_mean, t2np(Ytest_pred) * Y_std + Y_mean, test_aj_state_pred_array,Xtest[paras.bptt:,0]* X_std[0] + X_mean[0],
                            os.path.join(paras.save, 'predict_seq'),
                            seg_length=paras.visualization_len, dir_name='visualizations-test-%s/%s' % (str(best_dev_epoch) , everdata))

                        integral,error = calculation_ms(Ytest[paras.bptt:, 2] * Y_std[2] + Y_mean[2],
                                          t2np(Ytest_pred)[:, 2] * Y_std[2] + Y_mean[2],dttest,paras.powertime)

                        if (int(len(integral[0])) != 0):
                            draw_table(everdata, integral, error, paras.powertime,  os.path.join(paras.save, 'predict_seq'), dir_name='visualizations-test-%s/%s' % (str(best_dev_epoch) , everdata))
                        error_all_mae.append(str(error[0][0])+" ± "+str(error[1][0]))
                        error_all_mape.append(str(error[0][1])+" ± "+str(error[1][1]))
                    torch.save(model, os.path.join(paras.save, 'best_dev_model.pt'))
                    error_all=[error_all_mae,error_all_mape]
                    draw_table_all(datasets, error_all , os.path.join(paras.save, 'predict_seq'))
                elif epoch - best_dev_epoch > max_epochs_no_decrease:
                    logging('Development error did not decrease over %d epochs -- quitting.'%max_epochs_no_decrease)
                    break


    log_value('finished/best_dev_error', best_dev_error, 0)
    log_value('finished/corresp_test_error', error_test_sum, 0)

    logging('Finished: best dev error', best_dev_error,
              'at epoch', best_dev_epoch,
              'with corresp. test error', error_test_sum)

except:
    var = traceback.format_exc()
    logging(var)
