import os
import sys
import numpy as np
from time import time
import torch
GPU = torch.cuda.is_available()

parent = os.path.dirname(sys.path[0])#os.getcwd())
sys.path.append(parent)
from taho.model import MIMO, GRUCell, HOGRUCell, IncrHOGRUCell, HOARNNCell, IncrHOARNNCell
from dfa_ode.model_dfa import DFA_MIMO
from dfa_ode.train import EpochTrainer
from util import SimpleLogger, show_data, init_weights, array_operate_with_nan, process_dataset, visualize_prediction

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


parser.add_argument("--test", action="store_true", help="Testing model in para.save")
parser.add_argument("--time_aware", type=str, default='variable', choices=['no', 'input', 'variable'], help=methods)
parser.add_argument("--model", type=str, default='GRU', choices=['GRU', 'GRUinc', 'ARNN', 'ARNNinc', 'DFA'])
parser.add_argument("--interpol", type=str, default='constant', choices=['constant', 'linear'])


parser.add_argument("--gamma", type=float, default=1.0, help="diffusion parameter ARNN model")
parser.add_argument("--step_size", type=float, default=1.0, help="fixed step size parameter in the ARNN model")


# data
parser.add_argument("--missing", type=float, default=0.0, help="fraction of missing samples (0.0 or 0.5)")

# model architecture
parser.add_argument("--k_state", type=int, default=20, help="dimension of hidden state")

# in case method == 'variable'
RKchoices = ['Euler', 'Midpoint', 'Kutta3', 'RK4']
parser.add_argument("--scheme", type=str, default='Euler', choices=RKchoices, help='Runge-Kutta training scheme')

# training
parser.add_argument("--batch_size", type=int, default=4096, help="batch size")
parser.add_argument("--visualization_len", type=int, default=2000, help="The length of visualization.")
parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
parser.add_argument("--bptt", type=int, default=400, help="bptt")
parser.add_argument("--dropout", type=float, default=0., help="drop prob")
parser.add_argument("--l2", type=float, default=0., help="L2 regularization")


# admin
parser.add_argument("--save", type=str, default='results', help="experiment logging folder")
parser.add_argument("--eval_epochs", type=int, default=10, help="validation every so many epochs")
parser.add_argument("--seed", type=int, default=None, help="random seed")



# ODE_DFA
parser.add_argument("--dfa_yaml", type=str, default='dfa1', help="The setting of dfa ode")
parser.add_argument("--dfa_known", action="store_true",
                    help="The states of dfa for each positions are known in test and evaluation.")

parser.add_argument("--linear_decoder", action="store_true", help="Type of Ly")

# during development
parser.add_argument("--reset", action="store_true", help="reset even if same experiment already finished")
parser.add_argument("--short_encoding", action="store_true", help="Encoding short sequences to generate state0")
parser.add_argument("--debug", action="store_true", help="debug mode, for acceleration")


paras = parser.parse_args()

hard_reset = paras.reset
# if paras.save already exists and contains log.txt:
# reset if not finished, or if hard_reset
paras.save = os.path.join('results', paras.save)
if paras.test:
    model_test_path = os.path.join(paras.save, 'best_dev_model.pt')
    paras.save = os.path.join(paras.save, 'test')
    if not os.path.exists(paras.save):
        os.mkdir(paras.save)
    paras.eval_epochs = 1
    paras.epochs = 1

log_file = os.path.join(paras.save, 'log.txt')
if os.path.isfile(log_file) and not paras.test:
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
frac_dev = 15/100
frac_test = 15/100

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


def get_Dataset(path):
    df = pd.read_csv(path)
    df = process_dataset(df)
    return df[['Pserver', 'Tr']], df[['Ti', 'Pcooling', 'Power cooling']], df[['time']], df[['states']]

if paras.debug:
    Xtrain, Ytrain, ttrain, strain = [df.to_numpy(dtype=np.float32) for df in get_Dataset('./data/Data_train_debug.csv')]
    Xdev, Ydev, tdev, sdev = [df.to_numpy(dtype=np.float32) for df in get_Dataset('./data/Data_validate_short_debug.csv')]
else:
    # Xtrain, Ytrain, ttrain, strain = [df.to_numpy(dtype=np.float32) for df in get_Dataset('./data/Data_train.csv')]
    # Xdev, Ydev, tdev, sdev = [df.to_numpy(dtype=np.float32) for df in get_Dataset('./data/Data_validate_short.csv')]

    Xtrain, Ytrain, ttrain, strain = [df.to_numpy(dtype=np.float32) for df in get_Dataset('./data/train.csv')]
    Xdev, Ydev, tdev, sdev = [df.to_numpy(dtype=np.float32) for df in get_Dataset('./data/validate.csv')]

dttrain = np.append(ttrain[1:] - ttrain[:-1], ttrain[1]-ttrain[0]).reshape(-1, 1)
dtdev = np.append(tdev[1:] - tdev[:-1], tdev[1]-tdev[0]).reshape(-1, 1)

Xtest, Ytest, ttest, dttest, stest = Xdev, Ydev, tdev, dtdev, sdev


k_in = Xtrain.shape[1]
k_out = Ytrain.shape[1]

Ndev = Xdev.shape[0]
Ntest = Xtest.shape[0]
Ntrain = Xtrain.shape[0]

N = Ndev + Ntest + Ntrain

logging('first {} for training, then {} for development and {} for testing'.format(Ntrain, Ndev, Ntest))

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
    # each shape (sequence, n_outputs)
    # Root Relative Squared Error
    results = [prediction_error(truth[:, i], prediction[:, i]) for i in range(truth.shape[-1])]
    return np.mean(results)


"""
- model:
    GRU (compensated GRU to avoid linear increase of state; has standard GRU as special case for Euler scheme and equidistant data)
    GRUinc (incremental GRU, for baseline only)
- time_aware:
    no: ignore uneven spacing: for GRU use original GRU implementation
    input: use normalized next interval size as extra input feature
    variable: time-aware implementation
"""

#time_aware options

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

Y_mean, Y_std = array_operate_with_nan(Ytrain, np.mean), array_operate_with_nan(Ytrain, np.std)
Ytrain, Ydev, Ytest = [(Y - Y_mean) / Y_std for Y in [Ytrain, Ydev, Ytest]]

X_mean, X_std = array_operate_with_nan(Xtrain, np.mean), array_operate_with_nan(Xtrain, np.std)
Xtrain, Xdev, Xtest = [(X - X_mean) / X_std for X in [Xtrain, Xdev, Xtest]]

if paras.time_aware == 'no' or paras.time_aware == 'input':
    # in case 'input': variable intervals already in input X;
    # now set actual time intervals to 1 (else same effect as time_aware == 'variable')
    dttrain = np.ones(dttrain.shape)
    dtdev = np.ones(dtdev.shape)
    dttest = np.ones(dttest.shape)


dt_mean = np.mean(dttrain)
# set model:
if not paras.test:
    if paras.model == 'DFA':
        import yaml
        fs = open('./dfa_ode/transformations/{}.yaml'.format(paras.dfa_yaml), encoding='UTF-8', mode='r')
        dfa_setting = yaml.load(fs, Loader=yaml.FullLoader)
        model = DFA_MIMO(dfa_setting['ode_nums'], 1, k_in, k_out, paras.k_state, y_mean=Y_mean, y_std=Y_std,
                         odes_para=dfa_setting['odes'],
                         ode_2order=dfa_setting['ode_2order'],
                         transformations=dfa_setting['transformations'],
                         state_transformation_predictor=dfa_setting['predictors'], cell_type='merge',
                         Ly_share=dfa_setting['Ly_share'])

    else:

        if paras.model == 'GRU':
            cell_factory = GRUCell if paras.time_aware == 'no' else HOGRUCell
        elif paras.model == 'GRUinc':
            cell_factory = IncrHOGRUCell
        elif paras.model == 'ARNN':
            cell_factory = HOARNNCell
        elif paras.model == 'ARNNinc':
            cell_factory = IncrHOARNNCell
        else:
            raise NotImplementedError('unknown model type ' + paras.model)
        model = MIMO(k_in, k_out, paras.k_state, dropout=paras.dropout, cell_factory=cell_factory, meandt=dt_mean,
                     train_scheme=paras.scheme, eval_scheme=paras.scheme, gamma=paras.gamma, step_size=paras.step_size,
                     interpol=paras.interpol
                     )
    model.apply(init_weights)
else:
    model = torch.load(model_test_path)


if GPU:
    model = model.cuda()

params = sum([np.prod(p.size()) for p in model.parameters()])
logging('\nModel %s (time_aware: %s, scheme %s) with %d trainable parameters' % (paras.model, paras.time_aware, paras.scheme, params))
for n, p in model.named_parameters():
    p_params = np.prod(p.size())
    print('\t%s\t%d (cuda: %s)'%(n, p_params, str(p.is_cuda)))

logging('Architecture: ', model)
log_value('model/params', params, 0)

optimizer = torch.optim.Adam(model.parameters(), lr=paras.lr, weight_decay=paras.l2)


# prepare tensors for evaluation
Xtrain_tn = torch.tensor(Xtrain, dtype=torch.float).unsqueeze(0)  # (1, Ntrain, k_in)
Ytrain_tn = torch.tensor(Ytrain, dtype=torch.float).unsqueeze(0)  # (1, Ntrain, k_out)
dttrain_tn = torch.tensor(dttrain, dtype=torch.float).unsqueeze(0)  # (1, Ntrain, 1)
strain_tn = torch.tensor(strain, dtype=torch.int).unsqueeze(0)  # (1, Ntrain, 1)

Xdev_tn = torch.tensor(Xdev, dtype=torch.float).unsqueeze(0)  # (1, Ndev, k_in)
Ydev_tn = torch.tensor(Ydev, dtype=torch.float).unsqueeze(0)  # (1, Ndev, k_out)
dtdev_tn = torch.tensor(dtdev, dtype=torch.float).unsqueeze(0)  # (1, Ndev, 1)
sdev_tn = torch.tensor(sdev, dtype=torch.int).unsqueeze(0)  # (1, Ntrain, 1)

Xtest_tn = torch.tensor(Xtest, dtype=torch.float).unsqueeze(0)
Ytest_tn = torch.tensor(Ytest, dtype=torch.float).unsqueeze(0)
dttest_tn = torch.tensor(dttest, dtype=torch.float).unsqueeze(0)
stest_tn = torch.tensor(stest, dtype=torch.int).unsqueeze(0)  # (1, Ntrain, 1)

if GPU:
    Xtrain_tn = Xtrain_tn.cuda()
    Ytrain_tn = Ytrain_tn.cuda()
    dttrain_tn = dttrain_tn.cuda()
    Xdev_tn = Xdev_tn.cuda()
    Ydev_tn = Ydev_tn.cuda()
    dtdev_tn = dtdev_tn.cuda()
    Xtest_tn = Xtest_tn.cuda()
    Ytest_tn = Ytest_tn.cuda()
    dttest_tn = dttest_tn.cuda()

    strain_tn, sdev_tn, stest_tn = [s.cuda() for s in [strain_tn, sdev_tn, stest_tn]]


def t2np(tensor):
    return tensor.squeeze(dim=0).detach().cpu().numpy()


trainer = EpochTrainer(model, optimizer, paras.epochs, Xtrain, Ytrain, strain, dttrain,
                       batch_size=paras.batch_size, gpu=GPU, bptt=paras.bptt, save_dir=paras.save,
                       logging=logging, debug=paras.debug)  #dttrain ignored for all but 'variable' methods

t00 = time()

best_dev_error = 1.e5
best_dev_epoch = 0
error_test = -1

max_epochs_no_decrease = 30

try:  # catch error and redirect to logger

    for epoch in range(1, paras.epochs + 1):

        #train 1 epoch
        if not paras.test:
            mse_train = trainer(epoch)
        else:
            mse_train = 0


        if epoch % paras.eval_epochs == 0:
            with torch.no_grad():

                model.eval()
                # (1) forecast on train data steps
                # Ytrain_pred, htrain_pred = model(Xtrain_tn, dt=dttrain_tn)

                Ytrain_pred, htrain_pred = model.encoding_plus_predict(
                    Xtrain_tn,  dttrain_tn,  Ytrain_tn[:, :paras.bptt], strain_tn[:, :paras.bptt], paras.bptt,
                    strain_tn[:, paras.bptt:])
                error_train = prediction_error(Ytrain[paras.bptt:], t2np(Ytrain_pred))

                if not os.path.exists('{}/predict_seq'.format(paras.save)):
                    os.mkdir('{}/predict_seq'.format(paras.save))
                visualize_prediction(
                    Ytrain[paras.bptt:] * Y_std + Y_mean, t2np(Ytrain_pred) * Y_std + Y_mean, strain[paras.bptt:],
                    os.path.join(paras.save, 'predict_seq'),
                    seg_length=paras.visualization_len, dir_name='visualizations-train-%s' % str(best_dev_epoch))

                # (2) forecast on dev data
                # Ydev_pred, hdev_pred = model(Xdev_tn, state0=htrain_pred[:, -1, :], dt=dtdev_tn)

                Ydev_pred, hdev_pred = model.encoding_plus_predict(
                    Xdev_tn,  dtdev_tn,  Ydev_tn[:, :paras.bptt], sdev_tn[:, :paras.bptt], paras.bptt,
                    sdev_tn[:, paras.bptt:])

                mse_dev = model.criterion(Ydev_pred, Ydev_tn[:, paras.bptt:]).item()
                error_dev = prediction_error(Ydev[paras.bptt:], t2np(Ydev_pred))

                # report evaluation results
                log_value('train/mse', mse_train, epoch)
                log_value('train/error', error_train, epoch)
                log_value('dev/loss', mse_dev, epoch)
                log_value('dev/error', error_dev, epoch)

                logging('epoch %04d | loss %.3f (train), %.3f (dev) | error %.3f (train), %.3f (dev) | tt %.2fmin'%
                        (epoch, mse_train, mse_dev, error_train, error_dev, (time()-t00)/60.))
                show_data(ttrain[paras.bptt:], Ytrain[paras.bptt:], t2np(Ytrain_pred), paras.save, 'current_trainresults',
                               msg='train results (train error %.3f) at iter %d' % (error_train, epoch))
                show_data(tdev[paras.bptt:], Ydev[paras.bptt:], t2np(Ydev_pred), paras.save, 'current_devresults',
                               msg='dev results (dev error %.3f) at iter %d' % (error_dev, epoch))

                # update best dev model
                if error_dev < best_dev_error:
                    best_dev_error = error_dev
                    best_dev_epoch = epoch
                    log_value('dev/best_error', best_dev_error, epoch)

                    #corresponding test result:

                    Ytest_pred, test_state_pred = model.encoding_plus_predict(
                        Xtest_tn,  dttest_tn,  Ytest_tn[:, :paras.bptt], stest_tn[:, :paras.bptt], paras.bptt,
                        stest_tn[:, paras.bptt:] if paras.dfa_known else None)
                    # mse_test = model.criterion(Ytest_pred, Ytest_tn[paras.bptt:]).item()
                    error_test = prediction_error(Ytest[paras.bptt:], t2np(Ytest_pred))
                    test_dfa_state_pred_array = model.select_dfa_states(test_state_pred[0]).int().detach().cpu().numpy()

                    log_value('test/corresp_error', error_test, epoch)
                    logging('new best dev error %.3f'%best_dev_error)

                    np.save('{}/predict_seq/test_result_{}'.format(paras.save, epoch),
                            np.stack([Ytest[paras.bptt:], t2np(Ytest_pred)]))

                    np.save('{}/predict_seq/dev_result_{}'.format(paras.save, epoch),
                            np.stack([Ydev[paras.bptt:], t2np(Ydev_pred)]))

                    visualize_prediction(
                        Ytest[paras.bptt:] * Y_std + Y_mean, t2np(Ytest_pred) * Y_std + Y_mean, test_dfa_state_pred_array,
                        os.path.join(paras.save, 'predict_seq'),
                        seg_length=paras.visualization_len, dir_name='visualizations-test-%s' % str(best_dev_epoch))

                    visualize_prediction(
                        Ydev[paras.bptt:] * Y_std + Y_mean, t2np(Ydev_pred) * Y_std + Y_mean, stest[paras.bptt:],
                        os.path.join(paras.save, 'predict_seq'),
                        seg_length=paras.visualization_len, dir_name='visualizations-dev-%s' % str(best_dev_epoch))

                    # make figure of best model on train, dev and test set for debugging
                    show_data(tdev, Ydev, t2np(Ydev_pred), paras.save, 'best_dev_devresults',
                              msg='dev results (dev error %.3f) at iter %d' % (error_dev, epoch))
                    show_data(ttest, Ytest, t2np(Ytest_pred), paras.save,
                              'best_dev_testresults',
                              msg='test results (test error %.3f) at iter %d (=best dev)' % (error_test, epoch))

                    # save model
                    #torch.save(model.state_dict(), os.path.join(paras.save, 'best_dev_model_state_dict.pt'))
                    torch.save(model, os.path.join(paras.save, 'best_dev_model.pt'))

                    # save dev and test predictions of best dev model
                    pickle.dump({'t_dev': tdev, 'y_target_dev': Ydev, 'y_pred_dev': t2np(Ydev_pred),
                                 't_test': ttest, 'y_target_test': Ytest, 'y_pred_test': t2np(Ytest_pred)},
                                open(os.path.join(paras.save, 'data4figs.pkl'), 'wb'))

                elif epoch - best_dev_epoch > max_epochs_no_decrease:
                    logging('Development error did not decrease over %d epochs -- quitting.'%max_epochs_no_decrease)
                    break

    log_value('finished/best_dev_error', best_dev_error, 0)
    log_value('finished/corresp_test_error', error_test, 0)

    logging('Finished: best dev error', best_dev_error,
              'at epoch', best_dev_epoch,
              'with corresp. test error', error_test)




except:
    var = traceback.format_exc()
    logging(var)
