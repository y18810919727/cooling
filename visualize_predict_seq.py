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
from util import SimpleLogger, show_data, init_weights, array_operate_with_nan, process_dataset

import pandas as pd


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




def get_Dataset(path):
    df = pd.read_csv(path)
    df = process_dataset(df)
    return df[['Pserver', 'Tr']], df[['Ti', 'Pcooling', 'Power cooling']], df[['time']], df[['states']]

Xtrain, Ytrain, ttrain, strain = [df.to_numpy(dtype=np.float32) for df in get_Dataset('./data/Data_train.csv')]
Xdev, Ydev, tdev, sdev = [df.to_numpy(dtype=np.float32) for df in get_Dataset('./data/Data_validate_short.csv')]

dttrain = np.append(ttrain[1:] - ttrain[:-1], ttrain[1]-ttrain[0]).reshape(-1, 1)
dtdev = np.append(tdev[1:] - tdev[:-1], tdev[1]-tdev[0]).reshape(-1, 1)

Xtest, Ytest, ttest, dttest, stest = Xdev, Ydev, tdev, dtdev, sdev


k_in = Xtrain.shape[1]
k_out = Ytrain.shape[1]

Ndev = Xdev.shape[0]
Ntest = Xtest.shape[0]
Ntrain = Xtrain.shape[0]

N = Ndev + Ntest + Ntrain


"""
evaluation function
RRSE error
"""




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


Y_mean, Y_std = array_operate_with_nan(Ytrain, np.mean), array_operate_with_nan(Ytrain, np.std)

from util import visualize_prediction


if __name__ == '__main__':
    assert len(sys.argv) >= 3
    base_path = os.path.join('results', sys.argv[1], 'predict_seq')
    Y_merge = np.load(os.path.join(base_path, '%s.npy' % sys.argv[2]))
    Y_test, Y_pred = Y_merge[0], Y_merge[1]

    visualize_prediction(
        Y_test * Y_std + Y_mean, Y_pred * Y_std + Y_mean, stest[-len(Y_pred):], base_path,
        seg_length=2000, dir_name='visualizations-%s' % sys.argv[2])



