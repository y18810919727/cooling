#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=0 python main.py  --save  ode_rnn  --aj_yaml ode_rnn --mymodel rnn

