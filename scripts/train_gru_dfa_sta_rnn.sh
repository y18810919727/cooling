#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=7 python main.py  --save  1003_ode_rnn_not_t  --model DFA --dfa_yaml ode_rnn --mymodel rnn

