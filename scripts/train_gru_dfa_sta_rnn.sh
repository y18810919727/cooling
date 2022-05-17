#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=1 python main.py  --save  0506_ode_rnn  --model DFA --dfa_yaml ode_rnn --mymodel rnn

