#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=3 python main.py  --save  1005_ode_rnn  --model DFA --dfa_yaml ode_rnn --mymodel rnn

