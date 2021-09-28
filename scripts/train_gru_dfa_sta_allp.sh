#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=7 python main.py --save 0928_allp_alld_ode_rnn2  --model DFA --dfa_yaml dfa_allp

