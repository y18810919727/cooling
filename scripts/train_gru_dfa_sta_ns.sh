#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=1 python main.py --save  0519_new_ode_ns   --model DFA --dfa_yaml dfa_ns
