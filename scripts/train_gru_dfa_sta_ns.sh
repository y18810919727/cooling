#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=5 python main.py --save 0820_2_predict_ns_state4 --model DFA --dfa_yaml dfa_ns
