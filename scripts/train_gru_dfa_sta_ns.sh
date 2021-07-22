#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=4 python main.py --save 0722_predict_ns_state4 --model DFA --dfa_yaml dfa_ns