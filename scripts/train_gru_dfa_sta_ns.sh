#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=3 python main.py --save 0802_predict_ns_state4 --model DFA --dfa_yaml dfa_ns