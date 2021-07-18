#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=3 python main.py --save 0716_predict_diff --model DFA --dfa_yaml dfa_diff
