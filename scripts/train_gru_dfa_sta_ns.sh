#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=4 python main.py --save 0717_predict_ns --model DFA --dfa_yaml dfa_ns