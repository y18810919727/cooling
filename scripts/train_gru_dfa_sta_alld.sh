#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=1 python main.py --save 0722_predict_alld --model DFA --dfa_yaml dfa_alld
