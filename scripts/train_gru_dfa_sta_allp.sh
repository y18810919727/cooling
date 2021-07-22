#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=4 python main.py --save 0718_predict_allp --model DFA --dfa_yaml dfa_allp