#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=2 python main.py --save 0717_predict_tid --model DFA --dfa_yaml dfa_tid
