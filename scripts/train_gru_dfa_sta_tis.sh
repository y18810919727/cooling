#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=3 python main.py --save 0717_predict_tis --model DFA --dfa_yaml dfa_tis
