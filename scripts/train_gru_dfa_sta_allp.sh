#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=2 python main.py --save 0824_predict_allp --model DFA --dfa_yaml dfa_allp