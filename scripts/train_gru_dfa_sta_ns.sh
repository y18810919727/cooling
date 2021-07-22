#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=3 python main.py --save 0722_predict_ns_pbias --model DFA --dfa_yaml dfa_ns