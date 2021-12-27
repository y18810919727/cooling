#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=6 python main.py --save  1115_wendu_xin_ns4   --model DFA --dfa_yaml dfa_ns
