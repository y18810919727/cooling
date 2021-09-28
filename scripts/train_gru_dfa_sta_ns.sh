#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=2 python main.py --save  0927_wendu_ns_new_trans_newdata_state   --model DFA --dfa_yaml dfa_ns

