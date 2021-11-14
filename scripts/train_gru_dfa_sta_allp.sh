#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=3 python main.py --save 1010-allp_alld_12_41trans   --model DFA --dfa_yaml dfa_allp
