#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=1 python main.py --save 1122-allp_alld   --model DFA --dfa_yaml dfa_allp
