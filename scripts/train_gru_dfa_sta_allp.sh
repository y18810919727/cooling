#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=4 python main.py --save 1001_2-allp_alld  --model DFA --dfa_yaml dfa_allp

