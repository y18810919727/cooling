#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=1 python main.py --save 1005-allp_ns  --model DFA --dfa_yaml dfa_ns

