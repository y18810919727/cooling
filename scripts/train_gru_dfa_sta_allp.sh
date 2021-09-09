#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=4 python main.py --save 0909_ns_allp_net_ns --model DFA --dfa_yaml dfa_allp