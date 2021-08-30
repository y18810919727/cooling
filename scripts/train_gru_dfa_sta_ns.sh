#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=5 python main.py --save 0829_ns_bptt_1600 --model DFA --dfa_yaml dfa_ns

