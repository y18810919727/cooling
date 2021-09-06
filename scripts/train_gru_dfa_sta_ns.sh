#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=7 python main.py --save 0906_ns_all_1.7k --model DFA --dfa_yaml dfa_ns

