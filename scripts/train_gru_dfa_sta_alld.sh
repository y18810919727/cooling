#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=2 python main.py --save 0826_Time-Aware --model DFA --dfa_yaml Time-Aware
