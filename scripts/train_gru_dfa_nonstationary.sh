#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=7 python main.py --save 0825_non_stationary  --model DFA --dfa_yaml non_stationary

