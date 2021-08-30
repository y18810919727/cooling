#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=4 python main.py --save 0825_stationary  --model DFA --dfa_yaml stationary


