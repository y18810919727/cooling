#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=3 python main.py --save 0928_one_model --model DFA --dfa_yaml one_model