#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=5 python main.py  --save 1003_one_model --model DFA --dfa_yaml one_model --mymodel one