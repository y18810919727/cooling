#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=1 python main.py  --save 0519_-one-model --model DFA --dfa_yaml one_model --mymodel one
