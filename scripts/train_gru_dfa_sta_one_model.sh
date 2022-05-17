#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=0 python main.py  --save 0518_Y-one-model --model DFA --dfa_yaml one_model --mymodel one
