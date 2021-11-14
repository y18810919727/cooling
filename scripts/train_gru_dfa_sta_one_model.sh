#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=4 python main.py  --save 1008-one-model-alld --model DFA --dfa_yaml one_model --mymodel one