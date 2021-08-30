#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=3 python main.py --save 0825_Hierarchical --model DFA --dfa_yaml Hierarchical
