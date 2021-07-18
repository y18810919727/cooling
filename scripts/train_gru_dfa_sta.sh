#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=2 python main.py --dfa_known --save 0714dfa_known  --model DFA --seed 10
