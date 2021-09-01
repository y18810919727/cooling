#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=1 python main.py --save 0831_3.8_allp_ms --model DFA --dfa_yaml dfa_allp