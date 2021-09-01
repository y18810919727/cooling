#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=4 python main.py --save 0901_3.8_allp_pd --model DFA --dfa_yaml dfa_allp