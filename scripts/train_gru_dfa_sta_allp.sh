#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=0 python main.py --save 0508-allp_alld   --model DFA --dfa_yaml dfa_allp
