#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=2  python main.py --save 0816_predict_ns_state4 --model DFA --dfa_yaml dfa_ns --test --seed 632541
