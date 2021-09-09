#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=2 python main.py --save   0909_ns_predict_net  --model DFA --dfa_yaml dfa_ns

