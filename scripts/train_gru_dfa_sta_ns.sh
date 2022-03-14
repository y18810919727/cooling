#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=2 python main.py --save  0309_wendu_ns3   --model DFA --dfa_yaml dfa_ns
