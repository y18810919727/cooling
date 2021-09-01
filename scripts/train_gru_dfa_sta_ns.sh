#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=5 python main.py --save 0901_ns_1.7k --model DFA --dfa_yaml dfa_ns

