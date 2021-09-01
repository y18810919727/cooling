#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=2 python main.py --save 0831_train --model DFA --dfa_yaml dfa_ns

