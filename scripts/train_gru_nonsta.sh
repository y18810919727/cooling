#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=3 python main.py --save 0702inc  --model GRUinc
