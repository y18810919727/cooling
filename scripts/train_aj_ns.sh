#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=0 python main.py --save  ns  --aj_yaml aj_ns
