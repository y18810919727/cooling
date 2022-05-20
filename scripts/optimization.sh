#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=1 python optimization.py --save_dir optimization_test
