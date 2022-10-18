#!/usr/bin/env bash

cd ..

CUDA_VISIBLE_DEVICES=1 python main.py  --save one-cell-1  --aj_yaml one_model --mymodel one
