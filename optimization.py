#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
import argparse

parser = argparse.ArgumentParser(description='Models for Continuous Stirred Tank dataset')

# model definition
methods = """
set up model
- model:
    GRU (compensated GRU to avoid linear increase of state; has standard GRU as special case for Euler scheme and equidistant data)
    GRUinc (incremental GRU, for baseline only)
- time_aware:
    no: ignore uneven spacing: for GRU use original GRU implementation; ignore 'scheme' variable
    input: use normalized next interval size as extra input feature
    variable: time-aware implementation
"""


parser.add_argument("--low", action="store_true", help="Testing model in para.save")
parser.add_argument("--low", type=float, default=11, help='The temperature activating the work of cooling')
parser.add_argument("--high", type=float, default=20, help='The temperature stopping the cooling system')
parser.add_argument("--model", type=str, default='GRU', choices=['GRU', 'GRUinc', 'ARNN', 'ARNNinc', 'DFA'])
parser.add_argument("--interpol", type=str, default='constant', choices=['constant', 'linear'])

def main():
    pass

if __name__ == '__main__':
    main()

