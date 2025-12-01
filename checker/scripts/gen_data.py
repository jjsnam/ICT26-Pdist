#!/usr/bin/python3
# coding=utf-8

import numpy as np
import torch
import argparse
import sys, os


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Pdist Golden Data")
    parser.add_argument("--data_size", type=str, required=True, choices=['S', 'M', 'L'], help="Size: S(100,400), M(2024,3000), L(100000,100000)")
    parser.add_argument("--p", type=float, default=2.0, help="p-norm value")
    parser.add_argument("--data_type", type=str, default="float", choices=['float', 'float16', 'float32'], help="Data type (float/float32 are treated as float32)")
    parser.add_argument("--data_range", type=str, required=True, choices=['S', 'M', 'L'], help="Range: S(-1~1), M(1~10), L(-1000~1000)")
    return parser.parse_args()


def gen_golden_data():
    args = parse_args()
    
    size_map = {
        'S': (100, 400),
        'M': (2024, 3000),
        'L': (100000, 100000)
    }
    range_map = {
        'S': (-1.0, 1.0),
        'M': (1.0, 10.0),
        'L': (-1000.0, 1000.0)
    }
    
    N, M = size_map[args.data_size]
    min_val, max_val = range_map[args.data_range]
    
    if args.data_type == 'float16':
        torch_dtype = torch.float16
        np_dtype = np.float16
    else:
        torch_dtype = torch.float32
        np_dtype = np.float32
        
    print(f"Generating Data... Size: ({N}, {M}), Range: ({min_val}, {max_val}), P: {args.p}, Type: {args.data_type}")
    
    input_x = (torch.rand(N, M) * (max_val - min_val) + min_val).to(torch_dtype)
    golden = torch.pdist(input_x, p=args.p).to(torch_dtype)
    
    input_x_np = input_x.numpy().astype(np_dtype)
    golden_np = golden.numpy().astype(np_dtype)
    
    input_x_np.tofile("./input/input_x.bin")
    golden_np.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
