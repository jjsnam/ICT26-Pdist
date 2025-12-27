#!/usr/bin/python3
# coding=utf-8

import numpy as np
import torch, torch_npu
import argparse
import sys
import os
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Pdist Golden Data")
    parser.add_argument("--N", type=int, required=True, help="Size: S(100,400), M(2024,3000), L(100000,100000)")
    parser.add_argument("--M", type=int, required=True, help="Size: S(100,400), M(2024,3000), L(100000,100000)")
    parser.add_argument("--p", type=float, default=2.0, help="p-norm value")
    parser.add_argument("--data_type", type=str, default="float", choices=['float', 'float16', 'float32'], help="Data type (float/float32 are treated as float32)")
    parser.add_argument("--data_range", type=str, required=True, choices=['S', 'M', 'L'], help="Range: S(-1~1), M(1~10), L(-1000~1000)")
    parser.add_argument("--device", type=str, default="npu", choices=['cpu', 'npu'], help="Device to generate golden data")
    return parser.parse_args()

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def gen_golden_data():
    args = parse_args()

    range_map = {
        'S': (-1.0, 1.0),
        'M': (1.0, 10.0),
        'L': (-1000.0, 1000.0)
    }
    
    N, M = args.N, args.M
    min_val, max_val = range_map[args.data_range]
    
    device = torch.device('cpu')
    device_str = 'CPU'
    
    # 设置数据类型
    if args.data_type == 'float16':
        torch_dtype = torch.float16
        np_dtype = np.float16
    else:
        torch_dtype = torch.float32
        np_dtype = np.float32
        
    print("=" * 60)
    print(f"[Config] Size: ({N}, {M}), Range: ({min_val}, {max_val}), P: {args.p}, Type: {args.data_type}")
    print(f"[Config] Golden Generator Device: {device_str.upper()}")
    print("-" * 60)

    # 确保输出目录存在
    ensure_dir("./input")
    ensure_dir("./output")

    try:
        if device_str == 'npu':
            torch.npu.synchronize()

        print(f"[Info] Generating random input on {device_str.upper()}...")
        input_x = (torch.rand(N, M, device=device) * (max_val - min_val) + min_val).to(torch_dtype)
        
        print(f"[Info] Computing pdist (p={args.p}) on {device_str.upper()}...")
        start_time = time.time()
        if args.data_type == 'float16':
            golden = torch.pdist(input_x.float(), p=args.p).to(torch_dtype)
        else:
            golden = torch.pdist(input_x, p=args.p)

        if device_str == 'npu':
            torch.npu.synchronize()
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"[Success] Golden data generated successfully!")
        print(f"[Time] Duration: {elapsed_time:.4f} seconds")
        print(f"[Info] Golden Generator: {device_str.upper()}")
        print("-" * 60)

        print("[Info] Saving to disk (moving to CPU first)...")
        input_x_np = input_x.cpu().numpy().astype(np_dtype)
        golden_np = golden.cpu().numpy().astype(np_dtype)
        
        input_x_np.tofile("./input/input_x.bin")
        golden_np.tofile("./output/golden.bin")
        print("[Info] Files saved to ./input/input_x.bin and ./output/golden.bin")

    except RuntimeError as e:
        print(f"[Error] Runtime error occurred (likely OOM for Large size): {e}")
    except Exception as e:
        print(f"[Error] An unexpected error occurred: {e}")

if __name__ == "__main__":
    gen_golden_data()