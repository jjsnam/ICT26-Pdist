#!/usr/bin/python3
# coding=utf-8

import sys
import numpy as np
import argparse

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

def parse_args():
    parser = argparse.ArgumentParser(description="Verify Result")
    parser.add_argument("--output_path", type=str, help="Path to the operator output binary file")
    parser.add_argument("--golden_path", type=str, help="Path to the golden (ground truth) binary file")
    parser.add_argument("--data_type", type=str, default="float", choices=['float', 'float16', 'float32'], help="Data type (float/float32 are treated as float32)")
    return parser.parse_args()

def verify_result(args):
    if args.data_type == 'float16':
        np_dtype = np.float16
        TOL = 1e-3
    else:
        np_dtype = np.float32
        TOL = 1e-4
    
    output = np.fromfile(args.output_path, dtype=np_dtype).reshape(-1)
    golden = np.fromfile(args.golden_path, dtype=np_dtype).reshape(-1)
    
    eps = 1e-12
    denominator = np.where(np.abs(golden) < eps, eps, np.abs(golden))
    
    abs_err = np.abs(output - golden)
    rel_err = abs_err / denominator
    
    pass_check = (abs_err <= TOL) | (rel_err <= TOL)
    different_element_indexes = np.where(pass_check == False)[0]
    
    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print(
            "data index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f" %
            (real_index, golden_data, output_data,
             abs(output_data - golden_data) / golden_data))
        if index == 100:
            break
    error_ratio = float(different_element_indexes.size) / golden.size
    error_tol = 0
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, error_tol))
    return error_ratio <= error_tol


if __name__ == '__main__':
    args = parse_args()
    try:
        passed = verify_result(args)
        if passed:
            print(f"{Colors.GREEN}[TEST PASS].{Colors.RESET}")
            sys.exit(0)
        else:
            print(f"{Colors.RED}[TEST FAIL] Precision check failed.{Colors.RESET}")
            sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}[SYSTEM ERROR] {e}{Colors.RESET}")
        sys.exit(1)
