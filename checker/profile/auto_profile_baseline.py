import torch
import os
import time
import csv
import pandas as pd
import sys

# ================= Configuration =================
INPUT_CSVS = ["benchmark_results.csv", "benchmark_results_large.csv"]
OUTPUT_FILE = "baseline_results.csv"
DEVICE = "cpu"

# ================= Utils =================
def get_todo_cases():
    # 1. 读取所有任务
    all_cases = set()
    for f in INPUT_CSVS:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                for _, row in df.iterrows():
                    if row['Status'] in ['OK', 'Timeout']:
                        all_cases.add((int(row['N']), int(row['M']), str(row['P']), row['Type'], row['Group']))
            except: pass
            
    # 2. 读取已完成
    finished_cases = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df_done = pd.read_csv(OUTPUT_FILE)
            for _, row in df_done.iterrows():
                finished_cases.add((int(row['N']), int(row['M']), str(row['P']), row['Type'], row['Group']))
        except: pass

    # 3. 筛选
    todo = sorted(list(all_cases - finished_cases), key=lambda x: x[0]*x[0]*x[1])
    return todo

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ================= Main =================
def main():
    log(f"🚀 Starting CPU Baseline (Safe Float32 Mode)...")
    
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Group", "N", "M", "P", "Type", "Device", "Duration(us)", "Status"])
    
    todo_cases = get_todo_cases()
    log(f"📋 Total Tasks: {len(todo_cases)}")

    for i, (n, m, p_str, dtype_str, group) in enumerate(todo_cases):
        prefix = f"({i+1}/{len(todo_cases)})"
        case_info = f"{n}x{m} p={p_str} {dtype_str}"
        
        complexity = n * n * m
        # 大算子提示一下
        if complexity > 1e10:
            log(f"{prefix} Running {case_info} (Large case)...")
        else:
            print(f"{prefix} Running {case_info}...", end="", flush=True)

        try:
            # --- 【核心修改】类型自动升级 ---
            # 如果 CSV 里要求 float16，我们在 CPU 上用 float32 跑
            # 这样既能跑通，又能作为 Baseline
            if dtype_str == "float16":
                real_dtype = torch.float32
            else:
                real_dtype = torch.float32 # float32 还是 float32
            
            # 准备数据
            input_tensor = torch.randn(n, m, dtype=real_dtype, device=DEVICE)
            p = float('inf') if p_str == "inf" else float(p_str)
            
            # 运行
            start_t = time.time()
            torch.nn.functional.pdist(input_tensor, p=p)
            end_t = time.time()
            
            duration_us = (end_t - start_t) * 1e6
            status = "OK"
            
            # 打印结果
            if complexity > 1e10:
                log(f"    ✅ Done. {duration_us/1000:.2f} ms")
            else:
                print(f" ✅ {duration_us/1000:.2f} ms")

            del input_tensor

        except Exception as e:
            duration_us = -1
            status = f"Error: {str(e)}"
            print(f" ❌ {status}")

        # --- 【关键】存盘时依然写原始类型 ---
        # 这样你的画图脚本会认为这是 float16 的 Baseline，从而正确匹配
        with open(OUTPUT_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([group, n, m, p_str, dtype_str, "torch_cpu", duration_us, status])

    log("🎉 All Done!")

if __name__ == "__main__":
    main()