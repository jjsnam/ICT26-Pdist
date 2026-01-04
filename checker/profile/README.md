# Pdist 性能自动化测试工具集

本目录包含了一组用于自动化测试 `Pdist` 算子性能的 Python 脚本，基于 Google Gemini 3.0 Pro 编写。这些脚本能够自动生成测试用例、驱动 `msprof` 进行 NPU 性能采集、解析 profiling 数据，并与 PyTorch CPU 版本进行基准对比。

## 📂 脚本清单

| 脚本文件 | 用途 | 说明 | 输出文件 |
| :--- | :--- | :--- | :--- |
| **`auto_profile.py`** | **标准全量测试** | 覆盖 Group A (固定N扫M)、Group B (固定M扫N)、Group C (等比缩放) 三种场景。 | `benchmark_results.csv` |
| **`auto_profile_large.py`** | **超大规格压测** | 针对 **Huge Mode** 的专项测试。放宽了 N/M 限制 (M up to 200k)，超时时间延长至 5分钟。 | `benchmark_results_large.csv` |
| **`auto_profile_specified.py`** | **快速指定测试** | 仅运行代码中硬编码的几个典型尺寸（如 S/M/L 规模），用于快速验证环境或算子可用性。 | `performance_report.md` |
| **`auto_profile_baseline.py`** | **PyTorch CPU 基准** | 读取上述生成的 CSV 文件，提取所有已测试的 (N, M, p) 组合，调用 `torch.pdist` 跑一遍 CPU 基准作为对比。 | `baseline_results.csv` |

## 🚀 使用流程

请确保你的当前工作目录是 `checker/` 根目录（即 `run.sh` 所在的目录），然后通过路径调用这些脚本。

### 1. 环境准备
确保已安装必要的 Python 库：
```bash
pip install pandas numpy torch

```

确保环境变量 `ASCEND_DEVICE_ID` 已设置（脚本默认为设备 7，可视情况在脚本中修改）。

### 2. 运行 NPU 性能测试

**场景 A：运行标准全量测试（推荐）**

```bash
python3 profile/auto_profile.py

```

* 耗时：约 1-2 小时
* 测试点：覆盖从小规模到常规规模的各种形状。

**场景 B：运行超大规格测试（Huge Mode）**

```bash
python3 profile/auto_profile_large.py

```

* 耗时：较长（大算子执行慢）
* 测试点：重点覆盖 `N=5000, M=40000` 等触发分块逻辑的场景。

**场景 C：快速冒烟测试**

```bash
python3 profile/auto_profile_specified.py

```

### 3. 生成 Baseline 对比数据

在 NPU 测试完成后（生成了 csv 文件），运行此脚本来补充 PyTorch 的 CPU 耗时数据：

```bash
python3 profile/auto_profile_baseline.py

```

> **注意**：该脚本会自动读取 `benchmark_results.csv` 和 `benchmark_results_large.csv`，无需手动指定参数。

## 📊 输出结果说明

所有测试结果将保存为 CSV 格式，包含以下字段：

* **Group**: 测试组别（如 `A_Fix_N_Scan_M`）
* **N, M**: 输入形状
* **P**: 范数类型 (1.0, 2.0, inf...)
* **Type**: 数据类型 (float16/float32)
* **Duration(us)**: 算子核心执行耗时（不含 Host 发射开销）
* **Status**: 运行状态 (OK/Timeout/Error)

### 后续分析

你可以直接将生成的 CSV 文件导入 Excel 或使用 Python `pandas` / `matplotlib` 进行可视化绘图，分析算子在不同规模下的性能趋势（如线性度、吞吐量瓶颈等）。

## ⚠️ 注意事项

1. **运行路径**：脚本内部会检测 `./run.sh` 是否存在，**必须在 `checker` 目录下运行**，不要进入 `profile` 目录内运行。
2. **日志清理**：脚本启动时会自动清理目录下的 `OP*` profiling 日志文件夹，请注意备份重要数据。
3. **Float16 Baseline**：由于 PyTorch CPU 不支持 float16 的 pdist 计算，`auto_profile_baseline.py` 会自动将其转为 float32 执行，但在结果中仍标记为 float16 以便与 NPU 结果对齐。