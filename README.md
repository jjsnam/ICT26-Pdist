# Pdist 算子优化 (Ascend C)

本项目基于华为昇腾 CANN 架构，使用 Ascend C 编程语言实现了一个自定义工程化算子 `Pdist`。该算子用于计算输入矩阵中每两行向量之间的 -范数距离。

## 1. 算子介绍

`Pdist` 算子计算一个  矩阵中行向量之间的对等距离（Pairwise Distance）。

* **输入**: 一个形状为  的张量 。
* **输出**: 一个形状为  的张量 ，包含所有行对之间的距离。
* **属性 `p`**: 范数阶数。
    * : 曼哈顿距离 (L1)。
    * : 欧几里得距离 (L2)。
    * : 最大值距离 (Linf)。
    * : 通用 -范数距离。



### 算子规格

| 参数 | 类型 | 格式 | 说明 |
| --- | --- | --- | --- |
| 输入 x | float16, float32 | ND | 形状为 (N, M) |
| 输出 y | float16, float32 | ND | 形状为 (N*(N-1)/2, ) |
| 属性 p | float | - | 默认为 2.0 |

## 2. 项目结构

```text
├── Pdist.json            # 算子原型定义文件
├── Pdist/
│   ├── op_host/          # Host 侧代码：Tiling 实现及原型注册
│   ├── op_kernel/        # Device 侧代码：内核计算逻辑
│   ├── build.sh          # 编译一键化脚本
│   └── CMakeLists.txt    # 算子工程编译配置
└── checker/              # 算子功能验证工具 (AclNN 调用方式)
    ├── config.txt        # 验证参数配置文件
    ├── run.sh            # 运行验证脚本
    └── scripts/          # 数据生成与校验脚本

```

## 3. 环境准备

在编译运行之前，请确保已安装 CANN 软件栈并设置环境变量：

```bash
# 请根据实际安装路径修改
source /usr/local/Ascend/ascend-toolkit/set_env.bash

```

## 4. 编译与安装

请注意更改 `Pdist/CMakePresets.json` 的相关值，替换为运行环境的真实路径。

项目中的 `Pdist/build.sh` 已经过增强，支持编译后自动安装到系统路径。

1. 进入算子工程目录：
```bash
cd Pdist

```


2. 执行编译安装脚本：
```bash
bash build.sh

```


*该脚本会调用 CMake 完成编译，生成 `.run` 安装包，并自动执行静默安装，将算子 API 注册到 vendor 路径下。*

## 5. 算子验证 (Checker)

`checker` 目录提供了一个基于 `aclnn` 调用方式的验证环境。

### 5.1 配置参数

修改 `checker/config.txt` 来调整测试规格（如  的值）。

### 5.2 运行验证

1. 第一次运行前需编译验证程序：
```bash
cd checker
bash compile.sh

```


2. 执行测试：
```bash
bash run.sh

```


*该脚本会自动生成输入数据、调用算子 API、并对比算子输出与 Python 计算的真值。*


如果需要测试算子性能，可以直接运行 `msprof op ./run.sh` 并查看相关性能结果。

## 6. 实现细节

* **Tiling 策略**: 根据输入  和  的大小，自动计算每个 AI Core 处理的行对任务量，并充分利用 Double Buffer 提高搬运与计算的并行性。
* **内存优化**: 针对不同  值（1, 2, , 通用）实现了专门的内核计算分支，以获得最佳性能。
* **对齐处理**: 内部处理了 32 字节对齐，确保在昇腾平台上高效访问 Global Memory。