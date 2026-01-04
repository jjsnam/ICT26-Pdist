# AscendC Pdist 算子

基于华为昇腾 CANN 架构，使用 Ascend C 编程语言实现的高性能 `Pdist` (Pairwise Distance) 算子，与 Pytorch 相关算子严格对齐。该算子用于计算输入矩阵中任意两行向量之间的 p-范数距离，广泛应用于聚类分析、计算机视觉（ReID）及自然语言处理等领域。

本项目基于 [AscendC 工程化算子开发](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/opdevg/Ascendcopdevg/atlas_ascendc_10_00046.html) 构建。

* 项目背景：[Pdist 算子优化——任务书](./docs/Pdist%20算子优化——任务书.pdf)
* 技术报告：[Proposal-Pdist](./docs/ICT25_1_Proposal_Pdist.pdf)

## 1. 算子规格与定义

### 1.1 数学定义

给定输入矩阵 $X \in \mathbb{R}^{N \times M}$，其中 $N$ 为样本数，$M$ 为特征维度。算子计算任意两行 $x_i, x_j (0 \le i < j < N)$ 之间的 $p$-范数距离：

$$y_{k} = \|x_i - x_j\|_p = \left( \sum_{m=0}^{M-1} |x_{i,m} - x_{j,m}|^p \right)^{1/p}$$

输出 $Y$ 为一维压缩向量，长度为 $N(N-1)/2$。

### 1.2 支持属性
| 属性 | 值 | 说明 |
| :--- | :--- | :--- |
| **p** | $p=1$ | 曼哈顿距离 (Manhattan), 对应 L1 范数 |
| **p** | $p=2$ | 欧几里得距离 (Euclidean), 对应 L2 范数 |
| **p** | $p=\infty$ | 切比雪夫距离 (Chebyshev), 对应 Linf 范数 |
| **p** | $p \in (0, \infty)$ | 通用 Minkowski 距离（代码里称作 General） |

### 1.3 输入输出规格
| 参数 | 类型 | 格式 | 说明 |
| :--- | :--- | :--- | :--- |
| **Input x** | float16, float32 | ND | 形状 $(N, M)$, 支持超大 $M$ 维度 |
| **Output y** | float16, float32 | ND | 形状 $(\frac{N(N-1)}{2})$ |

---

## 2. 核心特性与优化

本项目针对 Ascend NPU 架构进行了深度优化，并尽可能保证了除 Ascend 910 外其他 Ascend 平台的支持性，核心技术点如下：

### 🚀 性能优化
* **基于逆向二分查找的负载均衡**:
    * 针对上三角矩阵计算任务随行号递减导致的负载不均问题，Kernel 内部实现了基于二分查找的坐标映射算法。
    * 将总任务线性均分给各 AI Core，核心内通过 O(log N) 复杂度快速反解 `(i, j)` 坐标，实现多核负载均衡。
* **多级 Tiling 策略**:
    * **Normal Mode**: 当一行数据能完整装入 UB (Unified Buffer) 时，采用 Batch 处理策略，最大化内存搬运与计算的并行度。
    * **Huge Mode**: 针对超长特征向量 (M 极大)，自动切换至分块累积 (Split-Accumulate) 模式，利用片上缓存暂存中间结果，突破硬件内存限制。
* **标量优化**: 减少常驻 UB 张量的入出队操作，减少标量同步和开销以及流水线打断。
* **批量处理**: 充分利用硬件性能，增加单次吞吐量，减少流水线中断，充分利用内存带宽。

### 🛠 工程化设计
* **混合精度计算**: 针对 Float16 输入，计算流采用 FP32 累加 (Cast-on-the-fly) 以保证精度，输出回转 FP16。
* **双缓冲流水线**: 全面启用 `Double Buffering`，实现 GM 数据搬运与 Vector 计算的完美并行掩盖。
* **内存对齐**: 强制 32 字节对齐访问，针对非对齐数据自动进行 Padding 处理，保障访存效率。
* **模板元编程 (Template Metaprogramming)**:
    * 利用 C++ 模板技术，在编译期生成针对不同 p 值（如 p=2 时移除幂运算改为乘法）和数据规模的特化内核，消除运行时分支判断开销。
    * 减少代码重复性，使项目更加整洁。

更详细的实现细节敬请参考 [op_host/pdist_tiling.h](./Pdist/op_host/pdist_tiling.h)、[op_host/pdist.cpp](./Pdist/op_host/pdist.cpp)、[op_kernel/pdist.cpp](./Pdist/op_kernel/pdist.cpp)，我们已经对关键部分进行了注释。

---

## 3. 项目结构

```text
├── Pdist.json             # 算子原型定义 (IR)
├── Pdist/
│   ├── op_host/           # Host 侧代码
│   │   ├── pdist.cpp      # Tiling 策略逻辑 (Normal/Huge 分支判定)
│   │   └── pdist_tiling.h # Tiling 数据结构定义
│   ├── op_kernel/         # Device 侧代码
│   │   └── pdist.cpp      # 核心计算逻辑 (二分查找、计算模板特化)
│   ├── build.sh           # 一键编译安装脚本
│   └── CMakeLists.txt     # CMake 编译配置
└── checker/               # 算子验证工具 (基于 AclNN)
    ├── config.txt         # 测试用例配置文件 (N, M, p, dtype)
    ├── run.sh             # 验证运行脚本
    ├── profile/           # 自动 Profile 脚本
    └── scripts/           # Python 数据生成与真值比对脚本

```

---

## 4. 编译与安装 (Pdist)

### 4.1 环境依赖

* Huawei CANN Toolkit (Version >= 7.0)
* Ascend 910B 单卡 (或仿真环境，理论上其他 Ascend 设备也能编译运行此项目)
* CMake >= 3.16
* ARM 架构 host 环境，否则需要更改 [CMakePresets.json](./Pdist/CMakePresets.json#L58) 并设置交叉编译工具链。

```bash
# 设置环境变量 (根据实际安装路径调整)
source /usr/local/Ascend/ascend-toolkit/set_env.bash
```

### 4.2 一键编译安装

项目提供了增强版构建脚本，支持编译、打包及自动注册算子到系统 Vendor 路径。

**重要**：在正式开始编译前，请将 [CMakePresets.json](./Pdist/CMakePresets.json#L42) 的 `ASCEND_CANN_PACKAGE_PATH` 值改为您的 CANN 工具链环境。

```bash
cd Pdist
bash build.sh
```

> **提示**: 脚本会自动生成 `custom_opp_*.run` 包并静默安装。安装成功后，算子即可通过 PyTorch 插件或 ACL 接口调用。

---

## 5. 算子验证 (Checker)

`checker` 模块基于 [AscendCLNNInvocation](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/1_add_frameworklaunch/AclNNInvocation) 进一步修改，提供了一个轻量级的 C++ 测试框架，无需编写完整应用即可验证算子正确性。

建议使用 CANN 相关工具包自带的 Python 环境。

### 5.1 配置测试用例

编辑 `checker/config.txt` 修改测试参数：

```ini
# data_type: float32, float16
# data_range: S(-1,1), M(1.0, 10), L(-1000, 1000), 
N=1024
M=512
p=2.0
data_type=float32
data_range=M

```

### 5.2 编译与运行

```bash
cd checker

# 1. 编译测试程序 (仅需执行一次)
bash compile.sh

# 2. 运行测试 (自动生成数据 -> 运行算子 -> 比对真值)
bash run.sh

```

### 5.3 性能分析

使用 `msprof` 及其子工具分析算子在 NPU 上的执行性能：

```bash
msprof op ./run.sh
```

或直接使用 ```checker/profile``` 下的各 `auto_profile` 脚本。使用方法详见 [#Link](./checker/profile/README.md)

---

## 6. 版本历史

* **v2.0 (Current)**:
    * 重构 Tiling 逻辑，引入 Huge Mode 支持超大数据。
    * 优化规约逻辑，使用 `BlockReduce + WholeReduce` 提升效率。
    * 进一步优化算子性能。


* **v1.0 (2025-12-23 checkpoint)**: 
    * 基础功能实现，支持 FP16/FP32 及各类特化和通用 p 范数计算。
    * 引入双缓冲等优化技术。

---