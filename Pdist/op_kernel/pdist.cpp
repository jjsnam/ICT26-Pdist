#include "kernel_operator.h"
#include <cmath>

class KernelPdist {
public:
    __aicore__ inline KernelPdist() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t N_, uint32_t D_) {
        N = N_;
        D = D_;

        // 绑定 GM buffer（全量）
        xGm.SetGlobalBuffer((__gm__ float*)x, (uint64_t)N * D);
        yGm.SetGlobalBuffer((__gm__ float*)y, (uint64_t)N * (N - 1) / 2);

        // 不用 pipe/queue：我们在每次 LoadRow 时直接 DataCopy 到 local 临时 buffer
        // 后续 AscendC 支持 LocalTensor 构造需替换下面的 local arrays
    }

    __aicore__ inline void Euclidean_process() {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        uint32_t base = N / blockNum;
        uint32_t rem = N % blockNum;
        uint32_t start = blockIdx * base + (blockIdx < rem ? blockIdx : rem);
        uint32_t myRows = base + (blockIdx < rem ? 1 : 0);
        uint32_t end = start + myRows;

        // LocalTensor 分配
        AscendC::LocalTensor<float> local_i;
        AscendC::LocalTensor<float> local_j;
  // TODO: 使用官方 vector API 或 queue / pipe 方式将 xGm 中对应行拷贝到 local_i/local_j
        // 示例：
        //   vector<float> vi(D);
        //   vector<float> vj(D);
        //   xGm.DataCopyTo(vi.data(), D * sizeof(float), i * D * sizeof(float));
        //   xGm.DataCopyTo(vj.data(), D * sizeof(float), j * D * sizeof(float));
        // 注意：上述 DataCopyTo 仅为示意，实际使用官方 sample API

    for (uint32_t i = start; i < end; i++) {
                for (uint32_t j = i + 1; j < N; j++) {
                    float sum = 0.0f;
                    // TODO: 计算局部差平方和
                    // for (uint32_t k = 0; k < D; k++) {
                    //     float diff = vi[k] - vj[k];
                    //     sum += diff * diff;
                    // }

                    // TODO: 写回全局 yGm 上三角输出
                    // yGm.DataCopyFrom(&sum, sizeof(float), UpperTriIndex(i,j) * sizeof(float));
                }
            }
        }


private:
    __aicore__ inline uint32_t UpperTriIndex(uint32_t i, uint32_t j) {
        // 映射 i<j 的上三角到线性索引
        // k = i*(2*N - i - 1)/2 + (j - i - 1)
        return i * (2 * N - i - 1) / 2 + (j - i - 1);
    }

private:
    AscendC::GlobalTensor<float> xGm, yGm;
    uint32_t N;
    uint32_t D;
};

extern "C" __global__ __aicore__ void pdist(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling ) {
    GET_TILING_DATA(TilingData, tiling);
    float p = TilingData.p;   
    KernelPdist op;
    op.Init(x, y, TilingData.N, TilingData.D);
    if (p == 2.0f) op.Euclidean_process();
    /*
    if (p == 0.0f) {// 固定分支，不依赖 p
        op.p0_process();
    } else if (p == 1.0f) {
        op.L1_process();
    } else if (p == 2.0f) {
        op.Euclidean_process();
    } else {
        op.Lp_process(p); // 我觉得这是最牢的，需要将p传入分支
    }
        */
}
