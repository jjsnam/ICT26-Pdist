#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelPdist{
public:
    __aicore__ inline KernelPdist() {}
    __aicore__ inline ~KernelPdist() {
        AscendC::LocalTensor<DTYPE_Y> outputBuffer = outQueBuffer.DeQue<DTYPE_Y>();
        outQueBuffer.FreeTensor(outputBuffer);
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t pType, float pVal, uint32_t N, uint32_t M, uint32_t alignNum) {
        this->blockIdx = AscendC::GetBlockIdx();
        this->N = N;
        this->M = M;
        this->alignNum = alignNum;
        uint64_t totalPairs = 1ull * N * (N - 1) / 2;
        uint64_t pairsPerBlock = (totalPairs + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
        uint64_t startPair = (blockIdx * pairsPerBlock + alignNum - 1) / alignNum * alignNum;
        uint64_t endPair = ((blockIdx + 1) * pairsPerBlock + alignNum - 1) / alignNum * alignNum;
        if (endPair > totalPairs) {
            endPair = totalPairs;
        }
        this->startPair = startPair;
        this->endPair = endPair;
        // this->i = floor((2 * N - 1 - sqrt((2 * N - 1) * (2 * N - 1) - 8 * startPair)) / 2.0);
        // this->j = startPair - i * (2 * N - i - 1) / 2 + i + 1;
        int l = 0, r = N - 1, mid, ans;
        while (l <= r){
            mid = (l + r) >> 1;
            uint64_t totalPairs = 1ull * (mid + 1) * (2 * N - mid - 2) / 2;
            if (totalPairs > startPair){
                ans = mid;
                r = mid - 1;
            }
            else{
                l = mid + 1;
            }
        }
        this->i = ans;
        this->j = startPair - 1ull * ans * (2 * N - ans - 1) / 2 + ans + 1;

        this->pType = pType;
        this->pVal = pVal;
        
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, 1ull * N * M);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, (1ull * N * (N - 1) / 2 + alignNum - 1) / alignNum * alignNum); // aligned output
        pipe.InitBuffer(inQueFirst, BUFFER_NUM, this->M * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueSecond, BUFFER_NUM, this->M * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueY, BUFFER_NUM, 1 * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueBuffer, BUFFER_NUM, this->M * sizeof(DTYPE_Y));
        pipe.InitBuffer(workQue, 1, this->M * sizeof(DTYPE_Y));

        AscendC::LocalTensor<DTYPE_Y> outputBuffer = outQueBuffer.AllocTensor<DTYPE_Y>();
        outQueBuffer.EnQue(outputBuffer);
        this->bufferNum = 0;
    }

    __aicore__ inline void Process(){
        switch (this->pType){
            case 2: {
                int i = this->i;
                int j = this->j;
                for (uint64_t pair = startPair; pair < endPair; pair ++){
                    if (j >= N){
                        i ++;
                        j = i + 1;
                    }
                    CopyInFirst(i);
                    CopyInSecond(j);
                    ComputeL2();
                    CopyOutAligned(pair);
                    j ++;
                }
                break;
            }
            default:
                break;
        }

    }
private:
    __aicore__ inline void CopyInFirst(int i){
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(x1Local, xGm[i * this->M], this->M);
        inQueFirst.EnQue(x1Local);
    }

    __aicore__ inline void CopyInSecond(int j){
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(x2Local, xGm[j * this->M], this->M);
        inQueSecond.EnQue(x2Local);
    }

    __aicore__ inline void CopyOut(int i, int j){
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[1ull * i * (2 * N - i - 1) / 2 + j - i - 1], yLocal, 1);
        outQueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutAligned(uint64_t pair){
        AscendC::LocalTensor<DTYPE_Y> outputBuffer = outQueBuffer.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.DeQue<DTYPE_Y>();
        DTYPE_Y val = yLocal.GetValue(0);
        outputBuffer.SetValue(this->bufferNum, val);
        this->bufferNum ++;
        if (this->bufferNum == this->alignNum || pair == this->endPair - 1){
            uint64_t startIdx = pair - this->bufferNum + 1;
            AscendC::DataCopy(yGm[startIdx], outputBuffer, this->alignNum);
            this->bufferNum = 0;
        }
        else if (pair == this->endPair - 1){

        }
        outQueBuffer.EnQue(outputBuffer);
        outQueY.FreeTensor(yLocal);
    }

    __aicore__ inline void ComputeL1(){

    }

    __aicore__ inline void ComputeL2(){
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Y> sharedTmpBuffer = workQue.AllocTensor<DTYPE_Y>();
        AscendC::Sub(x2Local, x1Local, x2Local, this->M);
        AscendC::Mul(x2Local, x2Local, x2Local, this->M);
        AscendC::ReduceSum(yLocal, x2Local, sharedTmpBuffer, this->M);
        AscendC::Sqrt(yLocal, yLocal, 1);
        outQueY.EnQue<DTYPE_Y>(yLocal);
        inQueFirst.FreeTensor(x1Local);
        inQueSecond.FreeTensor(x2Local);
        workQue.FreeTensor(sharedTmpBuffer);
    }

    __aicore__ inline void ComputeLinf(){

    }

    __aicore__ inline void ComputeLgeneral(){

    }

private:
    int blockIdx;
    int N, M;
    int i, j;
    int pType;
    float pVal;
    uint64_t startPair, endPair;
    int alignNum, bufferNum;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueFirst, inQueSecond;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueY, outQueBuffer;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> workQue;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
};

extern "C" __global__ __aicore__ void pdist(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    KernelPdist op;
    op.Init(x, y, tiling_data.pType, tiling_data.pVal, tiling_data.N, tiling_data.M, tiling_data.alignNum);
    op.Process();
}