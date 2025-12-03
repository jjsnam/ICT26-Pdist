#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

template<typename DTYPE>
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
        this->alignedM = (M + alignNum - 1) / alignNum * alignNum;
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
        pipe.InitBuffer(inQueFirst, BUFFER_NUM, this->alignedM * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueSecond, BUFFER_NUM, this->alignedM * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueY, BUFFER_NUM, 1 * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueBuffer, BUFFER_NUM, this->M * sizeof(DTYPE_Y));
        pipe.InitBuffer(workQue, 1, this->M * sizeof(float)); // shared work buffer for calculation
        pipe.InitBuffer(castQue1, 1, this->alignedM * sizeof(float)); // shared cast buffer
        pipe.InitBuffer(castQue2, 1, this->alignedM * sizeof(float)); // shared cast buffer

        AscendC::LocalTensor<DTYPE_Y> outputBuffer = outQueBuffer.AllocTensor<DTYPE_Y>();
        outQueBuffer.EnQue(outputBuffer);
        this->bufferNum = 0;
    }

    __aicore__ inline void Process(){
        switch (this->pType){
            case 0: {
                int i = this->i;
                int j = this->j;
                for (uint64_t pair = startPair; pair < endPair; pair ++){
                    if (j >= N){
                        i ++;
                        j = i + 1;
                    }
                    CopyInFirst(i);
                    CopyInSecond(j);
                    ComputeLgeneral(i, j, pVal);
                    CopyOutAligned(pair);
                    j ++;
                }
                break;
            }
            case 1: {
                int i = this->i;
                int j = this->j;
                for (uint64_t pair = startPair; pair < endPair; pair ++){
                    if (j >= N){
                        i ++;
                        j = i + 1;
                    }
                    CopyInFirst(i);
                    CopyInSecond(j);
                    ComputeL1(i, j);
                    CopyOutAligned(pair);
                    j ++;
                }
                break;
            }
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
                    ComputeL2(i, j);
                    CopyOutAligned(pair);
                    j ++;
                }
                break;
            }
            case 3: {
                int i = this->i;
                int j = this->j;
                for (uint64_t pair = startPair; pair < endPair; pair ++){
                    if (j >= N){
                        i ++;
                        j = i + 1;
                    }
                    CopyInFirst(i);
                    CopyInSecond(j);
                    ComputeLinf(i, j);
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
        AscendC::DataCopy(x1Local, xGm[i * this->M / this->alignNum * this->alignNum], this->alignedM);
        inQueFirst.EnQue(x1Local);
    }

    __aicore__ inline void CopyInSecond(int j){
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(x2Local, xGm[j * this->M / this->alignNum * this->alignNum], this->alignedM);
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

    __aicore__ inline void ComputeL1(int i, int j){
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.AllocTensor<DTYPE_Y>();
        if constexpr (std::is_same_v<DTYPE, half>){ // float16
            AscendC::LocalTensor<float> sharedTmpBuffer = workQue.AllocTensor<float>();
            AscendC::LocalTensor<float> castBufferx1 = castQue1.AllocTensor<float>();
            AscendC::LocalTensor<float> castBufferx2 = castQue2.AllocTensor<float>();
            uint64_t x1_offset = (1ull * i * this->M) % (this->alignNum);
            uint64_t x2_offset = (1ull * j * this->M) % (this->alignNum);
            AscendC::Cast(castBufferx1, x1Local, AscendC::RoundMode::CAST_NONE, this->alignedM);
            AscendC::Cast(castBufferx2, x2Local, AscendC::RoundMode::CAST_NONE, this->alignedM);
            AscendC::Sub(castBufferx2[x2_offset], castBufferx2[x2_offset], castBufferx1[x1_offset], this->M);
            AscendC::Abs(castBufferx2[x2_offset], castBufferx2[x2_offset], this->M);
            AscendC::ReduceSum(castBufferx2[x2_offset], castBufferx2[x2_offset], sharedTmpBuffer, this->M);
            AscendC::Cast(yLocal, castBufferx2[x2_offset], AscendC::RoundMode::CAST_NONE, 1);
            castQue1.FreeTensor(castBufferx1);
            castQue2.FreeTensor(castBufferx2);
            workQue.FreeTensor(sharedTmpBuffer);
        }
        else{ // float32
            AscendC::Sub(x2Local, x1Local, x2Local, this->M);
            AscendC::Abs(x2Local, x2Local, this->M);
            AscendC::LocalTensor<DTYPE_Y> sharedTmpBuffer = workQue.AllocTensor<DTYPE_Y>();
            AscendC::ReduceSum(yLocal, x2Local, sharedTmpBuffer, this->M);
            workQue.FreeTensor(sharedTmpBuffer);
        }
        outQueY.EnQue<DTYPE_Y>(yLocal);
        inQueFirst.FreeTensor(x1Local);
        inQueSecond.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeL2(int i, int j){
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.AllocTensor<DTYPE_Y>();
        if constexpr (std::is_same_v<DTYPE, half>){ // float16
            AscendC::LocalTensor<float> sharedTmpBuffer = workQue.AllocTensor<float>();
            AscendC::LocalTensor<float> castBufferx1 = castQue1.AllocTensor<float>();
            AscendC::LocalTensor<float> castBufferx2 = castQue2.AllocTensor<float>();
            uint64_t x1_offset = (1ull * i * this->M) % (this->alignNum);
            uint64_t x2_offset = (1ull * j * this->M) % (this->alignNum);
            AscendC::Cast(castBufferx1, x1Local, AscendC::RoundMode::CAST_NONE, this->alignedM);
            AscendC::Cast(castBufferx2, x2Local, AscendC::RoundMode::CAST_NONE, this->alignedM);
            AscendC::Sub(castBufferx2[x2_offset], castBufferx2[x2_offset], castBufferx1[x1_offset], this->M);
            AscendC::Mul(castBufferx2[x2_offset], castBufferx2[x2_offset], castBufferx2[x2_offset], this->M);
            AscendC::ReduceSum(castBufferx2[x2_offset], castBufferx2[x2_offset], sharedTmpBuffer, this->M);
            AscendC::Sqrt(castBufferx2[x2_offset], castBufferx2[x2_offset], 1);
            AscendC::Cast(yLocal, castBufferx2[x2_offset], AscendC::RoundMode::CAST_NONE, 1);
            castQue1.FreeTensor(castBufferx1);
            castQue2.FreeTensor(castBufferx2);
            workQue.FreeTensor(sharedTmpBuffer);
        }
        else{ // float32
            AscendC::Sub(x2Local, x1Local, x2Local, this->M);
            AscendC::LocalTensor<DTYPE_Y> sharedTmpBuffer = workQue.AllocTensor<DTYPE_Y>();
            AscendC::Mul(x2Local, x2Local, x2Local, this->M);
            AscendC::ReduceSum(yLocal, x2Local, sharedTmpBuffer, this->M);
            AscendC::Sqrt(yLocal, yLocal, 1);
            workQue.FreeTensor(sharedTmpBuffer);
        }
        outQueY.EnQue<DTYPE_Y>(yLocal);
        inQueFirst.FreeTensor(x1Local);
        inQueSecond.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeLinf(int i, int j){
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.AllocTensor<DTYPE_Y>();
        if constexpr (std::is_same_v<DTYPE, half>){ // float16
            AscendC::LocalTensor<float> sharedTmpBuffer = workQue.AllocTensor<float>();
            AscendC::LocalTensor<float> castBufferx1 = castQue1.AllocTensor<float>();
            AscendC::LocalTensor<float> castBufferx2 = castQue2.AllocTensor<float>();
            uint64_t x1_offset = (1ull * i * this->M) % (this->alignNum);
            uint64_t x2_offset = (1ull * j * this->M) % (this->alignNum);
            AscendC::Cast(castBufferx1, x1Local, AscendC::RoundMode::CAST_NONE, this->alignedM);
            AscendC::Cast(castBufferx2, x2Local, AscendC::RoundMode::CAST_NONE, this->alignedM);
            AscendC::Sub(castBufferx2[x2_offset], castBufferx2[x2_offset], castBufferx1[x1_offset], this->M);
            AscendC::Abs(castBufferx2[x2_offset], castBufferx2[x2_offset], this->M);
            AscendC::ReduceMax(castBufferx2[x2_offset], castBufferx2[x2_offset], sharedTmpBuffer, this->M);
            AscendC::Cast(yLocal, castBufferx2[x2_offset], AscendC::RoundMode::CAST_NONE, 1);
            castQue1.FreeTensor(castBufferx1);
            castQue2.FreeTensor(castBufferx2);
            workQue.FreeTensor(sharedTmpBuffer);
        }
        else{ // float32
            AscendC::Sub(x2Local, x1Local, x2Local, this->M);
            AscendC::Abs(x2Local, x2Local, this->M);
            AscendC::LocalTensor<DTYPE_Y> sharedTmpBuffer = workQue.AllocTensor<DTYPE_Y>();
            AscendC::ReduceMax(yLocal, x2Local, sharedTmpBuffer, this->M);
            workQue.FreeTensor(sharedTmpBuffer);
        }
        outQueY.EnQue<DTYPE_Y>(yLocal);
        inQueFirst.FreeTensor(x1Local);
        inQueSecond.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeLgeneral(int i, int j, float pVal){
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.AllocTensor<DTYPE_Y>();
        if constexpr (std::is_same_v<DTYPE, half>){ // float16
            AscendC::LocalTensor<float> sharedTmpBuffer = workQue.AllocTensor<float>();
            AscendC::LocalTensor<float> castBufferx1 = castQue1.AllocTensor<float>();
            AscendC::LocalTensor<float> castBufferx2 = castQue2.AllocTensor<float>();
            uint64_t x1_offset = (1ull * i * this->M) % (this->alignNum);
            uint64_t x2_offset = (1ull * j * this->M) % (this->alignNum);
            AscendC::Cast(castBufferx1, x1Local, AscendC::RoundMode::CAST_NONE, this->alignedM);
            AscendC::Cast(castBufferx2, x2Local, AscendC::RoundMode::CAST_NONE, this->alignedM);
            AscendC::Sub(castBufferx2[x2_offset], castBufferx2[x2_offset], castBufferx1[x1_offset], this->M);
            AscendC::Abs(castBufferx2[x2_offset], castBufferx2[x2_offset], this->M);
            float p = static_cast<float>(pVal);
            float Rp = static_cast<float>(1 / pVal);
            AscendC::Power(castBufferx2[x2_offset], castBufferx2[x2_offset], p, this->M);
            AscendC::ReduceSum(castBufferx2[x2_offset], castBufferx2[x2_offset], sharedTmpBuffer, this->M);
            AscendC::Power(castBufferx2[x2_offset], castBufferx2[x2_offset], Rp, 1);
            AscendC::Cast(yLocal, castBufferx2[x2_offset], AscendC::RoundMode::CAST_NONE, 1);
            castQue1.FreeTensor(castBufferx1);
            castQue2.FreeTensor(castBufferx2);
            workQue.FreeTensor(sharedTmpBuffer);
        }
        else{ // float32
            AscendC::Sub(x2Local, x1Local, x2Local, this->M);
            AscendC::Abs(x2Local, x2Local, this->M);
            AscendC::LocalTensor<DTYPE_Y> sharedTmpBuffer = workQue.AllocTensor<DTYPE_Y>();
            DTYPE_X p = static_cast<DTYPE_X>(pVal);
            DTYPE_Y Rp = static_cast<DTYPE_Y>(1 / pVal);
            AscendC::Power(x2Local, x2Local, p, this->M);
            AscendC::ReduceSum(yLocal, x2Local, sharedTmpBuffer, this->M);
            AscendC::Power(yLocal, yLocal, Rp, 1);
            workQue.FreeTensor(sharedTmpBuffer);
        }
        outQueY.EnQue<DTYPE_Y>(yLocal);
        inQueFirst.FreeTensor(x1Local);
        inQueSecond.FreeTensor(x2Local);
    }

private:
    int blockIdx;
    int N, M;
    int i, j;
    int pType;
    float pVal;
    uint64_t startPair, endPair;
    int alignNum, bufferNum;
    uint64_t alignedM;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueFirst, inQueSecond;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueY, outQueBuffer;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> workQue, castQue1, castQue2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
};

extern "C" __global__ __aicore__ void pdist(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    if (tiling_data.dataType == DT_FLOAT16){
        KernelPdist<half> op;
        op.Init(x, y, tiling_data.pType, tiling_data.pVal, tiling_data.N, tiling_data.M, tiling_data.alignNum);
        op.Process();
    }
    else{
        KernelPdist<float> op;
        op.Init(x, y, tiling_data.pType, tiling_data.pVal, tiling_data.N, tiling_data.M, tiling_data.alignNum);
        op.Process();
    }
}