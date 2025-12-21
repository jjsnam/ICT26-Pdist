#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

static constexpr uint32_t DEFAULT_BLK_STRIDE = 1;
static constexpr uint32_t DEFAULT_REP_STRIDE = 8;
static constexpr uint32_t REP_LEN = 256;
static constexpr uint32_t BLK_LEN = 32;
static constexpr uint32_t ONE_REPEAT_FLOAT_SIZE = REP_LEN / 4;
static constexpr uint32_t BINARY_BOUNDARY = DEFAULT_REP_STRIDE * 2;

template<typename DTYPE>
class KernelPdist{
    using DTYPE_CALC = std::conditional_t<std::is_same_v<DTYPE, half>, float, DTYPE_Y>;
public:
    __aicore__ inline KernelPdist() {}
    __aicore__ inline ~KernelPdist() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, PdistTilingData &tiling_data, AscendC::TPipe * pipeIn) {
        this->blockIdx = AscendC::GetBlockIdx();
        this->N = tiling_data.N;
        this->M = tiling_data.M;
        this->alignNum = tiling_data.alignNum;
        this->alignedM = tiling_data.alignedM;
        uint64_t totalPairs = 1ull * N * (N - 1) / 2;
        uint64_t pairsPerBlock = (totalPairs + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
        uint64_t startPair = (blockIdx * pairsPerBlock + alignNum - 1) / alignNum * alignNum;
        uint64_t endPair = ((blockIdx + 1) * pairsPerBlock + alignNum - 1) / alignNum * alignNum;
        if (endPair > totalPairs) {
            endPair = totalPairs;
        }
        this->startPair = startPair;
        this->endPair = endPair;
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
        
        this->copyOutBlock = tiling_data.copyOutBlock;

        this->pType = tiling_data.pType;
        this->pVal = tiling_data.pVal;

        this->batchSize = tiling_data.batchSize;

        this->copyParams.blockLen = this->M * sizeof(DTYPE_X);
        this->copyParams.srcStride = 0;
        this->copyParams.dstStride = 0;
        this->padParams.isPad = true;
        this->padParams.leftPadding = 0;
        this->padParams.rightPadding = (this->alignedM - this->M);
        this->padParams.paddingValue = 0;
        
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, 1ull * N * M);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, (1ull * N * (N - 1) / 2 + this->alignNum - 1) / this->alignNum * this->alignNum); // aligned output
        pipe = pipeIn;
        pipe->InitBuffer(inQueFirst, 1, this->alignedM * sizeof(DTYPE_X));
        pipe->InitBuffer(inQueSecond, BUFFER_NUM, 1ull * this->batchSize * this->alignedM * sizeof(DTYPE_X));
        pipe->InitBuffer(outQueY, 1ull * this->batchSize * sizeof(float));
        pipe->InitBuffer(outQueBuffer, this->copyOutBlock * sizeof(DTYPE_Y));
        pipe->InitBuffer(workBuf, this->M * sizeof(float)); // shared work buffer for calculation
        if constexpr(std::is_same_v<DTYPE, half>){
            pipe->InitBuffer(castBuf, 1ull * this->batchSize * this->alignedM * sizeof(float)); // shared cast buffer
        }
        this->bufferNum = 0;
    }

    __aicore__ inline void Process(){
        switch (this->pType){
            case 0: {
                int i = this->i;
                int j = this->j;
                uint64_t pair = startPair;
                int batch, next_batch;
                AscendC::LocalTensor<DTYPE_Y> outputBuffer = outQueBuffer.Get<DTYPE_Y>();
                AscendC::LocalTensor<DTYPE_CALC> yLocal = outQueY.Get<DTYPE_CALC>();
                AscendC::LocalTensor<DTYPE_CALC> sharedTmpBuffer = workBuf.Get<DTYPE_CALC>();
                AscendC::LocalTensor<DTYPE_CALC> castTmpBuffer;
                if constexpr(std::is_same_v<DTYPE, half>) {
                    castTmpBuffer = castBuf.Get<DTYPE_CALC>();
                }
                for (int i = this->i; pair < endPair; i ++) {
                    CopyInFirst(i);
                    AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
                    int j = i == this->i ? this->j : i + 1;
                    batch = min(min(N - j, (int)(endPair - pair)), this->batchSize);
                    CopyInSecondBatched(j, batch);
                    j += batch;
                    for (; j <= N && pair < endPair;) {
                        if (j < N) {
                            next_batch = min(min(N - j, (int)(endPair - pair)), this->batchSize);
                            CopyInSecondBatched(j, next_batch);
                        }
                        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
                        ComputeLgeneralBatched(i, j - batch, batch, this->pVal, x1Local, x2Local, yLocal, sharedTmpBuffer, castTmpBuffer);
                        for (int i = 0; i < batch; i ++) {
                            CopyOutAligned(pair + i, yLocal.GetValue(i), outputBuffer);
                        }
                        inQueSecond.FreeTensor(x2Local);
                        pair += batch;
                        batch = next_batch;
                        j += batch;
                    }
                    if (j <= N) {
                        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
                        inQueSecond.FreeTensor(x2Local);
                    }
                    inQueFirst.FreeTensor(x1Local);
                }
                break;
            }
            case 1: {
                int i = this->i;
                int j = this->j;
                uint64_t pair = startPair;
                int batch, next_batch;
                AscendC::LocalTensor<DTYPE_Y> outputBuffer = outQueBuffer.Get<DTYPE_Y>();
                AscendC::LocalTensor<DTYPE_CALC> yLocal = outQueY.Get<DTYPE_CALC>();
                AscendC::LocalTensor<DTYPE_CALC> sharedTmpBuffer = workBuf.Get<DTYPE_CALC>();
                AscendC::LocalTensor<DTYPE_CALC> castTmpBuffer;
                if constexpr(std::is_same_v<DTYPE, half>) {
                    castTmpBuffer = castBuf.Get<DTYPE_CALC>();
                }
                for (int i = this->i; pair < endPair; i ++) {
                    CopyInFirst(i);
                    AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
                    int j = i == this->i ? this->j : i + 1;
                    batch = min(min(N - j, (int)(endPair - pair)), this->batchSize);
                    CopyInSecondBatched(j, batch);
                    j += batch;
                    for (; j <= N && pair < endPair;) {
                        if (j < N) {
                            next_batch = min(min(N - j, (int)(endPair - pair)), this->batchSize);
                            CopyInSecondBatched(j, next_batch);
                        }
                        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
                        ComputeL1Batched(i, j - batch, batch, x1Local, x2Local, yLocal, sharedTmpBuffer, castTmpBuffer);
                        for (int i = 0; i < batch; i ++) {
                            CopyOutAligned(pair + i, yLocal.GetValue(i), outputBuffer);
                        }
                        inQueSecond.FreeTensor(x2Local);
                        pair += batch;
                        batch = next_batch;
                        j += batch;
                    }
                    if (j <= N) {
                        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
                        inQueSecond.FreeTensor(x2Local);
                    }
                    inQueFirst.FreeTensor(x1Local);
                }
                break;
            }
            case 2: {
                int i = this->i;
                int j = this->j;
                uint64_t pair = startPair;
                int batch, next_batch;
                AscendC::LocalTensor<DTYPE_Y> outputBuffer = outQueBuffer.Get<DTYPE_Y>();
                AscendC::LocalTensor<DTYPE_CALC> yLocal = outQueY.Get<DTYPE_CALC>();
                AscendC::LocalTensor<DTYPE_CALC> sharedTmpBuffer = workBuf.Get<DTYPE_CALC>();
                AscendC::LocalTensor<DTYPE_CALC> castTmpBuffer;
                if constexpr(std::is_same_v<DTYPE, half>) {
                    castTmpBuffer = castBuf.Get<DTYPE_CALC>();
                }
                for (int i = this->i; pair < endPair; i ++) {
                    CopyInFirst(i);
                    AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
                    int j = i == this->i ? this->j : i + 1;
                    batch = min(min(N - j, (int)(endPair - pair)), this->batchSize);
                    CopyInSecondBatched(j, batch);
                    j += batch;
                    for (; j <= N && pair < endPair;) {
                        if (j < N) {
                            next_batch = min(min(N - j, (int)(endPair - pair)), this->batchSize);
                            CopyInSecondBatched(j, next_batch);
                        }
                        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
                        ComputeL2Batched(i, j - batch, batch, x1Local, x2Local, yLocal, sharedTmpBuffer, castTmpBuffer);
                        for (int i = 0; i < batch; i ++) {
                            CopyOutAligned(pair + i, yLocal.GetValue(i), outputBuffer);
                        }
                        inQueSecond.FreeTensor(x2Local);
                        pair += batch;
                        batch = next_batch;
                        j += batch;
                    }
                    if (j <= N) {
                        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
                        inQueSecond.FreeTensor(x2Local);
                    }
                    inQueFirst.FreeTensor(x1Local);
                }
                break;
            }
            case 3: {
                int i = this->i;
                int j = this->j;
                uint64_t pair = startPair;
                int batch, next_batch;
                AscendC::LocalTensor<DTYPE_Y> outputBuffer = outQueBuffer.Get<DTYPE_Y>();
                AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.Get<DTYPE_Y>();
                AscendC::LocalTensor<DTYPE_Y> sharedTmpBuffer = workBuf.Get<DTYPE_Y>();
                for (int i = this->i; pair < endPair; i ++) {
                    CopyInFirst(i);
                    AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
                    int j = i == this->i ? this->j : i + 1;
                    batch = min(min(N - j, (int)(endPair - pair)), this->batchSize);
                    CopyInSecondBatched(j, batch);
                    j += batch;
                    for (; j <= N && pair < endPair;) {
                        if (j < N) {
                            next_batch = min(min(N - j, (int)(endPair - pair)), this->batchSize);
                            CopyInSecondBatched(j, next_batch);
                        }
                        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
                        ComputeLinfBatched(i, j - batch, batch, x1Local, x2Local, yLocal, sharedTmpBuffer);
                        for (int i = 0; i < batch; i ++) {
                            CopyOutAligned(pair + i, yLocal.GetValue(i), outputBuffer);
                        }
                        inQueSecond.FreeTensor(x2Local);
                        pair += batch;
                        batch = next_batch;
                        j += batch;
                    }
                    if (j <= N) {
                        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
                        inQueSecond.FreeTensor(x2Local);
                    }
                    inQueFirst.FreeTensor(x1Local);
                }
                break;
            }
        }
    }
private:
    __aicore__ inline void CopyInFirst(int i){
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(x1Local, xGm[1ull * i * this->M], this->alignedM);
        inQueFirst.EnQue(x1Local);
    }

    __aicore__ inline void CopyInSecondBatched(int j_start, int batch){
        this->copyParams.blockCount = batch;
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.AllocTensor<DTYPE_X>();
        AscendC::DataCopyPad(x2Local, xGm[1ull * j_start * this->M], this->copyParams, this->padParams);
        inQueSecond.EnQue(x2Local);
    }

    __aicore__ inline void CopyOutAligned(uint64_t pair, DTYPE_Y res, AscendC::LocalTensor<DTYPE_Y> &outputBuffer){
        outputBuffer.SetValue(this->bufferNum ++, res);
        if (this->bufferNum == this->copyOutBlock || pair == this->endPair - 1){
            uint64_t startIdx = pair - this->bufferNum + 1;
            AscendC::DataCopy(yGm[startIdx], outputBuffer, (this->bufferNum + this->alignNum - 1) / this->alignNum * this->alignNum);
            this->bufferNum = 0;
        }
    }

    __aicore__ inline void ComputeL2Batched(
        int i, int j, int batch,
        AscendC::LocalTensor<DTYPE_X> &x1Local, 
        AscendC::LocalTensor<DTYPE_X> &x2Local, 
        AscendC::LocalTensor<DTYPE_CALC> &yLocal,
        AscendC::LocalTensor<DTYPE_CALC> &sharedTmpBuffer,
        AscendC::LocalTensor<DTYPE_CALC> &castTmpBuffer
    ){
        if constexpr(std::is_same_v<DTYPE, half>){ // float16
            for (int i = 0; i < batch; i ++) {
                AscendC::Sub(x2Local[i * this->alignedM], x2Local[i * this->alignedM], x1Local, this->M);
            }
            AscendC::Cast(castTmpBuffer, x2Local, AscendC::RoundMode::CAST_NONE, batch * this->alignedM);
            AscendC::Mul(castTmpBuffer, castTmpBuffer, castTmpBuffer, batch * this->alignedM);
            for (int i = 0; i < batch; i ++) {
                AscendC::ReduceSum(yLocal[i], castTmpBuffer[i * this->alignedM], sharedTmpBuffer, this->M);
            }
            AscendC::Sqrt(yLocal, yLocal, batch);
        }
        else{ // float32
            for (int i = 0; i < batch; i ++) {
                AscendC::Sub(x2Local[i * this->alignedM], x2Local[i * this->alignedM], x1Local, this->M);
            }
            AscendC::Mul(x2Local, x2Local, x2Local, batch * this->alignedM);
            for (int i = 0; i < batch; i ++) {
                AscendC::ReduceSum(yLocal[i], x2Local[i * this->alignedM], sharedTmpBuffer, this->M);
            }
            AscendC::Sqrt(yLocal, yLocal, batch);
        }
    }

    __aicore__ inline void ComputeL1Batched(
        int i, int j, int batch,
        AscendC::LocalTensor<DTYPE_X> &x1Local, 
        AscendC::LocalTensor<DTYPE_X> &x2Local, 
        AscendC::LocalTensor<DTYPE_CALC> &yLocal,
        AscendC::LocalTensor<DTYPE_CALC> &sharedTmpBuffer,
        AscendC::LocalTensor<DTYPE_CALC> &castTmpBuffer
    ){
        if constexpr(std::is_same_v<DTYPE, half>){ // float16
            for (int i = 0; i < batch; i ++) {
                AscendC::Sub(x2Local[i * this->alignedM], x2Local[i * this->alignedM], x1Local, this->M);
            }
            AscendC::Abs(x2Local, x2Local, batch * this->alignedM);
            AscendC::Cast(castTmpBuffer, x2Local, AscendC::RoundMode::CAST_NONE, batch * this->alignedM);
            for (int i = 0; i < batch; i ++) {
                AscendC::ReduceSum(yLocal[i], castTmpBuffer[i * this->alignedM], sharedTmpBuffer, this->M);
            }
        }
        else{ // float32
            for (int i = 0; i < batch; i ++) {
                AscendC::Sub(x2Local[i * this->alignedM], x2Local[i * this->alignedM], x1Local, this->M);
            }
            AscendC::Abs(x2Local, x2Local, batch * this->alignedM);
            for (int i = 0; i < batch; i ++) {
                AscendC::ReduceSum(yLocal[i], x2Local[i * this->alignedM], sharedTmpBuffer, this->M);
            }
        }
    }

    __aicore__ inline void ComputeLinfBatched(
        int i, int j, int batch,
        AscendC::LocalTensor<DTYPE_X> &x1Local, 
        AscendC::LocalTensor<DTYPE_X> &x2Local, 
        AscendC::LocalTensor<DTYPE_Y> &yLocal,
        AscendC::LocalTensor<DTYPE_Y> &sharedTmpBuffer
    ){
        if constexpr(std::is_same_v<DTYPE, half>){ // float16
            for (int i = 0; i < batch; i ++) {
                AscendC::Sub(x2Local[i * this->alignedM], x2Local[i * this->alignedM], x1Local, this->M);
            }
            AscendC::Abs(x2Local, x2Local, batch * this->alignedM);
            for (int i = 0; i < batch; i ++) {
                AscendC::ReduceMax(yLocal[i], x2Local[i * this->alignedM], sharedTmpBuffer, this->M);
            }
        }
        else{ // float32
            for (int i = 0; i < batch; i ++) {
                AscendC::Sub(x2Local[i * this->alignedM], x2Local[i * this->alignedM], x1Local, this->M);
            }
            AscendC::Abs(x2Local, x2Local, batch * this->alignedM);
            for (int i = 0; i < batch; i ++) {
                AscendC::ReduceMax(yLocal[i], x2Local[i * this->alignedM], sharedTmpBuffer, this->M);
            }
        }
    }

    __aicore__ inline void ComputeLgeneralBatched(
        int i, int j, int batch, float pVal,
        AscendC::LocalTensor<DTYPE_X> &x1Local, 
        AscendC::LocalTensor<DTYPE_X> &x2Local, 
        AscendC::LocalTensor<DTYPE_CALC> &yLocal,
        AscendC::LocalTensor<DTYPE_CALC> &sharedTmpBuffer,
        AscendC::LocalTensor<DTYPE_CALC> &castTmpBuffer
    ){
        if constexpr(std::is_same_v<DTYPE, half>){ // float16
            for (int i = 0; i < batch; i ++) {
                AscendC::Sub(x2Local[i * this->alignedM], x2Local[i * this->alignedM], x1Local, this->M);
            }
            AscendC::Abs(x2Local, x2Local, batch * this->alignedM);
            DTYPE_CALC p = static_cast<DTYPE_CALC>(pVal);
            DTYPE_CALC Rp = static_cast<DTYPE_CALC>(1 / pVal);
            AscendC::Cast(castTmpBuffer, x2Local, AscendC::RoundMode::CAST_NONE, batch * this->alignedM);
            AscendC::Power(castTmpBuffer, castTmpBuffer, p, batch * this->alignedM);
            for (int i = 0; i < batch; i ++) {
                AscendC::ReduceSum(yLocal[i], castTmpBuffer[i * this->alignedM], sharedTmpBuffer, this->M);
            }
            AscendC::Power(yLocal, yLocal, Rp, batch);
        }
        else{ // float32
            for (int i = 0; i < batch; i ++) {
                AscendC::Sub(x2Local[i * this->alignedM], x2Local[i * this->alignedM], x1Local, this->M);
            }
            AscendC::Abs(x2Local, x2Local, batch * this->alignedM);
            DTYPE_X p = static_cast<DTYPE_X>(pVal);
            DTYPE_CALC Rp = static_cast<DTYPE_CALC>(1 / pVal);
            AscendC::Power(x2Local, x2Local, p/* , batch * this->alignedM */);
            for (int i = 0; i < batch; i ++) {
                AscendC::ReduceSum(yLocal[i], x2Local[i * this->alignedM], sharedTmpBuffer, this->M);
            }
            AscendC::Power(yLocal, yLocal, Rp/* , batch */);
        }
    }

private:
    int blockIdx;
    int N, M;
    int i, j;
    int pType;
    float pVal;
    uint64_t startPair, endPair;
    uint64_t copyOutBlock;
    int alignNum, bufferNum;
    uint64_t alignedM;
    int batchSize;

    AscendC::DataCopyParams copyParams;
    AscendC::DataCopyPadParams padParams;

    AscendC::TPipe *pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueFirst;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueSecond;
    AscendC::TBuf<AscendC::TPosition::VECOUT> outQueY;
    AscendC::TBuf<AscendC::TPosition::VECOUT> outQueBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castBuf;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
};

extern "C" __global__ __aicore__ void pdist(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    AscendC::TPipe pipe;
    if (tiling_data.dataType == DT_FLOAT16){
        KernelPdist<half> op;
        op.Init(x, y, tiling_data, &pipe);
        op.Process();
    }
    else{
        KernelPdist<float> op;
        op.Init(x, y, tiling_data, &pipe);
        op.Process();
    }
}