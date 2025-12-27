#include "kernel_operator.h"

static constexpr int BUFFER_NUM = 2;

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
        pipe->InitBuffer(outQueY, BUFFER_NUM, this->copyOutBlock * sizeof(DTYPE_Y));
        pipe->InitBuffer(outQueBuffer, this->copyOutBlock * sizeof(float));
        if constexpr(std::is_same_v<DTYPE, half>){
            pipe->InitBuffer(castBuf, 1ull * this->batchSize * this->alignedM * sizeof(float)); // shared cast buffer
        }
        this->bufferNum = 0;
    }

    __aicore__ inline void Process(){
        if (this->startPair >= this->endPair) return;

        switch (this->pType){
            case 0: { // L general
                int i = this->i;
                uint64_t pair = startPair;
                int batch, next_batch;
                AscendC::LocalTensor<DTYPE_CALC> outputBuffer = outQueBuffer.Get<DTYPE_CALC>();
                AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.AllocTensor<DTYPE_Y>();
                AscendC::LocalTensor<DTYPE_CALC> castTmpBuffer;
                if constexpr(std::is_same_v<DTYPE, half>) {
                    castTmpBuffer = castBuf.Get<DTYPE_CALC>();
                }
                CopyInFirst(i);
                AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
                int j = this->j, next_j;
                batch = min(min(N - j, (int)(endPair - pair)), this->batchSize);
                CopyInSecondBatched(j, batch);
                while (pair < endPair) {
                    next_j = j + batch;
                    if (next_j == N) next_j = (i + 1) + 1;
                    next_batch = min(min(N - next_j, (int)(endPair - pair - batch)), this->batchSize);
                    if (next_batch) CopyInSecondBatched(next_j, next_batch);
                    AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
                    ComputeLgeneralBatched(batch, pair, this->pVal, x1Local, x2Local, yLocal, outputBuffer, castTmpBuffer);
                    inQueSecond.FreeTensor(x2Local);
                    if (j + batch == N) {
                        inQueFirst.FreeTensor(x1Local);
                        if (pair < endPair) {
                            CopyInFirst(++ i);
                            x1Local = inQueFirst.DeQue<DTYPE_X>();    
                        }
                    }
                    else if (pair == endPair){
                        inQueFirst.FreeTensor(x1Local);
                    }
                    batch = next_batch;
                    j = next_j;
                }
                break;
            }
            case 1: { // L1
                int i = this->i;
                uint64_t pair = startPair;
                int batch, next_batch;
                AscendC::LocalTensor<DTYPE_CALC> outputBuffer = outQueBuffer.Get<DTYPE_CALC>();
                AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.AllocTensor<DTYPE_Y>();
                AscendC::LocalTensor<DTYPE_CALC> castTmpBuffer;
                if constexpr(std::is_same_v<DTYPE, half>) {
                    castTmpBuffer = castBuf.Get<DTYPE_CALC>();
                }
                CopyInFirst(i);
                AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
                int j = this->j, next_j;
                batch = min(min(N - j, (int)(endPair - pair)), this->batchSize);
                CopyInSecondBatched(j, batch);
                while (pair < endPair) {
                    next_j = j + batch;
                    if (next_j == N) next_j = (i + 1) + 1;
                    next_batch = min(min(N - next_j, (int)(endPair - pair - batch)), this->batchSize);
                    if (next_batch) CopyInSecondBatched(next_j, next_batch);
                    AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
                    ComputeL1Batched(batch, pair, x1Local, x2Local, yLocal, outputBuffer, castTmpBuffer);
                    inQueSecond.FreeTensor(x2Local);
                    if (j + batch == N) {
                        inQueFirst.FreeTensor(x1Local);
                        if (pair < endPair) {
                            CopyInFirst(++ i);
                            x1Local = inQueFirst.DeQue<DTYPE_X>();    
                        }
                    }
                    else if (pair == endPair){
                        inQueFirst.FreeTensor(x1Local);
                    }
                    batch = next_batch;
                    j = next_j;
                }
                break;
            }
            case 2: { // L2
                int i = this->i;
                uint64_t pair = startPair;
                int batch, next_batch;
                AscendC::LocalTensor<DTYPE_CALC> outputBuffer = outQueBuffer.Get<DTYPE_CALC>();
                AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.AllocTensor<DTYPE_Y>();
                AscendC::LocalTensor<DTYPE_CALC> castTmpBuffer;
                if constexpr(std::is_same_v<DTYPE, half>) {
                    castTmpBuffer = castBuf.Get<DTYPE_CALC>();
                }
                CopyInFirst(i);
                AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
                int j = this->j, next_j;
                batch = min(min(N - j, (int)(endPair - pair)), this->batchSize);
                CopyInSecondBatched(j, batch);
                while (pair < endPair) {
                    next_j = j + batch;
                    if (next_j == N) next_j = (i + 1) + 1;
                    next_batch = min(min(N - next_j, (int)(endPair - pair - batch)), this->batchSize);
                    if (next_batch) CopyInSecondBatched(next_j, next_batch);
                    AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
                    ComputeL2Batched(batch, pair, x1Local, x2Local, yLocal, outputBuffer, castTmpBuffer);
                    inQueSecond.FreeTensor(x2Local);
                    if (j + batch == N) {
                        inQueFirst.FreeTensor(x1Local);
                        if (pair < endPair) {
                            CopyInFirst(++ i);
                            x1Local = inQueFirst.DeQue<DTYPE_X>();    
                        }
                    }
                    else if (pair == endPair){
                        inQueFirst.FreeTensor(x1Local);
                    }
                    batch = next_batch;
                    j = next_j;
                }
                break;
            }
            case 3: { // L-inf
                int i = this->i;
                uint64_t pair = startPair;
                int batch, next_batch;
                AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.AllocTensor<DTYPE_Y>();
                CopyInFirst(i);
                AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
                int j = this->j, next_j;
                batch = min(min(N - j, (int)(endPair - pair)), this->batchSize);
                CopyInSecondBatched(j, batch);
                while (pair < endPair) {
                    next_j = j + batch;
                    if (next_j == N) next_j = (i + 1) + 1;
                    next_batch = min(min(N - next_j, (int)(endPair - pair - batch)), this->batchSize);
                    if (next_batch) CopyInSecondBatched(next_j, next_batch);
                    AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
                    ComputeLinfBatched(batch, pair, x1Local, x2Local, yLocal);
                    inQueSecond.FreeTensor(x2Local);
                    if (j + batch == N) {
                        inQueFirst.FreeTensor(x1Local);
                        if (pair < endPair) {
                            CopyInFirst(++ i);
                            x1Local = inQueFirst.DeQue<DTYPE_X>();    
                        }
                    }
                    else if (pair == endPair){
                        inQueFirst.FreeTensor(x1Local);
                    }
                    batch = next_batch;
                    j = next_j;
                }
                break;
            }
        }
    }
private:
    __aicore__ inline void CopyInFirst(int i){
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(x1Local, xGm[i * this->M], this->alignedM);
        inQueFirst.EnQue(x1Local);
    }

    __aicore__ inline void CopyInSecondBatched(int j_start, int batch){
        this->copyParams.blockCount = batch;
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.AllocTensor<DTYPE_X>();
        AscendC::DataCopyPad(x2Local, xGm[j_start * this->M], this->copyParams, this->padParams);
        inQueSecond.EnQue(x2Local);
    }

    __aicore__ inline void CopyOut(int startIdx, int outSizeB) {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.DeQue<DTYPE_Y>();
        DataCopy(yGm[startIdx], yLocal, (outSizeB + this->alignNum - 1) / this->alignNum * this->alignNum);
        outQueY.FreeTensor(yLocal);
    }

private:
    template <typename T>
    __aicore__ inline void reduceSum(const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src, const int &totalElements) {   
        constexpr int elemsPerBlock = 32 / sizeof(T);
        int currentLen = totalElements;
        AscendC::SetMaskCount();
        while (currentLen > (elemsPerBlock * 8)) {
            int blockCount = (currentLen + elemsPerBlock - 1) / elemsPerBlock;
            int repeat = (blockCount + 7) / 8;
            AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(currentLen);
            AscendC::BlockReduceSum<T, false>(src, src, repeat, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
            currentLen = blockCount;
        }
        AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(currentLen);
        AscendC::WholeReduceSum<T, false>(dst, src, AscendC::MASK_PLACEHOLDER, 1, 1, 1, 8);
        AscendC::SetMaskNorm();
        AscendC::ResetMask();  
    }

    template <typename T>
    __aicore__ inline void reduceMax(const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src, const int &totalElements) {   
        constexpr int elemsPerBlock = 32 / sizeof(T);
        int currentLen = totalElements;
        AscendC::SetMaskCount();
        while (currentLen > (elemsPerBlock * 8)) {
            int blockCount = (currentLen + elemsPerBlock - 1) / elemsPerBlock;
            int repeat = (blockCount + 7) / 8;
            AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(currentLen);
            AscendC::BlockReduceMax<T, false>(src, src, repeat, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
            currentLen = blockCount;
        }
        AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(currentLen);
        AscendC::WholeReduceMax<T, false>(dst, src, AscendC::MASK_PLACEHOLDER, 1, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::SetMaskNorm();
        AscendC::ResetMask();  
    }

private:
    __aicore__ inline void ComputeL2Batched(
        int &batch, uint64_t &pair,
        AscendC::LocalTensor<DTYPE_X> &x1Local, 
        AscendC::LocalTensor<DTYPE_X> &x2Local, 
        AscendC::LocalTensor<DTYPE_Y> &yLocal,
        AscendC::LocalTensor<DTYPE_CALC> &outputBuffer,
        AscendC::LocalTensor<DTYPE_CALC> &castTmpBuffer
    ){
        if constexpr(std::is_same_v<DTYPE, half>){ // float16
            for (int i = 0; i < batch; i ++) {
                AscendC::Sub(x2Local[i * this->alignedM], x2Local[i * this->alignedM], x1Local, this->M);
            }
            AscendC::Cast(castTmpBuffer, x2Local, AscendC::RoundMode::CAST_NONE, batch * this->alignedM);
            AscendC::Mul(castTmpBuffer, castTmpBuffer, castTmpBuffer, batch * this->alignedM);
            for (int i = 0; i < batch; i ++) {
                reduceSum(outputBuffer[this->bufferNum ++], castTmpBuffer[i * this->alignedM], this->M);
                pair ++;
                if (this->bufferNum == this->copyOutBlock || pair == endPair) {
                    AscendC::Sqrt(outputBuffer, outputBuffer, this->bufferNum);
                    AscendC::Cast(yLocal, outputBuffer, AscendC::RoundMode::CAST_NONE, this->bufferNum);
                    outQueY.EnQue(yLocal);
                    CopyOut(pair - this->bufferNum, this->bufferNum);
                    yLocal = outQueY.AllocTensor<DTYPE_Y>();
                    this->bufferNum = 0;
                }
            }
        }
        else{ // float32
            for (int i = 0; i < batch; i ++) {
                AscendC::Sub(x2Local[i * this->alignedM], x2Local[i * this->alignedM], x1Local, this->M);
            }
            AscendC::Mul(x2Local, x2Local, x2Local, batch * this->alignedM);
            for (int i = 0; i < batch; i ++) {
                reduceSum(yLocal[this->bufferNum ++], x2Local[i * this->alignedM], this->M);
                pair ++;
                if (this->bufferNum == this->copyOutBlock || pair == endPair) {
                    AscendC::Sqrt(yLocal, yLocal, this->bufferNum);
                    outQueY.EnQue(yLocal);
                    CopyOut(pair - this->bufferNum, this->bufferNum);
                    yLocal = outQueY.AllocTensor<DTYPE_Y>();
                    this->bufferNum = 0;
                }
            }
        }
    }

    __aicore__ inline void ComputeLinfBatched(
        int &batch, uint64_t &pair,
        AscendC::LocalTensor<DTYPE_X> &x1Local, 
        AscendC::LocalTensor<DTYPE_X> &x2Local, 
        AscendC::LocalTensor<DTYPE_Y> &yLocal
    ){
        for (int i = 0; i < batch; i ++) {
            AscendC::Sub(x2Local[i * this->alignedM], x2Local[i * this->alignedM], x1Local, this->M);
        }
        AscendC::Abs(x2Local, x2Local, batch * this->alignedM);
        for (int i = 0; i < batch; i ++) {
            reduceMax(yLocal[this->bufferNum ++], x2Local[i * this->alignedM], this->M);
            pair ++;
            if (this->bufferNum == this->copyOutBlock || pair == endPair) {
                outQueY.EnQue(yLocal);
                CopyOut(pair - this->bufferNum, this->bufferNum);
                yLocal = outQueY.AllocTensor<DTYPE_Y>();
                this->bufferNum = 0;
            }
        }
    }

    __aicore__ inline void ComputeL1Batched(
        int &batch, uint64_t &pair,
        AscendC::LocalTensor<DTYPE_X> &x1Local, 
        AscendC::LocalTensor<DTYPE_X> &x2Local, 
        AscendC::LocalTensor<DTYPE_Y> &yLocal,
        AscendC::LocalTensor<DTYPE_CALC> &outputBuffer,
        AscendC::LocalTensor<DTYPE_CALC> &castTmpBuffer
    ){
        if constexpr(std::is_same_v<DTYPE, half>){ // float16
            for (int i = 0; i < batch; i ++) {
                AscendC::Sub(x2Local[i * this->alignedM], x2Local[i * this->alignedM], x1Local, this->M);
            }
            AscendC::Abs(x2Local, x2Local, batch * this->alignedM);
            AscendC::Cast(castTmpBuffer, x2Local, AscendC::RoundMode::CAST_NONE, batch * this->alignedM);
            for (int i = 0; i < batch; i ++) {
                reduceSum(outputBuffer[this->bufferNum ++], castTmpBuffer[i * this->alignedM], this->M);
                pair ++;
                if (this->bufferNum == this->copyOutBlock || pair == endPair) {
                    AscendC::Cast(yLocal, outputBuffer, AscendC::RoundMode::CAST_NONE, this->bufferNum);
                    outQueY.EnQue(yLocal);
                    CopyOut(pair - this->bufferNum, this->bufferNum);
                    yLocal = outQueY.AllocTensor<DTYPE_Y>();
                    this->bufferNum = 0;
                }
            }
        }
        else{ // float32
            for (int i = 0; i < batch; i ++) {
                AscendC::Sub(x2Local[i * this->alignedM], x2Local[i * this->alignedM], x1Local, this->M);
            }
            AscendC::Abs(x2Local, x2Local, batch * this->alignedM);
            for (int i = 0; i < batch; i ++) {
                reduceSum(yLocal[this->bufferNum ++], x2Local[i * this->alignedM], this->M);
                pair ++;
                if (this->bufferNum == this->copyOutBlock || pair == endPair) {
                    outQueY.EnQue(yLocal);
                    CopyOut(pair - this->bufferNum, this->bufferNum);
                    yLocal = outQueY.AllocTensor<DTYPE_Y>();
                    this->bufferNum = 0;
                }
            }
        }
    }

    __aicore__ inline void ComputeLgeneralBatched(
        int &batch, uint64_t &pair, float &pVal,
        AscendC::LocalTensor<DTYPE_X> &x1Local, 
        AscendC::LocalTensor<DTYPE_X> &x2Local, 
        AscendC::LocalTensor<DTYPE_Y> &yLocal,
        AscendC::LocalTensor<DTYPE_CALC> &outputBuffer,
        AscendC::LocalTensor<DTYPE_CALC> &castTmpBuffer
    ){
        DTYPE_CALC p = static_cast<DTYPE_CALC>(pVal);
        DTYPE_CALC Rp = static_cast<DTYPE_CALC>(1 / pVal);
        if constexpr(std::is_same_v<DTYPE, half>){ // float16
            for (int i = 0; i < batch; i ++) {
                AscendC::Sub(x2Local[i * this->alignedM], x2Local[i * this->alignedM], x1Local, this->M);
            }
            AscendC::Abs(x2Local, x2Local, batch * this->alignedM);
            AscendC::Cast(castTmpBuffer, x2Local, AscendC::RoundMode::CAST_NONE, batch * this->alignedM);
            AscendC::Power(castTmpBuffer, castTmpBuffer, p, batch * this->alignedM);
            for (int i = 0; i < batch; i ++) {
                reduceSum(outputBuffer[this->bufferNum ++], castTmpBuffer[i * this->alignedM], this->M);
                pair ++;
                if (this->bufferNum == this->copyOutBlock || pair == endPair) {
                    AscendC::Power(outputBuffer, outputBuffer, Rp, this->bufferNum);
                    AscendC::Cast(yLocal, outputBuffer, AscendC::RoundMode::CAST_NONE, this->bufferNum);
                    outQueY.EnQue(yLocal);
                    CopyOut(pair - this->bufferNum, this->bufferNum);
                    yLocal = outQueY.AllocTensor<DTYPE_Y>();
                    this->bufferNum = 0;
                }
            }
        }
        else{ // float32
            for (int i = 0; i < batch; i ++) {
                AscendC::Sub(x2Local[i * this->alignedM], x2Local[i * this->alignedM], x1Local, this->M);
            }
            AscendC::Abs(x2Local, x2Local, batch * this->alignedM);
            AscendC::Power(x2Local, x2Local, p, batch * this->alignedM);
            for (int i = 0; i < batch; i ++) {
                reduceSum(yLocal[this->bufferNum ++], x2Local[i * this->alignedM], this->M);
                pair ++;
                if (this->bufferNum == this->copyOutBlock || pair == endPair) {
                    AscendC::Power(yLocal, yLocal, Rp, this->bufferNum);
                    outQueY.EnQue(yLocal);
                    CopyOut(pair - this->bufferNum, this->bufferNum);
                    yLocal = outQueY.AllocTensor<DTYPE_Y>();
                    this->bufferNum = 0;
                }
            }
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
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueY;
    AscendC::TBuf<AscendC::TPosition::VECOUT> outQueBuffer;
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