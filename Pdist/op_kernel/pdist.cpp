#include "kernel_operator.h"

static constexpr int BUFFER_NUM = 2;
static constexpr int MAX_ACC_BUF_SIZE = 256; // 256B (8 block)

enum DataScale { Normal = 0, Huge = 1 };
enum CalcType { General = 0, Manhattan = 1, Euclidean = 2, Chebyshev = 3 };

template<typename DTYPE, DataScale DSCALE, CalcType CTYPE>
class KernelPdist{
    using DTYPE_CALC = std::conditional_t<CTYPE == Chebyshev, DTYPE_Y, 
    std::conditional_t<std::is_same_v<DTYPE, half>, float, DTYPE_Y>>;
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
        if (startPair >= endPair) return;
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
        this->startRow = ans;
        this->startCol = startPair - 1ull * ans * (2 * N - ans - 1) / 2 + ans + 1;
        
        this->copyOutBlock = tiling_data.copyOutBlock;

        this->pVal = tiling_data.pVal;

        this->batchSize = tiling_data.batchSize;

        this->processM = tiling_data.processM;
        this->accBlock = MAX_ACC_BUF_SIZE / sizeof(float);
        this->accNum = this->bufferNum = 0;

        this->copyParams.blockLen = this->M * sizeof(DTYPE_X);
        this->copyParams.srcStride = 0;
        this->copyParams.dstStride = 0;
        this->padParams.isPad = Huge;
        this->padParams.leftPadding = 0;
        this->padParams.rightPadding = (this->alignedM - this->M);
        this->padParams.paddingValue = 0;
        
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, 1ull * N * M);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, (1ull * N * (N - 1) / 2 + this->alignNum - 1) / this->alignNum * this->alignNum); // aligned output
        pipe = pipeIn;
        pipe->InitBuffer(inQueFirst, BUFFER_NUM, this->processM * sizeof(DTYPE_X));
        pipe->InitBuffer(inQueSecond, BUFFER_NUM, 1ull * this->batchSize * this->processM * sizeof(DTYPE_X));
        pipe->InitBuffer(outQueY, BUFFER_NUM, this->copyOutBlock * sizeof(DTYPE_Y));
        pipe->InitBuffer(outQueBuffer, this->copyOutBlock * sizeof(float));
        if constexpr (std::is_same_v<DTYPE, half>) {
            pipe->InitBuffer(castBuffer, 1ull * this->batchSize * this->processM * sizeof(float)); // shared cast buffer
        }
        if constexpr (DSCALE == Huge) {
            pipe->InitBuffer(accBuffer, MAX_ACC_BUF_SIZE);
        }
        this->bufferNum = 0;
    }

    __aicore__ inline void Process(){
        if (this->startPair >= this->endPair) return;

        if constexpr (DSCALE == Huge) ProcessHuge();
        else ProcessNormal();
    }
private:
    __aicore__ inline void ProcessNormal() {
        AscendC::LocalTensor<DTYPE_CALC> outputBuffer = outQueBuffer.Get<DTYPE_CALC>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_CALC> castTmpBuffer;
        if constexpr(std::is_same_v<DTYPE, half>) {
            castTmpBuffer = castBuffer.Get<DTYPE_CALC>();
        }
        int pair = startPair;
        int batch;
        int i = this->startRow, j = this->startCol;
        for (; i < N && pair < endPair; i ++) {
            CopyInFirstNormal(i);
            AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
            for (; j < N && pair < endPair; j += batch) {
                batch = min(min(N - j, endPair - pair), this->batchSize);
                CopyInSecondNormal(j, batch);
                AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
                ComputeNormal(batch, pair, x1Local, x2Local, yLocal, outputBuffer, castTmpBuffer);
                inQueSecond.FreeTensor(x2Local);
            }
            inQueFirst.FreeTensor(x1Local);
            j = i + 2;
        }
    }

    __aicore__ inline void ProcessHuge() {
        AscendC::LocalTensor<DTYPE_CALC> outputBuffer = outQueBuffer.Get<DTYPE_CALC>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_CALC> castTmpBuffer;
        AscendC::LocalTensor<DTYPE_CALC> accumulateBuffer = accBuffer.Get<DTYPE_CALC>();
        if constexpr (std::is_same_v<DTYPE, half>) {
            castTmpBuffer = castBuffer.Get<DTYPE_CALC>();
        }
        int i = this->startRow, j = this->startCol;
        int pair = startPair;
        for (; i < N && pair < endPair; i ++) {
            for (; j < N && pair < endPair; j ++) {
                DTYPE_CALC zero = 0;
                AscendC::Duplicate(accumulateBuffer, zero, MAX_ACC_BUF_SIZE / sizeof(DTYPE_CALC));
                for (int offset = 0, totalElements; offset < this->M; offset += totalElements) {
                    totalElements = min(this->M - offset, this->processM);
                    CopyInFirstHuge(i, offset, totalElements);
                    CopyInSecondHuge(j, offset, totalElements);
                    AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
                    AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
                    ComputeHuge(pair, totalElements, offset, x1Local, x2Local, yLocal, outputBuffer, castTmpBuffer, accumulateBuffer);
                    inQueFirst.FreeTensor(x1Local);
                    inQueSecond.FreeTensor(x2Local);
                }
            }
            j = i + 2;
        }
    }

private:
    __aicore__ inline void CopyInFirstNormal(const int &i){
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(x1Local, xGm[1ull * i * this->M], this->alignedM);
        inQueFirst.EnQue(x1Local);
    }

    __aicore__ inline void CopyInSecondNormal(const int &j, const int &batch){
        this->copyParams.blockCount = batch;
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.AllocTensor<DTYPE_X>();
        AscendC::DataCopyPad(x2Local, xGm[1ull * j * this->M], this->copyParams, this->padParams);
        inQueSecond.EnQue(x2Local);
    }

    __aicore__ inline void CopyInFirstHuge(int i, int offset, int totalElements){
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(x1Local, xGm[1ull * i * this->M + offset], (totalElements + this->alignNum - 1) / this->alignNum * this->alignNum);
        inQueFirst.EnQue(x1Local);
    }

    __aicore__ inline void CopyInSecondHuge(int j, int offset, int totalElements){
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(x2Local, xGm[1ull * j * this->M + offset], (totalElements + this->alignNum - 1) / this->alignNum * this->alignNum);
        inQueSecond.EnQue(x2Local);
    }

    __aicore__ inline void CopyOut(int startIdx, int outSizeB) {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.DeQue<DTYPE_Y>();
        DataCopy(yGm[startIdx], yLocal, (outSizeB + this->alignNum - 1) / this->alignNum * this->alignNum);
        outQueY.FreeTensor(yLocal);
    }

private:
    template <typename T>
    __aicore__ inline void ReduceSumNormal(const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src, const int &totalElements) {   
        constexpr int elemsPerBlock = 32 / sizeof(T);
        int currentLen = totalElements;
        AscendC::SetMaskCount();
        while (currentLen > (elemsPerBlock * 8)) {
            int blockCount = (currentLen + elemsPerBlock - 1) / elemsPerBlock;
            int repeat = (blockCount + 7) / 8;
            AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(currentLen);
            AscendC::BlockReduceSum<T, Normal>(src, src, repeat, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
            currentLen = blockCount;
        }
        AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(currentLen);
        AscendC::WholeReduceSum<T, Normal>(dst, src, AscendC::MASK_PLACEHOLDER, 1, 1, 1, 8);
        AscendC::SetMaskNorm();
        AscendC::ResetMask();  
    }

    template <typename T>
    __aicore__ inline void ReduceMaxNormal(const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src, const int &totalElements) {   
        constexpr int elemsPerBlock = 32 / sizeof(T);
        int currentLen = totalElements;
        AscendC::SetMaskCount();
        while (currentLen > (elemsPerBlock * 8)) {
            int blockCount = (currentLen + elemsPerBlock - 1) / elemsPerBlock;
            int repeat = (blockCount + 7) / 8;
            AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(currentLen);
            AscendC::BlockReduceMax<T, Normal>(src, src, repeat, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
            currentLen = blockCount;
        }
        AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(currentLen);
        AscendC::WholeReduceMax<T, Normal>(dst, src, AscendC::MASK_PLACEHOLDER, 1, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        AscendC::SetMaskNorm();
        AscendC::ResetMask();  
    }

    template <typename T>
    __aicore__ inline void ReduceSumHuge(const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src, const int &totalElements) {   
        constexpr int elemsPerBlock = 32 / sizeof(T);
        int currentLen = totalElements;
        AscendC::SetMaskCount();
        while (currentLen > (elemsPerBlock * 8)) {
            int blockCount = (currentLen + elemsPerBlock - 1) / elemsPerBlock;
            int repeat = (blockCount + 7) / 8;
            AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(currentLen);
            AscendC::BlockReduceSum<T, Normal>(src, src, repeat, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
            currentLen = blockCount;
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();  
        AscendC::Add(dst, dst, src, currentLen);
    }

    template <typename T>
    __aicore__ inline void ReduceMaxHuge(const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src, const int &totalElements) {   
        constexpr int elemsPerBlock = 32 / sizeof(T);
        int currentLen = totalElements;
        AscendC::SetMaskCount();
        while (currentLen > (elemsPerBlock * 8)) {
            int blockCount = (currentLen + elemsPerBlock - 1) / elemsPerBlock;
            int repeat = (blockCount + 7) / 8;
            AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(currentLen);
            AscendC::BlockReduceMax<T, Normal>(src, src, repeat, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
            currentLen = blockCount;
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();  
        AscendC::Max(dst, dst, src, currentLen);
    }

private:
    __aicore__ inline void ComputeNormal(
        int &batch, int &pair,
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
            if constexpr (CTYPE != Euclidean) {
                AscendC::Abs(x2Local, x2Local, batch * this->alignedM);
            }
            if constexpr (CTYPE != Chebyshev) {
                AscendC::Cast(castTmpBuffer, x2Local, AscendC::RoundMode::CAST_NONE, batch * this->alignedM);
            }
            if constexpr (CTYPE == Euclidean) {
                AscendC::Mul(castTmpBuffer, castTmpBuffer, castTmpBuffer, batch * this->alignedM);
            }
            if constexpr (CTYPE == General) {
                DTYPE_CALC p = static_cast<DTYPE_CALC>(pVal);
                AscendC::Power(castTmpBuffer, castTmpBuffer, p, batch * this->alignedM);
            }
            for (int i = 0; i < batch; i ++) {
                if constexpr (CTYPE == Chebyshev) {
                    ReduceMaxNormal(yLocal[this->bufferNum ++], x2Local[i * this->alignedM], this->M);
                }
                else {
                    ReduceSumNormal(outputBuffer[this->bufferNum ++], castTmpBuffer[i * this->alignedM], this->M);
                }
                pair ++;
                if (this->bufferNum == this->copyOutBlock || pair == endPair) {
                    if constexpr (CTYPE == Euclidean) {
                        AscendC::Sqrt(outputBuffer, outputBuffer, this->bufferNum);
                    }
                    if constexpr (CTYPE == General) {
                        DTYPE_CALC Rp = static_cast<DTYPE_CALC>(1 / pVal);
                        AscendC::Power(outputBuffer, outputBuffer, Rp, this->bufferNum);
                    }
                    if constexpr (CTYPE != Chebyshev) {
                        AscendC::Cast(yLocal, outputBuffer, AscendC::RoundMode::CAST_NONE, this->bufferNum);
                    }
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
            if constexpr (CTYPE == Euclidean) {
                AscendC::Mul(x2Local, x2Local, x2Local, batch * this->alignedM);
            }
            else {
                AscendC::Abs(x2Local, x2Local, batch * this->alignedM);
            }
            if constexpr (CTYPE == General) {
                DTYPE_CALC p = static_cast<DTYPE_CALC>(pVal);
                AscendC::Power(x2Local, x2Local, p, batch * this->alignedM);
            }
            for (int i = 0; i < batch; i ++) {
                if constexpr (CTYPE == Chebyshev) {
                    ReduceMaxNormal(yLocal[this->bufferNum ++], x2Local[i * this->alignedM], this->M);
                }
                else {
                    ReduceSumNormal(yLocal[this->bufferNum ++], x2Local[i * this->alignedM], this->M);
                }
                pair ++;
                if (this->bufferNum == this->copyOutBlock || pair == endPair) {
                    if constexpr (CTYPE == Euclidean) {
                        AscendC::Sqrt(yLocal, yLocal, this->bufferNum);
                    }
                    if constexpr (CTYPE == General) {
                        DTYPE_CALC Rp = static_cast<DTYPE_CALC>(1 / pVal);
                        AscendC::Power(yLocal, yLocal, Rp, this->bufferNum);
                    }
                    outQueY.EnQue(yLocal);
                    CopyOut(pair - this->bufferNum, this->bufferNum);
                    yLocal = outQueY.AllocTensor<DTYPE_Y>();
                    this->bufferNum = 0;
                }
            }
        }
    }

    __aicore__ inline void ComputeHuge(
        int &pair, int &totalElements, int &offset, 
        AscendC::LocalTensor<DTYPE_X> &x1Local, 
        AscendC::LocalTensor<DTYPE_X> &x2Local, 
        AscendC::LocalTensor<DTYPE_Y> &yLocal,
        AscendC::LocalTensor<DTYPE_CALC> &outputBuffer,
        AscendC::LocalTensor<DTYPE_CALC> &castTmpBuffer,
        AscendC::LocalTensor<DTYPE_CALC> &accumulateBuffer
    ){
        if constexpr(std::is_same_v<DTYPE, half>){ // float16
            AscendC::Sub(x2Local, x2Local, x1Local, totalElements);
            if constexpr (CTYPE != Euclidean) {
                AscendC::Abs(x2Local, x2Local, totalElements);
            }
            if constexpr (CTYPE != Chebyshev) {
                AscendC::Cast(castTmpBuffer, x2Local, AscendC::RoundMode::CAST_NONE, totalElements);
            }
            if constexpr (CTYPE == Euclidean) {
                AscendC::Mul(castTmpBuffer, castTmpBuffer, castTmpBuffer, totalElements);
            }
            if constexpr (CTYPE == General) {
                DTYPE_CALC p = static_cast<DTYPE_CALC>(pVal);
                AscendC::Power(castTmpBuffer, castTmpBuffer, p, totalElements);
            }
            if constexpr (CTYPE == Chebyshev) {
                ReduceMaxHuge(accumulateBuffer, x2Local, totalElements);
            }
            else {
                ReduceSumHuge(accumulateBuffer, castTmpBuffer, totalElements);
            }
            if (offset + totalElements >= this->M) {
                if constexpr (CTYPE == Chebyshev) {
                    AscendC::WholeReduceMax<DTYPE_CALC, Normal>(yLocal[this->bufferNum ++], accumulateBuffer, MAX_ACC_BUF_SIZE / sizeof(DTYPE_CALC), 1, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
                }
                else {
                    AscendC::WholeReduceSum<DTYPE_CALC, Normal>(outputBuffer[this->bufferNum ++], accumulateBuffer, MAX_ACC_BUF_SIZE / sizeof(DTYPE_CALC), 1, 1, 1, 8);
                }
                pair ++;
                if (this->bufferNum == this->copyOutBlock || pair == endPair) {
                    if constexpr (CTYPE == Euclidean) {
                        AscendC::Sqrt(outputBuffer, outputBuffer, this->bufferNum);
                    }
                    if constexpr (CTYPE == General) {
                        DTYPE_CALC Rp = static_cast<DTYPE_CALC>(1 / pVal);
                        AscendC::Power(outputBuffer, outputBuffer, Rp, this->bufferNum);
                    }
                    if constexpr (CTYPE != Chebyshev) {
                        AscendC::Cast(yLocal, outputBuffer, AscendC::RoundMode::CAST_NONE, this->bufferNum);
                    }
                    outQueY.EnQue(yLocal);
                    CopyOut(pair - this->bufferNum, this->bufferNum);
                    yLocal = outQueY.AllocTensor<DTYPE_Y>();
                    this->bufferNum = 0;
                }
            }
        }
        else{ // float32
            AscendC::Sub(x2Local, x2Local, x1Local, totalElements);
            if constexpr (CTYPE == Euclidean) {
                AscendC::Mul(x2Local, x2Local, x2Local, totalElements);
            }
            else {
                AscendC::Abs(x2Local, x2Local, totalElements);
            }
            if constexpr (CTYPE == General) {
                DTYPE_CALC p = static_cast<DTYPE_CALC>(pVal);
                AscendC::Power(x2Local, x2Local, p, totalElements);
            }
            if constexpr (CTYPE == Chebyshev) {
                ReduceMaxHuge(accumulateBuffer, x2Local, totalElements);
            }
            else {
                ReduceSumHuge(accumulateBuffer, x2Local, totalElements);
            }
            if (offset + totalElements >= this->M) {
                if constexpr (CTYPE == Chebyshev) {
                    AscendC::WholeReduceMax<DTYPE_CALC, Normal>(yLocal[this->bufferNum ++], accumulateBuffer, MAX_ACC_BUF_SIZE / sizeof(DTYPE_CALC), 1, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
                }
                else {
                    AscendC::WholeReduceSum<DTYPE_CALC, Normal>(yLocal[this->bufferNum ++], accumulateBuffer, MAX_ACC_BUF_SIZE / sizeof(DTYPE_CALC), 1, 1, 1, 8);
                }
                pair ++;
                if (this->bufferNum == this->copyOutBlock || pair == endPair) {
                    if constexpr (CTYPE == Euclidean) {
                        AscendC::Sqrt(yLocal, yLocal, this->bufferNum);
                    }
                    if constexpr (CTYPE == General) {
                        DTYPE_CALC Rp = static_cast<DTYPE_CALC>(1 / pVal);
                        AscendC::Power(yLocal, yLocal, Rp, this->bufferNum);
                    }
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
    int startRow, startCol;
    float pVal;
    int startPair, endPair;
    int copyOutBlock;
    int alignNum, bufferNum;
    int alignedM;
    int batchSize;
    int processM;
    int accNum, accBlock; // the number of valid items in accBuffer and the block size

    AscendC::DataCopyParams copyParams;
    AscendC::DataCopyPadParams padParams;

    AscendC::TPipe *pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueFirst;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueSecond;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueY;
    AscendC::TBuf<AscendC::TPosition::VECOUT> outQueBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castBuffer;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuffer; // Accumulation Buffer for huge data
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
};

template<typename DTYPE, DataScale DSCALE, CalcType CTYPE>
__aicore__ inline void RunOp(GM_ADDR &x, GM_ADDR &y, PdistTilingData &tiling_data) {
    AscendC::TPipe pipe;
    KernelPdist<DTYPE, DSCALE, CTYPE> op;
    op.Init(x, y, tiling_data, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void pdist(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    AscendC::TPipe pipe;
    if (tiling_data.isHugeData) {
        if (tiling_data.dataType == DT_FLOAT16){
            switch (tiling_data.pType) {
                case General: RunOp<half, Huge, General>(x, y, tiling_data); break;
                case Manhattan: RunOp<half, Huge, Manhattan>(x, y, tiling_data); break;
                case Euclidean: RunOp<half, Huge, Euclidean>(x, y, tiling_data); break;
                case Chebyshev: RunOp<half, Huge, Chebyshev>(x, y, tiling_data); break;
                default: break;
            }
        }
        else{
            switch (tiling_data.pType) {
                case General: RunOp<float, Huge, General>(x, y, tiling_data); break;
                case Manhattan: RunOp<float, Huge, Manhattan>(x, y, tiling_data); break;
                case Euclidean: RunOp<float, Huge, Euclidean>(x, y, tiling_data); break;
                case Chebyshev: RunOp<float, Huge, Chebyshev>(x, y, tiling_data); break;
                default: break;
            }
        } 
    }
    else {
        if (tiling_data.dataType == DT_FLOAT16){
            switch (tiling_data.pType) {
                case General: RunOp<half, Normal, General>(x, y, tiling_data); break;
                case Manhattan: RunOp<half, Normal, Manhattan>(x, y, tiling_data); break;
                case Euclidean: RunOp<half, Normal, Euclidean>(x, y, tiling_data); break;
                case Chebyshev: RunOp<half, Normal, Chebyshev>(x, y, tiling_data); break;
                default: break;
            }
        }
        else{
            switch (tiling_data.pType) {
                case General: RunOp<float, Normal, General>(x, y, tiling_data); break;
                case Manhattan: RunOp<float, Normal, Manhattan>(x, y, tiling_data); break;
                case Euclidean: RunOp<float, Normal, Euclidean>(x, y, tiling_data); break;
                case Chebyshev: RunOp<float, Normal, Chebyshev>(x, y, tiling_data); break;
                default: break;
            }
        } 
    }
}