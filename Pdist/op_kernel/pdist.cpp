/**
 * @file pdist.cpp
 * @authors Tianyang Liu (@jjsnam), Chen Xu (@Sushidesu486)
 * @brief kernel implementation of AscendC p-distance operator
 * @version 2.0 
 * @note Updated after checkpoint at 2025-12-22
 * @date 2025-12-30
 * 
 * @copyright Copyright (c) 2025 ZJUSCT
 * This implementation is developed by the authors using the AscendC programming model and APIs provided by Huawei Technologies Co., Ltd.
 * AscendC and Ascend are trademarks of Huawei Technologies Co., Ltd.
 */
#include "kernel_operator.h"

static constexpr int BUFFER_NUM = 2; // Buffer number

static constexpr int ACC_BLOCK_SIZE = 256; // Huge data handling's accumulate buffer (256B as a block (8 datablocks))
static constexpr int TILE_HUGE = 16; // Tile size when computing huge scale data
static constexpr int ACC_BUF_SIZE = TILE_HUGE * ACC_BLOCK_SIZE; // Total buffer's size

/**
 * @brief Same declaration of DataScale and CalcType, see op_host/pdist.cpp for detail
 */
enum DataScale { Normal = 0, Huge = 1 };
enum CalcType { General = 0, Manhattan = 1, Euclidean = 2, Chebyshev = 3 };

/**
 * @brief p-dist Ascend C operator kernel implementation (class with template)
 * 
 * @tparam DTYPE: Major datatype (float or half)
 * @tparam DSCALE: Data scale (Normal or Huge)
 * @tparam CTYPE: Calculation kernel type (General / Manhattan / Euclidean / Chebyshev)
 *
 * @details Contains many constexpr() to reduce the total lines of code and repeated codes
 */
template<typename DTYPE, DataScale DSCALE, CalcType CTYPE>
class KernelPdist{
    // First determin the datatype used in calculation
    // Always the same type ad DTYPE, except for FP16 with cast requirements
    using DTYPE_CALC = std::conditional_t<CTYPE == Chebyshev, DTYPE_Y, 
    std::conditional_t<std::is_same_v<DTYPE, half>, float, DTYPE_Y>>;
public:
    __aicore__ inline KernelPdist() {}
    __aicore__ inline ~KernelPdist() {}

    /**
     * @brief Init the operator's info at te current ai-core
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, PdistTilingData &tiling_data, AscendC::TPipe * pipeIn) {
        this->blockIdx = AscendC::GetBlockIdx();
        this->N = tiling_data.N;
        this->M = tiling_data.M;
        this->alignNum = tiling_data.alignNum;
        this->alignedM = tiling_data.alignedM;

        // Calculate the task (start and end) on this kernel
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
        InitGetStartIndex(); // Get the start index (i, j)
        
        this->copyOutTile = tiling_data.copyOutTile;

        this->pVal = tiling_data.pVal;

        this->batchSize = tiling_data.batchSize;

        this->processM = tiling_data.processM;

        this->accBlock = ACC_BLOCK_SIZE / sizeof(DTYPE_CALC); // Total number of data blocks (8)
        this->accNum = this->bufferNum = 0;
        
        // Batched Copy Params (padding 0 for unaligned tail elements)
        this->copyParams.blockLen = this->M * sizeof(DTYPE_X);
        this->copyParams.srcStride = 0;
        this->copyParams.dstStride = 0;
        this->padParams.isPad = true;
        this->padParams.leftPadding = 0;
        this->padParams.rightPadding = (this->alignedM - this->M); // padding 0 for unaligned tail elements
        this->padParams.paddingValue = 0;
        
        // Initialize memory buffers
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, 1ull * N * M);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, (1ull * N * (N - 1) / 2 + this->alignNum - 1) / this->alignNum * this->alignNum); // aligned output
        this->pipe = pipeIn;
        pipe->InitBuffer(inQueFirst, BUFFER_NUM, this->processM * sizeof(DTYPE_X));
        pipe->InitBuffer(inQueSecond, BUFFER_NUM, 1ull * this->batchSize * this->processM * sizeof(DTYPE_X));
        pipe->InitBuffer(outQueY, BUFFER_NUM, this->copyOutTile * sizeof(DTYPE_Y));
        if constexpr (std::is_same_v<DTYPE, half> && std::is_same_v<DTYPE_CALC, float>) { // Cast is needed
            pipe->InitBuffer(castBuffer, 1ull * this->batchSize * this->processM * sizeof(float)); // shared cast buffer
            pipe->InitBuffer(outQueBuffer, this->copyOutTile * sizeof(float));
        }
        if constexpr (DSCALE == Huge) { // Accumulate buffer needed for Huge data
            pipe->InitBuffer(accBuffer, ACC_BUF_SIZE);
        }
        this->bufferNum = 0;
    }

    /**
     * @brief Major calculation pipeline
     */
    __aicore__ inline void Process(){
        if (this->startPair >= this->endPair) return; // No task on this core

        // Choose different process logic for different scale
        if constexpr (DSCALE == Huge) ProcessHuge();
        else ProcessNormal();
    }


private:
    /**
     * @brief Calculate the start indices (i, j) with binary search
     */
    __aicore__ inline void InitGetStartIndex() {
        // Using binary search to determine i
        int l = 0, r = N - 1, mid, ans;
        while (l <= r){
            mid = (l + r) >> 1;
            // Calculate the current total pairs for the first {mid + 1} rows
            uint64_t totalPairs = 1ull * (mid + 1) * (2 * N - mid - 2) / 2;
            if (totalPairs > startPair){
                ans = mid;
                r = mid - 1;
            }
            else{
                l = mid + 1;
            }
        }
        this->startI = ans;
        this->startJ = startPair - 1ull * ans * (2 * N - ans - 1) / 2 + ans + 1;
    }


private:
    /**
     * @brief Process logic for normal data scale
     */
    __aicore__ inline void ProcessNormal() {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_CALC> outputBuffer, castTmpBuffer;
        if constexpr(std::is_same_v<DTYPE, half> && std::is_same_v<DTYPE_CALC, float>) { // Cast calculation needed
            outputBuffer = outQueBuffer.Get<DTYPE_CALC>();
            castTmpBuffer = castBuffer.Get<DTYPE_CALC>();
        }
        int pair = startPair;
        int batch;
        int i = this->startI, j = this->startJ;
        for (; i < N && pair < endPair; i ++) { // Iterate over the first vector
            CopyInFirstNormal(i);
            AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
            for (; j < N && pair < endPair; j += batch) { // Iterate over the second vector
                batch = min(min(N - j, endPair - pair), this->batchSize); // Determine the batch size
                CopyInSecondNormal(j, batch);
                AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
                ComputeNormal(batch, pair, x1Local, x2Local, yLocal, outputBuffer, castTmpBuffer); // Compute
                inQueSecond.FreeTensor(x2Local);
            }
            inQueFirst.FreeTensor(x1Local);
            j = i + 2; // Update j for next i (j = (i + 1) + 1)
        }
    }

    /**
     * @brief Process logic for huge data scale
     *
     * @details Use tiling to improve data reuse and reduce memory access
     */
    __aicore__ inline void ProcessHuge() {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_CALC> outputBuffer, castTmpBuffer;
        AscendC::LocalTensor<DTYPE_CALC> accumulateBuffer = accBuffer.Get<DTYPE_CALC>(); // Accumulate buffer
        if constexpr (std::is_same_v<DTYPE, half> && std::is_same_v<DTYPE_CALC, float>) { // Cast calculation needed
            outputBuffer = outQueBuffer.Get<DTYPE_CALC>();
            castTmpBuffer = castBuffer.Get<DTYPE_CALC>();
        }
        int i = this->startI, j = this->startJ;
        int pair = startPair;
        for (; i < N && pair < endPair; i ++) {
            for (; j < N && pair < endPair; j += TILE_HUGE) {
                int tileSize = min(min(N - j, endPair - pair), TILE_HUGE); // Get the remain tile size

                DTYPE_CALC zero = 0;
                AscendC::Duplicate(accumulateBuffer, zero, this->accBlock * tileSize); // Initialize accumulate buffer

                for (int offset = 0, totalElements; offset < this->M; offset += totalElements) { // Iterate each tile of the vectors
                    totalElements = min(this->M - offset, this->processM);
                    // Copy in the first vector only once
                    CopyInFirstHuge(i, offset, totalElements);
                    AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
                    for (int k = 0; k < tileSize; k ++) {
                        CopyInSecondHuge(j + k, offset, totalElements);
                        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
                        ComputeHuge(pair, totalElements, offset, x1Local, x2Local, yLocal, outputBuffer, castTmpBuffer, accumulateBuffer[k * this->accBlock]);
                        inQueSecond.FreeTensor(x2Local);
                    }
                    inQueFirst.FreeTensor(x1Local);
                }
            }
            j = i + 2;
        }
    }


/**
 * @brief Data Copy Implementations
 */
private:
    __aicore__ inline void CopyInFirstNormal(const int i){
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(x1Local, xGm[1ull * i * this->M], this->alignedM);
        inQueFirst.EnQue(x1Local);
    }

    __aicore__ inline void CopyInSecondNormal(const int j, const int batch){
        this->copyParams.blockCount = batch;
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.AllocTensor<DTYPE_X>();
        // Batched copy with padding
        AscendC::DataCopyPad(x2Local, xGm[1ull * j * this->M], this->copyParams, this->padParams);
        inQueSecond.EnQue(x2Local);
    }

    __aicore__ inline void CopyInFirstHuge(int i, int offset, int totalElements){
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.AllocTensor<DTYPE_X>();
        // offset is needed, and copy number needs to be aligned in 32B, so using ceiling
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
        // Use ceiling to align the output size in 32B
        DataCopy(yGm[startIdx], yLocal, (outSizeB + this->alignNum - 1) / this->alignNum * this->alignNum);
        outQueY.FreeTensor(yLocal);
    }


/**
 * @brief Manual implemented reduce logics
 * @ref 选择低延迟指令，优化归约操作性能 "https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0031.html"
 *
 * @details First use BlockReduce to reduce the result into at most 256B (8 block)
 * For Normal Size, a WholeReduce is then needed to reduce to one result
 * But for Huge Size, we can use add/max to store the reduce into accumulate buffer, and finally WholeReduce only once
 */
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
            AscendC::BlockReduceSum<T, false>(src, src, repeat, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
            currentLen = blockCount;
        }
        AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(currentLen);
        // WholeReduce Once
        AscendC::WholeReduceSum<T, false>(dst, src, AscendC::MASK_PLACEHOLDER, 1, 1, 1, 8);
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
            AscendC::BlockReduceMax<T, false>(src, src, repeat, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
            currentLen = blockCount;
        }
        AscendC::SetVectorMask<T, AscendC::MaskMode::COUNTER>(currentLen);
        AscendC::WholeReduceMax<T, false>(dst, src, AscendC::MASK_PLACEHOLDER, 1, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
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
            AscendC::BlockReduceSum<T, false>(src, src, repeat, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
            currentLen = blockCount;
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();  
        // No WholeReduce here, do it later
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
            AscendC::BlockReduceMax<T, false>(src, src, repeat, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
            currentLen = blockCount;
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();  
        AscendC::Max(dst, dst, src, currentLen);
    }


/**
 * @brief Compute Implementations
 *
 * @note We use constexpr() to merge four kinds of calculation into one function
 * @details Below is each kind of calculation's explicit pipeline
 * - General: Sub -> Abs (-> Cast) -> Power(p) -> ReduceSum -> Power(1/p) (-> Cast)
 * - Manhattan: Sub -> Abs (-> Cast) -> ReduceSum (-> Cast)
 * - Euclidean: Sub (-> Cast) -> Mul -> ReduceSum -> Sqrt (-> Cast)
 * - Chebyshev: Sub -> Abs -> ReduceMax
 * @note Operations after Reduce is performed batched once in the output tensor
 */
private:
    /**
     * @brief Compute implementations of normal scale
     * 
     * @param batch The batch size of this calculation
     * @param pair The index of this solution
     * @param x1Local The first pair tensor
     * @param x2Local The second pair tensor
     * @param yLocal The output local tensor
     * @param outputBuffer Output Buffer when casting
     * @param castTmpBuffer Cast Buffer
     */
    __aicore__ inline void ComputeNormal(
        int &batch, int &pair,
        AscendC::LocalTensor<DTYPE_X> &x1Local, 
        AscendC::LocalTensor<DTYPE_X> &x2Local, 
        AscendC::LocalTensor<DTYPE_Y> &yLocal,
        AscendC::LocalTensor<DTYPE_CALC> &outputBuffer,
        AscendC::LocalTensor<DTYPE_CALC> &castTmpBuffer
    ){
        if constexpr(std::is_same_v<DTYPE, half>){ // float16 (check cast)
            for (int i = 0; i < batch; i ++) { // Note that x2 is batched but x1 isn't
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
            // Output logic, iterate each output result
            for (int i = 0; i < batch; i ++) {
                if constexpr (CTYPE == Chebyshev) {
                    ReduceMaxNormal(yLocal[this->bufferNum ++], x2Local[i * this->alignedM], this->M);
                }
                else {
                    ReduceSumNormal(outputBuffer[this->bufferNum ++], castTmpBuffer[i * this->alignedM], this->M);
                }
                pair ++;
                if (this->bufferNum == this->copyOutTile || pair == endPair) { // If one output size is reached
                    // Perform the remain operations in the output tensor
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
                    CopyOut(pair - this->bufferNum, this->bufferNum); // Copy out in a batch once
                    yLocal = outQueY.AllocTensor<DTYPE_Y>();
                    this->bufferNum = 0; // Clear the output tensor
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
                if (this->bufferNum == this->copyOutTile || pair == endPair) {
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

    /**
     * @brief Compute implementations of normal scale
     * 
     * @param totalElements Tile size (#elements) in this calculation
     * @param offset The current tile's offset to start
     * @param accumulateBuffer The accmulate buffer where stores the result temporarily
     * @param ... Same to Normal
     */
    __aicore__ inline void ComputeHuge(
        int &pair, int &totalElements, int &offset, 
        AscendC::LocalTensor<DTYPE_X> &x1Local, 
        AscendC::LocalTensor<DTYPE_X> &x2Local, 
        AscendC::LocalTensor<DTYPE_Y> &yLocal,
        AscendC::LocalTensor<DTYPE_CALC> &outputBuffer,
        AscendC::LocalTensor<DTYPE_CALC> &castTmpBuffer,
        AscendC::LocalTensor<DTYPE_CALC> accumulateBuffer
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
            // Reduce the tile's result into accumulate buffer
            if constexpr (CTYPE == Chebyshev) {
                ReduceMaxHuge(accumulateBuffer, x2Local, totalElements);
            }
            else {
                ReduceSumHuge(accumulateBuffer, castTmpBuffer, totalElements);
            }
            if (offset + totalElements >= this->M) { // Reached the last tile, it's time to summarize all the result
                // First use once WholeReduce to merge the results
                if constexpr (CTYPE == Chebyshev) {
                    AscendC::WholeReduceMax<DTYPE_CALC, Normal>(yLocal[this->bufferNum ++], accumulateBuffer, this->accBlock, 1, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
                }
                else {
                    AscendC::WholeReduceSum<DTYPE_CALC, Normal>(outputBuffer[this->bufferNum ++], accumulateBuffer, this->accBlock, 1, 1, 1, 8);
                }
                // Then the analogous logic like normal
                pair ++;
                if (this->bufferNum == this->copyOutTile || pair == endPair) {
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
                    AscendC::WholeReduceMax<DTYPE_CALC, Normal>(yLocal[this->bufferNum ++], accumulateBuffer, this->accBlock, 1, 1, 1, 8, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
                }
                else {
                    AscendC::WholeReduceSum<DTYPE_CALC, Normal>(yLocal[this->bufferNum ++], accumulateBuffer, this->accBlock, 1, 1, 1, 8);
                }
                pair ++;
                if (this->bufferNum == this->copyOutTile || pair == endPair) {
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

/**
 * @brief Private declarations
 */
private:
    int blockIdx;
    int N, M;
    int startI, startJ;
    float pVal;
    int startPair, endPair;
    int copyOutTile;
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

/**
 * @brief Operator Runner
 * 
 * @tparam DTYPE: Data type
 * @tparam DSCALE: Data scale
 * @tparam CTYPE: Calculation type (p-type)
 * @param x Gm x
 * @param y Gm y
 * @param tiling_data 
 */
template<typename DTYPE, DataScale DSCALE, CalcType CTYPE>
__aicore__ inline void RunOp(GM_ADDR &x, GM_ADDR &y, PdistTilingData &tiling_data) {
    AscendC::TPipe pipe;
    KernelPdist<DTYPE, DSCALE, CTYPE> op; // instantiate the corresponding operator kernel
    op.Init(x, y, tiling_data, &pipe); // Init first
    op.Process(); // And process (do calculations)
}

/**
 * @brief Dispatch operator running logic based on p-Type
 */
template<typename DTYPE, DataScale DSCALE>
__aicore__ inline void DispatchPType(GM_ADDR &x, GM_ADDR &y, PdistTilingData &tiling_data) {
    switch (tiling_data.pType) {
        case General: RunOp<DTYPE, DSCALE, General>(x, y, tiling_data); break;
        case Manhattan: RunOp<DTYPE, DSCALE, Manhattan>(x, y, tiling_data); break;
        case Euclidean: RunOp<DTYPE, DSCALE, Euclidean>(x, y, tiling_data); break;
        case Chebyshev: RunOp<DTYPE, DSCALE, Chebyshev>(x, y, tiling_data); break;
        default: break;
    }
}

/**
 * @brief Operator entry
 */
extern "C" __global__ __aicore__ void pdist(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling); // Get tiling data from host

    // Run operator based on different tiling infos
    if (tiling_data.dataScale == Huge) {
        if (tiling_data.dataType == DT_FLOAT16){
            DispatchPType<half, Huge>(x, y, tiling_data);
        }
        else{
            DispatchPType<float, Huge>(x, y, tiling_data);
        } 
    }
    else {
        if (tiling_data.dataType == DT_FLOAT16){
            DispatchPType<half, Normal>(x, y, tiling_data);
        }
        else{
            DispatchPType<float, Normal>(x, y, tiling_data);
        } 
    }
}