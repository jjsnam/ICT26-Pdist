#include "kernel_operator.h"
#include <cmath>

constexpr int32_t BUFFER_NUM = 2;

template<typename DTYPE>
class KernelPdist{
public:
    __aicore__ inline KernelPdist() {}
    __aicore__ inline ~KernelPdist() {
        FreeTensors();
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t pType, float pVal, uint32_t N, uint32_t M, uint32_t alignNum, int j_block) {
        this->blockIdx = AscendC::GetBlockIdx();
        this->N = N;
        this->M = M;
        this->alignNum = alignNum;
        this->j_block = j_block;
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
        pipe.InitBuffer(inQueSecond, BUFFER_NUM, this->j_block * this->alignedM * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueY, BUFFER_NUM, sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueBuffer, BUFFER_NUM, this->alignNum * sizeof(DTYPE_Y));
        pipe.InitBuffer(workQue, BUFFER_NUM, this->alignNum * sizeof(float)); // shared work buffer for calculation
        if constexpr (std::is_same_v<DTYPE, half>){
            pipe.InitBuffer(castQue1, BUFFER_NUM, this->alignedM * sizeof(float)); // shared cast buffer
            pipe.InitBuffer(castQue2, BUFFER_NUM, this->alignedM * sizeof(float)); // shared cast buffer
        }
        this->bufferNum = 0;
        AllocTensors();
    }

    __aicore__ inline void Process(){
        switch (this->pType){
            /*case 0: {
                int i = this->i;
                int j = this->j;

                CopyInFirst(i);
                CopyInSecond(j);

                
                for (uint64_t pair = startPair; pair < endPair; pair++) {

                    
                    ComputeLgeneralWithDoubleBuffer(i, j, pVal);
                    if (pair + 1 < endPair) {
                        int next_j = j + 1;
                        if (next_j < N) {
                            CopyInSecond(next_j);
                        }
                    }    
                    CopyOutAligned(pair);

                    j++;
                    if (j >= N) {
                        i++;
                        if (i >= N - 1) {
                            break;
                        }
                        j = i + 1;
                        CopyInFirst(i);

                        CopyInSecond(j);
                    }
                }
                break;
            }
            case 1: {
                int i = this->i;
                int j = this->j;
                CopyInFirst(i);
                for (uint64_t pair = startPair; pair < endPair; pair ++, j ++){
                    if (j >= N){
                        i ++;
                        j = i + 1;
                        CopyInFirst(i);
                    }
                    CopyInSecond(j);
                    ComputeL1(i, j);
                    CopyOutAligned(pair);
                }
                break;
            }
                */
            case 2: {
                int i = this->i;
                int j = this->j;
                int batchSize = this->j_block;
                
                CopyInFirst(i);
                int j_start = j;
                CopyInSecond(j_start, batchSize);
                
                for (uint64_t pair = startPair; pair < endPair; pair++) {
                    if (j >= N) {
                        i++;
                        j = i + 1;
                        CopyInFirst(i);
                        j_start = j;
                        CopyInSecond(j_start, batchSize);
                    }

                    if (j >= j_start + batchSize && j < N) {
                        j_start = j;
                        int remaining = (batchSize > N - j_start) ? (N - j_start) : batchSize;
                        if (remaining > 0) {
                            CopyInSecond(j_start, remaining);
                        }
                    }

                    ComputeL2(i, j, j_start);
                    CopyOutAligned(pair);
                    
                    j++;
                }
                break;
            }
/*
            case 3: {
                int i = this->i;
                int j = this->j;
                CopyInFirst(i);
                for (uint64_t pair = startPair; pair < endPair; pair ++, j ++){
                    if (j >= N){
                        i ++;
                        j = i + 1;
                        CopyInFirst(i);
                    }
                    CopyInSecond(j);
                    ComputeLinf(i, j);
                    CopyOutAligned(pair);
                }
                break;
            }
                */
            default:
                break;
        }
    }


private:
    __aicore__ inline void AllocTensors(){
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Y> outputBuffer = outQueBuffer.AllocTensor<DTYPE_Y>();
        if constexpr (std::is_same_v<DTYPE, half>){
            AscendC::LocalTensor<float> sharedTmpBuffer = workQue.AllocTensor<float>();
            AscendC::LocalTensor<float> castBufferx1 = castQue1.AllocTensor<float>();
            AscendC::LocalTensor<float> castBufferx2 = castQue2.AllocTensor<float>();
            workQue.EnQue(sharedTmpBuffer);
            castQue1.EnQue(castBufferx1);
            castQue2.EnQue(castBufferx2);
        }
        else{
            AscendC::LocalTensor<DTYPE_Y> sharedTmpBuffer = workQue.AllocTensor<DTYPE_Y>();
            workQue.EnQue(sharedTmpBuffer);
        }
        inQueFirst.EnQue(x1Local);
        inQueSecond.EnQue(x2Local);
        outQueY.EnQue(yLocal);
        outQueBuffer.EnQue(outputBuffer);
    }

    __aicore__ inline void FreeTensors(){
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Y> outputBuffer = outQueBuffer.DeQue<DTYPE_Y>();
        if constexpr (std::is_same_v<DTYPE, half>){
            AscendC::LocalTensor<float> sharedTmpBuffer = workQue.DeQue<float>();
            AscendC::LocalTensor<float> castBufferx1 = castQue1.DeQue<float>();
            AscendC::LocalTensor<float> castBufferx2 = castQue2.DeQue<float>();
            workQue.FreeTensor(sharedTmpBuffer);
            castQue1.FreeTensor(castBufferx1);
            castQue2.FreeTensor(castBufferx2);
        }
        else{
            AscendC::LocalTensor<DTYPE_Y> sharedTmpBuffer = workQue.DeQue<DTYPE_Y>();
            workQue.FreeTensor(sharedTmpBuffer);
        }
        inQueFirst.FreeTensor(x1Local);
        inQueSecond.FreeTensor(x2Local);
        outQueY.FreeTensor(yLocal);
        outQueBuffer.FreeTensor(outputBuffer);
    }

    __aicore__ inline void CopyInFirst(int i){
        // 仅在 i 变化时调用一次
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
        AscendC::DataCopy(x1Local, xGm[1ull * i * this->alignedM], this->alignedM);
        inQueFirst.EnQue(x1Local);
        if constexpr (std::is_same_v<DTYPE, half>) {
            AscendC::LocalTensor<DTYPE_X> x1LocalHalf = inQueFirst.DeQue<DTYPE_X>();
            AscendC::LocalTensor<float> castBufferx1 = castQue1.DeQue<float>();
            AscendC::Cast(castBufferx1, x1LocalHalf, AscendC::RoundMode::CAST_NONE, this->alignedM);
            castQue1.EnQue(castBufferx1);
            inQueFirst.EnQue(x1LocalHalf);
        }
    }

__aicore__ inline void CopyInSecond(int j_start, int j_count) {
    uint32_t rowSize = this->alignedM;
    uint32_t totalSize = j_count * rowSize;
    AscendC::LocalTensor<DTYPE_X> ubBuffer = inQueSecond.DeQue<DTYPE_X>();
    AscendC::DataCopy(ubBuffer, xGm[j_start * alignedM], totalSize);
    inQueSecond.EnQue(ubBuffer);
    if constexpr (std::is_same_v<DTYPE, half>) {
        AscendC::LocalTensor<DTYPE_X> tmpHalf = inQueSecond.DeQue<DTYPE_X>();
        AscendC::LocalTensor<float> castBuffer = castQue2.DeQue<float>();

        AscendC::Cast(castBuffer, tmpHalf, AscendC::RoundMode::CAST_NONE, totalSize);

        // 回收队列
        inQueSecond.EnQue(tmpHalf);
        castQue2.EnQue(castBuffer);
    }
}
    /* __aicore__ inline void CopyOut(int i, int j){
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[1ull * i * (2 * N - i - 1) / 2 + j - i - 1], yLocal, 1);
        outQueY.FreeTensor(yLocal);
    } */

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
        outQueBuffer.EnQue(outputBuffer);
        outQueY.EnQue(yLocal);
    }

    __aicore__ inline void ComputeL1(int i, int j){
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.DeQue<DTYPE_Y>();
        if constexpr (std::is_same_v<DTYPE, half>){ // float16
            AscendC::LocalTensor<float> sharedTmpBuffer = workQue.DeQue<float>();
            AscendC::LocalTensor<float> castBufferx1 = castQue1.DeQue<float>();
            AscendC::LocalTensor<float> castBufferx2 = castQue2.DeQue<float>();
            uint64_t x1_offset = (1ull * i * this->M) % (this->alignNum);
            uint64_t x2_offset = (1ull * j * this->M) % (this->alignNum);
            AscendC::Cast(castBufferx2, x2Local, AscendC::RoundMode::CAST_NONE, this->alignedM);
            AscendC::Sub(castBufferx2[x2_offset], castBufferx2[x2_offset], castBufferx1[x1_offset], this->M);
            AscendC::Abs(castBufferx2[x2_offset], castBufferx2[x2_offset], this->M);
            AscendC::ReduceSum(castBufferx2[x2_offset], castBufferx2[x2_offset], sharedTmpBuffer, this->M);
            AscendC::Cast(yLocal, castBufferx2[x2_offset], AscendC::RoundMode::CAST_NONE, 1);
            castQue1.EnQue(castBufferx1);
            castQue2.EnQue(castBufferx2);
            workQue.EnQue(sharedTmpBuffer);
        }
        else{ // float32
            AscendC::Sub(x2Local, x1Local, x2Local, this->M);
            AscendC::Abs(x2Local, x2Local, this->M);
            AscendC::LocalTensor<DTYPE_Y> sharedTmpBuffer = workQue.DeQue<DTYPE_Y>();
            AscendC::ReduceSum(yLocal, x2Local, sharedTmpBuffer, this->M);
            workQue.EnQue(sharedTmpBuffer);
        }
        outQueY.EnQue<DTYPE_Y>(yLocal);
        inQueFirst.EnQue(x1Local);
        inQueSecond.EnQue(x2Local);
    }

    __aicore__ inline void ComputeL2(int i, int j, int j_start){
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.DeQue<DTYPE_Y>();
        int offset = (j - j_start) * this->alignedM;
        AscendC::LocalTensor<DTYPE_X> x2Row = x2Local[offset];
        if constexpr (std::is_same_v<DTYPE, half>){ // float16
            AscendC::LocalTensor<float> sharedTmpBuffer = workQue.DeQue<float>();
            AscendC::LocalTensor<float> castBufferx1 = castQue1.DeQue<float>();
            AscendC::LocalTensor<float> castBufferx2 = castQue2.DeQue<float>();
            uint64_t x1_offset = (1ull * i * this->M) % (this->alignNum);
            uint64_t x2_offset = (1ull * j * this->M) % (this->alignNum);
            AscendC::Cast(castBufferx2, x2Local, AscendC::RoundMode::CAST_NONE, this->alignedM);
            AscendC::Sub(castBufferx2[x2_offset], castBufferx2[x2_offset], castBufferx1[x1_offset], this->M);
            AscendC::Mul(castBufferx2[x2_offset], castBufferx2[x2_offset], castBufferx2[x2_offset], this->M);
            AscendC::ReduceSum(castBufferx2[x2_offset], castBufferx2[x2_offset], sharedTmpBuffer, this->M);
            AscendC::Sqrt(castBufferx2[x2_offset], castBufferx2[x2_offset], 1);
            AscendC::Cast(yLocal, castBufferx2[x2_offset], AscendC::RoundMode::CAST_NONE, 1);
            castQue1.EnQue(castBufferx1);
            castQue2.EnQue(castBufferx2);
            workQue.EnQue(sharedTmpBuffer);
        }
        else{ // float32
            AscendC::Sub(x2Row, x1Local, x2Row, this->M);
            AscendC::LocalTensor<DTYPE_Y> sharedTmpBuffer = workQue.DeQue<DTYPE_Y>();
            AscendC::Mul(x2Row, x2Row, x2Row, this->M);
            AscendC::ReduceSum(yLocal, x2Row, sharedTmpBuffer, this->M);
            AscendC::Sqrt(yLocal, yLocal, 1);
            workQue.EnQue(sharedTmpBuffer);
        }
        outQueY.EnQue<DTYPE_Y>(yLocal);
        inQueFirst.EnQue(x1Local);
        inQueSecond.EnQue(x2Row);
    }

    __aicore__ inline void ComputeL2Batch(int i, int j_start, int jb, int startPair) { 
// 假设我们将计算结果存储在 outQue 中
    for (int k = 0; k < jb; ++k) {
        // 拿 x1, x2, yLocal
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.DeQue<DTYPE_Y>();

        // ComputeL2 逻辑
        if constexpr (std::is_same_v<DTYPE, half>) {
            // Perform calculations
            AscendC::LocalTensor<float> tmpBuffer = workQue.DeQue<float>();
            AscendC::LocalTensor<float> castBuf1 = castQue1.DeQue<float>();
            AscendC::LocalTensor<float> castBuf2 = castQue2.DeQue<float>();

            AscendC::Cast(castBuf2, x2Local, AscendC::RoundMode::CAST_NONE, M);
            AscendC::Cast(castBuf1, x1Local, AscendC::RoundMode::CAST_NONE, M); // x1Local -> castBuf1 (float)
            AscendC::Sub(castBuf2, castBuf2, castBuf1, M);
            AscendC::Mul(castBuf2, castBuf2, castBuf2, M);
            AscendC::ReduceSum(castBuf2, castBuf2, tmpBuffer, M);
            AscendC::Sqrt(castBuf2, castBuf2, 1);
            AscendC::Cast(yLocal, castBuf2, AscendC::RoundMode::CAST_NONE, 1);

            // 将计算结果存储到 outQue 中
            outQueY.EnQue(yLocal);

            // 回收资源
            castQue1.EnQue(castBuf1);
            castQue2.EnQue(castBuf2);
            workQue.EnQue(tmpBuffer);
        } else {
            // Perform calculations for non-half type
            AscendC::Sub(x2Local, x2Local, x1Local, M);
            AscendC::LocalTensor<DTYPE_Y> tmpBuffer = workQue.DeQue<DTYPE_Y>();
            AscendC::Mul(x2Local, x2Local, x2Local, M);
            AscendC::ReduceSum(yLocal, x2Local, tmpBuffer, M);
            AscendC::Sqrt(yLocal, yLocal, 1);

            // 将计算结果存储到 outQue 中
            outQueY.EnQue(yLocal);

            // 回收资源
            workQue.EnQue(tmpBuffer);
        }

        // 继续进行队列回收，保持循环流畅
        inQueFirst.EnQue(x1Local);
        inQueSecond.EnQue(x2Local);
    }

    // 在循环外，统一写回 GM
    for (int k = 0; k < jb; ++k) {
        // 从 outQue 取出数据
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.DeQue<DTYPE_Y>();

        // 计算写回位置
        int pairIdx = startPair + k;
        int l = 0, r = N-1, mid;
        while (l <= r) {
            mid = (l + r) >> 1;
            uint64_t count = 1ull * mid * (2*N - mid - 1) / 2;
            if (count <= pairIdx) l = mid + 1;
            else r = mid - 1;
        }
        int iIdx = r;
        int jIdx = pairIdx - iIdx*(2*N-iIdx-1)/2 + iIdx + 1;

        // 批量写回 GM
        AscendC::DataCopy(yGm[1ull * iIdx*(2*N-iIdx-1)/2 + jIdx - iIdx - 1], yLocal, 1);
    }
    }

    __aicore__ inline void ComputeLinf(int i, int j){
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.DeQue<DTYPE_Y>();
        if constexpr (std::is_same_v<DTYPE, half>){ // float16
            AscendC::LocalTensor<float> sharedTmpBuffer = workQue.DeQue<float>();
            AscendC::LocalTensor<float> castBufferx1 = castQue1.DeQue<float>();
            AscendC::LocalTensor<float> castBufferx2 = castQue2.DeQue<float>();
            uint64_t x1_offset = (1ull * i * this->M) % (this->alignNum);
            uint64_t x2_offset = (1ull * j * this->M) % (this->alignNum);
            AscendC::Cast(castBufferx2, x2Local, AscendC::RoundMode::CAST_NONE, this->alignedM);
            AscendC::Sub(castBufferx2[x2_offset], castBufferx2[x2_offset], castBufferx1[x1_offset], this->M);
            AscendC::Abs(castBufferx2[x2_offset], castBufferx2[x2_offset], this->M);
            AscendC::ReduceMax(castBufferx2[x2_offset], castBufferx2[x2_offset], sharedTmpBuffer, this->M);
            AscendC::Cast(yLocal, castBufferx2[x2_offset], AscendC::RoundMode::CAST_NONE, 1);
            castQue1.EnQue(castBufferx1);
            castQue2.EnQue(castBufferx2);
            workQue.EnQue(sharedTmpBuffer);
        }
        else{ // float32
            AscendC::Sub(x2Local, x1Local, x2Local, this->M);
            AscendC::Abs(x2Local, x2Local, this->M);
            AscendC::LocalTensor<DTYPE_Y> sharedTmpBuffer = workQue.DeQue<DTYPE_Y>();
            AscendC::ReduceMax(yLocal, x2Local, sharedTmpBuffer, this->M);
            workQue.EnQue(sharedTmpBuffer);
        }
        outQueY.EnQue<DTYPE_Y>(yLocal);
        inQueFirst.EnQue(x1Local);
        inQueSecond.EnQue(x2Local);
    }

    __aicore__ inline void ComputeLgeneral(int i, int j, float pVal){
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueY.DeQue<DTYPE_Y>();
        if constexpr (std::is_same_v<DTYPE, half>){ // float16
            AscendC::LocalTensor<float> sharedTmpBuffer = workQue.DeQue<float>();
            AscendC::LocalTensor<float> castBufferx1 = castQue1.DeQue<float>();
            AscendC::LocalTensor<float> castBufferx2 = castQue2.DeQue<float>();
            uint64_t x1_offset = (1ull * i * this->M) % (this->alignNum);
            uint64_t x2_offset = (1ull * j * this->M) % (this->alignNum);
            AscendC::Cast(castBufferx2, x2Local, AscendC::RoundMode::CAST_NONE, this->alignedM);
            AscendC::Sub(castBufferx2[x2_offset], castBufferx2[x2_offset], castBufferx1[x1_offset], this->M);
            AscendC::Abs(castBufferx2[x2_offset], castBufferx2[x2_offset], this->M);
            float p = static_cast<float>(pVal);
            float Rp = static_cast<float>(1 / pVal);
            AscendC::Power(castBufferx2[x2_offset], castBufferx2[x2_offset], p, this->M);
            AscendC::ReduceSum(castBufferx2[x2_offset], castBufferx2[x2_offset], sharedTmpBuffer, this->M);
            AscendC::Power(castBufferx2[x2_offset], castBufferx2[x2_offset], Rp, 1);
            AscendC::Cast(yLocal, castBufferx2[x2_offset], AscendC::RoundMode::CAST_NONE, 1);
            castQue1.EnQue(castBufferx1);
            castQue2.EnQue(castBufferx2);
            workQue.EnQue(sharedTmpBuffer);
        }
        else{ // float32
            AscendC::Sub(x2Local, x1Local, x2Local, this->M);
            AscendC::Abs(x2Local, x2Local, this->M);
            AscendC::LocalTensor<DTYPE_Y> sharedTmpBuffer = workQue.DeQue<DTYPE_Y>();
            DTYPE_X p = static_cast<DTYPE_X>(pVal);
            DTYPE_Y Rp = static_cast<DTYPE_Y>(1 / pVal);
            AscendC::Power(x2Local, x2Local, p, this->M);
            AscendC::ReduceSum(yLocal, x2Local, sharedTmpBuffer, this->M);
            AscendC::Power(yLocal, yLocal, Rp, 1);
            workQue.EnQue(sharedTmpBuffer);
        }
        outQueY.EnQue<DTYPE_Y>(yLocal);
        inQueFirst.EnQue(x1Local);
        inQueSecond.EnQue(x2Local);
    }
    __aicore__ inline void ComputeLgeneralWithDoubleBuffer(int /*i*/, int /*j*/, float pVal)
    {
        // 每个 DeQue 对应一个 pair
        AscendC::LocalTensor<DTYPE_X> x1Local = inQueFirst.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> x2Local = inQueSecond.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal  = outQueY.DeQue<DTYPE_Y>();

        if constexpr (std::is_same_v<DTYPE, half>) {
            // ===== float16 路径 =====
            AscendC::LocalTensor<float> tmpBuffer  = workQue.DeQue<float>();
            AscendC::LocalTensor<float> x1FloatBuf = castQue1.DeQue<float>();
            AscendC::LocalTensor<float> x2FloatBuf = castQue2.DeQue<float>();

            // 1. 显式 Cast（关键修复点）
            AscendC::Cast(x1FloatBuf, x1Local,
                        AscendC::RoundMode::CAST_NONE, this->M);
            AscendC::Cast(x2FloatBuf, x2Local,
                        AscendC::RoundMode::CAST_NONE, this->M);

            // 2. |x1 - x2|
            AscendC::Sub(x2FloatBuf, x2FloatBuf, x1FloatBuf, this->M);
            AscendC::Abs(x2FloatBuf, x2FloatBuf, this->M);

            float p  = static_cast<float>(pVal);
            float Rp = 1.0f / p;

            // 3. |x1 - x2|^p
            AscendC::Power(x2FloatBuf, x2FloatBuf, p, this->M);

            // 4. reduce sum
            AscendC::ReduceSum(x2FloatBuf, x2FloatBuf, tmpBuffer, this->M);

            // 5. (sum)^(1/p)
            AscendC::Power(x2FloatBuf, x2FloatBuf, Rp, 1);

            // 6. Cast 回输出
            AscendC::Cast(yLocal, x2FloatBuf,
                        AscendC::RoundMode::CAST_NONE, 1);

            // 7. 归还 buffer
            castQue1.EnQue(x1FloatBuf);
            castQue2.EnQue(x2FloatBuf);
            workQue.EnQue(tmpBuffer);
        }
        else {
            // ===== float32 路径 =====
            AscendC::LocalTensor<DTYPE_Y> tmpBuffer =
                workQue.DeQue<DTYPE_Y>();

            AscendC::Sub(x2Local, x2Local, x1Local, this->M);
            AscendC::Abs(x2Local, x2Local, this->M);

            DTYPE_X p  = static_cast<DTYPE_X>(pVal);
            DTYPE_Y Rp = static_cast<DTYPE_Y>(1.0f / pVal);

            AscendC::Power(x2Local, x2Local, p, this->M);
            AscendC::ReduceSum(yLocal, x2Local, tmpBuffer, this->M);
            AscendC::Power(yLocal, yLocal, Rp, 1);

            workQue.EnQue(tmpBuffer);
        }

        // 输出与 buffer 回收
        outQueY.EnQue<DTYPE_Y>(yLocal);
        inQueFirst.EnQue(x1Local);
        inQueSecond.EnQue(x2Local);
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
    int j_block;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueFirst, inQueSecond;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueY, outQueBuffer;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> workQue;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> castQue1, castQue2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
};

extern "C" __global__ __aicore__ void pdist(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    if (tiling_data.dataType == DT_FLOAT16){
        KernelPdist<half> op;
        op.Init(x, y, tiling_data.pType, tiling_data.pVal, tiling_data.N, tiling_data.M, tiling_data.alignNum, tiling_data.j_block);
        op.Process();
    }
    else{
        KernelPdist<float> op;
        op.Init(x, y, tiling_data.pType, tiling_data.pVal, tiling_data.N, tiling_data.M, tiling_data.alignNum, tiling_data.j_block);
        op.Process();
    }
}
