/**
 * @file pdist.cpp
 * @author Tianyang Liu (@jjsnam)
 * @brief host (mainly tiling) implementation of AscendC p-distance operator
 * @version 2.1.0 
 * @note Updated after training camp at 2026-01-26
 * @date 2026-01-26
 * 
 * @copyright Copyright (c) 2025 ZJUSCT
 * This implementation is developed by the authors using the AscendC programming model and APIs provided by Huawei Technologies Co., Ltd.
 * AscendC and Ascend are trademarks of Huawei Technologies Co., Ltd.
 */
#include "pdist_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cmath>

const double ZERO = 1e-12; // zero detect threshold
constexpr int sizeFP32 = 4; // sizeof(float)
constexpr int sizeFP16 = 2; // sizeof(half)
constexpr int alignSizeB = 32; // aligning with 32 bytes
constexpr int copyOutTileB = 4096; // Total bytes of a block when copy out (adjustable for better performance)
constexpr int BUFFER_NUM = 2; // Buffer number

static constexpr int ACC_BLOCK_SIZE = 256; // Huge data handling's accumulate buffer (256B as a block (8 datablocks))
static constexpr int TILE_HUGE = 16; // Tile size when computing huge scale data
static constexpr int ACC_BUF_SIZE = TILE_HUGE * ACC_BLOCK_SIZE; // Total buffer's size

/**
 * @brief Declaration of the p-dist kernel calculation type and the data's handle logic
 * @details Four type's of p-dist calculation is specified in our kernel:
 *          - General (p value not specially implemented. e.g p = 1.7);
 *          - Manhattan Distance (p = 1);
 *          - Euclidean Distance (p = 2);
 *          - Chebyshev Distance (p = \inf);
 * @details Two kind of data scale is suppoted now:
 *          - Normal: for which the UB's (Unified Buffer) size can accommodate 
 *                    at least one total row of size M's calculation
 *          - Huge: the other conditions where the Normal logic fails (more tiling is needed)
 */
enum CalcType { Hamming = 0, Manhattan = 1, Euclidean = 2, Chebyshev = -1, General = 3 };
enum DataScale { Normal = 0, Huge = 1 };

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) { // To get tiling infos
    PdistTilingData tiling;

    // Attaining and setting alignNum
    auto dataType = context->GetInputDesc(0)->GetDataType();
    int typeSizeB = dataType == ge::DT_FLOAT16 ? sizeFP16 : sizeFP32;
    int alignNum = alignSizeB / typeSizeB;
    tiling.set_alignNum(alignNum);
    tiling.set_dataType(dataType);

    // Some configs
    tiling.set_copyOutTile(copyOutTileB / typeSizeB); // 16KB

    // Dealing with attr p
    const gert::RuntimeAttrs * attrs = context->GetAttrs();
    const float * p_ptr = attrs->GetAttrPointer<float>(0);
    const float pVal = (p_ptr != nullptr) ? *p_ptr : 2.0f; // check if p is provided (else default 2.0f)
    tiling.set_pVal(pVal);

    // Choose calculation kernel's type
    CalcType pType;
    if (pVal < 0.0) return ge::GRAPH_FAILED; // p must be non-negative (as specified by pytorch)
    if (fabs(pVal) < ZERO) pType = Hamming; // p = 1
    else if (fabs(pVal - 1.0) < ZERO) pType = Manhattan; // p = 1
    else if (fabs(pVal - 2.0) < ZERO) pType = Euclidean; // p = 2
    else if (std::isinf(pVal)) pType = Chebyshev; // p = \inf
    else pType = General; // other p value
    tiling.set_pType(pType);

    // Dealing with input shape
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    const int64_t N = x_shape->GetStorageShape().GetDim(0);
    const int64_t M = x_shape->GetStorageShape().GetDim(1);
    tiling.set_N(N);
    tiling.set_M(M);
    const int64_t alignedM = (M + alignNum - 1) / alignNum * alignNum; // M's ceiling as a aligned number
    tiling.set_alignedM(alignedM);

    // Attaining platfom info
    uint64_t ubSize; // in byte
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNum(); // 40 for Ascend 910B4

    bool isNeedCast = typeSizeB == sizeFP16 && pType != Chebyshev; // if we need cast buffer and other related memory be allocated
    bool isNeedPower = pType == General; // is power needed? (need to reserve workspace)
    int64_t ubSizeRemain = static_cast<int64_t>(ubSize) - // Total ubSize
                           copyOutTileB * BUFFER_NUM - // copyOutTile's size in byte
                           (isNeedCast ? (copyOutTileB / typeSizeB * sizeof(float)) : 0) - // copyOut (buffer)'s size (if needed)
                           BUFFER_NUM * alignedM * typeSizeB; // inQueFirst's size in byte
    // The batchSize's calculation is due to the kernel implementation
    // Exclude the above sizes, only sizes related to inQueSecond is needed to be batched
    // For FP16 (not chebyshev dis) we need another 4 byte for cast calculation for each elements
    // For General calculation, workspace is needed to be reserved (here we assume 4 byte for each elements)
    int64_t batchSize = ubSizeRemain / (alignedM * 
                        ((isNeedCast ? sizeFP32 : 0) + BUFFER_NUM * typeSizeB + (isNeedPower ? sizeFP32 : 0)));
    if (batchSize <= 0) { // if the result <=0, then we can't handle a row in UB totally, which means huge scale logic
        std::cout << "Huge M detected." << std::endl;
        batchSize = 1; // One row at most
        tiling.set_dataScale(Huge);
        // InQueFirst needs to be considered again as well, first restore the size remained
        // and huge data requires an accumulate buffer
        ubSizeRemain += BUFFER_NUM * alignedM * typeSizeB - ACC_BUF_SIZE;
        // Now we calculate the maximum number of elements UB can hold (here we need to add inQueFirst's size)
        int maxM = ubSizeRemain / ((isNeedCast ? sizeFP32 : 0) + BUFFER_NUM * 2 * typeSizeB + (isNeedPower ? sizeFP32 : 0));
        int processM = maxM / alignNum * alignNum; // The elements we process at once need to be aligned
        tiling.set_processM(processM);
    }
    else { // Normal scale
        tiling.set_dataScale(Normal);
        tiling.set_processM(alignedM); // We can process M elements at a time
    }
    tiling.set_batchSize(batchSize);

    context->SetBlockDim(coreNum); // Set block num to core num

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    long N = x_shape->GetDim(0);
    *y_shape = gert::Shape({N * (N - 1) / 2}); // infer the output shape (1, N * (N - 1) / 2)
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class Pdist : public OpDef {
public:
    explicit Pdist(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("p").AttrType(OPTIONAL).Float(2.0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(Pdist);
}
