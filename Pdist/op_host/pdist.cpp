
#include "pdist_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cmath>
#include <cassert>

const double ZERO = 1e-12;
constexpr int alignSizeB = 32; // aligning with 32 bytes
constexpr int copyOutTileB = 16384; //1KB //16384; // 16KB
constexpr int32_t BUFFER_NUM = 2; // is Double Buffer ?

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    PdistTilingData tiling;

    // Attaining and setting alignNum
    auto dataType = context->GetInputDesc(0)->GetDataType();
    int typeSizeB = dataType == ge::DT_FLOAT16 ? 2 : 4;
    int alignNum = alignSizeB / typeSizeB;
    tiling.set_alignNum(alignNum);
    tiling.set_dataType(dataType);

    // Some configs
    tiling.set_copyOutBlock(copyOutTileB / typeSizeB); // 16KB

    // Dealing with attr p
    const gert::RuntimeAttrs * attrs = context->GetAttrs();
    const float * p_ptr = attrs->GetAttrPointer<float>(0);
    const float pVal = (p_ptr != nullptr) ? *p_ptr : 2.0f;
    tiling.set_pVal(pVal);

    if (pVal < 0.0){
        // p must be non-negative
        return ge::GRAPH_FAILED;
    }
    if (fabs(pVal - 1.0) < ZERO){
        tiling.set_pType(1);
    }
    else if (fabs(pVal - 2.0) < ZERO){
        tiling.set_pType(2);
    }
    else if (std::isinf(pVal)){
        tiling.set_pType(3);
    }
    else{
        tiling.set_pType(0);
    }

    // Dealing with input shape
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    const uint64_t N = x_shape->GetStorageShape().GetDim(0);
    const uint64_t M = x_shape->GetStorageShape().GetDim(1);
    tiling.set_N(N);
    tiling.set_M(M);
    const uint64_t alignedM = (M + alignNum - 1) / alignNum * alignNum;
    tiling.set_alignedM(alignedM);

    // Attaining platfom info
    uint64_t ubSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNum(); // 40 for Ascend 910B4

    uint64_t ubSizeRemain = ubSize - // Total ubSize
                            copyOutTileB * BUFFER_NUM - // copyOutTile's size in Byte
                            M * 4 - // work Buffer size in Byte
                            alignedM * typeSizeB; // inQue1's size in Byte
    int batchSize = ubSizeRemain / (alignedM * BUFFER_NUM * (typeSizeB == 2 ? 6 : 4) + 2);
    batchSize = (batchSize >> 1) << 1;
    if (tiling.get_pType() == 0) batchSize = 1;
    tiling.set_batchSize(batchSize);

    context->SetBlockDim(coreNum); // Set block num to core num

    // Code of Sample
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
    *y_shape = gert::Shape({N * (N - 1) / 2});
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
