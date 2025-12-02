
#include "pdist_tiling.h"
#include "register/op_def_registry.h"

// 后续需要修改tiling 目前只是验证编译
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    PdistTilingData tiling;

    // 1. 读取输入 shape
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    uint32_t N = x_shape->GetStorageShape().GetDim(0);
    uint32_t D = x_shape->GetStorageShape().GetDim(1);

    tiling.set_N(N);
    tiling.set_D(D);

    // 2. 设置 tile 大小（可根据你的核函数调整）
    // 推荐每次处理 8 行（N 方向）和全维度（D）
    uint32_t tileN = 8;
    uint32_t tileD = D; // 通常 pdist 没必要在 D 方向 tile

    tiling.set_tileN(tileN);
    tiling.set_tileD(tileD);

    // 3. 计算 blockN / loopN
    uint32_t loopN = (N + tileN - 1) / tileN;
    tiling.set_loopN(loopN);
    tiling.set_blockN(tileN);

    // 4. 计算 blockD / loopD（如果 tileD = D 则 loopD=1）
    tiling.set_blockD(tileD);
    tiling.set_loopD(1);

    // 5. workspace
    tiling.set_workspaceSize(0);
    tiling.set_reserved(0);

    // 6. 一个 block 处理一个 tile
    context->SetBlockDim(loopN);

    // 7. 写入 buffer
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity()
    );
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

// 以下的部分是自动模板自动生成的，主要是输入输出的shape，目前还未进行修改。
// 
namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
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
