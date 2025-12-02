
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(PdistTilingData)
    // 输入矩阵大小
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, D);
    // p 范数
    TILING_DATA_FIELD_DEF(float, p);

    // Tiling 参数
    TILING_DATA_FIELD_DEF(uint32_t, tileN);
    TILING_DATA_FIELD_DEF(uint32_t, tileD);

    // UB 分块大小
    TILING_DATA_FIELD_DEF(uint32_t, blockN);
    TILING_DATA_FIELD_DEF(uint32_t, blockD);

    // UB repeat 次数
    TILING_DATA_FIELD_DEF(uint32_t, loopN);
    TILING_DATA_FIELD_DEF(uint32_t, loopD);

    // workspace
    TILING_DATA_FIELD_DEF(uint32_t, workspaceSize);

    // 对齐字段
    TILING_DATA_FIELD_DEF(uint32_t, reserved);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Pdist, PdistTilingData)

} // namespace optiling