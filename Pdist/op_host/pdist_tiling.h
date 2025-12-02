
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(PdistTilingData)
    TILING_DATA_FIELD_DEF(float, pVal); // The p value for Pdist
    TILING_DATA_FIELD_DEF(uint32_t, pType); // The kernel type for different p value
    TILING_DATA_FIELD_DEF(uint32_t, N); // The first dimension of input tensor
    TILING_DATA_FIELD_DEF(uint32_t, M); // The second dimension of input tensor
    TILING_DATA_FIELD_DEF(int, alignNum); // The align number given the data type
    TILING_DATA_FIELD_DEF(int, dataType); // The data type
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Pdist, PdistTilingData)

} // namespace optiling