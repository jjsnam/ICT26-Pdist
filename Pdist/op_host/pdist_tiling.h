#include "register/tilingdata_base.h"

enum CalcType { General = 0, Manhattan = 1, Euclidean = 2, Chebyshev = 3 };

namespace optiling {
BEGIN_TILING_DATA_DEF(PdistTilingData)
    TILING_DATA_FIELD_DEF(float, pVal); // The p value for Pdist
    TILING_DATA_FIELD_DEF(int, pType); // The kernel type for different p value
    TILING_DATA_FIELD_DEF(uint32_t, N); // The first dimension of input tensor
    TILING_DATA_FIELD_DEF(uint32_t, M); // The second dimension of input tensor
    TILING_DATA_FIELD_DEF(int, alignNum); // The align number given the data type
    TILING_DATA_FIELD_DEF(int, dataType); // The data type
    TILING_DATA_FIELD_DEF(int, copyOutBlock); // copyOutBlock Size in number
    TILING_DATA_FIELD_DEF(int, alignedM);
    TILING_DATA_FIELD_DEF(int, batchSize); // The batch size of copyInSecond
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Pdist, PdistTilingData)
}