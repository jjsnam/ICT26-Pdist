/**
 * @file pdist.cpp
 * @author Tianyang Liu (@jjsnam)
 * @brief host tiling definition of AscendC p-distance operator
 * @version 2.0 
 * @note Updated after checkpoint at 2025-12-22
 * @date 2025-12-30
 * 
 * @copyright Copyright (c) 2025 ZJUSCT
 * This implementation is developed by the authors using the AscendC programming model and APIs provided by Huawei Technologies Co., Ltd.
 * AscendC and Ascend are trademarks of Huawei Technologies Co., Ltd.
 */
#include "register/tilingdata_base.h"

namespace optiling {
/**
 * @brief Definition of the p-dist tiling data
 */
BEGIN_TILING_DATA_DEF(PdistTilingData)
    // Basic op info
    TILING_DATA_FIELD_DEF(float, pVal); // The p value for p-dist (used for General calculation)
    TILING_DATA_FIELD_DEF(int, pType); // The op kernel type for different p value (see pdist.cpp's enum CalcType)
    TILING_DATA_FIELD_DEF(int, N); // The first dimension of input tensor
    TILING_DATA_FIELD_DEF(int, M); // The second dimension of input tensor
    TILING_DATA_FIELD_DEF(int, dataType); // The data type of p-dist

    // (Host-)processed info
    TILING_DATA_FIELD_DEF(int, alignNum); // The number of elements given the data type (aligned with 32 Byte)
    TILING_DATA_FIELD_DEF(int, alignedM); // M's ceiling as a 32Byte aligned number
    TILING_DATA_FIELD_DEF(int, processM); // Number of elements in dim M for once processing
    TILING_DATA_FIELD_DEF(int, copyOutTile); // Max number of elements once a copy out
    TILING_DATA_FIELD_DEF(int, batchSize); // The batch size (number of vectors) of CopyInSecond(Normal)
    TILING_DATA_FIELD_DEF(int, dataScale); // The processed data's scale (normal / huge)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Pdist, PdistTilingData)
}