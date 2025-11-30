/**
 * @file operator_desc.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "operator_desc.h"

#include "common.h"

using namespace std;

OperatorDesc::OperatorDesc() {}

OperatorDesc::~OperatorDesc()
{
    for (auto *desc : inputDesc) {
        aclDestroyTensorDesc(desc);
    }

    for (auto *desc : outputDesc) {
        aclDestroyTensorDesc(desc);
    }
}

OperatorDesc &OperatorDesc::AddInputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format)
{
    aclTensorDesc *desc = aclCreateTensorDesc(dataType, numDims, dims, format);
    if (desc == nullptr) {
        ERROR_LOG("create tensor failed");
        return *this;
    }
    inputDesc.emplace_back(desc);
    return *this;
}

OperatorDesc &OperatorDesc::AddOutputTensorDesc(aclDataType dataType, int numDims, const int64_t *dims,
                                                aclFormat format)
{
    aclTensorDesc *desc = aclCreateTensorDesc(dataType, numDims, dims, format);
    if (desc == nullptr) {
        ERROR_LOG("create tensor failed");
        return *this;
    }

    outputDesc.emplace_back(desc);
    return *this;
}
