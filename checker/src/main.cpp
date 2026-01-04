/**
 * @file main.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>

#include <cstdlib>

#include "acl/acl.h"
#include "common.h"
#include "op_runner.h"

bool g_isDevice = false;
int deviceId = 0;

struct Config {
    int N = 100;
    int M = 400;
    float p = 2.0;
    aclDataType dataType = ACL_FLOAT;
    std::string inputPath = "../input/input_x.bin";
    std::string outputPath = "./output_y.bin";
} g_conf;

OperatorDesc CreateOpDesc() {
    // define operator
    std::vector<int64_t> inputShape = {g_conf.N, g_conf.M};
    int64_t outLen = (int64_t)g_conf.N * (g_conf.N - 1) / 2;
    std::vector<int64_t> outputShape = {outLen};
    aclDataType dataType = g_conf.dataType;
    aclFormat format = ACL_FORMAT_ND;
    OperatorDesc opDesc;
    opDesc.AddInputTensorDesc(dataType, inputShape.size(), inputShape.data(), format);
    opDesc.AddOutputTensorDesc(dataType, outputShape.size(), outputShape.data(), format);
    return opDesc;
}

bool SetInputData(OpRunner &runner) {
    size_t fileSize = 0;
    ReadFile(g_conf.inputPath, fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0));
    INFO_LOG("Set input success");
    return true;
}

bool ProcessOutputData(OpRunner &runner) {
    WriteFile(g_conf.outputPath, runner.GetOutputBuffer<void>(0), runner.GetOutputSize(0));
    INFO_LOG("Write output success");
    return true;
}

void DestroyResource() {
    bool flag = false;
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        flag = true;
    }
    INFO_LOG("Reset Device success");
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag) {
        ERROR_LOG("Destroy resource failed");
    } else {
        INFO_LOG("Destroy resource success");
    }
}

bool InitResource() {
    std::string output = "../output";
    if (access(output.c_str(), 0) == -1) {
        int ret = mkdir(output.c_str(), 0700);
        if (ret == 0) {
            INFO_LOG("Make output directory successfully");
        } else {
            ERROR_LOG("Make output directory fail");
            return false;
        }
    }

    // acl.json is dump or profiling config file
    if (aclInit("../scripts/acl.json") != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. deviceId is %d", deviceId);
        (void)aclFinalize();
        return false;
    }
    INFO_LOG("Set device[%d] success", deviceId);

    // runMode is ACL_HOST which represents app is running in host
    // runMode is ACL_DEVICE which represents app is running in device
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestroyResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);

    return true;
}

bool RunOp() {
    // create op desc
    OperatorDesc opDesc = CreateOpDesc();

    // create Runner
    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed");
        return false;
    }

    opRunner.setPvalue(g_conf.p);

    // Load inputs
    if (!SetInputData(opRunner)) {
        ERROR_LOG("Set input data failed");
        return false;
    }

    // Run op
    if (!opRunner.RunOp()) {
        ERROR_LOG("Run op failed");
        return false;
    }

    // process output data
    if (!ProcessOutputData(opRunner)) {
        ERROR_LOG("Process output data failed");
        return false;
    }

    INFO_LOG("Run op success");
    return true;
}

aclDataType ParseDataType(const std::string &dataTypeStr) {
    if (dataTypeStr == "float32" || dataTypeStr == "float"){
        return ACL_FLOAT;
    }
    return ACL_FLOAT16;
}

int main(int argc, char **argv) {
    const char* env_device_id = std::getenv("ASCEND_DEVICE_ID");
    if (env_device_id != nullptr) {
        deviceId = std::atoi(env_device_id);
    }
    
    g_conf.N = std::stoi(argv[1]);
    g_conf.M = std::stoi(argv[2]);
    g_conf.p = atof(argv[3]);
    g_conf.dataType = ParseDataType(argv[4]);

    INFO_LOG("N=%d, M=%d, p=%.1f, dataType=%d", g_conf.N, g_conf.M, g_conf.p, g_conf.dataType);

    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    if (!RunOp()) {
        DestroyResource();
        return FAILED;
    }

    DestroyResource();

    return SUCCESS;
}
