#!/bin/bash
source config.txt
export ASCEND_DEVICE_ID=7

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
source $_ASCEND_INSTALL_PATH/bin/setenv.bash
export DDK_PATH=$_ASCEND_INSTALL_PATH
export NPU_HOST_LIB=$_ASCEND_INSTALL_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/devlib

function main {
    # 1. 清除遗留生成文件和日志文件
    rm -rf $HOME/ascend/log/*
    rm -f ./input/*.bin
    rm -f ./output/*.bin

    # 2. 生成输入数据和真值数据
    cd $CURRENT_DIR
    python3 scripts/gen_data.py --N $N --M $M --p $p --data_type $data_type --data_range $data_range
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Generate input data failed!"
        return 1
    fi
    echo "[INFO]: Generate input data success!"

    # 3. 编译可执行文件
    # skipped with compile.sh

    # 4. 运行可执行文件
    export LD_LIBRARY_PATH=$_ASCEND_INSTALL_PATH/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
    cd $CURRENT_DIR/output
    echo "[INFO]: Execute op!"
    ./execute_pdist_op $N $M $p $data_type
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Acl executable run failed! please check your project!"
        return 1
    fi
    echo "[INFO]: Acl executable run success!"

    # 5. 精度比对
    cd $CURRENT_DIR
    python3 scripts/verify_result.py --output_path output/output_y.bin --golden_path output/golden.bin --data_type $data_type
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Verify result failed!"
        return 1
    fi
}

main
