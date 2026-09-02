#!/bin/bash

# 流程：
# 从原始目录拷贝数据，做成特定结构，存储在指定目录t68_data_cvt

# 如果 $1 为空，就使用默认路径；否则使用 $1
if [ -z "$1" ]; then
    CONFIG="/home/motovis/dockerToolWorkspace/dataplatform/detzero/current/config.json"
else
    CONFIG="$1"
fi

# 输出看看结果（可选）
echo "使用配置文件：$CONFIG"

cd /home/motovis/dockerToolWorkspace/dataplatform/detzero/current/t68_data_cvt

python t68_2_navsim_v2.py \
    --config ${CONFIG}
