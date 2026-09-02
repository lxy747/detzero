#!/bin/bash

JSON_FILE=$1

# Linux原生解析JSON，无需安装jq，兼容任意格式
output_dir=$(grep -o '"output_dir" *: *"[^"]*"' ${JSON_FILE} | sed 's/.*: *"\([^"]*\)".*/\1/')
diffubox=$(grep -o '"diffubox" *: *[0-9]*' ${JSON_FILE} | awk '{print $NF}')
od_infer=$(grep -o '"od_infer" *: *[0-9]*' ${JSON_FILE} | awk '{print $NF}')
refining=$(grep -o '"refining" *: *[0-9]*' ${JSON_FILE} | awk '{print $NF}')
tracking=$(grep -o '"tracking" *: *[0-9]*' ${JSON_FILE} | awk '{print $NF}')
detection_ckpt=$(grep -o '"detection_ckpt" *: *"[^"]*"' ${JSON_FILE} | sed 's/.*: *"\([^"]*\)".*/\1/')


echo JSON_FILE: $JSON_FILE
echo output_dir: $output_dir
echo od_infer2: $od_infer
echo diffubox3: $diffubox
echo tracking4: $tracking
echo refining5: $refining
echo detection_ckpt6: $detection_ckpt

source /home/admin/miniconda3/etc/profile.d/conda.sh
conda activate detzero 

cd /home/motovis/dockerToolWorkspace/dataplatform/detzero/current
bash t68_data_cvt/data_cvt_v2.sh $JSON_FILE
bash scripts/premarking_pipelines_t68.sh $output_dir $od_infer $diffubox $tracking $refining $detection_ckpt
