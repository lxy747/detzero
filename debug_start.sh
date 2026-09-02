#!/bin/bash

docker rm -f detzero_v2_od

RUN_CMD="source /home/admin/miniconda3/etc/profile.d/conda.sh && conda activate detzero && cd /home/motovis/dockerToolWorkspace/dataplatform/detzero/current && bash start.sh /mnt/motovis/101_TestData/sunchaoTest/detzero/platform_config/20260410154535_generate_annos_with_detzero_a34eb95630657cc3a377f1ee09b422a3.json"

docker run -it \
    -v /home/motovis/sunchao/detzero:/home/motovis/dockerToolWorkspace/dataplatform/detzero/current \
    -v /mnt:/mnt --network=host --privileged --gpus all --name detzero_v2_od detzero:v2 /bin/bash -c "$RUN_CMD"