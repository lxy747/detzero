#! /bin/bash
export CUDA_VISIBLE_DEVICES=5,6
# 进入 mmdetection 根目录
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
                                           #
# -------------------------------------------------- #

# 1. 分布式训练参数
# MASTER_ADDR="10.24.2.8"  # 主节点IP地址
# MASTER_PORT="29503"          # 主节点端口
# NNODES=2                   # 总节点数
# NODE_RANK=1                  # 当前节点rank (机器1设0，机器2设1)
NPROC_PER_NODE=2             # 每个节点的GPU数
export MASTER_ADDR=10.24.2.8
export MASTER_PORT=29501
export NODE_RANK=1
export NNODES=2

# # 网卡配置
# export NCCL_SOCKET_IFNAME=ens11np0,ens13np0,ens15np0,ens17np0
# export NCCL_SOCKET_IFNAME=eth0

output_dir=/mnt/cfs/qiuhuan/qh_projects/DetZero-main/ckpts/detection
cfg_file=./cfgs/det_model_cfgs/centerpoint_motovis_3sweep_ontime_navsim.yaml
pretrained_model=/mnt/cfs/qiuhuan/qh_projects/DetZero-main/ckpts/detection/det_model_cfgs/centerpoint_motovis_3sweep_ontime_navsim/demotion_5.1k/ckpt/checkpoint_epoch_10.pth

sleep 5

# python -m torch.distributed.launch \
torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
  train.py \
  --launcher pytorch \
  --cfg_file ${cfg_file} \
    --extra_tag demotion_5.1k_dd \
    --output_dir ${output_dir} \
    --batch_size 16 --max_ckpt_save_num 2 --workers 4 --sync_bn --epochs 16 \
    --pretrained_model ${pretrained_model}


