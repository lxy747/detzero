#! /bin/bash

# 进入 mmdetection 根目录
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export MASTER_ADDR=10.24.2.4
export MASTER_PORT=29501
export NODE_RANK=0
export NNODES=2

# 网卡配置
export NCCL_SOCKET_IFNAME=ens11np0,ens13np0,ens15np0,ens17np0
# export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_0,mlx5_3,mlx5_6,mlx5_8   # 与以太网口一一对应
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=5
export NCCL_ALGO=Ring
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=0        # 先用 0 排除 TC 冲突
export NCCL_DEBUG=INFO

# export NCCL_SOCKET_FAMILY=AF_INET   # 强制 IPv4，禁 IPv6 解析
# export NCCL_IB_QPS_PER_CONNECTION=4 # 四端口时提高并发
export CUDA_VISIBLE_DEVICES=5,6

output_dir=/mnt/cfs/qiuhuan/qh_projects/DetZero-main/ckpts/detection
cfg_file=./cfgs/det_model_cfgs/centerpoint_motovis_3sweep_ontime_navsim.yaml
pretrained_model=/mnt/cfs/qiuhuan/qh_projects/DetZero-main/ckpts/detection/det_model_cfgs/centerpoint_motovis_3sweep_ontime_navsim_v1.1/demotion_5.1k/ckpt/checkpoint_epoch_26.pth

torchrun \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --nproc_per_node=2 \
  train.py \
  --cfg_file ${cfg_file} \
    --extra_tag demotion_5.1k_dd \
    --output_dir ${output_dir} \
    --batch_size 16 --max_ckpt_save_num 2 --workers 4 --sync_bn --epochs 16 \
    --pretrained_model ${pretrained_model}

exit
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=29987 train.py  --tcp_port 2923427  --launcher pytorch  \
--cfg_file ${cfg_file} \
--extra_tag demotion_5.1k_3 \
--output_dir ${output_dir} \
--batch_size 32 --max_ckpt_save_num 2 --workers 4 --sync_bn --epochs 16 \
--pretrained_model ${pretrained_model}
