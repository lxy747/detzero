#! /bin/bash



# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch \
# --nproc_per_node=7 --master_port=29988 train.py  --tcp_port 29988  --launcher pytorch  \
# --cfg_file ./cfgs/det_model_cfgs/centerpoint_motovis_1sweep.yaml \
# --extra_tag gpu7 \
# --batch_size 112 --max_ckpt_save_num 6 --workers 4 --sync_bn --epochs 36

# pretrained_model=/data/qh_projects/DetZero-main/ckpts/detection/det_model_cfgs/centerpoint_motovis_3sweep_ontime/f3/ckpt/checkpoint_epoch_20.pth
# pretrained_model=/data/qh_projects/LION/weights/checkpoint_epoch_36_nus_mamba.pth
output_dir=/mnt/cfs/qiuhuan/qh_projects/DetZero-main/ckpts/detection
# cfg_file=./cfgs/det_model_cfgs/centerpoint_motovis_3sweep_ontime_dy.yaml
# cfg_file=./cfgs/det_model_cfgs/centerpoint_motovis_3sweep_ontime_navsim.yaml
pretrained_model=/mnt/cfs/qiuhuan/qh_projects/DetZero-main/ckpts/detection/det_model_cfgs/centerpoint_motovis_3sweep_ontime_navsim_v1.1/demotion_5.1k/ckpt/checkpoint_epoch_26.pth
# cfg_file=./cfgs/det_model_cfgs/lion_mamb_motovis_3sweep_ontime_sparse.yaml

cfg_file=./cfgs/det_model_cfgs/centerpoint_motovis_3sweep_ontime_navsim_v1.1.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=29987 train.py  --tcp_port 2923427  --launcher pytorch  \
--cfg_file ${cfg_file} \
--extra_tag demotion_5.1k_3 \
--output_dir ${output_dir} \
--batch_size 32 --max_ckpt_save_num 2 --workers 4 --sync_bn --epochs 16 \
--pretrained_model ${pretrained_model}
