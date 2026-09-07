
export CUDA_VISIBLE_DEVICES=0
# data_dir=/data/dataset/bench2drive/base
# model_name=centerpoint_motovis_1sweep_with_vel
# extra_tag='gpu7'
# epoch_id=epoch_36
# cfg_file=cfgs/det_model_cfgs/${model_name}.yaml
# ckpt=../output/cfgs/det_model_cfgs/${model_name}/${extra_tag}/ckpt/checkpoint_${epoch_id}.pth
# python test.py --cfg_file ${cfg_file} --ckpt ${ckpt} --extra_tag ${extra_tag}

data_dir=/data/dataset/ontime
model_name=centerpoint_motovis_1sweep_ontime
extra_tag='gpu7_ft'
epoch_id=epoch_12
cfg_file=cfgs/det_model_cfgs/${model_name}.yaml
ckpt=../../output/det_model_cfgs/${model_name}/${extra_tag}/ckpt/checkpoint_${epoch_id}.pth
output_dir=../../output/det_model_cfgs/${model_name}/${extra_tag}/${epoch_id}
python test.py --cfg_file ${cfg_file} --ckpt ${ckpt} --extra_tag ${extra_tag} --output_dir ${output_dir} 

det_path=${output_dir}/eval/${epoch_id}/val/result.pkl
# python test.py --cfg_file ${cfg_file} --ckpt ${ckpt} --extra_tag ${extra_tag} --output_dir ${output_dir} --det_path ${det_path}

