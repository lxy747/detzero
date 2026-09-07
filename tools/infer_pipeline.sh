

data_dir=/data/dataset/bench2drive/base
model_name=centerpoint_motovis_1sweep_with_vel
extra_tag='gpu7'
epoch_id=epoch_36


cfg_file=cfgs/det_model_cfgs/${model_name}.yaml
ckpt=../output/cfgs/det_model_cfgs/${model_name}/${extra_tag}/ckpt/checkpoint_${epoch_id}.pth
# infer dataset
# python test.py --cfg_file ${cfg_file} --ckpt ${ckpt} --extra_tag ${extra_tag} --infer_mode 2 --output_dir ${output_dir}


infer_pkl_file=../output/cfgs/det_model_cfgs/${model_name}/${extra_tag}/infer/${epoch_id}/val/infer_result.pkl
save_dir=../vis/pred_with_vel_0.1
# vis prediction
python ../../motovis_visualizer/main.py --pkl_file ${infer_pkl_file} --data_dir ${data_dir} \
    --show_dt --show_bev_lidar --show_agents \
    --save_dir ${save_dir} --od_score 0.1