data_dir='/data/dataset/bench2drive/base'
pkl_file='../output/cfgs/det_model_cfgs/centerpoint_motovis_1sweep_with_vel/gpu7/infer/epoch_36/val/infer_result.pkl'
save_dir='../vis/pred_with_vel'
# pkl_file=${data_dir}/../infos/base/b2d_infos_merge_all_val03_with_gt_num_pts.pkl
# save_dir='../vis/gt'
python ../../motovis_visualizer/main.py --pkl_file ${pkl_file} --data_dir ${data_dir} \
    --show_dt --show_bev_lidar --show_agents \
    --save_dir ${save_dir}
