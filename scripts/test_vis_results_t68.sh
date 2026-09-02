#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

proj_dir=$(cd $(dirname $0)/../detzero_dev; pwd)
echo PROJECT_PATH: $proj_dir

od_test=0
od_vis=1


#data_dir=/mnt/cfs/e2e/datasets/t68_260403
data_dir="/mnt/motovis/101_TestData/sunchaoTest/detzero/sunchao/"
test_pkl="pkls_motovis_demotion/detzero/test.pkl"

test_basename=$(basename $test_pkl)
test_basename="${test_basename%.pkl}"

# output_dir=${proj_dir}/output_t68_new_30k_20ep/${test_basename}_pre1
output_dir=${proj_dir}/../output_t68_260403/${test_basename}



model_name=centerpoint_motovis_3sweep_ontime_navsim_ft_v2
extra_tag='ruqi_30k_5f'
epoch_id=epoch_20

cfg_file=${proj_dir}/detection/tools/cfgs/det_model_cfgs/${model_name}.yaml

# infer_pkl_file=/mnt/cfs/e2e/datasets/ontime/pkls_nuscenes_10hz_raw_lidar/ontime_1001clips_200070samples_world_clip.pkl
# infer_pkl_file=${output_dir}/infer/${epoch_id}/val/infer_result.pkl
# infer_pkl_file=${output_dir}/infer/${epoch_id}/val/infer_result_car.pkl
# infer_pkl_file=${output_dir}/infer/${epoch_id}/val/infer_result_diffu_final.pkl

# infer_pkl_file=${output_dir}/refining/v251009/infer_results_final_det+diff+tk.pkl
# infer_pkl_file=${output_dir}/refining/v251009/infer_results_final_det+tk.pkl
# infer_pkl_file=${output_dir}/refining/v251009/infer_results_final.pkl
# infer_pkl_file=${output_dir}/refining/v251009/20260309-17-45-24_dp_fmt.pkl
# infer_pkl_file=${output_dir}/refining/v251009/20260121-15-28-50_final_output.pkl
infer_pkl_file=${output_dir}/refining/v251009/${test_basename}.pkl

# infer dataset
if [ $od_test -ne 0 ]; then
    cd ${proj_dir}/detection/tools
    python test.py --cfg_file ${cfg_file} --det_path ${infer_pkl_file} --test_pkl ${test_pkl} --data_dir ${data_dir}
fi

# vis predictionsh s
if [ $od_vis -ne 0 ]; then
    echo "vis predictions from ${infer_pkl_file}"
    save_dir=${output_dir}/vis/pred_30k_pre1
    cd ${proj_dir}/motovis_visualizer
    fps=10
    skip_frame=5
    # python main.py --pkl_file ${infer_pkl_file} --data_dir ${data_dir} \
    #     --od_score 0.005 --show_axis --fps ${fps} \
    #     --save_dir ${save_dir}  --show_scene --format images --ref_cs lidar --show_key_frame 

    python main.py --pkl_file ${infer_pkl_file} --pts_order rfu --data_dir ${data_dir} --skip_frame ${skip_frame} \
        --show_dt --show_bev_lidar --show_agents --od_score 0.03 --show_axis --fps ${fps} --pt_dim 4 --show_cam_lidar \
        --save_dir ${save_dir} --format images --extra_path pkls_motovis_demotion --ref_cs lidar  --show_key_frame  
        # --show_cam_lidar 
        # --show_bev_only
    # --format clip_video
fi
