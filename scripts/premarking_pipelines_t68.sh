#!/bin/bash
proj_dir=$(cd $(dirname $0)/../detzero_dev; pwd)
echo PROJECT_PATH: $proj_dir
export CUDA_VISIBLE_DEVICES=0

export PYTHONPATH=${proj_dir}/detection:${PYTHONPATH}
export PYTHONPATH=${proj_dir}/tracking:${PYTHONPATH}
export PYTHONPATH=${proj_dir}/refining:${PYTHONPATH}
export PYTHONPATH=${proj_dir}/utils:${PYTHONPATH}

# switchModel inference switch
od_infer=$2
# Performance degradation if using diffubox. need further research.
diffubox=$3
# track switch
tracking=$4
# refining switch。now, only convert the output of tracking module to the format of test_pkl. no model infer
refining=$5

detection_ckpt=$6

data_dir=$1
test_pkl="pkls_motovis_demotion/batch_detzero/clip_detzero.pkl"
output_dir=${data_dir}


model_name=centerpoint_motovis_3sweep_ontime_navsim_ft_v2
extra_tag='ruqi_30k_5f'
epoch_id=epoch_20

if [ "$detection_ckpt" = "auto" ]; then
det_ckpt=${proj_dir}/../ckpts/detection/det_model_cfgs/${model_name}/${extra_tag}/ckpt/checkpoint_${epoch_id}.pth
else
det_ckpt=${detection_ckpt}
fi

echo "=== detection_ckpt : ${detection_ckpt} ==="
echo "=== det_ckpt : ${det_ckpt} ==="




# det_ckpt=${proj_dir}/ckpts/detection/det_model_cfgs/centerpoint_motovis_3sweep_ontime_navsim_ft/ruqi_10k/ckpt/checkpoint_epoch_30.pth
echo "=== all data will be saved in ${output_dir} ==="
# model_name=centerpoint_motovis_3sweep_ontime_navsim_ft_v1_rfu
model_name=centerpoint_motovis_3sweep_ontime_navsim_ft_v2_rfu
# infer dataset
if [ $od_infer -ne 0 ]; then
    echo "====== stage: od detection ======"
    cd ${proj_dir}/detection/tools
    cfg_file=${proj_dir}/detection/tools/cfgs/det_model_cfgs/${model_name}.yaml

    python test.py --cfg_file ${cfg_file} --ckpt ${det_ckpt} --extra_tag ${extra_tag} --infer_mode 2 --output_dir ${output_dir} \
        --data_dir ${data_dir} --test_pkl ${test_pkl} --workers 4 --infer_all --batch_size 4
fi

# the pkl format see https://bxdwf5h89k.feishu.cn/docx/R6nYdeJHZoruEfx4fmzcGHPxnNa#share-GoHRd7A0soh9NHxvYCbcpECgneG

infer_pkl_file=${output_dir}/infer/${epoch_id}/val/infer_result.pkl


if [ $diffubox -ne 0 ]; then
    echo "========= stage: do diffubox =========="
    cd ${proj_dir}
    cfg_file=${proj_dir}/detection/tools/cfgs/det_model_cfgs/${model_name}.yaml
    bash scripts/diffibox_infer.sh ${cfg_file} ${infer_pkl_file} ${data_dir} ${output_dir}/infer/${epoch_id}/val
    # 修框模块的输出文件
    infer_pkl_file=${output_dir}/infer/${epoch_id}/val/infer_result_diffu_final.pkl
fi

# split=val
split=test
if [ $tracking -ne 0 ]; then
    echo "========= stage: do tracking =========="
    cd ${proj_dir}/tracking/tools
    # tracking_cfg_file=cfgs/tk_model_cfgs/motovis_detzero_track_t4.yaml
    tracking_cfg_file=cfgs/tk_model_cfgs/motovis_detzero_track_t68.yaml
    # split==test: for only infer
    # split==train or val: 会根据模板id关联成轨迹，供后续训练
    # assign dt_instance id for each obj in infer_pkl_file
    python run_track_motovis.py --cfg_file ${tracking_cfg_file} \
        --data_path ${output_dir} --root_path ${data_dir} \
        --pkl_det_file ${infer_pkl_file} \
        --split ${split} --workers 1 --batch_size 1
    
    infer_tk_pkl_file=${output_dir}/tracking/${split}/track_data.pkl
    cd ${proj_dir}/daemon
    python track_filter.py ${infer_tk_pkl_file}
fi
# infer_tk_pkl_file=${output_dir}/tracking/${split}/track_data.pkl
# cd ${proj_dir}/daemon
# python track_filter.py ${infer_tk_pkl_file}
# exit

# 跟踪模块的输出文件
infer_tk_pkl_file=${output_dir}/tracking/${split}/track_data_tkf.pkl
# infer_tk_pkl_file=${output_dir}/tracking/${split}/track_data.pkl
infer_drop_pkl_file=${output_dir}/tracking/${split}/drop_data.pkl

if [ $refining -ne 0 ]; then
    # echo "========= stage: prepara refining data =========="
    cd ${proj_dir}/daemon
    # ref不能用，有副作用
    do_ref=0
    if [ $do_ref = 1 ]; then
        echo "========= stage: prepara refining data =========="
        python prepare_object_motovis_data.py --track_data_path ${infer_tk_pkl_file} --split ${split} --root_dir ${data_dir} \
            --output_dir ${output_dir}  --workers 4 --pt_dim 4 --remake True
    fi
    
    # echo "========= stage: infer refining =========="
    cd ${proj_dir}
    refining_data_dir=${output_dir}/refining
    
    source scripts/refining_infer.sh ${refining_data_dir} ${output_dir} ${infer_pkl_file} ${infer_tk_pkl_file} ${infer_drop_pkl_file} ${split} ${do_ref} ${data_dir}
    
    echo "final: ${final_pkl_file}"

    # cd ${proj_dir}/daemon
    # python filter_by_vlm.py ${data_dir} ${final_pkl_file} 'lidar'
    # final_pkl_file=${final_pkl_file}_vlm_f.pkl

    dst_dir=$(dirname $final_pkl_file)
    test_pkl_basename=$(basename $test_pkl)
    dst_pkl=${dst_dir}/${test_pkl_basename}
    echo "cp ${final_pkl_file} to ${dst_pkl}"
    cp ${final_pkl_file} ${dst_pkl}
fi


