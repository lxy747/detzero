# DetZero - Detection Module


## Intro
- This is the detection module of DetZero framework. It is very similiar to the usage of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), but only contains one detector, CenterPoint.

- We provide all the versions of our ensembled models' configs, one of the prerequisite is to generate multi-frame information and ground-truth database. Thanks to our flexible structure design of information, we only preprocess the raw data for only one time (refer to [Data Preprocess](../docs/DATA_PREPROCESS.md)).


## Data Preprocess
Please see [Data Preprocess](../docs/DATA_PREPROCESS.md) and make sure the dataset are placed at the required file folder.


## Running
- a. compile the module
```shell
cd DetZero/detection &&
python setup.py develop
```
- a1. if use mamba module, you should compile it first.  
>> 1. install casual-conv1d==1.2.0.post2 from source  
```shell
        cd detzero_det/ops/causal-conv1d-1.2.0.post2
        python setup.py install 
```
>> 2. install mamba-ssm  
```shell
        cd detzero_det/ops/mamba
        python setup.py install
```

- b0. create db infos for gt_sampling  
```shell
cd DetZero/detection/tools &&
python create_motovis_db_infos.py --cfg_file  cfgs/det_model_cfgs/centerpoint_motovis_1sweep.yaml --db_info_path ../infos/base
```

- b1. train one model
```shell
cd DetZero/detection/tools &&
python train.py --cfg_file cfgs/det_model_cfgs/centerpoint_1sweep.yaml
```

- c0. infer with the best model
```shell
cd DetZero/detection/tools &&
python test.py --cfg_file cfgs/det_model_cfgs/centerpoint_1sweep.yaml --ckpt <PATH_TO_CKPT>
```

- c1. infer with the best model, only infer
```shell
cd DetZero/detection/tools &&
python test.py --cfg_file cfgs/det_model_cfgs/centerpoint_1sweep.yaml --ckpt <PATH_TO_CKPT> --infer_mode 1
# 0: do test; 1: only save prediction; 2: merge meta to prediction

# visualize prediction
cd DetZero &&
python motovis_visualizer/main.py --pkl_file <pkl文件路径> --data_dir <原始数据路径> --save_dir ./vis --show_dt --show_bev_lidar --show_cam_lidar --show_agents
# pkl文件路径为上一步中保存的infer_result.pkl. 
# e.g. detection/output/cfgs/det_model_cfgs/centerpoint_motovis_1sweep/default/infer/epoch_36/val/infer_result.pkl
```


- d. infer with the best model TTA version
```shell
cd DetZero/detection/tools &&
python test.py --cfg_file cfgs/det_model_cfgs/centerpoint_1sweep.yaml --ckpt <PATH_TO_CKPT> --set DATA_CONFIG.TTA True
```



## Tips
- If you want to change the value of some common settings, please use `--set` following your training command rather than modify the yaml file straightly. For example, `--set OPTIMIZATION.BATCH_SIZE_PER_GPU 16` for changing batch size.

- If you want to use some tag to mark this specific experiment, please use `--extra_tag` following your command. For example, `--extra_tag bs16` for batch size 16 experiment.
