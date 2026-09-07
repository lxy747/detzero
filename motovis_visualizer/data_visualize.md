# 数据格式说明
[数据格式说明](https://bxdwf5h89k.feishu.cn/wiki/RlQlwFJ2Sij8EVkMeencYaOTnpd)

# 参数说明
- *pkl_file:* 需要可视化的文件路径
- *data_dir:* 原始数据路径, 包含图像, 点云文件夹
- *save_dir:* 可视化结果保存路径, 默认为None, 为Nond时默认保存到*pkl_file*所在路径
- *nuscenes:* 是否为nuscenes数据，默认为False
- *show_gt:* 是否可视化ground-truth, 默认为False
- *show_dt:* 是否可视化模型预测结果, 默认为False
- *show_axis:* BEV图像是否显示坐标系信息，默认为False
- *show_bev_lidar:* 是否将点云投影到BEV图像显示，默认为False
- *show_cam_lidar:* 是否将点云投影到CAMERA图像显示，默认为False
- *show_maps:* 是否显示地图信息，默认为False
- *show_agents:* 是否显示障碍物信息（包含轨迹），默认为False

# 启动脚本
```bash
python tools/visualization/motovis/main.py --pkl_file <pkl文件路径> --data_dir <原始数据路径> --show_gt --show_dt --show_bev_lidar --show_cam_lidar --show_maps --show_agents
```

# 其他信息
- 若可视化nuScenes数据集, 要将*nuscenes*参数设置为True.
- 可视化nuScenes数据集，map信息只在BEV图像显示，不在camera图像显示（nuScenes中map信息没有高度信息，投影到图像会比较偏）
- 当前脚本可自动适配nuScenes的6V图像和如祺的7V图像.