# DetZero

DetZero是一个用于3D目标检测、跟踪和优化的综合框架，主要用于自动驾驶场景下的感知任务。

## 项目简介

DetZero框架包含以下核心模块：

- **Detection**: 3D目标检测模块，基于CenterPoint架构
- **Tracking**: 离线跟踪模块，用于关联检测框并生成目标轨迹
- **Refining**: 优化模块，包含几何(Geometry)、位置(Position)和置信度(Confidence)三个优化模型
- **DiffuBox**: 基于扩散模型的3D框优化
- **Box Filter**: 基于视觉语言模型的框过滤器
- **Daemon**: 数据处理和转换工具集

## 项目结构

```
detzero/
├── detzero_dev/              # 核心开发目录
│   ├── detection/            # 检测模块
│   ├── tracking/             # 跟踪模块
│   ├── refining/             # 优化模块
│   ├── diffubox/             # 扩散模型模块
│   ├── box_filter/           # 框过滤模块
│   ├── daemon/               # 数据处理工具
│   ├── utils/                # 公共工具库
│   └── requirements.txt      # Python依赖
├── ckpts/                    # 模型权重文件
├── t68_data_cvt/             # T68数据转换工具
├── scripts/                  # 运行脚本
├── start.sh                  # 启动脚本
├── config.json               # 配置文件
└── generate_annos_with_detzero.py  # 标注生成主程序
```

## 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+
- PyTorch 1.10+

### Docker方式运行（推荐）

1. 加载Docker镜像
```bash
sudo docker load -i preanno_saved_detzero_v2.tar
```

2. 启动容器
```bash
sudo docker run -it \
    -v /local_code_dir:/code \
    -v /mnt:/mnt \
    --network host \
    --privileged \
    --gpus all \
    --name detzero_container \
    preanno:v2 /bin/bash
```

### 工具启动方式

使用以下命令启动DetZero工具：

```bash
/home/motovis/dockerToolWorkspace/dataplatform/detzero/current/start.sh /home/motovis/dockerToolWorkspace/dataplatform/detzero/current/config.json
```

或者简化命令：

```bash
bash start.sh config.json
```

### 配置文件说明

配置文件 `config.json` 包含以下主要参数：

```json
{
    "tools_config": {
        "generate_annos_with_detzero": {
            "calib_file": "标定文件路径",
            "detection_ckpt": "检测模型权重路径",
            "diffubox": "是否启用DiffuBox (0/1)",
            "lidar_name": "激光雷达名称",
            "lidar_root": "激光雷达数据根目录",
            "od_infer": "是否进行目标检测推理 (0/1)",
            "output_dir": "输出目录",
            "refining": "是否启用优化模块 (0/1)",
            "tracking": "是否启用跟踪模块 (0/1)",
            "trajectory_file": "轨迹文件路径",
            "workspace": "工作空间路径"
        }
    }
}
```

## 主要功能

### 1. 数据转换

将T68数据转换为NavSim格式：

```bash
bash t68_data_cvt/data_cvt_v2.sh config.json
```

### 2. 目标检测

运行3D目标检测：

```bash
cd detzero_dev/detection/tools
python test.py --cfg_file <CONFIG_FILE> --ckpt <CHECKPOINT> --data_dir <DATA_DIR>
```

### 3. 目标跟踪

运行跟踪模块：

```bash
cd detzero_dev/tracking/tools
python run_track.py --cfg_file cfgs/tk_model_cfgs/motovis_detzero_track.yaml --data_path <DETECTION_RESULT>
```

### 4. 结果优化

运行优化模块：

```bash
cd detzero_dev/refining/tools
python train.py --cfg_file <CONFIG_FILE>
```

## 模块编译

如果需要重新编译各模块：

```bash
# 编译工具库
cd detzero_dev/utils && python setup.py develop

# 编译检测模块
cd detzero_dev/detection && python setup.py develop

# 编译跟踪模块
cd detzero_dev/tracking && python setup.py develop

# 编译优化模块
cd detzero_dev/refining && python setup.py develop
```

## 依赖安装

安装Python依赖：

```bash
pip install -r detzero_dev/requirements.txt
```

主要依赖包括：
- numpy<=1.19.5
- numba==0.48.0
- tensorboardX
- easydict
- pyyaml
- tqdm
- opencv-python
- scipy

## 数据处理流程

完整的标注生成流程：

1. **数据准备**: 准备激光雷达数据、标定文件和轨迹文件
2. **数据转换**: 将原始数据转换为统一格式
3. **目标检测**: 运行3D目标检测模型
4. **目标跟踪**: 关联检测框生成轨迹
5. **结果优化**: 优化检测和跟踪结果
6. **输出生成**: 生成最终的标注文件

## 输出结果

输出目录结构：

```
output_dir/
├── infer/                    # 检测结果
│   └── epoch_20/
│       └── val/
│           └── infer_result.pkl
├── tracking/                 # 跟踪结果
└── refining/                 # 优化结果
```

## 注意事项

1. 确保CUDA环境正确配置
2. Docker运行时需要`--privileged`权限以访问设备
3. 网络挂载目录（/mnt/）可能需要特殊权限处理
4. 建议使用Docker方式运行以保证环境一致性

## 相关文档

- [部署说明](readme_deploy.md)
- [Docker配置](detzero_dev/docker.md)
- [跟踪模块详细说明](detzero_dev/tracking/README.md)
- [优化模块详细说明](detzero_dev/refining/README.md)

## 技术支持

如有问题，请联系开发团队或提交Issue。

## 版本信息

当前版本: v1.0

最后更新: 2026-04-10
