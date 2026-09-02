import json
import os
import os.path as op
import sys
sys.path.append('./')
import time
import argparse
import subprocess
import datetime
from easydict import EasyDict
from pathlib import Path
import yaml

CONDA_ACTIVATE = (
    "source /home/admin/miniconda3/etc/profile.d/conda.sh && "
    "conda activate detzero"
)


def run_cmd_realtime(cmd, timeout=1800, error_msg="命令执行失败"):
    """执行 shell 命令，实时打印输出，统一超时和错误处理"""
    proc = subprocess.Popen(
        cmd,
        shell=True,
        executable='/bin/bash',
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    try:
        start_time = time.time()
        for line in proc.stdout:
            if time.time() - start_time > timeout:
                proc.kill()
                raise RuntimeError(f"{error_msg}: 超时({timeout}秒)")
            print(line, end='')
            sys.stdout.flush()

        proc.wait()

    except KeyboardInterrupt:
        print("\n[用户中断]")
        proc.kill()
        proc.wait()
        raise

    if proc.returncode != 0:
        raise RuntimeError(f"{error_msg} (返回码: {proc.returncode})")

    return proc.returncode


def setup_output_directory(output_dir, sudo_password="123456"):
    """创建输出目录并设置权限"""
    print(f"\n{'='*60}")
    print(f"[步骤 1/5] 设置输出目录权限")
    print(f"{'='*60}")

    parent_dir = os.path.dirname(output_dir)
    is_network_mount = output_dir.startswith('/mnt/') or output_dir.startswith('/net/')

    print(f"- 输出目录: {output_dir}")
    print(f"- 父目录: {parent_dir}")

    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ 目录已创建: {output_dir}")

        if not is_network_mount:
            os.system(f"echo {sudo_password} | sudo -S chmod 777 {parent_dir}")
            os.system(f"echo {sudo_password} | sudo -S chmod 777 {output_dir}")
            print(f"✓ 目录权限已设置")

    except Exception as e:
        if is_network_mount and os.path.exists(output_dir):
            print(f"⚠️  目录已存在,继续执行")
        else:
            raise


def run_dataset_metadata_generator(params):
    """运行数据集元数据生成脚本"""
    print("\n" + "="*60)
    print("[步骤 2/5] 运行数据集元数据生成器")
    print("="*60)

    proj_dir="/home/motovis/dockerToolWorkspace/dataplatform/detzero/current/slam_tool"

    script_path = f"{proj_dir}/daemon/prepare_test_parking_data.py"
    if not op.exists(script_path):
        raise FileNotFoundError(f"脚本不存在: {script_path}")

    cmd_args = [
        "--data-root", params.data_root,
        "--scene-id",  params.scene_id,
        "--calib-file", params.calib_file,
        "--output-dir", params.output_dir,
        "--split",      params.split,
        "--lidar-name", params.lidar_name,
    ]
    if getattr(params, 'trajectory_file', None):
        cmd_args.extend(["--trajectory-file", params.trajectory_file])

    cmd = (
        f"{CONDA_ACTIVATE} && "
        f"python {script_path} {' '.join(cmd_args)}"
    )
    print(f"\n[执行命令] {cmd}\n")

    run_cmd_realtime(cmd, timeout=600, error_msg="数据集元数据生成失败")
    print("\n✅ 数据集元数据生成完成")


def update_dataset_config_yaml(params, sudo_password="123456"):
    """更新数据集配置 YAML 文件"""
    print("\n" + "="*60)
    print("[步骤 3/5] 更新数据集配置 YAML")
    print("="*60)
    
    proj_dir="/home/motovis/dockerToolWorkspace/dataplatform/detzero/current/detzero_dev"

    yaml_path = f"{proj_dir}/detection/tools/cfgs/det_dataset_cfgs/motovis_dataset_ontime_v1.yaml"

    temp_dir = os.path.dirname(params.output_dir)
    data_dir = os.path.dirname(temp_dir)
    test_pkl = os.path.join(params.output_dir, "motovis_infos_test.pkl")
    extra_path_test = os.path.join(params.output_dir, "extra_path_test")

    print(f"  - DATA_PATH: {data_dir}")
    print(f"  - test pkl : {test_pkl}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config['DATA_PATH'] = data_dir
    config.setdefault('INFO_PATH', {})
    config['INFO_PATH']['test'] = [test_pkl]
    config['INFO_PATH']['extra_path_test'] = extra_path_test

    backup_path = yaml_path + ".backup"
    if not os.path.exists(backup_path):
        try:
            import shutil
            shutil.copy(yaml_path, backup_path)
            print(f"  ✓ 备份成功: {backup_path}")
        except Exception as e:
            print(f"  ⚠️  备份失败(不影响主流程): {e}")

    dumped = yaml.dump(config, default_flow_style=False,
                       allow_unicode=True, sort_keys=False)
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(dumped)
        print(f"  ✓ YAML 配置文件已更新")
    except PermissionError:
        tmp = "/tmp/_pipeline_motovis_config.yaml"
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(dumped)
        result = subprocess.run(
            f"echo '{sudo_password}' | sudo -S cp {tmp} {yaml_path}",
            shell=True, executable='/bin/bash',
            capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            raise RuntimeError(f"sudo 写入 YAML 失败: {result.stderr}")
        try:
            os.remove(tmp)
        except OSError:
            pass
        print(f"  ✓ 已通过 sudo 更新配置文件")

    print("\n✅ YAML 配置更新完成")


def run_detzero_inference_pipeline(params):
    """运行 DetZero 推理流水线：检测、跟踪、精化"""
    print("\n" + "="*60)
    print("[步骤 4/5] 运行 DetZero 推理流水线")
    print("="*60)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    proj_dir="/home/motovis/dockerToolWorkspace/dataplatform/detzero/current/detzero_dev"
    temp_dir   = os.path.dirname(params.output_dir)
    data_dir   = os.path.dirname(temp_dir)
    output_dir = params.output_dir
    test_pkl   = os.path.join(output_dir, "motovis_infos_test.pkl")
    split      = "test"

    # ★ 从 params 读取开关（由命令行参数传入）
    od_infer = params.od_infer
    diffubox  = params.diffubox
    tracking  = params.tracking
    refining  = params.refining

    model_name = "centerpoint_motovis_3sweep_ontime_dy"
    extra_tag  = "parking50k"
    cfg_file   = f"{proj_dir}/detection/tools/cfgs/det_model_cfgs/centerpoint_motovis_3sweep_ontime.yaml"

    print(f"\n  模型: {model_name}")
    print(f"  流水线开关: OD={od_infer} | Diffubox={diffubox} | Tracking={tracking} | Refining={refining}")
    ckpt = params.detection_ckpt if params.detection_ckpt else \
    f"{proj_dir}/ckpts/detection/det_model_cfgs/{model_name}/dynamic/ckpt/checkpoint_{epoch_id}.pth"
    
    
    epoch_id   = os.path.splitext(os.path.basename(ckpt))[0].replace("checkpoint_", "")
    
    infer_pkl_file = f"{output_dir}/infer/{epoch_id}/val/infer_result.pkl"
    
    ckpt = f"{proj_dir}/../ckpts/detection/det_model_cfgs/centerpoint_motovis_3sweep_ontime_navsim_ft_v2/ruqi_30k_5f/ckpt/checkpoint_epoch_20.pth"

    # ── 4.1 目标检测 ──────────────────────────────
    if od_infer:
        print("\n" + "-"*60)
        print("[阶段 4.1] 目标检测推理")
        print("-"*60)

        # ckpt = f"{proj_dir}/ckpts/detection/det_model_cfgs/{model_name}/dynamic/ckpt/checkpoint_{epoch_id}.pth"
        

        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"模型权重文件不存在: {ckpt}")
        print(f"  使用权重: {ckpt}")
        
        # 拼接环境变量导出命令
        export_pythonpath = (
            f"export PYTHONPATH={proj_dir}/detection:$PYTHONPATH && "
            f"export PYTHONPATH={proj_dir}/tracking:$PYTHONPATH && "
            f"export PYTHONPATH={proj_dir}/refining:$PYTHONPATH && "
            f"export PYTHONPATH={proj_dir}/utils:$PYTHONPATH && "
        )
        
        cmd = (
            f"{CONDA_ACTIVATE} && "
            f"{export_pythonpath}"
            f"cd {proj_dir}/detection/tools && "
            f"python test.py "
            f"--cfg_file {cfg_file} --ckpt {ckpt} "
            f"--extra_tag {extra_tag} --infer_mode 2 "
            f"--output_dir {output_dir} --data_dir {data_dir} "
            f"--test_pkl {test_pkl} --workers 8 --infer_all --batch_size 2"
        )
        print(f"\n[执行命令] {cmd}\n")
        run_cmd_realtime(cmd, timeout=1800, error_msg="目标检测推理失败")
        print("\n✓ 目标检测推理完成")

    # ── 4.2 DiffuBox ──────────────────────────────
    if diffubox:
        print("\n" + "-"*60)
        print("[阶段 4.2] DiffuBox 精化")
        print("-"*60)

        val_dir = f"{output_dir}/infer/{epoch_id}/val"
        cmd = (
            f"{CONDA_ACTIVATE} && "
            f"cd {proj_dir} && "
            f"bash scripts/diffibox_infer.sh {cfg_file} {infer_pkl_file} {data_dir} {val_dir}"
        )
        run_cmd_realtime(cmd, timeout=1800, error_msg="DiffuBox 精化失败")
        infer_pkl_file = f"{output_dir}/infer/{epoch_id}/val/infer_result_diffu_final.pkl"
        print("✓ DiffuBox 精化完成")

    # ── 4.3 跟踪 ──────────────────────────────────
    if tracking:
        print("\n" + "-"*60)
        print("[阶段 4.3] 目标跟踪")
        print("-"*60)

        cmd_track = (
            f"{CONDA_ACTIVATE} && "
            f"cd {proj_dir}/tracking/tools && "
            f"python run_track_motovis.py "
            f"--cfg_file cfgs/tk_model_cfgs/motovis_detzero_track.yaml "
            f"--data_path {output_dir} --root_path {data_dir} "
            f"--pkl_det_file {infer_pkl_file} --split {split} "
            f"--workers 8 --batch_size 4"
        )
        print(f"\n[执行命令] {cmd_track}\n")
        run_cmd_realtime(cmd_track, timeout=1800, error_msg="目标跟踪失败")

        infer_tk_pkl = f"{output_dir}/tracking/{split}/track_data.pkl"
        cmd_filter = (
            f"{CONDA_ACTIVATE} && "
            f"cd {proj_dir}/daemon && "
            f"python track_filter.py {infer_tk_pkl}"
        )
        print(f"\n[执行命令] {cmd_filter}\n")
        run_cmd_realtime(cmd_filter, timeout=1800, error_msg="跟踪过滤失败")
        print("✅ 跟踪完成")

    # ── 4.4 精化 ──────────────────────────────────
    if refining:
        print("\n" + "-"*60)
        print("[阶段 4.4] 精化处理")
        print("-"*60)

        do_ref = 0
        infer_tk_pkl_file   = f"{output_dir}/tracking/{split}/track_data_tkf.pkl"
        infer_drop_pkl_file = f"{output_dir}/tracking/{split}/drop_data.pkl"

        if do_ref:
            cmd = (
                f"{CONDA_ACTIVATE} && "
                f"cd {proj_dir}/daemon && "
                f"python prepare_object_motovis_data.py "
                f"--track_data_path {infer_tk_pkl_file} --split {split} "
                f"--root_dir {data_dir} --output_dir {output_dir} "
                f"--workers 4 --pt_dim 3"
            )
            run_cmd_realtime(cmd, timeout=1800, error_msg="精化数据准备失败")

        cmd = (
            f"{CONDA_ACTIVATE} && "
            f"cd {proj_dir} && "
            f"bash scripts/refining_infer.sh "
            f"{output_dir}/refining {output_dir} {infer_pkl_file} "
            f"{infer_tk_pkl_file} {infer_drop_pkl_file} "
            f"{split} {do_ref} {data_dir}"
        )
        run_cmd_realtime(cmd, timeout=1800, error_msg="精化推理失败")
        print("✓ 精化处理完成")

    print("\n" + "="*60)
    print("✅ DetZero 推理流水线全部完成")
    print("="*60)


def run_prediction_conversion(params):
    """将预测结果转换为标注格式"""
    print("\n" + "="*60)
    print("[步骤 5/5] 转换预测结果为标注格式")
    print("="*60)

    split = "test"
    temp_dir  = os.path.dirname(params.output_dir)
    data_dir  = os.path.dirname(temp_dir)
    input_pkl = f"{params.output_dir}/tracking/{split}/track_data_tkf.pkl"
    output_path = os.path.join(params.output_dir, "annotations")
    convert_script = "/tools/unit_test/convert_predictions_with_pkl.py"

    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"转换脚本不存在: {convert_script}")
    if not os.path.exists(input_pkl):
        raise FileNotFoundError(f"输入 PKL 不存在: {input_pkl}")

    print(f"  - 数据目录: {data_dir}")
    print(f"  - 输入 PKL: {input_pkl}")
    print(f"  - 输出目录: {output_path}")

    os.makedirs(output_path, exist_ok=True)
    print(f"  ✅ 输出目录已创建: {output_path}")

    cmd_parts = [
        "python", convert_script,
        "--data-path",              data_dir,
        "--input-path",             input_pkl,
        "--output-path",            output_path,
        "--input-format",           "pickle",
        "--timestamp-tolerance-ms", "100",
    ]
    if getattr(params, 'trajectory_file', None) and os.path.exists(params.trajectory_file):
        cmd_parts.extend(["--trajectory-file", params.trajectory_file])
        print(f"  - 使用轨迹文件: {params.trajectory_file}")

    cmd = f"{CONDA_ACTIVATE} && {' '.join(cmd_parts)}"
    print(f"\n[执行命令]\n  {cmd}\n")

    run_cmd_realtime(cmd, timeout=1800, error_msg="预测结果转换失败")
    print("\n✅ 预测结果转换完成")

    _verify_conversion_output(output_path)

    print(f"\n  标注文件已保存至: {output_path}")
    return output_path


def _verify_conversion_output(output_path):
    """验证转换输出结果"""
    print("\n[验证输出结果]")
    if not os.path.exists(output_path):
        print("  ⚠️  输出目录不存在")
        return

    scene_dirs = [
        d for d in os.listdir(output_path)
        if os.path.isdir(os.path.join(output_path, d))
    ]
    if not scene_dirs:
        print(f"  ⚠️  未找到场景目录，内容: {os.listdir(output_path)}")
        return

    print(f"  ✅ 共生成 {len(scene_dirs)} 个场景的标注")
    total_static, total_frames = 0, 0

    for scene_dir in scene_dirs:
        scene_path = os.path.join(output_path, scene_dir)

        static_file = os.path.join(scene_path, "trajectory_temp_horizontal_pred.json")
        if os.path.exists(static_file):
            try:
                with open(static_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                n = len(data) if isinstance(data, list) else 0
                total_static += n
                print(f"    - {scene_dir}: {n} 个静态目标 ✅")
            except Exception:
                print(f"    - {scene_dir}: 静态文件读取失败 ⚠️")

        label_dir = os.path.join(scene_path, "label_frames")
        if os.path.exists(label_dir):
            frames = [f for f in os.listdir(label_dir) if f.endswith('.json')]
            total_frames += len(frames)
            print(f"    - {scene_dir}: {len(frames)} 帧动态标注 ✅")

    print(f"\n  汇总 - 静态目标: {total_static} | 动态帧: {total_frames}")


if __name__ == "__main__":
    from MitConfigTools import MitConfigTools

    print("\n" + "="*60)
    print("脚本启动信息")
    print("="*60)
    print(f"[当前工作目录] {os.getcwd()}")
    print(f"[当前用户] {os.getenv('USER') or os.getenv('USERNAME')}")
    print(f"[Python版本] {sys.version}")
    print("="*60)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str,
        default="/workspace/PreAnnotation/GenerateAnnoswithDetZero/generate_annos_with_detzero.json"
    )
    args = parser.parse_args()

    config_path = args.config
    tool_name   = "generate_annos_with_detzero"

    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    params = EasyDict()

    if MitConfigTools.init_config(config_path, tool_name):
        print("\n[INFO] 成功读取配置文件")

        try:
            params.data_root       = MitConfigTools.get_param("lidar_root")
            params.calib_file      = MitConfigTools.get_param("calib_file")
            params.output_dir      = MitConfigTools.get_param("output_dir")
            params.split           = "test"
            params.lidar_name      = MitConfigTools.get_param("lidar_name")
            params.trajectory_file = MitConfigTools.get_param("trajectory_file")

            # ★ 把命令行开关写入 params
            params.od_infer = bool(MitConfigTools.get_param("od_infer"))
            params.diffubox  = bool(MitConfigTools.get_param("diffubox"))
            params.tracking  = bool(MitConfigTools.get_param("tracking"))
            params.refining  = bool(MitConfigTools.get_param("refining"))
            params.detection_ckpt      = MitConfigTools.get_param("detection_ckpt")

            output_path     = Path(params.output_dir).resolve()
            params.scene_id = output_path.parent.parent.name
            print(f"[INFO] 自动推导 scene_id: {params.scene_id}")

            print("\n" + "="*60)
            print("配置参数:")
            print("="*60)
            for k, v in params.items():
                print(f"  - {k}: {v}")
            print("="*60)

            setup_output_directory(params.output_dir)
            run_dataset_metadata_generator(params)
            update_dataset_config_yaml(params)
            run_detzero_inference_pipeline(params)
            if params.tracking:
                annotation_output_path = run_prediction_conversion(params)

            print("\n" + "="*60)
            print("✅ 全部流程执行完成")
            print(f"   标注输出: {annotation_output_path}")
            print("="*60)

            end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            run_msg  = {"success": True, "errMsg": None}

        except Exception as e:
            end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            run_msg  = {"success": False, "errMsg": str(e)}
            print(f"\n❌ {tool_name} 执行失败! 错误: {e}")

            import traceback
            traceback.print_exc()
            raise

    else:
        end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        run_msg  = {"success": False, "errMsg": "配置初始化失败"}
        print("❌ 配置初始化失败")

    config_data['tools_config']['toolsRunMsg'] = [{
        "toolName":  tool_name,
        "startTime": start_time,
        "endTime":   end_time,
        **run_msg
    }]

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)

    print(f"\n[INFO] 执行结果已写入: {config_path}")
