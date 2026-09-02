import os
import time
import json
import shutil
import pickle
import argparse
import numpy as np
import multiprocessing
from pathlib import Path
from pyquaternion import Quaternion

t4_proj_dir='/data/qh_projects/t4-devkit'
import sys
sys.path.append(t4_proj_dir)
from t4_devkit import Tier4

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.stores.ros2_foxy import *
from typing import List, Tuple, Dict, Any

"""
工具说明:
    将T4格式的数据转换成Motovis格式
"""


default_visibility = [
    {
        "token": "404bd3c60e97a1795206f4ee5b737e44",
        "level": "partial",
        "description": "The object is occluded by more than 50% (but not completely)."
    },
    {
        "token": "a850b0ca5f1d7a87384ef6bba1d825c0",
        "level": "most",
        "description": "Object is occluded, but by less than 50%."
    },
    {
        "token": "b8f07fdf7a82c82babfd55582dcab261",
        "level": "none",
        "description": "The object is 90-100% occluded and no points/pixels are visible in the label."
    },
    {
        "token": "102103621edaa261b4f8a466a8a72264",
        "level": "full",
        "description": "No occlusion of the object."
    }
]


cam_types = {
    'CAM_FRONT': 33.34,
    'CAM_FRONT_WIDE': 51.34,
    'CAM_FRONT_LEFT_WIDE': 119,
    'CAM_FRONT_RIGHT_WIDE': 50,
    'CAM_BACK_LEFT_WIDE': 110.18,
    'CAM_BACK_RIGHT_WIDE': 92.96,   
}


class TargetPointExtractor:
    def __init__(self,
                 timestamps: List[float],
                 ego_positions: List[List[float]],
                 turn_angle_threshold: float = 3.0,
                 straight_distance_threshold: float = 40) -> None:
        """
        初始化参数
        Args:
            turn_angle_threshold: 转弯角度阈值(度)
            straight_distance_threshold: 直行分段距离阈值(米)
        """
        self.timestamps = timestamps

        # smooth
        self.ego_positions = self._smooth_positions(ego_positions)

        self.turn_angle_threshold = turn_angle_threshold
        self.straight_distance_threshold = straight_distance_threshold
        
        # 提取所有target-points
        self.target_points = self._extract_target_points()
        self.target_points_timestamps = list(self.target_points.keys())

    def _smooth_positions(self, ego_positions: list):
        smoothed_positions = []
        prev_position = ego_positions[0]
        for position in ego_positions:
            dist = np.linalg.norm(np.array(position[:2]) - np.array(prev_position[:2]))

            if dist <= 0.2:
                position = prev_position
            
            prev_position = position

            smoothed_positions.append(position)
        
        return smoothed_positions

    def _extract_target_points(self):
        """提取所有类型的目标点"""
        target_points = {}

        num_pts = len(self.ego_positions)
        if num_pts < 3:
            return self.ego_positions, self.timestamps

        accumulated_distance = 0.0
        for pt_ind in range(1, num_pts - 1):
            timestamp = self.timestamps[pt_ind]
            prev_pt = self.ego_positions[pt_ind - 1][:2]
            curr_pt = self.ego_positions[pt_ind][:2]
            next_pt = self.ego_positions[pt_ind + 1][:2]

            # ------------ 转弯点 --------------
            deviation_angle = self._calculate_deviation_angle(prev_pt, curr_pt, next_pt)
            if deviation_angle > self.turn_angle_threshold:
                target_points[timestamp] = {
                        "type": "turning",
                        "point": self.ego_positions[pt_ind],
                        "timestamp": timestamp,
                        "deviation_angle": deviation_angle,
                    }
                accumulated_distance = 0
                continue

            #  ------------ 直行分段点 --------------
            # 计算当前点与前一点的距离
            distance = np.linalg.norm(np.array(curr_pt[:2]) - np.array(prev_pt[:2]))
            accumulated_distance += distance

            # 如果累积距离超过阈值，添加分段点
            if accumulated_distance >= self.straight_distance_threshold:
                target_points[timestamp] = {
                        "type": "straight",
                        "point": self.ego_positions[pt_ind],
                        "timestamp": timestamp,
                        "segment_distance": accumulated_distance,
                    }
                accumulated_distance = 0.0
                continue

        # 将路径的最后一个点也当作target-point
        if self.timestamps[-1] not in target_points:
            target_points[self.timestamps[-1]] = {
                    "type": "ending",
                    "point": self.ego_positions[-1],
                    "timestamp": self.timestamps[-1]
                }
        return target_points
        
    def _calculate_deviation_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """计算三个点的偏离角度"""
        # 以p1为原点，转换其他点的坐标
        p2_rel = np.array(p2) - np.array(p1)
        p3_rel = np.array(p3) - np.array(p1)

        # 避免除零错误
        if np.linalg.norm(p2_rel) == 0 or np.linalg.norm(p3_rel) == 0:
            return 0

        # 计算两个向量的夹角
        cos_angle = np.dot(p2_rel, p3_rel) / (
            np.linalg.norm(p2_rel) * np.linalg.norm(p3_rel)
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return np.degrees(angle)

    def find_closest_target_point(self, timestamp: float = None) -> Dict[str, Any]:
        """找到车辆前进方向上的下一个目标点
        """

        idx = np.searchsorted(self.target_points_timestamps, timestamp, side='right')
        if idx < len(self.target_points_timestamps):
            target_timestamp = self.target_points_timestamps[idx]
            return self.target_points[target_timestamp]
        else:
            return {
                "type": "ending",
                "point": self.ego_positions[-1],
                "timestamp": self.timestamps[-1]
            }

    def debug(self, global2local, mode='flu'):
        """
        mode: 坐标系方向， flu -> 前左上
        """
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import matplotlib.colors as colors

        if mode == 'flu':
            xmin, xmax = -10, 200
            ymin, ymax = -100, 100

            fig_width = ymax - ymin
            fig_height = xmax - xmin
        else:
            xmin, xmax = -150, 150
            ymin, ymax = -50, 300

            fig_width = xmax - xmin
            fig_height = ymax - ymin

        figdpi = 300
        figsize_scale = 0.1        
        figsize = (fig_width * figsize_scale, fig_height * figsize_scale)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=figdpi)
        if mode == 'flu':
            ax.invert_xaxis()

        fig.tight_layout(pad=0)  # 自动留最小边距
    
        # 设置坐标轴范围
        if mode == 'flu':
            ax.set_xlim(ymin, ymax)  # 横向实际为Ego的Y轴
            ax.set_ylim(xmin, xmax)
            ax.set_xlabel('Ego Y (m)')
            ax.set_ylabel('Ego X (m)')
        else:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

            ax.set_xlabel('LiDAR X (m)')
            ax.set_ylabel('LiDAR Y (m)')

        # 设置坐标轴
        ax.grid(True, alpha=0.7)
        
        # 设置网格间隔为10米
        ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        
        # 设置等比例坐标轴
        ax.set_aspect('equal')

        # path
        # 收集所有路径点的坐标
        path_xs = []
        path_ys = []
        for pt in self.ego_positions:
            global_pt_x, global_pt_y, global_pt_z = pt
            global_pt = np.array([global_pt_x, global_pt_y, global_pt_z, 1]).reshape([1, 4])
            local_pt = (global_pt @ global2local.T).reshape(-1)
            local_x, local_y = local_pt[:2]
            if mode == 'flu':
                path_xs.append(-local_y)
                path_ys.append(local_x)
            else:
                path_xs.append(local_x)
                path_ys.append(local_y)

        # 绘制路径
        ax.plot(path_xs, path_ys, color='blue', linewidth=1, label='Path')
        for i, (x, y) in enumerate(zip(path_xs, path_ys)):
            ax.scatter(x, y, s=10, c='red', marker='o', edgecolors='black', linewidth=0.5)

        # target-points
        print(f'{len(self.target_points)} target points')
        xs = []
        ys = []
        for ts, tp in self.target_points.items():
            global_tp_x, global_tp_y, global_tp_z = tp['point']
            global_tp = np.array([global_tp_x, global_tp_y, global_tp_z, 1]).reshape([1, 4])
            local_tp = (global_tp @ global2local.T).reshape(-1)
            local_x, local_y = local_tp[:2]

            if mode == 'flu':
                xs.append(-local_y)
                ys.append(local_x)
            else:
                xs.append(local_x)
                ys.append(local_y)
        
        # 使用渐变色根据点的顺序绘制散点图
        if xs and ys:  # 确保有点数据
            # 创建颜色映射（从蓝色到红色）
            cmap = cm.get_cmap('viridis')
            normalize = colors.Normalize(vmin=0, vmax=len(xs)-1)
            
            # 绘制每个点，使用不同的颜色
            for i, (x, y) in enumerate(zip(xs, ys)):
                color = cmap(normalize(i))
                ax.scatter(x, y, s=30, c=[color], marker='o', edgecolors='black', linewidth=0.5)
                
                # 在点旁边添加序号标注
                ax.annotate(f'{i}', (x, y), xytext=(3, 3), textcoords='offset points', 
                        fontsize=8, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        plt.savefig('/Users/ace/workspace/aceLab/kit/data_converter/utils/tp.png')
        plt.close(fig)


class T4Converter:
    def __init__(self, args):
        self.args = args

        # 数据根目录
        self.dataroot = os.path.abspath(self.args.dataroot)
        self.batch_names = self.args.batch_names

        # 获取每个批次的clip信息
        self.task_infos = self._get_task_infos()
        self.num_tasks = len(self.task_infos)
        print(f'Found {self.num_tasks} clips')

        # 设置并行处理的进程数量
        self.num_workers = self.args.num_workers
        if self.num_workers <= 0:
            self.num_workers = multiprocessing.cpu_count()
        
        self.num_workers = min(self.num_workers, self.num_tasks)

        self.ego_fut_ts = self.args.ego_fut_ts
        self.agent_fut_ts = self.args.agent_fut_ts

        self.save_dir = self.args.save_dir

        # 相机名称: 相机的延后时间ms
        self.cam_types = {
            'CAM_FRONT': 33.34,
            'CAM_FRONT_WIDE': 51.34,
            'CAM_FRONT_LEFT_WIDE': 119,
            'CAM_FRONT_RIGHT_WIDE': 50,
            'CAM_BACK_LEFT_WIDE': 110.18,
            'CAM_BACK_RIGHT_WIDE': 92.96,   
        }
    
    def _get_task_infos(self):
        task_infos = []

        for batch_name in self.batch_names:
            batch_root = os.path.join(self.dataroot, batch_name)
            if not os.path.isdir(batch_root):
                continue

            for sub_name in os.listdir(batch_root):
                sub_root = os.path.join(batch_root, sub_name)
                if not os.path.isdir(sub_root):
                    continue

                data_root = os.path.join(sub_root, 't4-dataset-creation.t4_converter', 't4_datasets')
                if not os.path.exists(data_root):
                    continue

                for clip_name in os.listdir(data_root):
                    clip_root = os.path.join(data_root, clip_name)
                    task_infos.append((clip_root, batch_name, clip_name))

        return task_infos

    @staticmethod
    def _rebuild_sample_annotation_json(dataroot):
        def _find_annotation(dataroot):
            dataroot = Path(dataroot)
            # 情况1：直接子目录
            direct = dataroot / "annotation"
            if direct.is_dir():
                return direct

            # 情况2：再下一层，如 dataroot/0/annotation
            for subdir in dataroot.iterdir():
                if subdir.is_dir():
                    nested = subdir / "annotation"
                    if nested.is_dir():
                        return nested
            return None
        
        annotation_root = _find_annotation(dataroot)

        # step1. 先判断是否要重新保存visibility.json
        reset_visibility = False
        visibility_filepath = annotation_root / 'visibility.json'
        if not visibility_filepath.exists():
            reset_visibility = True
            with open(visibility_filepath, "w", encoding="utf-8") as f:
                json.dump(default_visibility, f)
        else:
            with open(visibility_filepath, 'r') as f:
                cur_visibility = json.load(f)
            
            if len(cur_visibility) != 4:
                # 备份原来的visibility
                visibility_filepath_raw = annotation_root / 'visibility_raw.json'
                shutil.copy2(visibility_filepath, visibility_filepath_raw)

                reset_visibility = True
                with open(visibility_filepath, "w", encoding="utf-8") as f:
                    json.dump(default_visibility, f)

        # step2. 修改sample_annotation
        reset_sample_annotation = False
        sample_annotation_filepath = annotation_root / 'sample_annotation.json'
        with open(sample_annotation_filepath, 'r') as f:
            sample_annotations_raw = json.load(f)
        
        tmp_ann = sample_annotations_raw[0]
        if 'score' in tmp_ann:
            reset_sample_annotation = True

        if reset_sample_annotation or reset_visibility:
            print(f'reset_visibility: {reset_visibility}')
            print(f'reset_sample_annotation: {reset_sample_annotation}')
            # 备份
            sample_annotation_filepath_raw = annotation_root / 'sample_annotation_raw.json'
            shutil.copy2(sample_annotation_filepath, sample_annotation_filepath_raw)

            sample_annotations_new = []
            for ann in sample_annotations_raw:
                ann.pop('score', None)
                ann.pop('autolabel_metadata', None)

                if ann['automatic_annotation']:
                    ann['automatic_annotation'] = False
                
                if len(ann['visibility_token']) == 0 or reset_visibility: 
                    ann['visibility_token'] = default_visibility[-1]['token']
                    reset_visibility = True

                sample_annotations_new.append(ann)

            assert len(sample_annotations_new) == len(sample_annotations_raw)

            # 重新保存
            with open(sample_annotation_filepath, "w", encoding="utf-8") as f:
                json.dump(sample_annotations_new, f)

    @staticmethod
    def _get_ego_velocities(rosbag_filepath):
        velocities = {}
        accelerations = {}
        
        # 1. 创建类型存储
        typestore = get_typestore(Stores.ROS2_FOXY)

        # 2. 打开 bag
        rosbag_filepath = Path(rosbag_filepath)
        with AnyReader([rosbag_filepath], default_typestore=typestore) as reader:
            topics = [
                '/localization/kinematic_state', 
                '/localization/acceleration'
            ]

            # 获取所有连接
            connections = [x for x in reader.connections if x.topic in topics]
            for conn, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, conn.msgtype)

                # 纳秒 -> 微秒
                timestamp = int(timestamp / 1e3)
                
                if conn.topic == '/localization/acceleration':
                    ax = msg.accel.accel.linear.x
                    ay = msg.accel.accel.linear.y
                    accelerations[timestamp] = [ax, ay]
                else:
                    vx = msg.twist.twist.linear.x
                    vy = msg.twist.twist.linear.y
                    velocities[timestamp] = [vx, vy]
                    
        return velocities, accelerations

    def _rematch_cam_tokens(self, t4: Tier4):
        """
        对每个相机的时间戳做补偿, 然后重新匹配
        """
        timestamps_raw = {cam_name: {} for cam_name in self.cam_types}  # 保存原始的时间戳和对应的token
        timestamps_new = {cam_name: [] for cam_name in self.cam_types}  # 保存补偿后的时间戳

        for sample in t4.sample:
            sample_data_tokens = sample.data
            
            for cam_name, cam_delay in self.cam_types.items():
                if cam_name not in sample_data_tokens:
                    continue

                cam_token = sample_data_tokens[cam_name]
                cam_data = t4.get('sample_data', cam_token)
                cam_timestamp_delay = cam_data.timestamp
                timestamps_raw[cam_name][cam_timestamp_delay] = cam_token

                cam_timestamp = cam_timestamp_delay - cam_delay * 1e3  # 做补偿
                timestamps_new[cam_name].append(cam_timestamp)

        cam_tokens = {}
        for cam_name in self.cam_types:
            cam_tokens[cam_name] = {}
            # 根据补偿后的时间戳重新排序
            timestamps_sorted = list(sorted(timestamps_new[cam_name], key=lambda e: e))
            timestamps_sorted_array = np.array(timestamps_sorted, dtype=np.float64)

            # 对应相机原始的时间戳
            timestamps_old = list(timestamps_raw[cam_name].keys())
            timestamps_old_array = np.array(timestamps_old, dtype=np.float64)

            # 找到离补偿后时间戳最近的token
            for ts_id, ts in enumerate(timestamps_sorted_array):
                diff = np.abs(ts - timestamps_old_array)
                match_id = np.argmin(diff)

                match_token = timestamps_raw[cam_name][timestamps_old[match_id]]
                cam_tokens[cam_name][timestamps_sorted[ts_id]] = match_token
        
        return cam_tokens

    @staticmethod
    def _get_cam_info(cam_name: str, cam_token: str, t4: Tier4, ego2lidar: np.ndarray) -> dict:
        sd_record = t4.get('sample_data', cam_token)
        cs_record = t4.get('calibrated_sensor', sd_record.calibrated_sensor_token)

        cam_info = {
            'type': cam_name,
            'sample_data_token': cam_token,
            'sensor2ego_translation': np.array(cs_record.translation, dtype=np.float64).tolist(),
            'sensor2ego_rotation': np.array(Quaternion(cs_record.rotation).rotation_matrix, dtype=np.float64).tolist()
        }

        cam2ego = np.eye(4, dtype=np.float64)
        cam2ego[:3, :3] = cam_info['sensor2ego_rotation']
        cam2ego[:3, 3] = cam_info['sensor2ego_translation']

        cam2lidar = ego2lidar @ cam2ego
        cam_info['sensor2lidar'] = cam2lidar.tolist()
        cam_info['sensor2lidar_rotation'] = cam2lidar[:3, :3].tolist()
        cam_info['sensor2lidar_translation'] = cam2lidar[:3, 3].tolist()

        return cam_info

    def _get_ego_trajs(self, sample_id, ego_headings, ego_translations):
        def _normalize_angle(angle):
            """
            Map a angle in range [-π, π]
            :param angle: any angle as float
            :return: normalized angle
            """
            return np.arctan2(np.sin(angle), np.cos(angle))
    
        ego_heading = ego_headings[sample_id]
        ego_translation = ego_translations[sample_id]

        cur_pose = np.array([[ego_translation[0], ego_translation[1], ego_heading]], dtype=np.float64)
        theta = -ego_heading
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        fut_poses = []
        for fut_id in range(1, self.ego_fut_ts + 1):
            fut_keyframe_id = sample_id + fut_id
            if fut_keyframe_id >= len(ego_headings):
                break
            
            fut_heading = ego_headings[fut_keyframe_id]
            fut_translation = ego_translations[fut_keyframe_id]
            fut_poses.append([fut_translation[0], fut_translation[1], fut_heading])
        
        command = 'unknown'

        ego_fut_trajs = np.zeros([self.ego_fut_ts, 3], dtype=np.float32)
        ego_fut_masks = np.zeros([self.ego_fut_ts, ], dtype=np.bool_)

        if len(fut_poses):
            fut_poses = np.array(fut_poses, dtype=np.float64)
            
            ego_trajs = fut_poses - cur_pose
            ego_trajs[:, :2] = ego_trajs[:, :2] @ R.T
            ego_trajs[:, 2] = _normalize_angle(ego_trajs[:, 2])

            if ego_trajs[-1][1] >= 2:
                command = 'left'    # 左转
            elif ego_trajs[-1][1] <= -2:
                command = 'right'    # 右转
            else:
                command = 'straight'    # 直行

            ego_fut_trajs[:len(ego_trajs)] = ego_trajs
            ego_fut_masks[:len(ego_trajs)] = True
        
        return (ego_fut_trajs, ego_fut_masks, command)

    def _get_agent_trajs(self, sample_id, clip_infos):
        cur_infos = clip_infos[sample_id]
        cur_boxes = cur_infos['anns']['gt_boxes']
        cur_names = cur_infos['anns']['gt_names']
        cur_trackids = cur_infos['anns']['track_tokens']

        num_instances = np.array(cur_boxes).shape[0]
        if num_instances == 0:
            gt_agent_fut_trajs = np.zeros([0, self.agent_fut_ts, 2], np.float32),
            gt_agent_fut_masks = np.zeros([0, self.agent_fut_ts], np.bool_)
            return (gt_agent_fut_trajs, gt_agent_fut_masks)
        
        global2cur_ego = np.array(cur_infos['utm2ego'])

        # 缓存每个目标的当前位置
        agent_positions = {}
        for agent_id, (cur_box, cur_name, cur_trackid) in enumerate(zip(cur_boxes, cur_names, cur_trackids)):
            agent_token = f'{cur_name}&{cur_trackid}'
            if agent_token not in agent_positions:
                agent_positions[agent_token] = {'id': agent_id, 'pos': [cur_box[:2]]}  # 缓存xy坐标, ego坐标系
        
        # 缓存拿到未来帧所有agent在当前坐标系下的位置
        for fut_id in range(1, self.agent_fut_ts + 1):
            fut_keyframe_id = sample_id + fut_id
            if fut_keyframe_id >= len(clip_infos):
                break

            fut_infos = clip_infos[fut_keyframe_id]

            fut_ego2global = np.array(fut_infos['ego2utm'])
            T_fut2cur = global2cur_ego @ fut_ego2global

            fut_boxes = fut_infos['anns']['gt_boxes']
            fut_names = fut_infos['anns']['gt_names']
            fut_trackids = fut_infos['anns']['track_tokens']

            for agent_token in agent_positions:
                flag = False   # 表示是否匹配到未来帧的目标
                for (fut_box, fut_name, fut_trackid) in zip(fut_boxes, fut_names, fut_trackids):
                    if agent_token != f'{fut_name}&{fut_trackid}':
                        continue

                    # fut_ego -> global -> cur_ego
                    fut_position = np.array(fut_box[:3])
                    fut_position_cur = (fut_position @ T_fut2cur[:3, :3].T) + T_fut2cur[:3, 3]
                    fut_cx, fut_cy = fut_position_cur[:2]

                    agent_positions[agent_token]['pos'].append([fut_cx, fut_cy])
                    flag = True
                    break

                if not flag:
                    agent_positions[agent_token]['pos'].append(None)

        # 计算每一个目标的轨迹
        agent_fut_trajs = np.zeros([num_instances, self.agent_fut_ts, 2], dtype=np.float32)
        agent_fut_masks = np.zeros([num_instances, self.agent_fut_ts], dtype=np.bool_)

        for agent_token, agent_infos in agent_positions.items():
            agent_id = agent_infos['id']
            agent_fut_infos = agent_infos['pos']

            cur_pose = np.array(agent_fut_infos[0])
            fut_poses = agent_fut_infos[1:]

            for fut_id, fut_pose in enumerate(fut_poses):
                if fut_pose is None:
                    break

                offset = np.array(fut_pose) - cur_pose
                agent_fut_trajs[agent_id, fut_id] = offset
                agent_fut_masks[agent_id, fut_id] = True
        
        return (agent_fut_trajs, agent_fut_masks)
    
    def _process_single_clip(self, clip_info):
        clip_root, batch_name, clip_name = clip_info
        self._rebuild_sample_annotation_json(clip_root)

        # 初始化
        t4 = Tier4(clip_root, verbose=False)

        # 重新匹配每个时间戳的cam-token
        cam_tokens = self._rematch_cam_tokens(t4)

        # 逐个样本处理
        clip_infos = []
        ego_headings = []
        ego_translations = []
        for sample_id, sample in enumerate(t4.sample):
            sample_data = sample.data
            lidar_token = sample_data['LIDAR_CONCAT']
            lidar_record = t4.get('sample_data', lidar_token)
            key_frame = lidar_record.is_key_frame
            if not key_frame:
                continue

            timestamp = sample.timestamp  # 微秒
            # scene_token = sample.scene_token

            cs_record = t4.get('calibrated_sensor', lidar_record.calibrated_sensor_token)
            pose_record = t4.get("ego_pose", lidar_record.ego_pose_token)

            # lidar -> ego
            lidar2ego = np.eye(4, dtype=np.float64)
            lidar2ego[:3, :3] = np.array(Quaternion(cs_record.rotation).rotation_matrix, dtype=np.float64)
            lidar2ego[:3, 3] = np.array(cs_record.translation, dtype=np.float64)
            ego2lidar = np.linalg.inv(lidar2ego)

            # get cam infos
            cams = {}
            for cam_name in self.cam_types:
                if cam_name not in sample_data:
                    break

                # 找到与当前lidar时间戳最接近的cam_token
                cam_timestamps = list(cam_tokens[cam_name].keys())
                cam_timestamps_array = np.array(cam_timestamps, dtype=np.float64)
                lidar_timestamp_array = np.array([timestamp], dtype=np.float64)

                diff = np.abs(lidar_timestamp_array - cam_timestamps_array)
                match_ind = np.argmin(diff)
                cam_timestamp = cam_timestamps[match_ind]
                cam_token = cam_tokens[cam_name][cam_timestamp]

                # 根据cam_token去解析信息
                cam_path, _, cam_intrinsic = t4.get_sample_data(cam_token)

                cam_info = self._get_cam_info(cam_name, cam_token, t4, ego2lidar)
                cam_info['data_path'] = cam_path
                cam_info['cam_intrinsic'] = cam_intrinsic.tolist()
                cam_info['timestamp'] = cam_timestamp

                cams[cam_name] = cam_info

            if len(cams) != len(self.cam_types):
                continue

            # lidar信息
            lidar_path, boxes3d, _ = t4.get_sample_data(lidar_token)

            # ego -> utm
            ego2utm = np.eye(4, dtype=np.float64)
            ego2utm[:3, :3] = np.array(Quaternion(pose_record.rotation).rotation_matrix, dtype=np.float64)
            ego2utm[:3, 3] = np.array(pose_record.translation, dtype=np.float64)
            utm2ego = np.linalg.inv(ego2utm)

            # lidar -> utm
            lidar2utm = ego2utm @ lidar2ego
            utm2lidar = np.linalg.inv(lidar2utm)

            # 获取3d标注 (lidar坐标系)
            locs = np.array([b.position for b in boxes3d]).reshape(-1, 3)
            dims = np.array([b.shape.size for b in boxes3d]).reshape(-1, 3)   # wlh
            rots = np.array([Quaternion(b.rotation).yaw_pitch_roll[0] for b in boxes3d]).reshape(-1, 1)
            names = np.array([b.semantic_label.name for b in boxes3d]).reshape(-1)
            num_lidar_pts = np.array([b.num_points for b in boxes3d]).reshape(-1)
            valid_flag = np.array(num_lidar_pts > 0, dtype=np.bool_)
            
            # lidar -> ego
            locs = np.concatenate([locs, np.ones((locs.shape[0], 1))], axis=1)
            locs = (locs @ ego2lidar.T)[:, :3]

            # which is x_size, y_size, z_size (corresponding to l, w, h)
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)

            # velocity
            velocity = np.array([t4.box_velocity(token)[:3] for token in sample.ann_3ds])   # global vel
            for i in range(len(boxes3d)):
                velo = np.array([*velocity[i]])
                velo = velo @ np.linalg.inv(ego2utm[:3, :3]).T    # ego
                velocity[i] = velo

            # object tracking annos: instance_ids
            annotations = [
                t4.get('sample_annotation', ann3d_token)
                for ann3d_token in sample.ann_3ds
            ]
            trackids = np.array([t4.get_idx('instance', anno.instance_token) for anno in annotations])

            info = {
                'sweeps': [],
                'key_frame': sample_id % 5 == 0,
                'token': sample.token,   # 样本名称
                'timestamp': timestamp,
                'lidar_path': lidar_path,
                'scene_token': clip_name,
                'lidar2ego': lidar2ego.tolist(),
                'ego2lidar': ego2lidar.tolist(),
                'lidar2ego_rotation': lidar2ego[:3, :3].tolist(),
                'lidar2ego_translation': lidar2ego[:3, 3].tolist(),
                'cams': cams,
                'ego2utm': ego2utm.tolist(),
                'utm2ego': utm2ego.tolist(),
                'lidar2utm': lidar2utm.tolist(),
                'utm2lidar': utm2lidar.tolist(),
                'anns': {
                    'gt_names': names.tolist(),
                    'gt_boxes': gt_boxes.tolist(),
                    'gt_velocity_3d': velocity.tolist(),
                    'num_lidar_pts': num_lidar_pts.tolist(),
                    'valid_flag': valid_flag.tolist(),
                    'track_tokens': trackids.tolist()
                }
            }
            clip_infos.append(info)
            
            if info['key_frame']:
                ego_translations.append(pose_record.translation)
                ego_headings.append(Quaternion(pose_record.rotation).yaw_pitch_roll[0])

        # 逐帧制作motion & trajs & ego status & target-points
        # clip内按时间戳排序
        clip_infos = list(sorted(clip_infos, key=lambda e: e["timestamp"]))

        # 获取所有关键帧的infos
        keyframe_infos = [info for info in clip_infos if info.get('key_frame', False)]

        # 初始化target-points提取器
        targetpoints_extractor = TargetPointExtractor(
            timestamps=[info['timestamp'] for info in keyframe_infos],
            ego_positions=ego_translations,
        )

        # 获取自车速度、加速度
        rosbag_filepath = os.path.join(t4.data_root, 'input_bag')
        assert os.path.exists(rosbag_filepath)

        velocities, accelerations = self._get_ego_velocities(rosbag_filepath)
        velocityies_timestamps = list(velocities.keys())
        velocityies_timestamps_array = np.array(velocityies_timestamps, dtype=np.float64)
        accelerations_timestamps = list(accelerations.keys())
        accelerations_timestamps_arry = np.array(accelerations_timestamps, dtype=np.float64)

        keyframe_id = 0
        dump_infos = []
        utm2first_ego = None
        for sample_id, info in enumerate(clip_infos):
            # 将clip第1帧的ego坐标系当作global坐标系
            if sample_id == 0:
                utm2first_ego = np.linalg.inv(np.array(info['ego2utm']))
            
            cur_ego2utm = np.array(info['ego2utm'])

            cur2first_ego = utm2first_ego @ cur_ego2utm   # 当前ego到第1帧ego
            info['ego2global'] = cur2first_ego.tolist()
            info['global2ego'] = np.linalg.inv(cur2first_ego).tolist()
            info['ego2global_rotation'] = cur2first_ego[:3, :3].tolist()
            info['ego2global_translation'] = cur2first_ego[:3, 3].tolist()

            lidar2ego = np.array(info['lidar2ego'])      # 当前lidar到当前ego
            lidar2global = cur2first_ego @ lidar2ego     # 当前lidar到第1帧ego
            info['lidar2global'] = lidar2global.tolist()
            info['global2lidar'] = np.linalg.inv(lidar2global).tolist()
            info['lidar2global_rotation'] = lidar2global[:3, :3].tolist()
            info['lidar2global_translation'] = lidar2global[:3, 3].tolist()

            if not info.get('key_frame', False):
                dump_infos.append(info)
                continue

            # ego trajs
            (ego_fut_trajs, ego_fut_masks, command) = self._get_ego_trajs(keyframe_id, ego_headings, ego_translations)
            info['anns']['gt_ego_fut_trajs'] = ego_fut_trajs.tolist()
            info['anns']['gt_ego_fut_masks'] = ego_fut_masks.tolist()
            info['command'] = command

            # 计算自车未来轨迹有效帧数
            num_traj_frames = np.sum(ego_fut_masks)
            info['ego_fut_trajs_num'] = int(num_traj_frames)

            # agent trajs
            (agent_fut_trajs, agent_fut_masks) = self._get_agent_trajs(keyframe_id, keyframe_infos)
            info['anns']['gt_agent_fut_trajs'] = agent_fut_trajs.tolist()
            info['anns']['gt_agent_fut_masks'] = agent_fut_masks.tolist()

            # 自车状态
            cur_timestamp = info['timestamp']
            diff = np.abs(cur_timestamp - velocityies_timestamps_array)
            match_ind = np.argmin(diff)
            ego_velocity = velocities[velocityies_timestamps[match_ind]]

            diff = np.abs(cur_timestamp - accelerations_timestamps_arry)
            match_ind = np.argmin(diff)
            ego_accleration = accelerations[accelerations_timestamps[match_ind]]

            info['ego_velocity'] = [ego_velocity[0], ego_velocity[1], 0]
            info['ego_accleration'] = [ego_accleration[0], ego_accleration[1], 0]

            # target-point
            closest_tp_info = targetpoints_extractor.find_closest_target_point(cur_timestamp)
            utm_tp_x, utm_tp_y, utm_tp_z = closest_tp_info['point']
            utm_tp = np.array([utm_tp_x, utm_tp_y, utm_tp_z, 1]).reshape([1, 4])
            ego_tp = (utm_tp @ np.array(info['utm2ego']).T).reshape(-1)
            target_point = ego_tp[:2]
            info['target_point'] = target_point.tolist()

            dump_infos.append(info)
            keyframe_id += 1

        # 保存
        save_dir = os.path.join(self.save_dir, batch_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, f'{clip_name}.pkl'), 'wb') as f:
            pickle.dump(dump_infos, f)

        num_frames = 0
        num_keyframes = 0
        for info in dump_infos:
            num_frames += 1
            if info.get('key_frame', False):
                num_keyframes += 1

        return (True, clip_name, batch_name, num_frames, num_keyframes)

    def run(self):

        failed_clips = {name: [] for name in self.batch_names}
        successed_clips = {name: [] for name in self.batch_names}

        avg_times = []
        num_processed = 0

        if self.num_workers > 1 and self.num_tasks > 1:
            print(f'Using {self.num_workers} processes to process {self.num_tasks} clips.')
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                s = time.time()
                for (process_flag, process_clip, process_batch_name, num_frames, num_keyframes) in pool.imap_unordered(self._process_single_clip, self.task_infos):
                    e = time.time()
                    num_processed += 1

                    avg_times.append(e - s)
                    eta = round(np.mean(avg_times) * (self.num_tasks - num_processed), 2)
                    s = time.time()

                    if process_flag:
                        successed_clips[process_batch_name].append(process_clip)
                        print(f'[{num_processed}/{self.num_tasks}] - {process_clip} success. Frames: {num_frames}, Key-Frames: {num_keyframes}. ETA: {eta}s')
                    else:
                        failed_clips[process_batch_name].append(process_clip)
                        print(f'[{num_processed}/{self.num_tasks}] - {process_clip} failed. ETA: {eta}s')
                    
        else:
            for task_info in self.task_infos:
                s = time.time()
                (process_flag, process_clip, process_batch_name, num_frames, num_keyframes) = self._process_single_clip(task_info)
                e = time.time()

                num_processed += 1
                avg_times.append(e - s)
                eta = round(np.mean(avg_times) * (self.num_tasks - num_processed), 2)

                if process_flag:
                    successed_clips[process_batch_name].append(process_clip)
                    print(f'[{num_processed}/{self.num_tasks}] - {process_clip} success. Frames: {num_frames}, Key-Frames: {num_keyframes}. ETA: {eta}s')
                else:
                    failed_clips[process_batch_name].append(process_clip)
                    print(f'[{num_processed}/{self.num_tasks}] - {process_clip} failed. ETA: {eta}s')

        # save
        print('\n ----------- Results -----------')
        for batch_name in self.batch_names:
            if len(successed_clips[batch_name]):
                print(f'{batch_name} - {len(successed_clips[batch_name])} clips processed successfully.')

            if len(failed_clips[batch_name]):
                print(f'{batch_name} - {len(failed_clips[batch_name])} clips processed failed.')
                save_failed_dir = os.path.join(args.save_dir, 'processed_logs', batch_name)
                os.makedirs(save_failed_dir, exist_ok=True)
                with open(os.path.join(save_failed_dir, 'failed_clips.txt'), 'w') as f:
                    f.write('\n'.join(failed_clips[batch_name]))


if __name__ == '__main__':
    '''
        python t4_converter.py --dataroot /mnt/cfs/e2e/datasets/t4 --batch_name 2025-10-17 --save_dir /mnt/cfs/e2e/datasets/t4/pkls_motovis
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot", 
        help="原始数据目录."
    )
    parser.add_argument(
        "--save_dir", 
        help="pkl结果保存路径."
    )
    parser.add_argument(
        "--batch_names", 
        nargs="+", 
        help="需要处理的场景名称, 使用空格隔开"
    )
    parser.add_argument(
        '--ego_fut_ts',
        type=int,
        default=8,  # 4秒 * 10hz
    )
    parser.add_argument(
        '--agent_fut_ts',
        type=int,
        default=8,  # 4秒 * 10hz
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4, 
        help="并行处理的进程数，默认为4，设置为0或1表示不使用多进程"
    )
    parser.add_argument(
        "--max_sweeps", 
        type=int, 
        default=10, 
        help="最大扫描帧数量"
    )
    parser.add_argument('--trans_pc', action='store_false')
    args = parser.parse_args()

    # main(args)
    converter = T4Converter(args)
    converter.run()