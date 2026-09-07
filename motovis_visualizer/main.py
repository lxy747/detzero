
import os
# import mmcv
import pickle
import cv2
import argparse
import numpy as np

from tqdm import tqdm
from copy import deepcopy

from colors import LINE_COLORS, ROAD_MARKER_COLORS

from utils import (load_point_cloud, get_bbox_corners3d, 
                   get_matrix4x4, get_matrix4x4_2, proj_lidar2img, create_color_maps)
import matplotlib.pyplot as plt

import pdb

__all__ = ['Visualizer']


class Visualizer:
    def __init__(self,
                 show_gt=True,
                 show_dt=False,
                 show_axis=False,
                 show_maps=True,
                 show_bev_only=False,
                 show_bev_lidar=False,
                 show_cam_lidar=False,
                 show_agents=True,
                 is_nuscenes=False,
                 pts_order='rfu',
                 bev_canvas_size=(3840, 2160),
                 point_cloud_range=(150, 70, 60, 60),
                 od_score = 0.001,
                 format='images',
                 fps=10,
                 ref_cs_is_lidar=True,
                 pt_dim=4,
                 ):
        self.pt_dim = pt_dim
        self.pts_order=pts_order
        self.ref_cs_is_lidar = ref_cs_is_lidar
        self.show_gt = show_gt
        self.show_dt = show_dt
        self.show_axis = show_axis
        self.show_maps = show_maps
        self.show_bev_only=show_bev_only
        self.show_bev_lidar = show_bev_lidar
        self.show_cam_lidar = show_cam_lidar
        self.show_agents = show_agents
        # assert format in ['images', 'clip_video', 'video'], f'{format} must be in [images, clip_video, video]'
        self.format = format
        self.fps = fps
        self.is_nuscenes = is_nuscenes
        self.bev_canvas_size = bev_canvas_size
        self.point_cloud_range_org = point_cloud_range

        self.point_cloud_range = point_cloud_range

        self.bev_scale = None          # 实际距离转换到bev-canvas像素的scale
        self.bev_lidar_origin = None   # bev canvas下lidar的原点

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        # self.font_sclae = 0.5
        self.font_sclae = 1.0
        self.font_thickness = 2
        self.interval = 10

        self.od_score = od_score
        
        # 初始化bev图像
        self.bev_canvas = self._create_bev_canvas()

    def _create_bev_canvas(self, point_cloud_range=None):
        """
        绘制lidar坐标系下的BEV图像
        lidar: x: 右, y: 前
        """
        if point_cloud_range is not None:
            self.point_cloud_range = point_cloud_range
        # 获取bev前后左右的距离, 单位m
        if isinstance(self.point_cloud_range, int):
            distance_front = distance_back = distance_left = distance_right = self.point_cloud_range

        elif len(self.point_cloud_range) == 2:
            distance_front = distance_back = self.point_cloud_range[0]
            distance_left = distance_right = self.point_cloud_range[1]
        else:
            distance_front, distance_back, distance_left, distance_right = self.point_cloud_range[:4]
        self.point_cloud_range = (distance_front, distance_back, distance_left, distance_right)

        # 纵向, 横向距离, 单位m
        bev_range_h = distance_front + distance_back
        bev_range_w = distance_left + distance_right

        # 绘制bev图像
        if isinstance(self.bev_canvas_size, int):
            bev_canvas_h = bev_canvas_w = self.bev_canvas_size
        else:
            bev_canvas_h, bev_canvas_w = self.bev_canvas_size
        self.bev_canvas_size = (bev_canvas_h, bev_canvas_w)

        bev_canvas = np.ones((bev_canvas_h, bev_canvas_w, 3), dtype=np.uint8) * 255

        # 计算实际位置到bev-canvas的映射比例  m -> pixel
        self.bev_scale = max(bev_canvas_h, bev_canvas_w) / max(bev_range_h, bev_range_w)  # bev_range[0] + bev_range[1]m 映射到 img_size 像素

        # 绘制自车位置
        self.bev_lidar_origin = (bev_canvas_w // 2, 
                                 int(distance_front * self.bev_scale))  # bev canvas中lidar坐标系原点位置
        cv2.circle(bev_canvas, 
                   self.bev_lidar_origin, 
                   radius=5, 
                   color=(0, 0, 0), 
                   thickness=-1)
        
        vehicle_size = [1.6 * self.bev_scale, 0.9 * self.bev_scale]  # [length, width]

        cv2.rectangle(bev_canvas, 
                      (int(self.bev_lidar_origin[0] - vehicle_size[1]), int(self.bev_lidar_origin[1] - vehicle_size[0])),
                      (int(self.bev_lidar_origin[0] + vehicle_size[1]), int(self.bev_lidar_origin[1] + vehicle_size[0])),
                      (0, 0, 255),
                      thickness=2)

        # 画坐标轴
        if self.show_axis:
            color = (79, 79, 47)
            # 画距离范围
            for d in range(0, distance_front + 1, self.interval):  # 竖直方向
                radius = int(d * self.bev_scale)
                cv2.circle(bev_canvas, self.bev_lidar_origin, radius=radius, color=color)

            # 画x轴,
            cv2.arrowedLine(bev_canvas,
                            self.bev_lidar_origin, 
                            (bev_canvas_w, self.bev_lidar_origin[1]), 
                            color=color,
                            tipLength=0.03,
                            thickness=2)
            for d in range(self.interval, distance_right + 1, self.interval):
                x = int(self.bev_lidar_origin[0] + d * self.bev_scale)
                cv2.line(bev_canvas, (x, self.bev_lidar_origin[1]), (x, self.bev_lidar_origin[1] - 10), color, 2)

                mark = str(d)
                mark_size = cv2.getTextSize(mark, self.font, self.font_sclae, self.font_thickness)[0]
                cv2.putText(bev_canvas, 
                            mark, 
                            (int(x - mark_size[0] // 2), self.bev_lidar_origin[1] + mark_size[1] + 5), 
                            self.font, 
                            self.font_sclae, 
                            color, 
                            self.font_thickness)

            # 画y轴
            cv2.arrowedLine(bev_canvas,
                            self.bev_lidar_origin, 
                            (self.bev_lidar_origin[0], 0), 
                            color=color,
                            tipLength=0.01,
                            thickness=2)
            
            for d in range(self.interval, distance_front, self.interval):
                y = int(self.bev_lidar_origin[1] - d * self.bev_scale)
                cv2.line(bev_canvas, (self.bev_lidar_origin[0], y), (self.bev_lidar_origin[0] + 10, y), color, 2)

                mark = str(d)
                mark_size = cv2.getTextSize(mark, self.font, self.font_sclae, self.font_thickness)[0]
                cv2.putText(bev_canvas, 
                            mark, 
                            (self.bev_lidar_origin[0] - mark_size[0] - 5, int(y + mark_size[1] // 2)), 
                            self.font, 
                            self.font_sclae, 
                            color, 
                            self.font_thickness)
        
        return bev_canvas

    def _lidar_point_to_bev_canvas(self, lidar_x, lidar_y):
        """
        x, y: lidar坐标系下, x, y轴的坐标, 单位m, x纵向距离, y横向距离
        """
        if self.pts_order=='rfu':
            # bev视角下: y -> 前向, x -> 向右
            u = int(lidar_x * self.bev_scale + self.bev_lidar_origin[0])
            v = int(self.bev_lidar_origin[1] - lidar_y * self.bev_scale)
        elif self.pts_order=='flu':
            # bev视角下: x -> 前向, y -> 向左
            u = int(self.bev_lidar_origin[0] - lidar_y * self.bev_scale)
            v = int(self.bev_lidar_origin[1] - lidar_x * self.bev_scale)
        else:
            assert False
        return [u, v]
    
    def _lidar_points_to_bev_canvas(self, lidar_x, lidar_y):
        """
        x, y: lidar坐标系下, x, y轴的坐标, 单位m, x纵向距离, y横向距离
        bev视角下: y -> 前向, x -> 向右
        """
        if self.pts_order=='rfu':
            # bev视角下: y -> 前向, x -> 向右
            u = (lidar_x * self.bev_scale + self.bev_lidar_origin[0]).astype(np.int32)
            v = (self.bev_lidar_origin[1] - lidar_y * self.bev_scale).astype(np.int32)
        elif self.pts_order=='flu':
            # bev视角下: x -> 前向, y -> 向左
            u = (self.bev_lidar_origin[0] - lidar_y * self.bev_scale).astype(np.int32)
            v = (self.bev_lidar_origin[1] - lidar_x * self.bev_scale).astype(np.int32)
        else:
            assert False
        # u = (lidar_x * self.bev_scale + self.bev_lidar_origin[0]).astype(np.int32)
        # v = (self.bev_lidar_origin[1] - lidar_y * self.bev_scale).astype(np.int32)
        return [u, v]
    
    @staticmethod
    def yaw_filter(yaw):
        """
        filter the heading into -pi ~ pi
        Args:
            yaw: np.ndarray or float, raw heading
        Returns:
            yaw: np.ndarray or float, filtered heading
        """
        pi2 = np.pi * 2
        if isinstance(yaw, np.ndarray):
            mask = np.abs(yaw) >= pi2
            yaw[mask] = yaw[mask] - np.floor(yaw[mask]/pi2)*pi2
            yaw[yaw > np.pi] -= pi2
            yaw[yaw <= -np.pi] += pi2
        else:
            if np.abs(yaw) >= pi2:
                yaw = yaw - np.floor(yaw/pi2)*pi2
                if yaw > np.pi: yaw -= pi2
                if yaw <= -np.pi: yaw += pi2
        return yaw

    @staticmethod
    def bev_points_cmap(lidar_points, min_v = -5, max_v=16):
        cmap = plt.get_cmap('viridis')
        z = lidar_points[:, 2]
        z = np.clip(z, a_min=min_v, a_max=max_v)
        norm_z = (z - min_v + 0.0001)# / (max_v - min_v)
        color = cmap(norm_z)[:, :3] * 255
        color = color.astype(np.uint8)
        return color
    
    @staticmethod
    def get_inverse_transform_mat(src_pose):
        """ 
        Args:
            src_pose: 4*4 transform pose include rotate matrix and translation
        Returns:
            reverse_pose: 4*4 inverse of transform pose
        """
        reverse_pose = np.zeros((4, 4), dtype=np.float32)
        reverse_pose[:3, :3] = src_pose[:3, :3].T
        reverse_pose[:3, 3:] = -(src_pose[:3, :3].T @ src_pose[:3, 3:])
        reverse_pose[3, 3] = 1
        return reverse_pose

    @staticmethod
    def transform_boxes3d(boxes, pose, inverse=False, with_vel=False):
        """
        Args:
            boxes: N*7 x,y,z,dx,dy,dz,heading
            pose: 4*4 transform pose include rotate matrix and translation
            inverse: using inverse of transform pose if True, Fasle otherwise
        Returns:
            transformed_boxes: N*7 x,y,z,dx,dy,dz,heading
        """
        is_one_dim = len(boxes.shape)==1
        if is_one_dim:
            boxes = boxes.reshape(1, -1)
        
        center = boxes[:, :3]
        center = np.concatenate([center, np.ones((center.shape[0], 1))], axis=-1)
        if inverse:
            pose = Visualizer.get_inverse_transform_mat(pose)
        center = center @ pose.T
        heading = Visualizer.yaw_filter(boxes[:, [6]] + np.arctan2(pose[1, 0], pose[0, 0]))
        if with_vel:
            vel = np.concatenate([boxes[:, 7:9], np.ones((center.shape[0], 1))], axis=-1)
            vel = vel @ pose[:3,:3].T
            out = np.concatenate([center[:, :3], boxes[:, 3:6], heading, vel[:, :2]], axis=-1)
        else:
            out = np.concatenate([center[:, :3], boxes[:, 3:6], heading], axis=-1)
        if is_one_dim:
            out = out.reshape(-1)
        return out

    def _porj_lidar_to_bev_canvas(self, bev_canvas, lidar_points):
        # pdb.set_trace()
        num_pt = lidar_points.shape[0]
        mask = np.random.rand(num_pt,) < 0.5
        lidar_points = lidar_points[mask, :]
        ord_id = np.argsort(lidar_points[:, 2])
        lidar_points = lidar_points[ord_id, :]
        
        bev_canvas_xs, bev_canvas_ys = self._lidar_points_to_bev_canvas(lidar_points[:, 0], lidar_points[:, 1])
        mask = np.logical_and(bev_canvas_xs >=0, bev_canvas_xs<self.bev_canvas_size[1]) 
        mask = np.logical_and(mask, bev_canvas_ys>=0)
        mask = np.logical_and(mask, bev_canvas_ys<self.bev_canvas_size[0])
        bev_canvas_xs = bev_canvas_xs[mask]
        bev_canvas_ys = bev_canvas_ys[mask]
        color_map = Visualizer.bev_points_cmap(lidar_points[mask, :], min_v= -2, max_v=10)
        # pdb.set_trace()
        bev_canvas[bev_canvas_ys, bev_canvas_xs, :] = color_map
        # bev_canvas[bev_canvas_ys, bev_canvas_xs, :] = np.array([190, 190, 190], dtype=np.uint8)
        return bev_canvas
    
    # ------------- render on bev -------------
    def _draw_bbox_on_bev(self, bev_canvas, bboxes3d, instance_inds=None, color=(0, 255, 0)):
        for idx, bbox in enumerate(bboxes3d):
            corners3d = get_bbox_corners3d(bbox, on_images=False, on_lidar=True, return_front_idx=True, pts_orders=self.pts_order)
            # pdb.set_trace()
            corners3d, front_idx = corners3d
            corners3d = corners3d.T[:4, :2]
            corners3d = np.vstack((corners3d, corners3d[0]))

            # 绘制多条边界线
            poly_pixels = np.array([self._lidar_point_to_bev_canvas(x, y) for x, y in corners3d], dtype=np.int32)
            cv2.polylines(bev_canvas, [poly_pixels], isClosed=False, color=color, thickness=3)

            front_pt =list(np.mean(poly_pixels[front_idx, :], axis=0).astype(np.int32))
            center_pt = list(np.mean(poly_pixels[:4, :], axis=0).astype(np.int32))

            cv2.line(bev_canvas, front_pt, center_pt, color=[0,0,255], thickness=3)
            if instance_inds is not None:
                track_id = instance_inds[idx]
                x, y = poly_pixels[0]
                cv2.putText(bev_canvas, f'id: {track_id}', (x, y), self.font, 
                            self.font_sclae, 
                            (0, 0, 255), 
                            self.font_thickness)

        return bev_canvas
    
    def _draw_map_on_bev(self, bev_canvas, maps, task='line'):
        """
        lanes: lidar坐标系
        """
        is_closed = task == 'rm'

        for cls, polylines in maps.items():
            if task == 'line':
                color = LINE_COLORS[cls]
            else:
                color = ROAD_MARKER_COLORS[cls]
            
            for ployline in polylines:
                ployline = ployline[:, :2]  # 只取(x, y)
                
                poly_pixels = np.array([self._lidar_point_to_bev_canvas(x, y) for x, y in ployline], dtype=np.int32)
                cv2.polylines(bev_canvas, [poly_pixels], isClosed=is_closed, color=color, thickness=4)
        
        return bev_canvas
    
    def _draw_ego_trajs_on_bev(self, bev_canvas, trajs, command=None, start_point=None, colormap='autumn', thickness=4):
        # 起始点
        if start_point is None:   # lidar坐标原点
            x = 0
            y = 0
            trajs_points = [self.bev_lidar_origin]
        else:
            x, y = start_point
            obj_x, obj_y = self._lidar_point_to_bev_canvas(x, y)
            trajs_points = [[obj_x, obj_y]]

        # 添加起始点 & 投影到bev
        for traj in trajs:
            x += traj[0]
            y += traj[1]
            trajs_points.append(self._lidar_point_to_bev_canvas(x, y))

        # colormap = 'winter'
        colors = create_color_maps(len(trajs_points), colormap)

        # 绘制轨迹
        for idx in range(len(trajs_points) - 1):
            x1 = int(trajs_points[idx][0])
            y1 = int(trajs_points[idx][1])

            x2 = int(trajs_points[idx + 1][0])
            y2 = int(trajs_points[idx + 1][1])
            cv2.line(bev_canvas, (x1, y1), (x2, y2), color=colors[idx], thickness=thickness)

        if command is not None:
            command = list(command)
            if command.index(1) == 0:
                cv2.putText(bev_canvas, 'Turn Right', (5, 120), self.font, 4, (0, 0, 255), 6)
            elif command.index(1) == 1:
                cv2.putText(bev_canvas, 'Turn Left', (5, 120), self.font, 4, (0, 0, 255), 6)
            elif command.index(1) == 2:
                cv2.putText(bev_canvas, 'Go Straight', (5, 120), self.font, 4, (0, 0, 255), 6)

        return bev_canvas

    def _render_on_bev(self, sample_info, data_dir):
        bev_canvas = deepcopy(self.bev_canvas)
    
        # 渲染点云
        pc_points = None
        if 'lidar_path' in sample_info and self.show_bev_lidar:
            lidar_path = sample_info['lidar_path']
            if self.is_nuscenes:
                lidar_path = lidar_path.replace('./data/nuscenes/', '')

            lidar_path = os.path.join(data_dir, lidar_path)
            
            if os.path.exists(lidar_path):
                pc_points = load_point_cloud(lidar_path, dim=self.pt_dim)
                bev_canvas = self._porj_lidar_to_bev_canvas(bev_canvas, pc_points)
        else:
            pc_points = None
            
        if self.show_gt:
            anns = sample_info['anns'] if 'anns' in sample_info.keys() else sample_info
            # 障碍物
            if 'gt_boxes' in anns and self.show_agents:
                bev_canvas = self._draw_bbox_on_bev(bev_canvas, anns['gt_boxes'])
            
            # 地图
            if 'map_annos' in anns and self.show_maps:
                map_annos = anns['map_annos']
                if self.is_nuscenes:
                    bev_canvas = self._draw_map_on_bev(bev_canvas, map_annos, task='line')
                else:
                    for task in ['line', 'rm']:
                        if task in map_annos:
                            bev_canvas = self._draw_map_on_bev(bev_canvas, map_annos[task], task=task)

            # 自车轨迹
            if 'gt_ego_fut_path' in anns and len(anns['gt_ego_fut_path']):
                bev_canvas = self._draw_ego_trajs_on_bev(bev_canvas, anns['gt_ego_fut_path'], colormap='winter')

            if 'gt_ego_fut_trajs' in anns:
                bev_canvas = self._draw_ego_trajs_on_bev(bev_canvas, anns['gt_ego_fut_trajs'], command=anns.get('gt_ego_fut_cmd', None), thickness=8)
    
            # target point
            if 'gt_target_point' in anns:
                bev_canvas = self._draw_target_point_on_bev(bev_canvas, anns['gt_target_point'])

            # agent trajs
            if 'gt_agent_fut_trajs' in anns and self.show_agents:
                num_objs = len(anns['gt_agent_fut_trajs'])
                for idx in range(num_objs):
                    trajs = anns['gt_agent_fut_trajs'][idx]
                    start_point = anns['gt_boxes'][idx][:2]
                    bev_canvas = self._draw_ego_trajs_on_bev(bev_canvas, trajs, start_point=start_point, thickness=6, colormap='summer')
        
        if self.show_dt:
            # 障碍物
            if 'dt_boxes' in sample_info and self.show_agents:
                bev_canvas = self._draw_bbox_on_bev(bev_canvas, sample_info['dt_boxes'], sample_info.get('dt_instance_inds', None), color=(255, 0, 0))
            
            # maps
            if 'map_preds' in sample_info and self.show_maps:
                map_preds = sample_info['map_preds']
                for task in ['line', 'rm']:
                    if task in map_preds:
                        bev_canvas = self._draw_map_on_bev(bev_canvas, map_preds[task], task=task)
            
            # agent trajs
            if 'dt_agent_fut_trajs' in sample_info and self.show_agents:
                num_objs = len(sample_info['dt_agent_fut_trajs'])
                for idx in range(num_objs):
                    trajs = sample_info['dt_agent_fut_trajs'][idx]
                    start_point = sample_info['dt_boxes'][idx][:2]
                    bev_canvas = self._draw_ego_trajs_on_bev(bev_canvas, trajs, start_point=start_point, thickness=6, colormap='summer')

            if 'dt_ego_fut_trajs' in sample_info:
                bev_canvas = self._draw_ego_trajs_on_bev(bev_canvas, sample_info['dt_ego_fut_trajs'], thickness=8)

        return bev_canvas, pc_points

    def _draw_target_point_on_bev(self, bev_canvas, point):
        x, y = point
        bev_x, bev_y = self._lidar_point_to_bev_canvas(x, y)
        cv2.circle(bev_canvas, (int(bev_x), int(bev_y)), radius=32, color=(0, 0, 255), thickness=-1)
        return bev_canvas

    # ------------- render on camera -------------
    def _draw_bbox3d_on_camera(self, image, bboxes3d, sensor2lidar, cam_intrinsic, color):
        """
        将 3D 目标边界框转换为 2D 投影，并绘制到图像上。

        Args:
            image (np.ndarray): 输入图像 (H, W, 3)。
            bboxes3d (np.ndarray): 3D 边界框 (N, 7)，格式 (x, y, z, dx, dy, dz, yaw)。
            sensor2lidar (np.ndarray): 4x4 传感器到激光雷达的变换矩阵。
            cam_intrinsic (np.ndarray): 3x3 相机内参矩阵。
            color (tuple): 线条颜色 (B, G, R)，默认绿色。

        Returns:
            np.ndarray: 叠加 3D 目标框后的图像。
        """
        for box in bboxes3d:
            projected_corners = get_bbox_corners3d(box, cam_intrinsic, sensor2lidar, on_images=True, on_lidar=False, return_front_idx=True, pts_orders=self.pts_order)
            projected_corners, front_idx = projected_corners
            # 画 3D Box 的 12 条边
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # 上方 4 条边
                (4, 5), (5, 6), (6, 7), (7, 4),  # 下方 4 条边
                (0, 4), (1, 5), (2, 6), (3, 7)   # 连接上下的 4 条边
            ]
            front_color = [0, 0, 255]
            if len(projected_corners) == 8:
                for i, j in edges:
                    pt1 = tuple(projected_corners[i])
                    pt2 = tuple(projected_corners[j])
                    if (i in front_idx) and (j in front_idx):
                        tmp_color = front_color
                    else:
                        tmp_color = color
                    cv2.line(image, pt1, pt2, tmp_color, 3)
           
        return image

    def _draw_bbox2d(self, image, points, color=(0, 0, 255)):
        if points.shape[0] == 0:
            return image
        
        h, w, c = image.shape

        points[:, 0] = np.clip(points[:, 0], a_max=w, a_min=0)
        points[:, 1] = np.clip(points[:, 1], a_max=h, a_min=0)
        
        x1 = int(min(points[:, 0]))
        y1 = int(min(points[:, 1]))
        x2 = int(max(points[:, 0]))
        y2 = int(max(points[:, 1]))

        w = x2 - x1
        h = y2 - y1

        if w > 0 and h > 0:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

        return image

    def _draw_map_on_camera(self, image, maps, cam_intrinsic, cam2lidar, task='line'):
        img_h, img_w = image.shape[:-1]

        is_close = task == 'rm'
        for cls, map_polylines in maps.items():
            if task == 'line':
                color = LINE_COLORS[cls]
            else:
                color = ROAD_MARKER_COLORS[cls]

            for ployline in map_polylines:
                img_points = proj_lidar2img(ployline, cam_intrinsic, cam2lidar)  # [n, 2]
                # img_points = img_points[img_points[:, 2] > 0][:, :2].astype(np.int32)

                if task == 'rm':
                    image = self._draw_bbox2d(image, img_points, color)
                else:
                    # 只保留在图像范围内的点
                    valid_points = []
                    for pt in img_points:
                        x, y = pt
                        if 0 <= x < img_w and 0 <= y < img_h:
                            valid_points.append((int(x), int(y)))
                    
                    valid_points = np.array(valid_points)
                    cv2.polylines(image, [valid_points], isClosed=is_close, color=color, thickness=4)
        
        return image

    def _draw_ego_trajs_on_camera(self, img, trajs, cam_intrinsic, sensor2lidar, colormap='autumn', thickness=16):
        img_h, img_w, _ = img.shape

        x, y, z = np.linalg.inv(sensor2lidar)[:3, 3]
        x = 0
        y = 0

        # lidar -> image
        lidar_points = [[x, y, z]]
        for traj in trajs:
            x += traj[0]
            y += traj[1]
            lidar_points.append([x, y, z])

        lidar_points = np.array(lidar_points)
        img_points = proj_lidar2img(lidar_points, cam_intrinsic, sensor2lidar)
        # img_points = img_points[img_points[:, 2] > 0][:, :2].astype(np.int32)
        img_points[:, 0] = np.clip(img_points[:, 0], a_max=img_w, a_min=0)
        img_points[:, 1] = np.clip(img_points[:, 1], a_max=img_h, a_min=0)

        colors = create_color_maps(len(img_points), colormap=colormap)

        for idx in range(len(img_points) - 1):
            x1, y1 = [int(x) for x in img_points[idx]]
            x2, y2 = [int(x) for x in img_points[idx + 1]]
            cv2.line(img, (x1, y1), (x2, y2), color=colors[idx], thickness=thickness)
        return img
    
    def _draw_target_point_on_camera(self, img, point, cam_intrinsic, sensor2lidar):
        # 2d -> 3d
        lidar_point = np.array([
            point[0],
            point[1],
            np.linalg.inv(sensor2lidar)[2, 3]
        ])

        lidar_point = np.array(lidar_point)
        lidar_point = lidar_point.reshape([1, 3])

        img_point = proj_lidar2img(lidar_point, cam_intrinsic, sensor2lidar)
        # img_point = img_point[img_point[:, 2] > 0][:, :2].astype(np.int32)
        if img_point.shape[0]:
            x, y = int(img_point[0, 0]), int(img_point[0, 1])
            cv2.circle(img, (x, y), radius=16, color=(0, 0, 255), thickness=-1)

        return img

    def _draw_lidar_points_on_camera(self, img, lidar_points, cam_intrinsic, lidar2sensor, radius=1):
        H, W = img.shape[:-1]
        pts_2d = proj_lidar2img(lidar_points[:, :3], cam_intrinsic, lidar2sensor, result_dim=3)

        U = np.round(pts_2d[:, 0]).astype(np.int32)
        V = np.round(pts_2d[:, 1]).astype(np.int32)
        
        depths = pts_2d[:, 2]

        mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.1,
                ]
        )
        V, U, depths = V[mask], U[mask], depths[mask]
        # sort_idx = np.argsort(depths)[::-1]
        sort_idx = np.argsort(depths)
        V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]
        depths = np.clip(depths, 0.1, self.point_cloud_range[0])

        depth_map = np.ones([H, W], dtype=np.float32) * -1
        depth_map[V, U] = depths
        depth_map = depth_map.reshape(-1)

        norm_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color_map = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)

        mask = depth_map<0.1
        for pixel_id in range(0, len(depth_map), 1):
            if mask[pixel_id]:
                continue
            # depth = depth_map[pixel_id]
            # if depth == -1:
            #     continue
            y = int(pixel_id // W)
            x = int(pixel_id % W)
            color = color_map[pixel_id, 0, :].tolist()
            img = cv2.circle(img, (x, y), radius=radius, color=color, thickness=-1)  # 绘制点
        # pdb.set_trace()
        return img

    def _render_on_camera(self, sample_info, data_dir, lidar_points=None):
        cam_imgs = dict()

        for cam, cam_info in sample_info['cams'].items():
            cam_path = cam_info['data_path']
            if self.is_nuscenes:
                cam_path = cam_path.replace('./data/nuscenes/', '')
            img = cv2.imread(os.path.join(data_dir, cam_path))
            
            
            cam_intrinsic = get_matrix4x4(np.array(cam_info['cam_intrinsic']).reshape(3,3))
            # pdb.set_trace()
            if self.ref_cs_is_lidar:
                sensor2lidar_rotation = cam_info['sensor2lidar_rotation']
                sensor2lidar_translation = cam_info['sensor2lidar_translation']
            else:
                sensor2lidar_rotation = cam_info['sensor2ego_rotation']
                sensor2lidar_translation = cam_info['sensor2ego_translation']
            cam2lidar = get_matrix4x4_2(sensor2lidar_rotation, sensor2lidar_translation, quat=False, inverse=False)

            # 点云投影到图像上
            if lidar_points is not None and self.show_cam_lidar:
                img = self._draw_lidar_points_on_camera(img, lidar_points, cam_intrinsic, cam2lidar, radius=1)

            if self.show_gt:
                
                anns = sample_info['anns'] if 'anns' in sample_info.keys() else sample_info
                # 在图像上绘制障碍物
                if 'gt_boxes' in anns and self.show_agents:
                    img = self._draw_bbox3d_on_camera(img, anns['gt_boxes'], cam2lidar, cam_intrinsic, color=(100, 255, 100))

                # 在图像上绘制map
                if 'map_annos' in anns and self.show_maps and not self.is_nuscenes:
                    map_annos = anns['map_annos']
                    for task in ['line', 'rm']:
                        if task in map_annos:
                            img = self._draw_map_on_camera(img, map_annos[task], cam_intrinsic, cam2lidar, task)

                # 图像上绘制trajs
                if 'gt_ego_fut_path' in anns and not self.is_nuscenes:
                    if cam in ['CAM_FRONT', 'CAM_FRONT_NARROW']:
                        img = self._draw_ego_trajs_on_camera(img, anns['gt_ego_fut_path'], cam_intrinsic, cam2lidar, colormap='winter')

                if 'gt_ego_fut_trajs' in anns and not self.is_nuscenes:
                    if cam in ['CAM_FRONT', 'CAM_FRONT_NARROW']:
                        img = self._draw_ego_trajs_on_camera(img, anns['gt_ego_fut_trajs'], cam_intrinsic, cam2lidar, thickness=32)
            
            if self.show_dt:
                # 在图像上绘制障碍物
                if 'dt_boxes' in sample_info and self.show_agents:
                    img = self._draw_bbox3d_on_camera(img, sample_info['dt_boxes'], cam2lidar, cam_intrinsic, color=(255, 0, 0))

            cam_imgs[cam] = img

        return cam_imgs 

    def _stitch_image(self, cam_imgs):
        # 计算拼接后的高度 & 每张相机图片的高度
        if 'CAM_FRONT_NARROW' in cam_imgs:
            cam_img_h = self.bev_canvas_size[0] // 3
            canvas_h = cam_img_h * 3 
        else:
            cam_img_h = self.bev_canvas_size[0] // 2
            canvas_h = cam_img_h * 2 
        
        # 计算相机图片的宽度
        front_cam_h, front_cam_w = cam_imgs['CAM_FRONT'].shape[:-1]
        scale = cam_img_h / front_cam_h
        cam_img_w = int(front_cam_w * scale)

        # 保证每张图片高度一致
        for cam in cam_imgs:
            cam_imgs[cam] = cv2.resize(cam_imgs[cam], (cam_img_w, cam_img_h))
            cv2.putText(cam_imgs[cam], cam, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if 'CAM_FRONT_NARROW' in cam_imgs:
            tmp_img = np.ones([cam_img_h, cam_img_w, 3], dtype=np.uint8) * 255
            row1 = np.hstack([tmp_img, cam_imgs['CAM_FRONT_NARROW'], tmp_img])
            row2 = np.hstack([cam_imgs[cam] for cam in ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']])
            row3 = np.hstack([cam_imgs[cam] for cam in ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']])
            stitched_image = np.vstack([row1, row2, row3])  # 纵向拼接
        else:
            row1 = np.hstack([cam_imgs[cam] for cam in ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']])
            row2 = np.hstack([cam_imgs[cam] for cam in ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']])
            stitched_image = np.vstack([row1, row2])  # 纵向拼接
        return stitched_image, canvas_h

    def _stitch_navsim_image(self, cam_imgs):
        # 计算拼接后的高度 & 每张相机图片的高度
        # if 'CAM_FRONT_NARROW' in cam_imgs:
        if 'CAM_FRONT' in cam_imgs:
            cam_img_h = self.bev_canvas_size[0] // 3
            canvas_h = cam_img_h * 3 
        else:
            cam_img_h = self.bev_canvas_size[0] // 2
            canvas_h = cam_img_h * 2 

        # 计算相机图片的宽度
        front_cam_h, front_cam_w = cam_imgs['CAM_FRONT_WIDE'].shape[:-1]
        scale = cam_img_h / front_cam_h
        cam_img_w = int(front_cam_w * scale)

        # 保证每张图片高度一致
        for cam in cam_imgs:
            cam_imgs[cam] = cv2.resize(cam_imgs[cam], (cam_img_w, cam_img_h))
            cv2.putText(cam_imgs[cam], cam, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if 'CAM_FRONT' in cam_imgs:
            # pdb.set_trace()
            tmp_img = np.ones([cam_img_h, cam_img_w, 3], dtype=np.uint8) * 255
            row1 = np.hstack([tmp_img, cam_imgs['CAM_FRONT'], tmp_img])
            row2 = np.hstack([cam_imgs[cam] for cam in ['CAM_FRONT_LEFT_WIDE', 'CAM_FRONT_WIDE', 'CAM_FRONT_RIGHT_WIDE']])
            # row3 = np.hstack([cam_imgs[cam] for cam in ['CAM_BACK_LEFT_WIDE', 'CAM_BACK_WIDE', 'CAM_BACK_RIGHT_WIDE']])
            row3 = np.hstack([cam_imgs['CAM_BACK_LEFT_WIDE'], tmp_img, cam_imgs['CAM_BACK_RIGHT_WIDE']])
            stitched_image = np.vstack([row1, row2, row3])  # 纵向拼接
        else:
            row1 = np.hstack([cam_imgs[cam] for cam in ['CAM_FRONT_LEFT_WIDE', 'CAM_FRONT_WIDE', 'CAM_FRONT_RIGHT_WIDE']])
            # row2 = np.hstack([cam_imgs[cam] for cam in ['CAM_BACK_LEFT_WIDE', 'CAM_BACK_WIDE', 'CAM_BACK_RIGHT_WIDE']])
            tmp_img = np.ones([cam_img_h, cam_img_w, 3], dtype=np.uint8) * 255
            row2 = np.hstack([cam_imgs['CAM_BACK_LEFT_WIDE'], tmp_img, cam_imgs['CAM_BACK_RIGHT_WIDE']])
            stitched_image = np.vstack([row1, row2])  # 纵向拼接
        return stitched_image, canvas_h

    def _cat_camera_and_bev(self, cam_imgs, bev_canvas, sample_info):
        # pdb.set_trace()
        if 'CAM_FRONT_WIDE' in cam_imgs.keys():
            # stitch navsim, t4
            stitched_image, canvas_h  = self._stitch_navsim_image(cam_imgs)
        else:
            # stitch nuscenes
            stitched_image, canvas_h  = self._stitch_image(cam_imgs)
        
        # 拼接 BEV 图像
        bev_scale = canvas_h / bev_canvas.shape[0]
        bev_canvas = cv2.resize(bev_canvas, (int(self.bev_canvas_size[1] * bev_scale), canvas_h))
        
        stitched_image = np.hstack([stitched_image, bev_canvas])  # 横向拼接

        if sample_info is not None:
            cv2.putText(stitched_image, sample_info['token'], (10, 100), self.font,  2, (0, 0, 255), 2)

        return stitched_image.astype(np.uint8)

    def render(self, sample_info, data_dir):
        sample_info = self.filter_dt_by_score(sample_info)
        # render on bev
        bev_canvas, pc_points = self._render_on_bev(sample_info, data_dir)
        if self.show_bev_only:
            return bev_canvas
        
        # render on camera
        cam_imgs = self._render_on_camera(sample_info, data_dir, pc_points)
        
        # concate
        fig = self._cat_camera_and_bev(cam_imgs, bev_canvas, sample_info)
        return fig

    def render_scene(self, sample_info, draw_canvas=False):
        sample_info = self.filter_dt_by_score(sample_info)
        scene_token = sample_info['scene_token']
        cur_scene_token = getattr(self, 'scene_token', 'None')
        
        bev_canvas = None
        if (cur_scene_token != scene_token):
            self.obj_trajs = dict()
            self.obj_trajs['track_ego']=dict()
            self.obj_trajs['track_ego'][0] = []
            self.frame_in_scene = 0
            self.scene_token = scene_token
            if self.ref_cs_is_lidar:
                self.global2lidar = Visualizer.get_inverse_transform_mat(np.array(sample_info['lidar2global']))
            else:
                self.global2lidar = Visualizer.get_inverse_transform_mat(np.array(sample_info['ego2global']))
        
 

        self._update_trajs(sample_info)
        if draw_canvas:
            bev_canvas = self.show_trajs()
        return bev_canvas

    def _update_trajs(self, sample_info):
        def enrich_trajs(boxes_global, instance_inds, traj_key):
            assert instance_inds is not None
            if traj_key not in self.obj_trajs.keys():
                self.obj_trajs[traj_key] = dict()
            for i, idx in enumerate(instance_inds):
                if idx not in self.obj_trajs[traj_key].keys():
                    self.obj_trajs[traj_key][idx] = []
                self.obj_trajs[traj_key][idx].append(boxes_global[i])
        if self.ref_cs_is_lidar:
            lidar2global = sample_info['lidar2global']
        else:
            lidar2global = sample_info['ego2global']
        cur_lidar2init_lidar = self.global2lidar @ lidar2global
        
        ego_bb = np.array([0, 0, 0, 4,4,2, 0]).reshape(-1,7)
        ego_bb_global = Visualizer.transform_boxes3d(ego_bb, cur_lidar2init_lidar, inverse=False)
        self.obj_trajs['track_ego'][0].append(ego_bb_global[0])
        
        if self.show_gt:
            # 障碍物
            if 'gt_boxes' in sample_info and self.show_agents:
                # if not sample_info['key_frame']:
                #     return
                # print(sample_info['gt_names'])
                # # pdb.set_trace()
                # velc = sample_info['gt_velocity']
                # mask = (velc*velc).sum(1) < 0.1
                # if mask.sum()==0:
                #     return
                # pdb.set_trace()
                instance_inds = sample_info['instance_inds']
                gt_boxes_scene = Visualizer.transform_boxes3d(sample_info['gt_boxes'], cur_lidar2init_lidar, inverse=False)
                enrich_trajs(gt_boxes_scene, instance_inds, 'track_gt')
                
        if self.show_dt:
            # 障碍物
            if 'dt_boxes' in sample_info and self.show_agents:
                dt_instance_inds = sample_info.get('dt_instance_inds', None)
                if dt_instance_inds is not None:
                    dt_boxes_scene = Visualizer.transform_boxes3d(sample_info['dt_boxes'], cur_lidar2init_lidar, inverse=False)
                    enrich_trajs(dt_boxes_scene, dt_instance_inds, 'track_dt')



    def _show_grajs(self, trajs):
        def _expand_area(traj, pcr):
            # left right
            x = np.max(np.abs(traj[:, 0]))
            if x>pcr[2]:
                pcr[2] = x
                pcr[3] = x
            # back
            y1 = 0.0-np.min(traj[:, 1])
            if y1>pcr[1]:
                pcr[1] = y1
            # front
            y2 = np.max(traj[:, 1])
            if y2>pcr[0]:
                pcr[0] = y2
            return pcr
        
        pcr = list(self.point_cloud_range_org)

        for idx in trajs.keys():
            trajs[idx] = np.stack(trajs[idx], axis=0)
            pcr = _expand_area(trajs[idx], pcr)
        pcr = [int(v) for v in pcr]
        bev_canvas = self._create_bev_canvas(pcr)
        for idx, traj in trajs.items():
            instance = np.zeros((traj.shape[0],), dtype=np.int32) + idx
            color = [np.random.randint(0, 255) for i in range(3)]
            bev_canvas = self._draw_bbox_on_bev(bev_canvas, traj, instance_inds=instance, color=color)
        return bev_canvas
        

    def show_trajs(self):
        # pdb.set_trace()
        canvas = []
        if args.show_dt:
            bev_canvas_dt = self._show_grajs(self.obj_trajs['track_dt'])
            cv2.putText(bev_canvas_dt, 'track_dt', (10, 10), cv2.FONT_HERSHEY_COMPLEX, self.font_sclae*4, (0,0,255), thickness=self.font_thickness)
            canvas.append(bev_canvas_dt)
        if args.show_gt:
            bev_canvas_gt = self._show_grajs(self.obj_trajs['track_gt'])
            cv2.putText(bev_canvas_gt, 'track_gt', (10, 10), cv2.FONT_HERSHEY_COMPLEX, self.font_sclae*4, (0,0,255), thickness=self.font_thickness)
            canvas.append(bev_canvas_gt)
        bev_canvas_ego = self._show_grajs(self.obj_trajs['track_ego'])
        cv2.putText(bev_canvas_ego, 'track_ego', (10, 10), cv2.FONT_HERSHEY_COMPLEX, self.font_sclae*4, (0,0,255), thickness=self.font_thickness)
        canvas.append(bev_canvas_ego)
        if len(canvas)==0:
            return None
        if len(canvas)>1:
            canvas = np.concatenate(canvas, axis=1)
        else:
            canvas = canvas[0]
        return canvas
        

    def filter_dt_by_score(self, sample_info):
        if self.show_dt:
            if 'score' in sample_info.keys():
                mask = sample_info['score'] > self.od_score
                sample_info['score'] = sample_info['score'][mask]
                sample_info['boxes_lidar'] = sample_info['boxes_lidar'][mask]
                dt_instance_inds = sample_info.get('dt_instance_inds', None)
                if dt_instance_inds is not None:
                    sample_info['dt_instance_inds'] = dt_instance_inds[mask]
                # sample_info['dt_instance_inds'] = None
        return self.dt_key_remap(sample_info)

    def dt_key_remap(self, sample_info):
        map_keys = {
            # od
            'boxes_lidar':'dt_boxes', 'score': 'dt_scores', 'label':'dt_labels', 'dt_instance_inds':'dt_instance_inds',
        }
        for k, v in map_keys.items():
            if k in sample_info.keys():
                sample_info[v] = sample_info.pop(k)
        return sample_info
        
    def dump_images(self, fig, save_dir, sample_token):
        if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f'{save_dir}/{sample_token}.png', fig)

    def dump_video(self, fig, save_dir):
        if save_dir[-4:] in ['.mp4', '.avi']:
            tmp_dir = os.path.split(save_dir)[0]
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir, exist_ok=True)

        if not hasattr(self, 'video'):
            h, w = fig.shape[:2]
            self.video = cv2.VideoWriter(save_dir, cv2.VideoWriter_fourcc(*'mp4v'), int(self.fps), (w, h))
        self.video.write(fig)

    def dump_clip_video(self, fig, save_dir, scene_token):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        clip_scene_token = getattr(self, 'clip_scene_token', None)
        if (clip_scene_token is None) or (clip_scene_token != scene_token):
            self.release()
            h, w = fig.shape[:2]
            # clip_video_path = os.path.join(save_dir, scene_token+'.mp4')
            # self.video = cv2.VideoWriter(clip_video_path, cv2.VideoWriter_fourcc(*'mp4v'), int(self.fps), (w, h))
            clip_video_path = os.path.join(save_dir, scene_token+'.avi')
            self.video = cv2.VideoWriter(clip_video_path, cv2.VideoWriter_fourcc(*'XVID'), int(self.fps), (w, h))
            self.clip_scene_token = scene_token
        self.video.write(fig)
        

    def dump(self, fig, save_dir, sample_token, scene_token=None):
        # resize
        img_h, img_w = fig.shape[:-1]
        fig = cv2.resize(fig, (int(img_w * 0.5), int(img_h * 0.5)))
        
        if 'images' in self.format:
            self.dump_images(fig, save_dir, sample_token)
        if 'video' in self.format:
            self.dump_video(fig, save_dir)
        if 'clip_video' in self.format:
            self.dump_clip_video(fig, save_dir, scene_token)
   
            
    def release(self):
        if hasattr(self, 'video'):
            self.video.release()
            del self.video

def load_extra_info(info, root_dir):
    extra_info = dict()
    if 'filepath' in info.keys():
        filepath = os.path.join(root_dir, info['filepath'])
        extra_info = pickle.load(open(filepath, 'rb'))
    return extra_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_file', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--extra_path', type=str, default='pkls_nuscenes_10hz_raw_lidar')
    
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--nuscenes', action='store_true')
    parser.add_argument('--pts_order', type=str, default='rfu', help='rfu: 右前上, flu: 前左上')
    

    parser.add_argument('--show_gt', action='store_true')
    parser.add_argument('--show_dt', action='store_true')
    parser.add_argument('--show_axis', action='store_true')
    parser.add_argument('--show_bev_only', action='store_true')
    parser.add_argument('--show_bev_lidar', action='store_true')
    parser.add_argument('--show_cam_lidar', action='store_true')
    parser.add_argument('--show_maps', action='store_true')
    parser.add_argument('--show_agents', action='store_true')
    parser.add_argument('--show_scene', action='store_true', default=False)
    parser.add_argument('--ref_cs', type=str, default='lidar')
    parser.add_argument('--od_score', type=float, default=0.001)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--skip_frame', type=int, default=1)
    parser.add_argument('--pt_dim', type=int, default=4)
    parser.add_argument('--format', type=str, nargs='+', choices=['images', 'clip_video', 'video'], default='clip_video')
    parser.add_argument('--show_key_frame', action='store_true')
    args = parser.parse_args()

    # if args.show_scene:
    #     assert 'clip_video' in args.format
    # point_cloud_range=(201, 101, 101, 101) 
    point_cloud_range=(101, 101, 101, 101)
    visualizer = Visualizer(show_gt=args.show_gt,
                            show_dt=args.show_dt,
                            is_nuscenes=args.nuscenes, 
                            pts_order=args.pts_order,
                            show_axis=args.show_axis, 
                            show_agents=args.show_agents, 
                            show_bev_only=args.show_bev_only,
                            show_bev_lidar=args.show_bev_lidar,
                            show_cam_lidar=args.show_cam_lidar,
                            show_maps=args.show_maps,
                            od_score=args.od_score,
                            format=args.format,
                            fps=args.fps,
                            point_cloud_range=point_cloud_range,
                            ref_cs_is_lidar=args.ref_cs=='lidar',
                            pt_dim=args.pt_dim
                            )
    
    # load pkl
    # infos = mmcv.load(args.pkl_file, file_format="pkl")   # 加载pkl
    infos = pickle.load(open(args.pkl_file, 'rb'))

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(args.pkl_file), "vis")
    print('vis results are saved in ', save_dir)
    if isinstance(infos, dict):
        if 'infos' in infos.keys():
            infos = infos['infos']
    bar = tqdm(range(len(infos)))
    scene_list = set()
    for index in bar:
        if index % args.skip_frame > 0:
            continue
        info = infos[index]
        if args.show_key_frame and (not info['key_frame']):
            continue
        scene_token = info['scene_token']
        # pdb.set_trace()
        # if scene_token not in ['20260121-11-00-44_final_output_4']:
        #     continue
        scene_list.add(scene_token)

        bar.set_description(desc='{}: {}'.format(scene_token, len(scene_list)))
     
        if 'filepath' in info.keys():
            info.update(load_extra_info(info, root_dir=os.path.join(args.data_dir, args.extra_path)))
        # pdb.set_trace()
        sample_token = info['token']
        if args.show_scene:
            if (index + 1 == len(infos)):
                draw_canvas = True
            elif (infos[index]['scene_token'] != infos[index+1]['scene_token']):
                draw_canvas = True
            else:
                draw_canvas = False
            
            result_fig = visualizer.render_scene(info, draw_canvas=draw_canvas)   # 可视化
            if result_fig is not None:
                visualizer.dump(result_fig, save_dir, info['scene_token'], scene_token=info['scene_token'])  # 保存
        else:
            result_fig = visualizer.render(info, args.data_dir)   # 可视化
            img_fn = str(index).rjust(5, '0')
            visualizer.dump(result_fig, save_dir, f'{img_fn}_{sample_token}', scene_token=info['scene_token'])  # 保存
    visualizer.release()
    print('######### vis Done ##########')